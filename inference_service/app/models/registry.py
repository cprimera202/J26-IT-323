"""Model registry: loads each trained model independently and tracks real readiness.

Rules enforced here:
* checkpoints are loaded with strict=True (never strict=False)
* a model that fails to load is marked "unavailable" with a sanitized reason
* no fabricated outputs -- predict() raises if the model is not ready
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

from ..config import Settings
from .custom_architectures import CUSTOM_BUILDERS

log = logging.getLogger("dr.registry")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GRADES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# code -> definition. Thresholds/colours here are the documented defaults; metadata wins for threshold.
LESIONS: dict[str, dict[str, Any]] = {
    "MA": dict(name="Microaneurysms", file="MA/MA_BEST_deeplabv3plus.pth", meta="MA/MA_BEST_model_metadata.json",
               arch="deeplabv3plus", encoder="resnet50", threshold=0.50, color=[255, 0, 0]),
    "HE": dict(name="Haemorrhages", file="HE/HE_BEST_deeplabv3plus.pth", meta="HE/HE_BEST_model_metadata.json",
               arch="deeplabv3plus", encoder="resnet50", threshold=0.55, color=[0, 128, 255]),
    "EX": dict(name="Hard Exudates", file="EX/EX_BEST_unet.pth", meta="EX/EX_BEST_model_metadata.json",
               arch="unet", encoder="resnet34", threshold=0.80, color=[255, 215, 0]),
    "SE": dict(name="Soft Exudates", file="SE/SE_BEST_unetplusplus.pth", meta="SE/SE_BEST_model_metadata.json",
               arch="unetplusplus", encoder="resnet34", threshold=0.65, color=[0, 200, 0]),
}

CLASSIFICATION_FILES = ("fundus_classification/DR/dr_grading.pth", "fundus_classification/DR/grading_config.json")


class ModelUnavailable(Exception):
    def __init__(self, key: str, reason: str):
        super().__init__(reason)
        self.key = key
        self.reason = reason


@dataclass
class Slot:
    key: str
    label: str
    status: str = "unavailable"  # "ready" | "unavailable"
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    predictor: Any = None
    load_seconds: float | None = None

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "ready": self.status == "ready",
            "error": self.error,
            **self.details,
        }


# ----------------------------------------------------------------------------- helpers
def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _meta_get(meta: dict, aliases: tuple[str, ...], default=None):
    """Look up the first matching alias at top level, then one level of nesting."""
    for a in aliases:
        if a in meta and meta[a] is not None:
            return meta[a]
    for v in meta.values():
        if isinstance(v, dict):
            for a in aliases:
                if a in v and v[a] is not None:
                    return v[a]
    return default


def _hw(value, fallback: int | None) -> tuple[int, int] | None:
    if value is None:
        return (fallback, fallback) if fallback else None
    if isinstance(value, (int, float)):
        return int(value), int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])  # (H, W)
    return (fallback, fallback) if fallback else None



def _extract_state_dict(
    obj: Any,
    state_dict_key: str | None = None,
) -> dict[str, torch.Tensor]:
    # Select the explicitly configured model from a multi-model checkpoint.
    if state_dict_key and isinstance(obj, dict) and state_dict_key in obj:
        obj = obj[state_dict_key]

    # Support common checkpoint wrappers.
    elif isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict) or not obj:
        raise ValueError("Checkpoint does not contain a recognisable state_dict")

    if not all(torch.is_tensor(value) for value in obj.values()):
        raise ValueError(
            "Selected checkpoint entry is not a flat tensor state_dict"
        )

    state = dict(obj)

    # Remove a consistent wrapper prefix, when present.
    for prefix in ("module.", f"{state_dict_key}." if state_dict_key else ""):
        if prefix and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}

    return state



def _load_checkpoint(
    path: Path,
    state_dict_key: str | None = None,
) -> dict[str, torch.Tensor]:
    allow_pickle = os.environ.get("ALLOW_UNSAFE_PICKLE", "0") == "1"

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_pickle:
            raise RuntimeError(
                "Checkpoint could not be loaded safely. "
                "Only enable unsafe pickle loading for trusted files. "
                + str(exc).splitlines()[0][:200]
            ) from exc

        raw = torch.load(path, map_location="cpu", weights_only=False)

    return _extract_state_dict(raw, state_dict_key)



def _strict_load(model: nn.Module, state: dict[str, torch.Tensor], hint: str) -> None:
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        own = set(model.state_dict().keys())
        theirs = set(state.keys())
        missing = sorted(own - theirs)
        unexpected = sorted(theirs - own)
        msg = (
            f"Checkpoint does not match the reconstructed architecture "
            f"({len(missing)} missing keys, {len(unexpected)} unexpected keys). "
            f"e.g. missing={missing[:3]} unexpected={unexpected[:3]}. {hint}"
        )
        raise RuntimeError(msg) from exc


def _pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize(img: Image.Image, size_hw: tuple[int, int], mean, std) -> torch.Tensor:
    h, w = size_hw
    arr = cv2.resize(np.asarray(img, dtype=np.uint8), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


# ----------------------------------------------------------------------------- classification
class Classifier:
    def __init__(self, model: nn.Module, device: torch.device, cfg: dict):
        self.model = model.eval().to(device)
        self.device = device
        self.cfg = cfg

    def predict(self, img: Image.Image) -> dict[str, Any]:
        cfg = self.cfg
        x = _normalize(img, cfg["size_hw"], cfg["mean"], cfg["std"]).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite outputs")
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy().astype(float)
        grade = int(probs.argmax())
        confidence = float(probs[grade])
        min_grade = cfg["referable_min_grade"]
        p_ref = float(probs[min_grade:].sum())
        if cfg["referable_prob_threshold"] is not None:
            referable = p_ref >= cfg["referable_prob_threshold"]
            basis = f"P(grade >= {min_grade}) = {p_ref:.3f} vs threshold {cfg['referable_prob_threshold']}"
        else:
            referable = grade >= min_grade
            basis = f"predicted grade {grade} vs referable from grade {min_grade}"
        names = cfg["class_names"]
        return {
            "grade": grade,
            "class_name": names[grade],
            "confidence": confidence,
            "probabilities": [{"grade": i, "class_name": names[i], "probability": float(p)} for i, p in enumerate(probs)],
            "referable_dr": bool(referable),
            "referable_basis": basis,
            "referable_probability": p_ref,
            "low_confidence": confidence < cfg["low_confidence_threshold"],
            "low_confidence_threshold": cfg["low_confidence_threshold"],
        }


def _load_classifier(settings: Settings, device: torch.device) -> tuple[Classifier, dict]:
    ckpt = settings.model_store / CLASSIFICATION_FILES[0]
    cfg_path = settings.model_store / CLASSIFICATION_FILES[1]
    if not ckpt.is_file():
        raise ModelUnavailable("classification", "Checkpoint file dr_grading.pth not found in model store.")
    if not cfg_path.is_file():
        raise ModelUnavailable("classification", "Config file grading_config.json not found in model store.")

    raw_cfg = _read_json(cfg_path)
    assumptions: list[str] = []

    names = _meta_get(raw_cfg, ("class_names", "classes", "labels", "grade_names"))
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    if not (isinstance(names, list) and names):
        names = DEFAULT_GRADES
        assumptions.append("class names not in config; using standard DR grade names")
    names = [str(n) for n in names]

    size_hw = _hw(_meta_get(raw_cfg, ("image_size", "input_size", "img_size", "resize")), None)
    if size_hw is None:
        size_hw = (224, 224)
        assumptions.append("input size not in config; using 224x224")
    mean = _meta_get(raw_cfg, ("mean", "imagenet_mean", "normalize_mean"), IMAGENET_MEAN)
    std = _meta_get(raw_cfg, ("std", "imagenet_std", "normalize_std"), IMAGENET_STD)

    min_grade = _meta_get(raw_cfg, ("referable_min_grade", "referable_grade", "referable_threshold_grade"))
    if min_grade is None:
        min_grade = 2
        assumptions.append("referable grade not in config; using grade >= 2 (moderate or worse)")
    prob_thr = _meta_get(raw_cfg, ("referable_prob_threshold", "referable_threshold"))
    if isinstance(prob_thr, (int, float)) and 0 < float(prob_thr) < 1:
        prob_thr = float(prob_thr)
    else:
        if isinstance(prob_thr, (int, float)) and float(prob_thr) >= 1 and "referable_min_grade" not in raw_cfg:
            min_grade = int(prob_thr)  # an integer "threshold" is read as a grade cut-off
        prob_thr = None
    low_conf = _meta_get(raw_cfg, ("low_confidence_threshold", "confidence_threshold", "low_confidence"))
    if not isinstance(low_conf, (int, float)) or not 0 < float(low_conf) < 1:
        low_conf = 0.60
        assumptions.append("low-confidence threshold not in config; using 0.60")

    state_key = raw_cfg.get("state_dict_key")
    state = _load_checkpoint(ckpt, state_dict_key=state_key)
    # Infer head shape from the checkpoint instead of assuming it.
    if "fc.weight" in state:
        n_out, head = state["fc.weight"].shape[0], "linear"
    elif "fc.1.weight" in state:
        n_out, head = state["fc.1.weight"].shape[0], "dropout_linear"
    else:
        raise RuntimeError("Checkpoint has no recognisable ResNet50 'fc' head (expected fc.weight or fc.1.weight).")
    if n_out != len(names):
        raise RuntimeError(f"Checkpoint has {n_out} outputs but config defines {len(names)} classes.")

    model = torchvision.models.resnet50(weights=None)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, n_out) if head == "linear" else nn.Sequential(nn.Dropout(0.0), nn.Linear(in_f, n_out))
    _strict_load(model, state, "The classifier must be a torchvision ResNet50 with a replaced fc head.")

    cfg = dict(size_hw=size_hw, mean=list(mean), std=list(std), class_names=names,
               referable_min_grade=int(min_grade), referable_prob_threshold=prob_thr,
               low_confidence_threshold=float(low_conf))
    details = {
        "architecture": "resnet50", "num_classes": n_out, "input_size": list(size_hw),
        "class_names": names, "referable_min_grade": int(min_grade),
        "referable_prob_threshold": prob_thr, "low_confidence_threshold": float(low_conf),
        "assumptions": assumptions,
    }
    return Classifier(model, device, cfg), details


# ----------------------------------------------------------------------------- segmentation
class Segmenter:
    def __init__(self, code: str, model: nn.Module, device: torch.device, cfg: dict):
        self.code = code
        self.model = model.eval().to(device)
        self.device = device
        self.cfg = cfg

    def predict(self, img: Image.Image) -> dict[str, Any]:
        cfg = self.cfg
        w0, h0 = img.size
        x = _normalize(img, cfg["size_hw"], cfg["mean"], cfg["std"]).to(self.device)
        with torch.inference_mode():
            out = self.model(x)
        if out.ndim != 4 or out.shape[1] != 1:
            raise RuntimeError(f"Unexpected output shape {tuple(out.shape)}; expected (1, 1, H, W)")
        if not torch.isfinite(out).all():
            raise RuntimeError("Model produced non-finite outputs")
        prob = torch.sigmoid(out.float())[0, 0].cpu().numpy()
        prob_full = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)  # back to original size
        mask = (prob_full >= cfg["threshold"]).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG", optimize=True)
        positive = int((mask > 0).sum())
        return {
            "mask_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": w0, "height": h0,
            "threshold": cfg["threshold"],
            "positive_pixels": positive,
            "positive_fraction": positive / float(w0 * h0),
            "max_probability": float(prob_full.max()),
        }


def _build_segmentation_arch(code: str, arch: str, encoder: str, meta: dict) -> tuple[nn.Module, str]:
    if code in CUSTOM_BUILDERS:
        return CUSTOM_BUILDERS[code](meta), "custom (custom_architectures.py)"
    import segmentation_models_pytorch as smp

    if arch == "deeplabv3plus":
        kwargs: dict[str, Any] = dict(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
        stride = _meta_get(meta, ("encoder_output_stride", "output_stride"))
        if stride in (8, 16):
            kwargs["encoder_output_stride"] = int(stride)  # passed ONCE, via the supported argument
        return smp.DeepLabV3Plus(**kwargs), f"smp.DeepLabV3Plus({encoder})"
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1), f"smp.Unet({encoder}) [standard]"
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1), f"smp.UnetPlusPlus({encoder}) [standard]"
    raise RuntimeError(f"Unsupported architecture '{arch}'.")


def _load_segmenter(code: str, settings: Settings, device: torch.device) -> tuple[Segmenter, dict]:
    d = LESIONS[code]
    base = settings.model_store / "fundus_segmentation"
    ckpt, meta_path = base / d["file"], base / d["meta"]
    if not ckpt.is_file():
        raise ModelUnavailable(code, f"Checkpoint {Path(d['file']).name} not found in model store.")
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    assumptions: list[str] = []
    if not meta_path.is_file():
        assumptions.append("metadata file missing; using documented defaults")

    arch = str(_meta_get(meta, ("architecture", "arch", "model_name", "model"), d["arch"]) or d["arch"]).lower().replace("-", "").replace("_", "").replace("+", "plus")
    if "deeplab" in arch:
        arch = "deeplabv3plus"
    elif "unetplusplus" in arch or "unetpp" in arch:
        arch = "unetplusplus"
    elif "unet" in arch:
        arch = "unet"
    else:
        arch = d["arch"]
    encoder = str(_meta_get(meta, ("encoder", "encoder_name", "backbone"), d["encoder"]) or d["encoder"]).lower()

    thr = _meta_get(meta, ("threshold", "best_threshold", "optimal_threshold", "thr"))
    if not isinstance(thr, (int, float)) or not 0 < float(thr) < 1:
        thr = d["threshold"]
        assumptions.append(f"threshold not in metadata; using documented default {thr}")
    size_hw = _hw(_meta_get(meta, ("input_size", "image_size", "img_size", "resolution", "input_resolution", "resize")), None)
    if size_hw is None:
        size_hw = (512, 512)
        assumptions.append("input size not in metadata; using 512x512 (verify against training code)")
    if size_hw[0] % 32 or size_hw[1] % 32:
        raise RuntimeError(f"Input size {size_hw} is not divisible by 32 as the encoder requires.")
    mean = _meta_get(meta, ("mean", "normalize_mean"), IMAGENET_MEAN)
    std = _meta_get(meta, ("std", "normalize_std"), IMAGENET_STD)

    state = _load_checkpoint(ckpt)
    model, arch_desc = _build_segmentation_arch(code, arch, encoder, meta)
    _strict_load(
        model, state,
        "Run scripts/inspect_checkpoints.py, then add the original training class to "
        "app/models/custom_architectures.py (see the README).",
    )
    cfg = dict(size_hw=size_hw, mean=list(mean), std=list(std), threshold=float(thr))
    details = {
        "code": code, "name": d["name"], "architecture": arch_desc, "encoder": encoder,
        "threshold": float(thr), "input_size": list(size_hw), "color": d["color"], "assumptions": assumptions,
    }
    return Segmenter(code, model, device, cfg), details


# ----------------------------------------------------------------------------- registry
class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = _pick_device(settings.device_preference)
        self.slots: dict[str, Slot] = {"classification": Slot("classification", "Fundus DR classification")}
        for code, d in LESIONS.items():
            self.slots[code] = Slot(code, f"{d['name']} segmentation",
                                    details={"code": code, "name": d["name"], "threshold": d["threshold"], "color": d["color"]})

    def _sanitize(self, msg: str) -> str:
        return msg.replace(str(self.settings.model_store), "<model_store>")

    def load_all(self) -> None:
        log.info("Loading models (device=%s)", self.device)
        self._load("classification", lambda: _load_classifier(self.settings, self.device))
        for code in LESIONS:
            self._load(code, lambda c=code: _load_segmenter(c, self.settings, self.device))
        for s in self.slots.values():
            log.info("model %-14s %s%s", s.key, s.status, f" -- {s.error}" if s.error else "")

    def _load(self, key: str, fn) -> None:
        slot = self.slots[key]
        t0 = time.perf_counter()
        try:
            predictor, details = fn()
            slot.predictor, slot.details, slot.status, slot.error = predictor, {**slot.details, **details}, "ready", None
        except ModelUnavailable as exc:
            slot.status, slot.error, slot.predictor = "unavailable", self._sanitize(exc.reason), None
        except Exception as exc:  # noqa: BLE001 -- any load failure => model unavailable
            slot.status, slot.error, slot.predictor = "unavailable", self._sanitize(f"{type(exc).__name__}: {exc}")[:600], None
        slot.load_seconds = round(time.perf_counter() - t0, 2)

    def get_ready(self, key: str) -> Any:
        slot = self.slots[key]
        if slot.status != "ready" or slot.predictor is None:
            raise ModelUnavailable(key, slot.error or "Model is not ready.")
        return slot.predictor

    def status(self) -> dict[str, Any]:
        slots = {k: s.public() for k, s in self.slots.items()}
        seg = [s for k, s in self.slots.items() if k in LESIONS]
        return {
            "device": str(self.device),
            "classification": slots["classification"],
            "segmentation": {k: slots[k] for k in LESIONS},
            "all_ready": all(s.status == "ready" for s in self.slots.values()),
            "any_segmentation_ready": any(s.status == "ready" for s in seg),
            "ready_count": sum(s.status == "ready" for s in self.slots.values()),
            "total_count": len(self.slots),
        }
