"""Test fixtures. Real (randomly initialised) checkpoints are generated in a temp dir, so the loading,
strict-matching, forward-pass and postprocessing code paths are exercised without your trained weights.
Your real weights are verified separately with scripts/verify_models.py."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
smp = pytest.importorskip("segmentation_models_pytorch")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def make_settings(store: Path, max_mb: int = 15) -> Settings:
    return Settings(model_store=store, max_upload_bytes=max_mb * 1024 * 1024, min_image_side=128,
                    max_image_side=8000, device_preference="cpu", allowed_origins=("http://localhost:3000",))


def fundus_like(w: int = 300, h: int = 240, fmt: str = "PNG") -> bytes:
    """Synthetic red/orange disc on a black background (passes the plausibility heuristic)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(w, h) * 0.47) ** 2
    arr = np.zeros((h, w, 3), np.uint8)
    arr[disc] = (200, 110, 45)
    arr = np.clip(arr.astype(int) + rng.integers(-12, 12, arr.shape) * disc[..., None], 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def solid(color, size=(300, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def full_store(tmp_path_factory) -> Path:
    """Model store with a classifier (ResNet50) and MA/HE DeepLabV3+ (ResNet18 encoder) checkpoints."""
    root = tmp_path_factory.mktemp("model_store")
    cls_dir = root / "fundus_classification" / "DR"
    cls_dir.mkdir(parents=True)
    m = torchvision.models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 5)
    torch.save(m.state_dict(), cls_dir / "dr_grading.pth")
    (cls_dir / "grading_config.json").write_text(json.dumps({
        "image_size": 224, "class_names": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "referable_min_grade": 2,
    }))
    for code, name, thr in (("MA", "MA", 0.5), ("HE", "HE", 0.55)):
        d = root / "fundus_segmentation" / code
        d.mkdir(parents=True)
        seg = smp.DeepLabV3Plus(encoder_name="resnet18", encoder_weights=None, classes=1)
        torch.save(seg.state_dict(), d / f"{name}_BEST_deeplabv3plus.pth")
        (d / f"{name}_BEST_model_metadata.json").write_text(json.dumps({
            "architecture": "DeepLabV3+", "encoder": "resnet18", "threshold": thr, "input_size": 64}))
    # EX: checkpoint with custom (non-SMP) keys -> must be reported unavailable, never loaded loosely
    d = root / "fundus_segmentation" / "EX"
    d.mkdir(parents=True)
    torch.save({"up4.conv.weight": torch.zeros(4, 4, 3, 3), "head.weight": torch.zeros(1, 4, 1, 1)}, d / "EX_BEST_unet.pth")
    (d / "EX_BEST_model_metadata.json").write_text(json.dumps({"encoder": "resnet18", "threshold": 0.8, "input_size": 64}))
    return root


@pytest.fixture(scope="session")
def client(full_store):
    with TestClient(create_app(make_settings(full_store))) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c
