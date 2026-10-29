"""Per-model verification against the REAL trained checkpoints.

For each model: checkpoint exists -> architecture rebuilt -> strict load -> forward pass ->
output shape/finite -> postprocessed mask matches the input image size -> READY / NOT READY.

Usage (PowerShell, from inference_service):
    py scripts\\verify_models.py --image "C:\\path\\to\\fundus.jpg"
    py scripts\\verify_models.py --model-store "C:\\...\\Backend\\model_store" --image "C:\\...\\fundus.png"
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-store", default=None)
    ap.add_argument("--image", required=True, help="a real retinal fundus image")
    args = ap.parse_args()
    if args.model_store:
        os.environ["MODEL_STORE_DIR"] = args.model_store

    from app.config import Settings
    from app.models.registry import LESIONS, ModelRegistry

    settings = Settings.load()
    reg = ModelRegistry(settings)
    reg.load_all()
    img = Image.open(args.image).convert("RGB")
    print(f"model store: {settings.model_store}\ndevice: {reg.device}\nimage: {img.size[0]}x{img.size[1]}\n")

    rows = []
    for key, slot in reg.slots.items():
        row = {"model": key, "loaded": slot.status == "ready", "forward": "-", "postprocess": "-", "note": slot.error or ""}
        if slot.status == "ready":
            try:
                out = slot.predictor.predict(img)
                if key == "classification":
                    probs = [p["probability"] for p in out["probabilities"]]
                    ok = all(math.isfinite(p) for p in probs) and abs(sum(probs) - 1) < 1e-3
                    row.update(forward="ok" if ok else "FAIL", postprocess=f"grade {out['grade']} ({out['confidence']:.2f})")
                else:
                    mask = np.array(Image.open(io.BytesIO(base64.b64decode(out["mask_png_base64"]))))
                    ok = mask.shape == (img.size[1], img.size[0]) and set(np.unique(mask)) <= {0, 255}
                    row.update(forward="ok", postprocess=("mask aligned to image size" if ok else "MASK SIZE/FORMAT WRONG")
                               + f", {out['positive_fraction']*100:.3f}% positive @thr {out['threshold']}")
            except Exception as exc:  # noqa: BLE001
                row.update(forward="FAIL", note=f"{type(exc).__name__}: {exc}")
        row["READY"] = row["loaded"] and row["forward"] == "ok"
        rows.append(row)

    for r in rows:
        print(f"[{'READY' if r['READY'] else 'NOT READY':9s}] {r['model']:15s} loaded={r['loaded']!s:5s} forward={r['forward']:5s} {r['postprocess']}")
        if r["note"]:
            print(f"             {r['note']}")
    return 0 if all(r["READY"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
