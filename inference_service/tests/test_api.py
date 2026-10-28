from __future__ import annotations

import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from tests.conftest import fundus_like, make_settings, solid


def up(data: bytes, name="x.png", ctype="image/png"):
    return {"file": (name, data, ctype)}


# ---------------------------------------------------------------- health / readiness
def test_health_is_independent_of_models(empty_client):
    r = empty_client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_false_when_models_missing_and_no_path_leak(empty_client, tmp_path):
    r = empty_client.get("/ready")
    assert r.status_code == 503 and r.json()["ready"] is False
    assert str(tmp_path) not in r.text
    models = empty_client.get("/models").json()
    assert models["classification"]["ready"] is False
    assert all(not v["ready"] for v in models["segmentation"].values())


def test_models_report_per_model_status(client):
    m = client.get("/models").json()
    assert m["classification"]["ready"] is True
    assert m["segmentation"]["MA"]["ready"] is True and m["segmentation"]["HE"]["ready"] is True
    assert m["segmentation"]["SE"]["ready"] is False  # file absent
    assert m["all_ready"] is False
    assert client.get("/ready").status_code == 503


def test_mismatched_checkpoint_is_unavailable_not_loaded_loosely(client):
    ex = client.get("/models").json()["segmentation"]["EX"]
    assert ex["ready"] is False
    assert "does not match" in ex["error"] and "missing keys" in ex["error"]


# ---------------------------------------------------------------- missing models
def test_classify_returns_503_when_model_missing(empty_client):
    r = empty_client.post("/v1/fundus/classify", files=up(fundus_like()))
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_segment_503_when_no_requested_model_ready(client):
    r = client.post("/v1/fundus/segment?lesions=EX,SE", files=up(fundus_like()))
    assert r.status_code == 503
    body = r.json()
    assert body["lesions"]["EX"]["status"] == "unavailable"


# ---------------------------------------------------------------- classification
def test_classification_output_format(client):
    r = client.post("/v1/fundus/classify", files=up(fundus_like(), "f.png"))
    assert r.status_code == 200, r.text
    body = r.json()
    res = body["result"]
    assert 0 <= res["grade"] <= 4 and res["class_name"]
    probs = [p["probability"] for p in res["probabilities"]]
    assert len(probs) == 5 and abs(sum(probs) - 1) < 1e-4 and all(np.isfinite(probs))
    assert abs(res["confidence"] - max(probs)) < 1e-9
    assert isinstance(res["referable_dr"], bool) and isinstance(res["low_confidence"], bool)
    assert body["inference_ms"] > 0 and "Research use only" in body["disclaimer"]
    assert body["image"] == {"width": 300, "height": 240, "format": "png"}


def test_classification_accepts_jpeg(client):
    r = client.post("/v1/fundus/classify", files=up(fundus_like(fmt="JPEG"), "f.jpg", "image/jpeg"))
    assert r.status_code == 200 and r.json()["image"]["format"] == "jpeg"


# ---------------------------------------------------------------- segmentation
def test_segmentation_masks_match_original_dimensions(client):
    r = client.post("/v1/fundus/segment", files=up(fundus_like(300, 240)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"]["width"] == 300 and body["image"]["height"] == 240
    for code in ("MA", "HE"):
        les = body["lesions"][code]
        assert les["status"] == "ok"
        mask = np.array(Image.open(io.BytesIO(base64.b64decode(les["mask_png_base64"]))))
        assert mask.shape == (240, 300)  # (H, W) == original image, not model input size
        assert set(np.unique(mask)) <= {0, 255}
        assert 0.0 <= les["positive_fraction"] <= 1.0
        assert les["threshold"] in (0.5, 0.55)
    # unavailable lesions are reported explicitly, never as empty masks
    assert body["lesions"]["EX"]["status"] == "unavailable" and "mask_png_base64" not in body["lesions"]["EX"]
    assert body["lesions"]["SE"]["status"] == "unavailable"


def test_segmentation_subset_and_invalid_lesion(client):
    r = client.post("/v1/fundus/segment?lesions=ma", files=up(fundus_like()))
    assert r.status_code == 200 and list(r.json()["lesions"]) == ["MA"]
    r = client.post("/v1/fundus/segment?lesions=XX", files=up(fundus_like()))
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_lesion"


# ---------------------------------------------------------------- invalid uploads
def test_rejects_non_image_bytes(client):
    r = client.post("/v1/fundus/classify", files=up(b"hello this is not an image", "a.png"))
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_format"


def test_rejects_unsupported_format_gif(client):
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 40)).save(buf, format="GIF")
    r = client.post("/v1/fundus/classify", files=up(buf.getvalue(), "a.png"))
    assert r.status_code == 415


def test_rejects_corrupt_image_with_valid_signature(client):
    r = client.post("/v1/fundus/classify", files=up(b"\x89PNG\r\n\x1a\n" + b"garbage" * 50))
    assert r.status_code == 422 and r.json()["error"]["code"] == "corrupt_image"


def test_rejects_empty_file(client):
    assert client.post("/v1/fundus/classify", files=up(b"")).status_code == 400


def test_rejects_tiny_image(client):
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (200, 100, 40)).save(buf, format="PNG")
    r = client.post("/v1/fundus/classify", files=up(buf.getvalue()))
    assert r.status_code == 422 and r.json()["error"]["code"] == "image_too_small"


def test_rejects_non_fundus_images(client):
    for colour in ((20, 40, 220), (128, 128, 128), (0, 0, 0)):
        r = client.post("/v1/fundus/classify", files=up(solid(colour)))
        assert r.status_code == 422 and r.json()["error"]["code"] == "not_fundus_image", colour


def test_rejects_oversized_file(full_store):
    with TestClient(create_app(make_settings(full_store, max_mb=1))) as c:
        big = fundus_like(300, 240) + b"\x00" * (1024 * 1024 + 10)
        r = c.post("/v1/fundus/classify", files=up(big))
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_missing_file_field(client):
    r = client.post("/v1/fundus/classify")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
