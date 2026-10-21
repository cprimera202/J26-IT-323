import os
import io
import torch
from PIL import Image
from fastapi.testclient import TestClient

from main import app, load_oct_model

print("Testing backend startup...")
load_oct_model()

client = TestClient(app)

print("Testing GET /health...")
resp = client.get("/health")
print("Health response:", resp.status_code, resp.json())
assert resp.status_code == 200

# Create dummy OCT image
dummy_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
buf = io.BytesIO()
dummy_img.save(buf, format="PNG")
buf.seek(0)

print("Testing POST /api/predict...")
files = {"file": ("test_oct.png", buf.getvalue(), "image/png")}
pred_resp = client.post("/api/predict", files=files)
print("Predict response code:", pred_resp.status_code)
data = pred_resp.json()
print("Predicted class:", data.get("predicted_class_id"), data.get("predicted_label"))
print("Confidence:", data.get("confidence_percentage"))
print("Probabilities:", data.get("class_probabilities"))
print("Heatmap base64 length:", len(data.get("gradcam_heatmap_base64", "")))

assert pred_resp.status_code == 200
assert "predicted_class_id" in data
assert "predicted_label" in data
assert "confidence_percentage" in data
assert "class_probabilities" in data
assert "gradcam_heatmap_base64" in data
print("ALL BACKEND TESTS PASSED!")
