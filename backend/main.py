import io
import base64
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------
# Application Initialization & Setup
# ---------------------------------------------------------
app = FastAPI(
    title="DME OCT Classification API",
    description="Diabetic Macular Edema (DME) Optical Coherence Tomography (OCT) AI Diagnostic Platform",
    version="1.0.0"
)

# Enable CORS for React frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target Classes Mapping
CLASS_NAMES = {
    0: "No DME",
    1: "NCI-DME",
    2: "CI-DME"
}

# Image Preprocessing Transform (Must match PyTorch validation transforms)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global variables for model and execution device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
cam = None

def get_model_path() -> Path:
    """Resolve the location of the pre-trained weights."""
    current_file_dir = Path(__file__).resolve().parent
    potential_paths = [
        current_file_dir.parent / "saved_models" / "best_convnext_tiny.pth",
        Path("C:/DME_OCT_Research/saved_models/best_convnext_tiny.pth"),
        Path("./saved_models/best_convnext_tiny.pth")
    ]
    for path in potential_paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Model weights not found. Searched paths: {[str(p) for p in potential_paths]}"
    )

def load_oct_model():
    """Load convnext_tiny architecture and weights."""
    global model, cam, device
    print(f"[*] Initializing model on device: {device}")
    model_path = get_model_path()
    print(f"[*] Loading model weights from: {model_path}")

    # Build ConvNeXt-Tiny architecture with 3 output classes
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=3)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Target normalization layer for Grad-CAM
    target_layers = [model.stages[-1].blocks[-1].norm]
    cam = GradCAM(model=model, target_layers=target_layers)
    print("[+] ConvNeXt-Tiny model and Grad-CAM loaded successfully!")

@app.on_event("startup")
def startup_event():
    load_oct_model()

# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model": "convnext_tiny",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "is_cuda": torch.cuda.is_available()
    }

# ---------------------------------------------------------
# Prediction & Explainability Endpoint
# ---------------------------------------------------------
@app.post("/api/predict")
async def predict_oct_scan(file: UploadFile = File(...)):
    global model, cam, device

    if model is None or cam is None:
        try:
            load_oct_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model failed to initialize: {str(e)}"
            )

    # Validate uploaded file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload an OCT B-scan (.png, .jpg, .jpeg, .tif)."
        )

    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {str(e)}"
        )

    try:
        # Prepare RGB normalized image for Grad-CAM overlay (224x224 float32 in [0, 1])
        resized_image = raw_image.resize((224, 224), Image.Resampling.BILINEAR)
        rgb_normalized = np.float32(resized_image) / 255.0

        # Apply PyTorch validation transforms
        input_tensor = val_transforms(raw_image).unsqueeze(0).to(device)

        # 1. Compute Model Predictions & Softmax Probabilities
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_class_id = int(np.argmax(probabilities))
        predicted_label = CLASS_NAMES[pred_class_id]
        confidence_percentage = round(float(probabilities[pred_class_id]) * 100, 2)

        class_probabilities = {
            "No DME": round(float(probabilities[0]) * 100, 2),
            "NCI-DME": round(float(probabilities[1]) * 100, 2),
            "CI-DME": round(float(probabilities[2]) * 100, 2)
        }

        # 2. Compute Grad-CAM Explainability Heatmap
        # Input tensor needs gradient tracking for Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        gradcam_overlay = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)

        # 3. Encode Grad-CAM Visualization to PNG Base64
        overlay_pil = Image.fromarray(gradcam_overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        gradcam_heatmap_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Return standardized clinical decision support JSON payload
        return {
            "predicted_class_id": pred_class_id,
            "predicted_label": predicted_label,
            "confidence_percentage": confidence_percentage,
            "class_probabilities": class_probabilities,
            "gradcam_heatmap_base64": gradcam_heatmap_base64
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error during OCT analysis: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
