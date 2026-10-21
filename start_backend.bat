@echo off
echo =======================================================
echo Starting Diabetic Macular Edema (DME) AI Backend Server
echo Architecture: ConvNeXt-Tiny + Grad-CAM Explainability
echo =======================================================

call "C:\Users\user\anaconda3\Scripts\activate.bat" oct_dme
cd /d "C:\DME_OCT_Research\backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
