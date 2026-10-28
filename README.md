# Diabetic Retinopathy Analysis System

Research dashboard with four components:

| Component | Status |
|---|---|
| Fundus image classification (DR grade 0–4, ResNet50) | Implemented — uses your trained checkpoint |
| Fundus lesion segmentation (MA, HE, EX, SE) | Implemented — each lesion model loads independently |
| OCTA analysis | **Under development** — UI + API hook only, no model, no fake output |
| OCT analysis | **Under development** — UI + API hook only, no model, no fake output |

> **Research use only. Not a medical device. Not clinically validated. Not for standalone clinical diagnosis.**

This is a standalone folder. It does **not** modify your `J26-IT-323` repository; it only *reads* your trained models
from `Backend\model_store` through an environment variable.

```
dr-analysis-system/
├─ inference_service/   Python · FastAPI · PyTorch   (port 8001)  – loads models, runs inference
├─ backend/             Node · Express                (port 5000)  – frontend-facing API, validation, proxy
├─ frontend/            Next.js · TypeScript · Tailwind (port 3000) – dashboard UI
└─ start-all.ps1        opens the three servers in separate PowerShell windows
```

Request flow: `Browser → Node (5000) → FastAPI (8001) → PyTorch models`. Inference never runs in the browser.

---

## 1. Required software

- Windows 10/11 with PowerShell
- Python 3.10 – 3.12 (use the `py` launcher) — use the **same PyTorch/TorchVision versions you trained with**
- Node.js 20 or newer (`node --version`)

## 2. Python environment (inference service)

```powershell
cd C:\path\to\dr-analysis-system\inference_service
py -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install PyTorch **first**, matching your training environment (CPU shown; pick your CUDA build from pytorch.org if you use a GPU).
If you already have a working PyTorch/TorchVision in this environment, skip this line — `requirements.txt` deliberately does not pin or upgrade them.

```powershell
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
py -m pip install -r requirements.txt
```

## 3. Node dependencies

```powershell
cd C:\path\to\dr-analysis-system\backend
npm install
cd ..\frontend
npm install
```

## 4. Environment variables

| Variable | Service | Default | Purpose |
|---|---|---|---|
| `MODEL_STORE_DIR` | inference | `<this folder>\model_store` | **Set this** to your `Backend\model_store` |
| `INFERENCE_DEVICE` | inference | `auto` | `auto`, `cpu`, or `cuda` |
| `MAX_UPLOAD_MB` | inference | `15` | upload limit (Python boundary) |
| `ALLOW_UNSAFE_PICKLE` | inference | `0` | set `1` only if a trusted checkpoint cannot load in safe `weights_only` mode |
| `PORT` | backend | `5000` | |
| `INFERENCE_BASE_URL` | backend | `http://127.0.0.1:8001` | |
| `INFERENCE_TIMEOUT_MS` | backend | `120000` | per-inference timeout |
| `MAX_UPLOAD_MB` | backend | `10` | upload limit (Node boundary) |
| `FRONTEND_ORIGINS` | backend | `http://localhost:3000,http://127.0.0.1:3000` | CORS allow-list |
| `NEXT_PUBLIC_API_BASE_URL` | frontend | `http://127.0.0.1:5000` | set in `frontend\.env.local` |

## 5. Start everything

**Option A – one command** (edit the model-store path at the top of the script first):

```powershell
cd C:\path\to\dr-analysis-system
.\start-all.ps1
```

**Option B – three terminals**

```powershell
# Terminal 1 — inference service
cd C:\path\to\dr-analysis-system\inference_service
.\.venv\Scripts\Activate.ps1
$env:MODEL_STORE_DIR = "C:\Users\mhjha\Documents\GitHub\J26-IT-323\Backend\model_store"
py -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Terminal 2 — Node backend
cd C:\path\to\dr-analysis-system\backend
npm start

# Terminal 3 — frontend (dev)
cd C:\path\to\dr-analysis-system\frontend
npm run dev
```

Open <http://localhost:3000>.

### Health checks

| URL | Meaning |
|---|---|
| <http://127.0.0.1:8001/health> | FastAPI process is running (says nothing about models) |
| <http://127.0.0.1:8001/models> | **Per-model status** (ready / unavailable + reason) |
| <http://127.0.0.1:8001/ready> | `200` only if **all** models are ready, otherwise `503` |
| <http://127.0.0.1:5000/api/health> | Node is up + whether it can reach the inference service |
| <http://127.0.0.1:5000/api/components> | What the dashboard shows (real availability) |

On startup the inference service logs one line per model (`ready` or `unavailable -- reason`).

## 6. API routes (identical in frontend, Node and FastAPI)

| Frontend → Node | Node → FastAPI |
|---|---|
| `GET /api/health` | `GET /health`, `GET /models` |
| `GET /api/components` | `GET /models` |
| `POST /api/fundus/classify` (multipart field `file`) | `POST /v1/fundus/classify` |
| `POST /api/fundus/segment?lesions=MA,HE` (optional filter) | `POST /v1/fundus/segment` |
| `POST /api/octa/analyze`, `POST /api/oct/analyze` | *(none yet — returns `501 model_not_available`)* |

Errors always look like `{"error": {"code": "...", "message": "..."}}`. A test (`frontend/tests/routes.test.ts`) fails if the three layers drift apart.

## 7. Model files

```
model_store\
  fundus_classification\DR\dr_grading.pth
  fundus_classification\DR\grading_config.json
  fundus_segmentation\MA\MA_BEST_deeplabv3plus.pth   + MA_BEST_model_metadata.json
  fundus_segmentation\HE\HE_BEST_deeplabv3plus.pth   + HE_BEST_model_metadata.json
  fundus_segmentation\EX\EX_BEST_unet.pth            + EX_BEST_model_metadata.json
  fundus_segmentation\SE\SE_BEST_unetplusplus.pth    + SE_BEST_model_metadata.json
```

Checkpoints are always loaded with `strict=True`. If the architecture does not match, that model is marked **unavailable** with the reason and the
API returns an error for it — never an empty or invented mask.

### ⚠️ EX and SE need your original training classes

Your EX checkpoint uses custom U-Net keys (`up4 … up0`, `head`) and SE uses custom U-Net++ keys (`x_0_0`, `x_0_1`, …). Those are not standard
`segmentation-models-pytorch` modules, and **this project does not guess them**. Until you add the original classes, EX and SE will show as *unavailable*.
MA and HE use `smp.DeepLabV3Plus` (ResNet50) with `output_stride` passed once, via `encoder_output_stride`.

To enable EX / SE:

1. See the real keys and shapes: `py scripts\inspect_checkpoints.py --model-store "<your model_store>"`
2. Open the training notebook/script that produced the checkpoint and copy the `nn.Module` classes into
   `inference_service\app\models\custom_architectures.py` **without renaming attributes** (state-dict keys come from attribute names).
3. Register a builder at the bottom of that file, e.g. `CUSTOM_BUILDERS["EX"] = lambda meta: MyUNet(encoder="resnet34", classes=1)`.
4. Restart the service and check `/models`. If keys still mismatch, the error lists missing/unexpected keys to compare against.

If you send me those two classes I can wire them in.

### Assumptions that could not be verified without your files

Config/metadata keys are read flexibly, and every fallback is reported in `/models` under `assumptions`. Please check these against your training code:

- Segmentation input size is read from metadata; if absent it falls back to **512×512** and images are resized directly (no letterbox/crop). If training used a different resize/pad strategy, set it in the metadata or adapt `_normalize` in `registry.py`.
- Referable-DR rule: from `grading_config.json` if present, otherwise grade ≥ 2. Low-confidence threshold: from config, otherwise 0.60.
- The classifier is built as `torchvision.models.resnet50` with the head shape inferred from the checkpoint (`fc` or `Dropout→fc`).

## 8. Verify your real models

```powershell
cd C:\path\to\dr-analysis-system\inference_service
.\.venv\Scripts\Activate.ps1
py scripts\verify_models.py --model-store "C:\Users\mhjha\Documents\GitHub\J26-IT-323\Backend\model_store" --image "C:\path\to\a_real_fundus.jpg"
```

For each model it reports: loaded (strict) → forward pass → finite outputs → mask size equals the image size → `READY` / `NOT READY`.
Exit code is non-zero if any model is not ready.

## 9. Tests

```powershell
# Inference service (uses small temporary checkpoints it generates itself)
cd inference_service ; py -m pytest

# Node backend (uses a fake inference server)
cd ..\backend ; npm test

# Frontend unit tests + type check + production build
cd ..\frontend ; npm test ; npm run typecheck ; npm run build
```

These tests exercise the code paths with generated weights. They do **not** prove your trained weights work — `verify_models.py` does that.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard says "Service unreachable" | Start FastAPI (port 8001) and Node (5000); check `/api/health` |
| Everything "unavailable", log mentions `not found in model store` | `MODEL_STORE_DIR` is wrong — it must point to `Backend\model_store`, not `dr_inference_service_starter\model_store` |
| EX/SE "does not match the reconstructed architecture" | Expected until you add the original classes (section 7) |
| "could not be loaded in safe (weights_only) mode" | Re-save weights as a plain `state_dict`, or set `$env:ALLOW_UNSAFE_PICKLE = "1"` for a file you trust |
| Browser shows a CORS error | Add your frontend origin to `FRONTEND_ORIGINS` for the Node backend |
| `413` on upload | Raise `MAX_UPLOAD_MB` on Node (and on FastAPI if above 15) |
| `422 not_fundus_image` | The plausibility check (colour profile, not blank) rejected the image; it is a heuristic input filter, not a medical check |
| `504 inference_timeout` | CPU inference is slow; raise `INFERENCE_TIMEOUT_MS` |
| Port already in use | Change `PORT` (backend) / `--port` (uvicorn) and the matching URL variables |

## 11. Adding OCT / OCTA models later

1. Add the weights under `model_store\oct\...` / `model_store\octa\...`.
2. In `inference_service\app\models\`, add a loader (same pattern as `registry.py`: strict load, readiness flag) and an endpoint such as `POST /v1/octa/analyze`.
3. In `backend\src\providers\index.js`, implement `providers.octa = { analyze: async (file, sniffed, client) => ... }`; the route `/api/octa/analyze` already exists and switches from `501` to the provider automatically.
4. In `backend\src\app.js` (`/api/components`), report the real status instead of `under_development`.
5. In the frontend `components\ModalityPage.tsx`, replace the disabled button with a call through `lib\api.ts`.

## 12. Privacy and safety behaviour

- Uploads are checked by file signature (JPEG/PNG only), size, full decode, dimensions, and a fundus plausibility heuristic. Passing it does not mean the image is clinically adequate or says anything about disease.
- Images are processed in memory and never written to disk. Image content and filenames are not logged. Client filenames are never forwarded.
- Error responses never include file-system paths. No route accepts a path from the user.
- No history page: nothing is stored, so there is nothing to list.

## 13. Known limitations

- EX and SE segmentation stay unavailable until the original architectures are added (section 7).
- Not tested on Windows or against your real checkpoints in the environment where this was built; run section 8 first.
- No clinical validation; the fundus plausibility check is heuristic.
