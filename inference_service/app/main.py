"""FastAPI inference service for the DR analysis system (research use only)."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .models.registry import LESIONS, ModelRegistry, ModelUnavailable
from .validation import ValidationError, assess_fundus, decode_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dr.api")

DISCLAIMER = "Research use only. Not a medical device and not for standalone clinical diagnosis."
_infer_lock = threading.Semaphore(1)  # one forward pass at a time keeps CPU usage predictable


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None, load_models: bool = True) -> FastAPI:
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reg = registry or ModelRegistry(settings)
        app.state.registry = reg
        if load_models and registry is None:
            await run_in_threadpool(reg.load_all)
        st = reg.status()
        log.info("Model store: %s", settings.model_store)
        log.info("Models ready: %d/%d", st["ready_count"], st["total_count"])
        yield

    app = FastAPI(title="DR Inference Service", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.registry = registry  # replaced in lifespan when not injected
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    @app.exception_handler(ValidationError)
    async def _val(_: Request, exc: ValidationError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(ModelUnavailable)
    async def _unavail(_: Request, exc: ModelUnavailable):
        return error_response(503, "model_unavailable", f"Model '{exc.key}' is not available: {exc.reason}")

    @app.exception_handler(RequestValidationError)
    async def _req(_: Request, exc: RequestValidationError):
        return error_response(422, "invalid_request", "Invalid request. Send a multipart form with an image in the 'file' field.")

    @app.exception_handler(Exception)
    async def _any(_: Request, exc: Exception):
        log.exception("Unhandled error: %s", type(exc).__name__)
        return error_response(500, "inference_failed", "Inference failed due to an internal error.")

    async def _read_image(request: Request, file: UploadFile):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 1024 * 1024:
            raise ValidationError("file_too_large", f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.", 413)
        data = await file.read(settings.max_upload_bytes + 1)
        v = decode_image(data, settings.max_upload_bytes, settings.min_image_side, settings.max_image_side)
        assess_fundus(v)
        return v

    # ------------------------------------------------------------------ health / readiness
    @app.get("/health")
    def health():
        """Liveness only: the API process is running. Says nothing about model readiness."""
        return {"status": "ok", "service": "dr-inference", "disclaimer": DISCLAIMER}

    @app.get("/models")
    def models():
        return {**app.state.registry.status(), "disclaimer": DISCLAIMER}

    @app.get("/ready")
    def ready():
        st = app.state.registry.status()
        code = 200 if st["all_ready"] else 503
        return JSONResponse(status_code=code, content={"ready": st["all_ready"], **st})

    # ------------------------------------------------------------------ classification
    @app.post("/v1/fundus/classify")
    async def classify(request: Request, file: UploadFile = File(...)):
        model = app.state.registry.get_ready("classification")
        v = await _read_image(request, file)

        def run():
            with _infer_lock:
                t0 = time.perf_counter()
                out = model.predict(v.image)
                return out, (time.perf_counter() - t0) * 1000

        result, ms = await run_in_threadpool(run)
        return {
            "result": result,
            "image": {"width": v.width, "height": v.height, "format": v.format},
            "quality_notes": v.quality_notes,
            "inference_ms": round(ms, 1),
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------ segmentation
    @app.post("/v1/fundus/segment")
    async def segment(
        request: Request,
        file: UploadFile = File(...),
        lesions: str | None = Query(default=None, description="Comma separated subset of MA,HE,EX,SE"),
    ):
        reg: ModelRegistry = app.state.registry
        wanted = list(LESIONS) if not lesions else [c.strip().upper() for c in lesions.split(",") if c.strip()]
        bad = [c for c in wanted if c not in LESIONS]
        if bad:
            raise ValidationError("invalid_lesion", "Unknown lesion code. Use MA, HE, EX or SE.", 422)
        if not any(reg.slots[c].status == "ready" for c in wanted):
            reasons = {c: reg.slots[c].error for c in wanted}
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "model_unavailable", "message": "None of the requested segmentation models are available."},
                         "lesions": {c: {"status": "unavailable", "error": reasons[c]} for c in wanted}},
            )
        v = await _read_image(request, file)

        def run():
            results: dict[str, dict] = {}
            t_all = time.perf_counter()
            for code in wanted:
                slot = reg.slots[code]
                meta = {"code": code, "name": LESIONS[code]["name"], "color": LESIONS[code]["color"]}
                if slot.status != "ready":
                    results[code] = {**meta, "status": "unavailable", "error": slot.error}
                    continue
                try:
                    with _infer_lock:
                        t0 = time.perf_counter()
                        out = slot.predictor.predict(v.image)
                    results[code] = {**meta, "status": "ok", **out, "inference_ms": round((time.perf_counter() - t0) * 1000, 1)}
                except Exception as exc:  # noqa: BLE001
                    log.error("segmentation %s failed: %s", code, type(exc).__name__)
                    results[code] = {**meta, "status": "error", "error": "Inference failed for this lesion model."}
            return results, (time.perf_counter() - t_all) * 1000

        lesion_results, total_ms = await run_in_threadpool(run)
        return {
            "image": {"width": v.width, "height": v.height, "format": v.format},
            "lesions": lesion_results,
            "quality_notes": v.quality_notes,
            "inference_ms": round(total_ms, 1),
            "disclaimer": DISCLAIMER,
        }

    return app


app = create_app()
