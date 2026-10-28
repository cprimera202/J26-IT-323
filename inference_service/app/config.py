"""Runtime configuration for the inference service (environment variables only)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# <project root>/inference_service/app/config.py -> parents[2] == <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_store: Path
    max_upload_bytes: int
    min_image_side: int
    max_image_side: int
    device_preference: str
    allowed_origins: tuple[str, ...]

    @staticmethod
    def load() -> "Settings":
        raw = os.environ.get("MODEL_STORE_DIR", "").strip()
        # Default: <project root>/model_store. Set MODEL_STORE_DIR to point at your
        # real Backend\model_store folder (recommended).
        store = Path(raw).expanduser() if raw else PROJECT_ROOT / "model_store"
        origins = os.environ.get("INFERENCE_ALLOWED_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000")
        return Settings(
            model_store=store.resolve(),
            max_upload_bytes=_int_env("MAX_UPLOAD_MB", 15) * 1024 * 1024,
            min_image_side=_int_env("MIN_IMAGE_SIDE", 128),
            max_image_side=_int_env("MAX_IMAGE_SIDE", 8000),
            device_preference=os.environ.get("INFERENCE_DEVICE", "auto").lower(),  # auto | cpu | cuda
            allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )
