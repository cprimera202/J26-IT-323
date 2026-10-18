"""Django settings for the retinal lesion API. Stateless: no database, sessions or auth are used."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_list(name, default):
    return [v.strip() for v in os.environ.get(name, default).split(",") if v.strip()]


DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
if not DEBUG and SECRET_KEY == "dev-only-insecure-key-change-me":
    raise RuntimeError("Set DJANGO_SECRET_KEY when DJANGO_DEBUG=0.")
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")

INSTALLED_APPS = ["corsheaders", "lesions"]
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
DATABASES = {}
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Origins of the React app (Vite dev server by default).
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

# ---- lesion models ---------------------------------------------------------------------------------------------
# Folder layout:  model_store/MA/, model_store/HE/, model_store/EX/, model_store/SE/  (each = that notebook's best_model/)
MODEL_STORE_DIR = Path(os.environ.get("MODEL_STORE_DIR", BASE_DIR / "model_store"))
LESION_CODES = ["MA", "HE", "EX", "SE"]
# Outline colours come from each model's metadata (overlay_color_rgb). Override here if a colour is hard to see, e.g.
# LESION_COLOR_OVERRIDES = {"MA": (255, 0, 255)}. The web app also lets the user recolour each type.
LESION_COLOR_OVERRIDES = {}
LESION_MIN_AREA_PX = int(os.environ.get("LESION_MIN_AREA_PX", "0"))   # drop outlines smaller than this (original pixels)

# ---- DR severity grading (optional) ------------------------------------------------------------------------------
# One image-level classifier, trained elsewhere. Put its weights and grading_config.json in model_store/DR/ (see model_store/README.md).
GRADING_ENABLED = os.environ.get("GRADING_ENABLED", "1") == "1"
GRADING_FOLDER = os.environ.get("GRADING_FOLDER", "DR")

# ---- upload limits ---------------------------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(60_000_000)))
DATA_UPLOAD_MAX_MEMORY_SIZE = (MAX_UPLOAD_MB + 1) * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = DATA_UPLOAD_MAX_MEMORY_SIZE

LOGGING = {"version": 1, "disable_existing_loggers": False,
           "handlers": {"console": {"class": "logging.StreamHandler"}},
           "root": {"handlers": ["console"], "level": os.environ.get("LOG_LEVEL", "INFO")}}
