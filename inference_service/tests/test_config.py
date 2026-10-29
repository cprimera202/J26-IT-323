from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT, Settings


def test_model_store_defaults_to_repository_backend_directory(monkeypatch):
    monkeypatch.delenv("MODEL_STORE_DIR", raising=False)

    settings = Settings.load()

    assert settings.model_store == (PROJECT_ROOT / "backend" / "model_store").resolve()
    assert settings.model_store.is_dir()


def test_model_store_environment_override_takes_priority(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_STORE_DIR", str(tmp_path))

    assert Settings.load().model_store == tmp_path.resolve()
