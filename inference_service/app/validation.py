"""Upload validation: signature, size, decoding, dimensions and a fundus plausibility check."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 120_000_000  # decompression-bomb guard

SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
}


class ValidationError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class ValidatedImage:
    image: Image.Image  # RGB
    format: str
    width: int
    height: int
    quality_notes: list[str]


def sniff_format(data: bytes) -> str | None:
    for sig, name in SIGNATURES.items():
        if data.startswith(sig):
            return name
    return None


def decode_image(data: bytes, max_bytes: int, min_side: int, max_side: int) -> ValidatedImage:
    if not data:
        raise ValidationError("empty_file", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(
            "file_too_large", f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.", 413
        )
    fmt = sniff_format(data)
    if fmt is None:
        raise ValidationError("unsupported_format", "Only JPEG and PNG images are supported.", 415)
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()  # structural check
        img = Image.open(io.BytesIO(data))
        img.load()  # full decode
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ValidationError("corrupt_image", "The file could not be decoded as a valid image.", 422)
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < min_side:
        raise ValidationError("image_too_small", f"Image is too small (minimum side {min_side}px).", 422)
    if max(w, h) > max_side:
        raise ValidationError("image_too_large", f"Image dimensions exceed {max_side}px.", 422)
    return ValidatedImage(img, fmt, w, h, [])


def assess_fundus(v: ValidatedImage) -> None:
    """Cheap plausibility check that the image resembles a colour fundus photograph.

    This is a heuristic input filter only. Passing it does NOT mean the image is
    clinically adequate and says nothing about the presence or absence of disease.
    """
    small = v.image.resize((128, 128))
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    brightness = float(arr.mean())
    spread = float(arr.std())

    if brightness < 12 or spread < 8:
        raise ValidationError(
            "not_fundus_image", "The image is nearly blank and does not look like a retinal fundus photograph.", 422
        )
    # Colour fundus photographs are strongly red/orange dominant.
    if not (mean_r > mean_g * 1.05 and mean_r > mean_b * 1.3):
        raise ValidationError(
            "not_fundus_image",
            "The image does not look like a colour retinal fundus photograph (unexpected colour profile).",
            422,
        )
    # Most fundus photos have dark corners around the circular field of view.
    corners = np.concatenate(
        [arr[:12, :12].reshape(-1, 3), arr[:12, -12:].reshape(-1, 3), arr[-12:, :12].reshape(-1, 3), arr[-12:, -12:].reshape(-1, 3)]
    )
    if float(corners.mean()) > 0.8 * brightness and brightness > 60:
        v.quality_notes.append(
            "No dark border detected around the field of view; results may be less reliable."
        )
    if brightness < 25:
        v.quality_notes.append("The image is very dark; results may be unreliable.")
