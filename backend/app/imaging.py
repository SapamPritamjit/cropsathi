"""Image decoding and validation (no torch dependency).

Validation trusts actual pixel decoding, not the filename or MIME type.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ApiError

# Formats we accept. Anything PIL can decode but that is not listed here is
# rejected as an unsupported image type.
ALLOWED_FORMATS: frozenset[str] = frozenset(
    {"JPEG", "PNG", "BMP", "WEBP", "TIFF"}
)


def decode_image(
    data: bytes,
    *,
    min_side: int,
    max_side: int,
    allowed_formats: frozenset[str] = ALLOWED_FORMATS,
) -> Image.Image:
    """Decode ``data`` into an RGB ``PIL.Image`` or raise :class:`ApiError`.

    Performs a real decode (``verify``) and enforces format and dimension
    limits. Returns an EXIF-corrected RGB image.
    """
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")

    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise ApiError(
            400,
            "corrupt_image",
            "The uploaded file is not a valid or readable image.",
        )

    try:
        image = Image.open(BytesIO(data))
        image_format = image.format
        if image_format not in allowed_formats:
            raise ApiError(
                415,
                "unsupported_image_format",
                "Unsupported image format. Allowed: "
                + ", ".join(sorted(allowed_formats))
                + ".",
                details={"format": image_format},
            )
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise ApiError(
            400,
            "corrupt_image",
            "The uploaded file is not a valid or readable image.",
        )

    width, height = image.size
    if min(width, height) < min_side:
        raise ApiError(
            400,
            "image_too_small",
            f"Image is too small. Minimum side is {min_side}px.",
            details={"width": width, "height": height, "min_side": min_side},
        )
    if max(width, height) > max_side:
        raise ApiError(
            400,
            "image_too_large",
            f"Image is too large. Maximum side is {max_side}px.",
            details={"width": width, "height": height, "max_side": max_side},
        )
    return image
