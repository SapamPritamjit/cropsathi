"""Image prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from .. import inference
from ..config import SUPPORTED_CROPS, settings
from ..errors import ApiError
from ..imaging import decode_image
from ..schemas import ErrorResponse, PredictResponse

router = APIRouter(tags=["prediction"])

SUPPORTED_FAMILIES: tuple[str, ...] = ("baseline", "fine_tuned")


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(
    crop: str = Form(..., description="Crop identifier: maize, rice, potato, tomato."),
    file: UploadFile = File(..., description="Leaf image (JPEG/PNG/BMP/WEBP/TIFF)."),
    family: str = Form(
        "baseline",
        description="Model family: 'baseline' (default) or 'fine_tuned'.",
    ),
) -> PredictResponse:
    """Predict the disease class for a leaf image.

    Uses the crop's verified baseline ConvNeXt-Small model by default. The
    optional ``family`` field selects the fine-tuned EfficientNet-B0 model, which
    stays disabled until the training preprocessing is confirmed.
    """
    crop_key = (crop or "").strip().lower()
    if crop_key not in SUPPORTED_CROPS:
        raise ApiError(
            400,
            "unsupported_crop",
            f"Unsupported crop '{crop}'. Supported crops: "
            + ", ".join(SUPPORTED_CROPS)
            + ".",
        )

    family_key = (family or "baseline").strip().lower()
    if family_key not in SUPPORTED_FAMILIES:
        raise ApiError(
            400,
            "unsupported_family",
            f"Unsupported model family '{family}'. Supported families: "
            + ", ".join(SUPPORTED_FAMILIES)
            + ".",
        )
    if family_key == "fine_tuned" and not settings.enable_fine_tuned_predictions:
        raise ApiError(
            503,
            "fine_tuned_preprocessing_unverified",
            "Fine-tuned predictions are disabled: the training preprocessing "
            "(resize/normalization) is not recorded in the supplied checkpoints. "
            "Confirm the training pipeline before enabling "
            "(set ENABLE_FINE_TUNED_PREDICTIONS=1).",
        )

    data = await file.read(settings.max_upload_bytes + 1)
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ApiError(
            413,
            "file_too_large",
            f"The uploaded file exceeds the {settings.max_upload_bytes}-byte limit.",
        )

    image = decode_image(
        data,
        min_side=settings.min_image_side,
        max_side=settings.max_image_side,
    )

    payload = inference.predict(crop_key, image, family=family_key)
    return PredictResponse(**payload)
