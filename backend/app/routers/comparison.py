"""Model inventory and comparison endpoints.

Exposes read-only, deterministic metadata assembled from verified artifacts.
No model is loaded for inference here; live predictions remain on ``POST /predict``.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..comparison import build_comparison, build_models
from ..config import SUPPORTED_CROPS
from ..errors import ApiError
from ..schemas import (
    ComparisonResponse,
    ErrorResponse,
    ModelEntry,
    ModelsResponse,
)

router = APIRouter(tags=["comparison"])


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """Inventory every available model artifact with its verification status."""
    models = [ModelEntry(**entry) for entry in build_models()]
    return ModelsResponse(count=len(models), models=models)


@router.get(
    "/comparison/{crop}",
    response_model=ComparisonResponse,
    responses={400: {"model": ErrorResponse}},
)
def compare_crop(crop: str) -> ComparisonResponse:
    """Metadata, class-set overlap and comparison eligibility for one crop."""
    crop_key = (crop or "").strip().lower()
    if crop_key not in SUPPORTED_CROPS:
        raise ApiError(
            400,
            "unsupported_crop",
            f"Unsupported crop '{crop}'. Supported crops: "
            + ", ".join(SUPPORTED_CROPS)
            + ".",
        )
    return ComparisonResponse(**build_comparison(crop_key))
