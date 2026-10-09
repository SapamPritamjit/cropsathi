"""Crop information endpoint.

Reports which crops are supported and, separately, whether a baseline model and
a fine-tuned model are present. Availability reflects artifacts on disk only;
no model is loaded or activated here.
"""

from __future__ import annotations

from fastapi import APIRouter

from ..config import CROP_DISPLAY_NAMES, SUPPORTED_CROPS
from ..model_metadata import read_baseline_metadata, read_fine_tuned_metadata
from ..schemas import (
    BaselineModelInfo,
    CropInfo,
    CropsResponse,
    FineTunedModelInfo,
)

router = APIRouter(tags=["crops"])


@router.get("/crops", response_model=CropsResponse)
def list_crops() -> CropsResponse:
    """List supported crops with baseline and fine-tuned availability."""
    crops: list[CropInfo] = []
    for crop in SUPPORTED_CROPS:
        crops.append(
            CropInfo(
                crop=crop,
                display_name=CROP_DISPLAY_NAMES[crop],
                baseline=BaselineModelInfo(**read_baseline_metadata(crop)),
                fine_tuned=FineTunedModelInfo(**read_fine_tuned_metadata(crop)),
            )
        )
    return CropsResponse(count=len(crops), crops=crops)
