import json

import pytest
from safetensors import safe_open

from app.config import BASELINE_MODEL_DIRS, FINE_TUNED_MODEL_DIRS
from app.model_verification import verify_all

BASELINE_CROPS = ["maize", "rice", "potato", "tomato"]
# Expected baseline classifier dimensions, from config.json (verified against
# the safetensors head shape).
BASELINE_CLASSES = {"maize": 11, "rice": 14, "potato": 3, "tomato": 13}
FINE_TUNED_CLASSES = {"maize": 9, "rice": 13}


@pytest.mark.parametrize("crop", BASELINE_CROPS)
def test_baseline_head_matches_config(crop: str) -> None:
    model_dir = BASELINE_MODEL_DIRS[crop]
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    with safe_open(model_dir / "model.safetensors", framework="pt") as handle:
        shape = handle.get_slice("head.fc.weight").get_shape()
    assert shape[0] == config["num_classes"] == BASELINE_CLASSES[crop]
    assert shape[1] == 768


@pytest.mark.parametrize("crop,expected", list(FINE_TUNED_CLASSES.items()))
def test_fine_tuned_checkpoint_output_dim(crop: str, expected: int) -> None:
    import torch
    from torchvision.models import efficientnet_b0

    model_dir = FINE_TUNED_MODEL_DIRS[crop]
    checkpoint_path = next(model_dir.glob("*.pth"))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert ckpt["architecture"] == "efficientnet_b0"
    assert ckpt["num_classes"] == expected

    model = efficientnet_b0(weights=None, num_classes=ckpt["num_classes"])
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing and not unexpected
    assert tuple(model.classifier[1].weight.shape) == (expected, 1280)


@pytest.mark.parametrize("crop,expected", list(FINE_TUNED_CLASSES.items()))
def test_fine_tuned_classes_metadata_matches_checkpoint(crop: str, expected: int) -> None:
    import torch

    model_dir = FINE_TUNED_MODEL_DIRS[crop]
    checkpoint_path = next(model_dir.glob("*.pth"))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes_json = json.loads((model_dir / "classes.json").read_text(encoding="utf-8"))

    assert ckpt["classes"] == classes_json
    assert len(classes_json) == expected


def test_verify_all_reports_six_verified_artifacts() -> None:
    results = verify_all()
    assert len(results) == 6
    by_key = {(r["crop"], r["family"]): r for r in results}

    for crop in BASELINE_CROPS:
        assert by_key[(crop, "baseline")]["status"] == "verified", crop

    for crop in FINE_TUNED_CLASSES:
        entry = by_key[(crop, "fine_tuned")]
        assert entry["status"] == "verified", crop
        assert entry["implementation"] == "torchvision:efficientnet_b0", crop
        assert entry["classes_match_json"] is True, crop


def test_known_integrity_differences_are_preserved() -> None:
    from app.config import SUPPORTED_CROPS
    from app.model_metadata import read_baseline_metadata, read_fine_tuned_metadata

    # Maize: fine-tuned 9 vs baseline 11. Rice: fine-tuned 13 vs baseline 14.
    assert read_fine_tuned_metadata("maize")["num_classes"] == 9
    assert read_baseline_metadata("maize")["num_classes"] == 11
    assert read_fine_tuned_metadata("rice")["num_classes"] == 13
    assert read_baseline_metadata("rice")["num_classes"] == 14

    # Potato and tomato have no fine-tuned artifacts.
    for crop in ("potato", "tomato"):
        assert crop in SUPPORTED_CROPS
        assert read_fine_tuned_metadata(crop)["present"] is False
