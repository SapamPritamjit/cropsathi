"""Real fine-tuned checkpoint smoke tests (marked ``smoke``).

These actually load the maize/rice EfficientNet-B0 checkpoints and run
inference. Contrast with ``test_fine_tuned_validation.py`` (fakes, no weights).
"""

import json
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import inference
from app.config import FINE_TUNED_MODEL_DIRS, settings
from app.main import app

pytestmark = pytest.mark.smoke

FINE_TUNED_CLASSES = {"maize": 9, "rice": 13}


def synthetic_image(size=(224, 224), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def classes_json(crop: str) -> list[str]:
    return json.loads(
        (FINE_TUNED_MODEL_DIRS[crop] / "classes.json").read_text(encoding="utf-8")
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    inference.clear_cache()
    yield
    inference.clear_cache()


@pytest.mark.parametrize("crop,expected", list(FINE_TUNED_CLASSES.items()))
def test_fine_tuned_adapter_metadata(crop: str, expected: int) -> None:
    adapter = inference.get_fine_tuned_adapter(crop)
    assert adapter.family == "fine_tuned"
    assert adapter.implementation == "torchvision:efficientnet_b0"
    assert adapter.num_classes == expected
    assert adapter.labels == classes_json(crop)
    assert adapter.input_size == [3, 224, 224]
    assert adapter.weights.endswith(".pth")
    assert not adapter.weights.startswith("D:")
    assert adapter.preprocessing_verified is False
    assert "not recorded" in adapter.preprocessing_note


@pytest.mark.parametrize("crop,expected", list(FINE_TUNED_CLASSES.items()))
def test_strict_checkpoint_loading_and_classifier_dimensions(
    crop: str, expected: int
) -> None:
    import torch
    from torchvision.models import efficientnet_b0

    checkpoint_path = next(FINE_TUNED_MODEL_DIRS[crop].glob("*.pth"))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = efficientnet_b0(weights=None, num_classes=ckpt["num_classes"])
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing and not unexpected
    assert tuple(model.classifier[1].weight.shape) == (expected, 1280)


@pytest.mark.parametrize("crop,expected", list(FINE_TUNED_CLASSES.items()))
def test_fine_tuned_prediction_probabilities(crop: str, expected: int) -> None:
    result = inference.predict(crop, synthetic_image(seed=3), family="fine_tuned")
    probabilities = result["probabilities"]
    assert result["model"]["family"] == "fine_tuned"
    assert result["model"]["implementation"] == "torchvision:efficientnet_b0"
    assert result["model"]["num_classes"] == expected
    assert len(probabilities) == expected
    assert sum(p["probability"] for p in probabilities) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p["probability"] <= 1.0 for p in probabilities)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["predicted_label"] == probabilities[0]["label"]
    assert result["predicted_index"] == probabilities[0]["index"]

    # Model-specific mapping: labels come from the fine-tuned classes.json.
    assert {p["label"] for p in probabilities} == set(classes_json(crop))


def test_fine_tuned_labels_differ_from_baseline_ordering() -> None:
    # Maize: 9 fine-tuned classes vs 11 baseline classes -> never reused.
    fine = inference.get_fine_tuned_adapter("maize")
    baseline = inference.get_baseline_adapter("maize")
    assert fine.num_classes == 9
    assert baseline.num_classes == 11
    assert fine.labels != baseline.labels


def test_baseline_prediction_unchanged_regression() -> None:
    result = inference.predict("maize", synthetic_image(seed=1))
    assert result["model"]["family"] == "baseline"
    assert result["model"]["implementation"] == "timm:convnext_small"
    assert len(result["probabilities"]) == 11


def test_http_fine_tuned_enabled_end_to_end(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_fine_tuned_predictions", True)
    client = TestClient(app)
    buffer = BytesIO()
    synthetic_image(seed=5).save(buffer, format="PNG")

    response = client.post(
        "/predict",
        data={"crop": "maize", "family": "fine_tuned"},
        files={"file": ("leaf.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["family"] == "fine_tuned"
    assert len(body["probabilities"]) == 9
    assert body["predicted_label"] in set(classes_json("maize"))
