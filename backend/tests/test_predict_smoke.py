"""Real model-inference smoke tests (marked ``smoke``).

These actually load each baseline checkpoint and run inference. Contrast with
``test_predict_validation.py``, which uses a fake adapter and loads no weights.
"""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import inference
from app.config import SUPPORTED_CROPS
from app.main import app
from app.model_metadata import read_baseline_metadata

pytestmark = pytest.mark.smoke


def synthetic_image(size=(224, 224), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")


@pytest.mark.parametrize("crop", SUPPORTED_CROPS)
def test_real_prediction_all_baselines(crop: str) -> None:
    expected_classes = read_baseline_metadata(crop)["num_classes"]

    result = inference.predict(crop, synthetic_image())

    probabilities = result["probabilities"]
    assert len(probabilities) == expected_classes
    assert sum(p["probability"] for p in probabilities) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= p["probability"] <= 1.0 for p in probabilities)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model"]["implementation"] == "timm:convnext_small"
    assert result["predicted_label"] == probabilities[0]["label"]
    assert result["predicted_index"] == probabilities[0]["index"]


def test_predict_http_end_to_end_real_model() -> None:
    client = TestClient(app)
    buffer = BytesIO()
    synthetic_image(seed=7).save(buffer, format="PNG")

    response = client.post(
        "/predict",
        data={"crop": "potato"},
        files={"file": ("leaf.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["crop"] == "potato"
    assert body["model"]["implementation"] == "timm:convnext_small"
    assert len(body["probabilities"]) == 3
    assert body["predicted_label"] in {"Early Blight", "Healthy", "Late Blight"}
