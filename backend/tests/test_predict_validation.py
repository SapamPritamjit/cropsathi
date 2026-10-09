"""Mocked, fast tests for /predict validation and response behaviour.

No real checkpoint is loaded here; a fake adapter is injected so these tests
exercise request validation, the response schema, error envelopes and the
probability->response math.
"""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import inference
from app.config import settings
from app.errors import ApiError
from app.main import app

client = TestClient(app)


class FakeAdapter:
    crop = "maize"
    family = "baseline"
    implementation = "fake:test"
    labels = ["alpha", "beta", "gamma"]
    num_classes = 3
    input_size = [3, 224, 224]
    weights = "fake/model.safetensors"
    label_source = "fake/config.json"

    def __init__(self, probs):
        self._probs = probs

    def predict_proba(self, image):
        return np.asarray(self._probs, dtype=np.float64)


def png_bytes(size=(224, 224), color=(120, 180, 90)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def gif_bytes(size=(224, 224)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="GIF")
    return buffer.getvalue()


def post(crop, data: bytes, filename="leaf.png", content_type="image/png"):
    return client.post(
        "/predict",
        data={"crop": crop},
        files={"file": (filename, data, content_type)},
    )


@pytest.fixture
def fake_adapter(monkeypatch):
    adapter = FakeAdapter([0.1, 0.7, 0.2])
    monkeypatch.setattr(inference, "_resolve_adapter", lambda crop, family: adapter)
    return adapter


def test_valid_prediction_schema_and_math(fake_adapter) -> None:
    response = post("maize", png_bytes())
    assert response.status_code == 200
    body = response.json()

    assert body["crop"] == "maize"
    assert body["display_name"] == "Maize"
    assert body["predicted_index"] == 1
    assert body["predicted_label"] == "beta"
    assert body["confidence"] == pytest.approx(0.7, abs=1e-9)
    assert body["model"]["implementation"] == "fake:test"
    assert body["model"]["num_classes"] == 3
    assert body["disclaimer"]

    probs = body["probabilities"]
    assert len(probs) == 3
    assert [p["label"] for p in probs] == ["beta", "gamma", "alpha"]
    assert sum(p["probability"] for p in probs) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= p["probability"] <= 1.0 for p in probs)
    assert probs[0]["label"] == body["predicted_label"]
    assert probs[0]["index"] == body["predicted_index"]


def test_unsupported_crop(fake_adapter) -> None:
    response = post("wheat", png_bytes())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_crop"


def test_missing_file_is_validation_error(fake_adapter) -> None:
    response = client.post("/predict", data={"crop": "maize"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_missing_crop_is_validation_error(fake_adapter) -> None:
    response = client.post(
        "/predict", files={"file": ("leaf.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_empty_file(fake_adapter) -> None:
    response = post("maize", b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_corrupt_image(fake_adapter) -> None:
    response = post("maize", b"this is not an image at all")
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "corrupt_image"
    assert "Traceback" not in body["error"]["message"]
    assert "D:\\" not in body["error"]["message"]


def test_unsupported_image_format(fake_adapter) -> None:
    response = post("maize", gif_bytes(), filename="leaf.gif", content_type="image/gif")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_image_format"


def test_image_too_small(fake_adapter) -> None:
    response = post("maize", png_bytes(size=(8, 8)))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "image_too_small"


def test_oversized_upload(fake_adapter, monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 50)
    response = post("maize", png_bytes())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_model_load_failure(monkeypatch) -> None:
    def boom(crop):
        raise ApiError(503, "model_unavailable", "nope")

    monkeypatch.setattr(inference, "get_baseline_adapter", boom)
    response = post("maize", png_bytes())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_malformed_model_output(monkeypatch) -> None:
    adapter = FakeAdapter([0.5, 0.5])  # wrong length for num_classes == 3
    monkeypatch.setattr(inference, "_resolve_adapter", lambda crop, family: adapter)
    response = post("maize", png_bytes())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_failed"


def test_response_does_not_leak_paths(fake_adapter) -> None:
    body = post("maize", png_bytes()).json()
    assert body["model"]["weights"] == "fake/model.safetensors"
    assert "D:" not in body["model"]["weights"]
