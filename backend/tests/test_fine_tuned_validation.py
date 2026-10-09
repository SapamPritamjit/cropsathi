"""Mocked, fast tests for model-family selection and the fine-tuned gate.

No real checkpoint is loaded here. The baseline / fine-tuned adapters are
replaced with fakes so these tests exercise request validation, the family
activation gate, family passthrough, error envelopes and cache behaviour.
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


class FakeBaselineAdapter:
    crop = "maize"
    family = "baseline"
    implementation = "fake:baseline"
    labels = ["a", "b", "c"]
    num_classes = 3
    input_size = [3, 224, 224]
    weights = "fake/model.safetensors"
    label_source = "fake/config.json"

    def predict_proba(self, image):
        return np.asarray([0.2, 0.5, 0.3], dtype=np.float64)


class FakeFineTunedAdapter:
    crop = "maize"
    family = "fine_tuned"
    implementation = "fake:efficientnet_b0"
    labels = [f"ft{i}" for i in range(9)]
    num_classes = 9
    input_size = [3, 224, 224]
    weights = "fake/net.pth"
    label_source = "fake/classes.json"

    def predict_proba(self, image):
        probs = np.zeros(self.num_classes, dtype=np.float64)
        probs[4] = 1.0
        return probs


def png_bytes(size=(224, 224), color=(120, 180, 90)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def post(crop, data: bytes, family=None, filename="leaf.png", content_type="image/png"):
    fields = {"crop": crop}
    if family is not None:
        fields["family"] = family
    return client.post(
        "/predict",
        data=fields,
        files={"file": (filename, data, content_type)},
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    inference.clear_cache()
    yield
    inference.clear_cache()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_fine_tuned_predictions", True)


def test_default_family_is_baseline(monkeypatch) -> None:
    adapter = FakeBaselineAdapter()
    seen = {}

    def resolve(crop, family):
        seen["f"] = family
        return adapter

    monkeypatch.setattr(inference, "_resolve_adapter", resolve)
    response = post("maize", png_bytes())
    assert response.status_code == 200
    assert seen["f"] == "baseline"
    assert response.json()["model"]["family"] == "baseline"


def test_unsupported_family(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_resolve_adapter", lambda crop, family: FakeBaselineAdapter())
    response = post("maize", png_bytes(), family="ensemble")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_family"


def test_fine_tuned_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_fine_tuned_predictions", False)
    response = post("maize", png_bytes(), family="fine_tuned")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "fine_tuned_preprocessing_unverified"
    assert "Traceback" not in body["error"]["message"]


def test_fine_tuned_gate_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_fine_tuned_predictions", False)
    response = post("maize", png_bytes(), family="  Fine_Tuned  ")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "fine_tuned_preprocessing_unverified"


def test_fine_tuned_enabled_passes_family_and_labels(monkeypatch, enabled) -> None:
    adapter = FakeFineTunedAdapter()
    seen = {}

    def resolve(crop, family):
        seen["f"] = family
        return adapter

    monkeypatch.setattr(inference, "_resolve_adapter", resolve)
    response = post("maize", png_bytes(), family="fine_tuned")
    assert response.status_code == 200
    assert seen["f"] == "fine_tuned"
    body = response.json()
    assert body["model"]["family"] == "fine_tuned"
    assert body["model"]["num_classes"] == 9
    assert body["predicted_label"] == "ft4"
    assert len(body["probabilities"]) == 9
    assert sum(p["probability"] for p in body["probabilities"]) == pytest.approx(1.0, abs=1e-9)


def test_fine_tuned_unavailable_crop_when_enabled(monkeypatch, enabled) -> None:
    def unavailable(crop):
        raise ApiError(503, "model_unavailable", "nope")

    monkeypatch.setattr(inference, "get_fine_tuned_adapter", unavailable)
    response = post("potato", png_bytes(), family="fine_tuned")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_fine_tuned_load_error_maps_to_503(monkeypatch) -> None:
    class Boom:
        def __init__(self, crop):
            raise ValueError("bad checkpoint")

    monkeypatch.setattr(inference, "TorchvisionEfficientNetAdapter", Boom)
    with pytest.raises(ApiError) as excinfo:
        inference.get_fine_tuned_adapter("maize")
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "model_load_failed"


def test_adapter_cache_identity_and_clear(monkeypatch) -> None:
    class Counting:
        count = 0

        def __init__(self, crop):
            type(self).count += 1

    monkeypatch.setattr(inference, "BaselineConvNeXtAdapter", Counting)
    first = inference.get_baseline_adapter("maize")
    second = inference.get_baseline_adapter("maize")
    assert first is second
    assert Counting.count == 1

    inference.clear_cache()
    third = inference.get_baseline_adapter("maize")
    assert third is not first
    assert Counting.count == 2


def test_baseline_and_fine_tuned_caches_are_independent(monkeypatch) -> None:
    class CountingBaseline:
        count = 0

        def __init__(self, crop):
            type(self).count += 1

    class CountingFineTuned:
        count = 0

        def __init__(self, crop):
            type(self).count += 1

    monkeypatch.setattr(inference, "BaselineConvNeXtAdapter", CountingBaseline)
    monkeypatch.setattr(inference, "TorchvisionEfficientNetAdapter", CountingFineTuned)

    inference.get_baseline_adapter("maize")
    inference.get_fine_tuned_adapter("maize")
    assert CountingBaseline.count == 1
    assert CountingFineTuned.count == 1

    inference.clear_cache()
    inference.get_baseline_adapter("maize")
    inference.get_fine_tuned_adapter("maize")
    assert CountingBaseline.count == 2
    assert CountingFineTuned.count == 2
