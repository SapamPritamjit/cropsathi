"""Integration tests across prediction -> response builder -> TTS (Part 6F-1).

These exercise the real ``build_prediction_speech`` and the real TTS service,
but every upstream HTTP request is fulfilled by ``httpx.MockTransport`` so the
suite makes **no** real network calls and needs no real Groq key. No new route
or production code is added; existing interfaces are reused as-is.

``test_real_predict_to_speech_*`` is marked ``smoke`` because it loads the real
baseline checkpoint.
"""

import asyncio
import json
from io import BytesIO

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.errors import ApiError
from app.main import app
from app.schemas import PredictionModelMeta, PredictResponse
from app.speech import tts
from app.speech.text import build_prediction_speech

AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt "

PREDICTION = {
    "crop": "maize",
    "display_name": "Maize",
    "predicted_index": 1,
    "predicted_label": "Northern Leaf Blight",
    "confidence": 0.929,
    "model": {
        "crop": "maize",
        "family": "baseline",
        "implementation": "timm:convnext_small",
        "num_classes": 11,
        "input_size": [3, 224, 224],
        "weights": "model.safetensors",
        "label_source": "config.json",
    },
    "disclaimer": "Confidence is the model's softmax probability.",
}


def run(coro):
    return asyncio.run(coro)


def capturing_client(captured: dict, status: int = 200, content: bytes = AUDIO):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            status, content=content, headers={"content-type": "audio/wav"}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-abc")


def synthetic_png() -> bytes:
    rng = np.random.default_rng(3)
    array = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(array, "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# --- happy path: prediction -> builder -> TTS ------------------------------- #


def test_prediction_payload_flows_into_tts_request() -> None:
    text = build_prediction_speech(PREDICTION)
    captured: dict = {}
    client = capturing_client(captured)

    audio = run(tts.synthesize_speech(text, client=client))

    assert captured["json"]["input"] == text
    assert captured["url"] == settings.groq_tts_url
    assert captured["json"]["model"] == settings.groq_tts_model
    assert audio.data == AUDIO
    assert audio.content_type == "audio/wav"


def test_generated_text_is_within_limit_and_keeps_uncertainty() -> None:
    text = build_prediction_speech(PREDICTION)
    assert len(text) <= settings.groq_tts_max_input_chars
    assert len(text) <= 200
    assert "prediction" in text.lower()
    assert "not a confirmed diagnosis" in text


def test_predict_response_model_flows_into_tts_request() -> None:
    model = PredictResponse(
        crop="rice",
        display_name="Rice",
        model=PredictionModelMeta(
            crop="rice",
            family="baseline",
            implementation="timm:convnext_small",
            num_classes=14,
            input_size=[3, 224, 224],
            weights="model.safetensors",
            label_source="config.json",
        ),
        predicted_index=3,
        predicted_label="Neck Blast",
        confidence=0.94,
        probabilities=[],
        disclaimer="softmax probability, not a guarantee.",
    )
    text = build_prediction_speech(model)
    captured: dict = {}

    audio = run(tts.synthesize_speech(text, client=capturing_client(captured)))

    assert captured["json"]["input"] == text
    assert "Rice" in text and "Neck Blast" in text
    assert audio.data == AUDIO


@pytest.mark.smoke
def test_real_predict_to_speech_end_to_end() -> None:
    response = TestClient(app).post(
        "/predict",
        data={"crop": "potato"},
        files={"file": ("leaf.png", synthetic_png(), "image/png")},
    )
    assert response.status_code == 200

    text = build_prediction_speech(response.json())
    captured: dict = {}
    audio = run(tts.synthesize_speech(text, client=capturing_client(captured)))

    assert captured["json"]["input"] == text
    assert len(text) <= 200
    assert "not a confirmed diagnosis" in text
    assert audio.data == AUDIO


# --- boundary failures ------------------------------------------------------ #


def test_invalid_prediction_fails_before_any_tts_call() -> None:
    with pytest.raises(ApiError) as excinfo:
        build_prediction_speech({"predicted_label": "X"})

    # builder raises before any TTS payload is constructed
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "response_missing_crop"


def test_missing_tts_configuration_raises_503(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ-API-KEY", raising=False)
    text = build_prediction_speech(PREDICTION)
    captured: dict = {}
    client = capturing_client(captured)

    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech(text, client=client))

    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "tts_not_configured"
    assert captured == {}


def test_tts_upstream_failure_raises_502() -> None:
    text = build_prediction_speech(PREDICTION)
    captured: dict = {}
    client = capturing_client(captured, status=500, content=b"{}")

    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech(text, client=client))

    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_upstream_error"
    assert captured["json"]["input"] == text
