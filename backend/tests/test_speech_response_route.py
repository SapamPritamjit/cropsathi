"""Route tests for ``POST /speech/response`` (Part 6H).

Exercises the real ``build_prediction_speech`` builder through the HTTP layer.
No network and no external (Groq) calls: the speech services are monkeypatched
to fail loudly if the endpoint ever tried to reach them.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import speech as speech_router
from app.speech.text import build_prediction_speech

client = TestClient(app)

PREDICTION = {
    "crop": "maize",
    "display_name": "Maize",
    "predicted_index": 1,
    "predicted_label": "Northern Leaf Blight",
    "confidence": 0.929,
    "model": {"crop": "maize", "family": "baseline"},
    "probabilities": [{"index": 1, "label": "Northern Leaf Blight", "probability": 0.929}],
    "disclaimer": "Confidence is the model's softmax probability.",
}


@pytest.fixture(autouse=True)
def _no_external_calls(monkeypatch):
    async def _boom(*_args, **_kwargs):
        raise AssertionError("the endpoint must not call speech services")

    monkeypatch.setattr(speech_router, "synthesize_speech", _boom)
    monkeypatch.setattr(speech_router, "transcribe_audio", _boom)


# --- success ---------------------------------------------------------------- #


def test_valid_full_prediction_returns_builder_text() -> None:
    response = client.post("/speech/response", json=PREDICTION)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body == {"text": build_prediction_speech(PREDICTION)}
    assert "Maize" in body["text"]
    assert "Northern Leaf Blight" in body["text"]
    assert "92.9%" in body["text"]
    assert "not a confirmed diagnosis" in body["text"]


def test_minimal_prediction_omits_confidence() -> None:
    response = client.post(
        "/speech/response",
        json={"display_name": "Rice", "predicted_label": "Neck Blast"},
    )
    assert response.status_code == 200
    text = response.json()["text"]
    assert "Rice" in text and "Neck Blast" in text
    assert "%" not in text and "confidence" not in text
    assert "not a confirmed diagnosis" in text


def test_extra_prediction_fields_are_ignored() -> None:
    response = client.post("/speech/response", json=PREDICTION)
    assert response.status_code == 200
    assert response.json()["text"] == build_prediction_speech(PREDICTION)


# --- builder validation surfaced through the API ---------------------------- #


def test_missing_crop_is_envelope_400() -> None:
    response = client.post("/speech/response", json={"predicted_label": "Late Blight"})
    assert response.status_code == 400
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert body["error"]["code"] == "response_missing_crop"


def test_missing_class_is_envelope_400() -> None:
    response = client.post("/speech/response", json={"display_name": "Rice"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "response_missing_class"


@pytest.mark.parametrize("bad", [1.5, -0.1, True, "high", {"x": 1}])
def test_invalid_confidence_is_rejected_by_builder(bad) -> None:
    response = client.post(
        "/speech/response",
        json={"display_name": "Maize", "predicted_label": "Smut", "confidence": bad},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "response_invalid_confidence"


def test_builder_length_error_is_mapped() -> None:
    response = client.post(
        "/speech/response",
        json={"display_name": "a" * 120, "predicted_label": "b" * 120},
    )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "response_text_too_long"
    assert body["details"] == {"max_characters": 200}


# --- request validation / envelope ------------------------------------------ #


def test_non_object_body_is_validation_error() -> None:
    response = client.post("/speech/response", json=["not", "an", "object"])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_error_envelope_has_no_traceback() -> None:
    response = client.post("/speech/response", json={"predicted_label": "x"})
    assert response.status_code == 400
    assert "traceback" not in response.text.lower()
