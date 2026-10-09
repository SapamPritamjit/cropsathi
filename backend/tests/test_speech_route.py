"""Route tests for ``POST /speech/speak`` (Part 6C-2).

The TTS service is monkeypatched on the router module, so no test makes a real
network request.
"""

import pytest
from fastapi.testclient import TestClient

from app.errors import ApiError
from app.main import app
from app.routers import speech as speech_router
from app.speech.tts import SynthesizedAudio

client = TestClient(app)

WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt "


@pytest.fixture
def recorded_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def fake_synthesize(text, **_kwargs) -> SynthesizedAudio:
        calls.append(text)
        return SynthesizedAudio(
            data=WAV_BYTES,
            content_type="audio/wav",
            model="canopylabs/orpheus-v1-english",
            voice="hannah",
            format="wav",
        )

    monkeypatch.setattr(speech_router, "synthesize_speech", fake_synthesize)
    return calls


# --- success ---------------------------------------------------------------- #


def test_speak_returns_wav_bytes(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": "Maize leaf looks healthy."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == WAV_BYTES
    assert recorded_calls == ["Maize leaf looks healthy."]


def test_speak_normalizes_text_before_service(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": "  Hello\t farmer \n"})
    assert response.status_code == 200
    assert recorded_calls == ["Hello farmer"]


def test_speak_exactly_max_length_allowed(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": "a" * 200})
    assert response.status_code == 200
    assert recorded_calls == ["a" * 200]


# --- local validation ------------------------------------------------------- #


def test_speak_empty_text_is_400(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "tts_empty_text"
    assert recorded_calls == []


def test_speak_whitespace_only_is_400(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": "   \n\t "})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "tts_empty_text"


def test_speak_over_limit_is_413(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": "a" * 201})
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "tts_text_too_long"
    assert body["error"]["details"] == {"max_characters": 200, "characters": 201}
    assert recorded_calls == []


def test_speak_missing_field_is_422_envelope() -> None:
    response = client.post("/speech/speak", json={})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert "traceback" not in response.text.lower()


# --- upstream / configuration mapping -------------------------------------- #


@pytest.mark.parametrize(
    "status,code",
    [
        (503, "tts_not_configured"),
        (502, "tts_upstream_auth_error"),
        (503, "tts_rate_limited"),
        (502, "tts_upstream_error"),
        (502, "tts_upstream_unavailable"),
        (504, "tts_timeout"),
        (502, "tts_empty_audio"),
    ],
)
def test_speak_maps_service_errors(monkeypatch, status: int, code: str) -> None:
    async def fake_synthesize(text, **_kwargs):
        raise ApiError(status, code, "safe message", {"upstream_status": 418})

    monkeypatch.setattr(speech_router, "synthesize_speech", fake_synthesize)

    response = client.post("/speech/speak", json={"text": "hello"})
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert "test-key" not in response.text


def test_speak_error_envelope_shape(monkeypatch) -> None:
    async def fake_synthesize(text, **_kwargs):
        raise ApiError(502, "tts_upstream_error", "The service failed.")

    monkeypatch.setattr(speech_router, "synthesize_speech", fake_synthesize)
    response = client.post("/speech/speak", json={"text": "hello"})
    assert set(response.json().keys()) == {"error"}
    assert set(response.json()["error"].keys()) == {"code", "message", "details"}


def test_speak_introspection_is_json_not_audio(recorded_calls) -> None:
    response = client.post("/speech/speak", json={"text": "hi"})
    assert response.headers["content-type"].startswith("audio/")
    assert not response.headers["content-type"].startswith("application/json")
