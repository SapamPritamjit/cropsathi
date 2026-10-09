"""Route tests for ``POST /speech/transcribe`` (Part 6D).

The STT service is monkeypatched on the router module; upload validation runs
for real. No test makes a real network request.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.errors import ApiError
from app.main import app
from app.routers import speech as speech_router
from app.speech.stt import Transcription

client = TestClient(app)

AUDIO = b"RIFF....WAVEfmt fake-audio-bytes"


@pytest.fixture
def fake_service(monkeypatch) -> dict:
    captured: dict = {}

    async def fake_transcribe(data, **kwargs) -> Transcription:
        captured["data"] = data
        captured.update(kwargs)
        return Transcription(text="rice leaves show blast", language="english")

    monkeypatch.setattr(speech_router, "transcribe_audio", fake_transcribe)
    return captured


def test_transcribe_success_returns_json(fake_service) -> None:
    response = client.post(
        "/speech/transcribe", files={"file": ("clip.wav", AUDIO, "audio/wav")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"text": "rice leaves show blast", "language": "english"}
    assert fake_service["data"] == AUDIO
    assert fake_service["filename"] == "clip.wav"


def test_transcribe_forwards_language_hint(fake_service) -> None:
    response = client.post(
        "/speech/transcribe",
        files={"file": ("clip.wav", AUDIO, "audio/wav")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    assert fake_service["language"] == "en"


def test_transcribe_empty_upload_is_400(fake_service) -> None:
    response = client.post(
        "/speech/transcribe", files={"file": ("clip.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "stt_empty_audio"


def test_transcribe_oversized_upload_is_413(fake_service, monkeypatch) -> None:
    monkeypatch.setattr(settings, "stt_max_upload_bytes", 4)
    response = client.post(
        "/speech/transcribe", files={"file": ("clip.wav", b"12345", "audio/wav")}
    )
    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "stt_audio_too_large"
    assert body["error"]["details"] == {"max_upload_bytes": 4}


def test_transcribe_invalid_media_type_is_415(fake_service) -> None:
    response = client.post(
        "/speech/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "stt_unsupported_media_type"


def test_transcribe_octet_stream_with_audio_extension_ok(fake_service) -> None:
    response = client.post(
        "/speech/transcribe",
        files={"file": ("clip.mp3", AUDIO, "application/octet-stream")},
    )
    assert response.status_code == 200


def test_transcribe_missing_file_is_422_envelope() -> None:
    response = client.post("/speech/transcribe")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "status,code",
    [
        (503, "stt_not_configured"),
        (502, "stt_upstream_auth_error"),
        (503, "stt_rate_limited"),
        (502, "stt_upstream_error"),
        (502, "stt_upstream_unavailable"),
        (504, "stt_timeout"),
        (422, "stt_empty_transcription"),
        (400, "stt_invalid_language"),
    ],
)
def test_transcribe_maps_service_errors(monkeypatch, status: int, code: str) -> None:
    async def fake_transcribe(data, **kwargs):
        raise ApiError(status, code, "safe message", {"upstream_status": 418})

    monkeypatch.setattr(speech_router, "transcribe_audio", fake_transcribe)
    response = client.post(
        "/speech/transcribe", files={"file": ("clip.wav", AUDIO, "audio/wav")}
    )
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert "test-key" not in response.text
