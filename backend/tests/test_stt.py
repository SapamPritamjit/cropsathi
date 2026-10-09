"""Unit tests for the Groq Whisper speech-to-text service (Part 6D).

Upstream calls use ``httpx.MockTransport`` so the default suite makes no real
network requests.
"""

import asyncio
import json

import httpx
import pytest

from app.config import settings
from app.errors import ApiError
from app.speech import stt

AUDIO = b"RIFF....WAVEfmt fake-audio-bytes"


def run(coro):
    return asyncio.run(coro)


def client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


def ok_handler(captured: dict | None = None, payload: dict | None = None):
    body = payload if payload is not None else {"text": "maize leaves have rust", "language": "English"}

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["content_type"] = request.headers.get("content-type", "")
            captured["body"] = request.content
        return httpx.Response(200, content=json.dumps(body).encode(), headers={"content-type": "application/json"})

    return handler


def status_handler(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"{}", headers={"content-type": "application/json"})

    return handler


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-abc")


def call(handler, **kwargs):
    return run(
        stt.transcribe_audio(
            kwargs.pop("data", AUDIO),
            filename=kwargs.pop("filename", "clip.wav"),
            content_type=kwargs.pop("content_type", "audio/wav"),
            client=client_with(handler),
            **kwargs,
        )
    )


# --- success ---------------------------------------------------------------- #


def test_transcribe_success_returns_text_and_language() -> None:
    result = call(ok_handler())
    assert result.text == "maize leaves have rust"
    assert result.language == "english"


def test_transcribe_missing_language_is_none() -> None:
    result = call(ok_handler(payload={"text": "hello"}))
    assert result.text == "hello"
    assert result.language is None


def test_transcribe_builds_multipart_request() -> None:
    captured: dict = {}
    call(ok_handler(captured))
    assert captured["auth"] == "Bearer test-key-abc"
    assert captured["url"] == settings.groq_stt_url
    assert "multipart/form-data" in captured["content_type"]
    blob = captured["body"]
    assert b'name="model"' in blob
    assert settings.groq_stt_model.encode() in blob
    assert b'name="file"' in blob
    assert b"fake-audio-bytes" in blob


def test_transcribe_passes_language_hint() -> None:
    captured: dict = {}
    call(ok_handler(captured), language="EN")
    assert b'name="language"' in captured["body"]
    assert b"\r\nen\r\n" in captured["body"]


def test_transcribe_omits_language_when_absent() -> None:
    captured: dict = {}
    call(ok_handler(captured))
    assert b'name="language"' not in captured["body"]


# --- upload validation ------------------------------------------------------ #


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("a.flac", "application/octet-stream"),
        ("a.mp3", "audio/mpeg"),
        ("a.mp4", "video/mp4"),
        ("a.mpeg", None),
        ("a.mpga", None),
        ("a.m4a", "audio/mp4"),
        ("a.ogg", "audio/ogg"),
        ("a.wav", "audio/wav"),
        ("a.webm", "audio/webm"),
    ],
)
def test_supported_audio_types_accepted(filename: str, content_type) -> None:
    result = call(ok_handler(), filename=filename, content_type=content_type)
    assert result.text


def test_unsupported_media_type_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler(), filename="notes.txt", content_type="text/plain")
    assert excinfo.value.status_code == 415
    assert excinfo.value.code == "stt_unsupported_media_type"


def test_empty_audio_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler(), data=b"")
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "stt_empty_audio"


def test_oversized_audio_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "stt_max_upload_bytes", 4)
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler(), data=b"12345")
    assert excinfo.value.status_code == 413
    assert excinfo.value.code == "stt_audio_too_large"


def test_invalid_language_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler(), language="english")
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "stt_invalid_language"


def test_blank_language_is_none() -> None:
    assert stt.normalize_language("   ") is None
    assert stt.normalize_language(None) is None


# --- configuration ---------------------------------------------------------- #


def test_missing_configuration(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ-API-KEY", raising=False)
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler())
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "stt_not_configured"


def test_unsupported_model_config(monkeypatch) -> None:
    monkeypatch.setattr(settings, "groq_stt_model", "not-a-model")
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler())
    assert excinfo.value.status_code == 500
    assert excinfo.value.code == "stt_configuration_error"


# --- upstream results and errors ------------------------------------------- #


def test_empty_transcription_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(ok_handler(payload={"text": "   ", "language": "English"}))
    assert excinfo.value.status_code == 422
    assert excinfo.value.code == "stt_empty_transcription"


def test_unreadable_json_is_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "text/plain"})

    with pytest.raises(ApiError) as excinfo:
        call(handler)
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "stt_upstream_error"


def test_upstream_auth_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(status_handler(401))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "stt_upstream_auth_error"


def test_rate_limited() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(status_handler(429))
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "stt_rate_limited"


def test_upstream_server_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(status_handler(500))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "stt_upstream_error"


def test_timeout_maps_to_gateway_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with pytest.raises(ApiError) as excinfo:
        call(handler)
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "stt_timeout"


def test_transport_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ApiError) as excinfo:
        call(handler)
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "stt_upstream_unavailable"


def test_secret_not_in_errors() -> None:
    with pytest.raises(ApiError) as excinfo:
        call(status_handler(401))
    err = excinfo.value
    blob = err.message + json.dumps(err.details) + str(err)
    assert "test-key-abc" not in blob
