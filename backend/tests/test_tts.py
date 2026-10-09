"""Unit tests for the Groq text-to-speech service (Part 6B).

All upstream calls use ``httpx.MockTransport`` so the default suite makes **no**
real network requests and needs no API key of its own.
"""

import asyncio
import json

import httpx
import pytest

from app.config import settings
from app.errors import ApiError
from app.speech import tts

AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt "


def run(coro):
    return asyncio.run(coro)


def client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


def ok_handler(captured: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["json"] = json.loads(request.content.decode())
            captured["auth"] = request.headers.get("authorization")
            captured["url"] = str(request.url)
            captured["accept"] = request.headers.get("accept")
        return httpx.Response(
            200, content=AUDIO, headers={"content-type": "audio/wav"}
        )

    return handler


def status_handler(status: int, body: bytes = b"{}"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body, headers={"content-type": "application/json"}
        )

    return handler


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-abc")


# --- input normalization ---------------------------------------------------- #


def test_normalize_text_collapses_whitespace_and_controls() -> None:
    assert tts.normalize_text("  Hello\t\tfarmer\n\n world  ") == "Hello farmer world"
    assert tts.normalize_text("a\x00b\x07c") == "a b c"


def test_normalize_text_empty_returns_blank() -> None:
    assert tts.normalize_text(None) == ""
    assert tts.normalize_text("   \n\t ") == ""


def test_normalize_text_rejects_non_string() -> None:
    with pytest.raises(ApiError) as excinfo:
        tts.normalize_text(123)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "tts_invalid_text"


# --- success path ----------------------------------------------------------- #


def test_synthesize_success_builds_payload_and_returns_audio() -> None:
    captured: dict = {}
    result = run(
        tts.synthesize_speech(
            "  Hello   farmer\n\nworld ", client=client_with(ok_handler(captured))
        )
    )
    assert result.data == AUDIO
    assert result.content_type == "audio/wav"
    assert result.format == "wav"
    assert result.model == settings.groq_tts_model
    assert result.voice == settings.groq_tts_voice

    assert captured["json"] == {
        "model": settings.groq_tts_model,
        "input": "Hello farmer world",
        "voice": settings.groq_tts_voice,
        "response_format": "wav",
    }
    assert captured["auth"] == "Bearer test-key-abc"
    assert captured["url"] == settings.groq_tts_url
    assert captured["accept"] == "audio/wav"


def test_synthesize_respects_explicit_voice_override() -> None:
    captured: dict = {}
    result = run(
        tts.synthesize_speech(
            "hi", voice="troy", client=client_with(ok_handler(captured))
        )
    )
    assert result.voice == "troy"
    assert captured["json"]["voice"] == "troy"


# --- local validation ------------------------------------------------------- #


def test_empty_text_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("   \n  ", client=client_with(ok_handler())))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "tts_empty_text"


def test_non_string_text_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech(42, client=client_with(ok_handler())))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "tts_invalid_text"


def test_text_too_long_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("a" * 201, client=client_with(ok_handler())))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "tts_text_too_long"
    assert excinfo.value.details == {"max_characters": 200, "characters": 201}


def test_exactly_max_length_allowed() -> None:
    result = run(tts.synthesize_speech("a" * 200, client=client_with(ok_handler())))
    assert result.data == AUDIO


# --- configuration ---------------------------------------------------------- #


def test_missing_configuration(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ-API-KEY", raising=False)
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(ok_handler())))
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "tts_not_configured"


def test_unsupported_model_config(monkeypatch) -> None:
    monkeypatch.setattr(settings, "groq_tts_model", "playai-tts")
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(ok_handler())))
    assert excinfo.value.status_code == 500
    assert excinfo.value.code == "tts_configuration_error"


def test_unsupported_voice_config(monkeypatch) -> None:
    monkeypatch.setattr(settings, "groq_tts_voice", "not-a-real-voice")
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(ok_handler())))
    assert excinfo.value.status_code == 500
    assert excinfo.value.code == "tts_configuration_error"


def test_unsupported_format_config(monkeypatch) -> None:
    monkeypatch.setattr(settings, "groq_tts_response_format", "mp3")
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(ok_handler())))
    assert excinfo.value.status_code == 500
    assert excinfo.value.code == "tts_configuration_error"


# --- upstream error mapping ------------------------------------------------- #


def test_upstream_auth_error(monkeypatch) -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(status_handler(401))
            )
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_upstream_auth_error"


def test_rate_limited() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(status_handler(429))
            )
        )
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "tts_rate_limited"


def test_upstream_server_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(status_handler(500))
            )
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_upstream_error"


def test_bad_request_maps_to_upstream_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(status_handler(400))
            )
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_upstream_error"


def test_timeout_maps_to_gateway_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow", request=request)

    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(handler)))
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "tts_timeout"


def test_transport_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(handler)))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_upstream_unavailable"


def test_empty_audio_body_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"", headers={"content-type": "audio/wav"})

    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(handler)))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_empty_audio"


def test_oversized_audio_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "groq_tts_max_audio_bytes", 4)
    with pytest.raises(ApiError) as excinfo:
        run(tts.synthesize_speech("hello", client=client_with(ok_handler())))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_audio_too_large"


# --- upstream error diagnostics (Part 6F-3) --------------------------------- #


def error_handler(status: int, body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body, headers={"content-type": "application/json"}
        )

    return handler


def test_bad_request_surfaces_safe_diagnostics() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Invalid value for 'voice'.",
                "type": "invalid_request_error",
                "code": "invalid_value",
            }
        }
    ).encode()
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(400, body))
            )
        )
    err = excinfo.value
    assert err.status_code == 502
    assert err.code == "tts_upstream_error"
    assert err.details["upstream_status"] == 400
    assert err.details["upstream_type"] == "invalid_request_error"
    assert err.details["upstream_code"] == "invalid_value"
    assert err.details["upstream_message"] == "Invalid value for 'voice'."


def test_malformed_json_body_does_not_break_mapping() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(400, b"{not json"))
            )
        )
    err = excinfo.value
    assert err.code == "tts_upstream_error"
    assert err.details == {"upstream_status": 400}


def test_non_object_json_body_is_ignored() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(400, b'["oops"]'))
            )
        )
    assert excinfo.value.details == {"upstream_status": 400}


def test_missing_diagnostic_fields_are_omitted() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(400, b'{"error": {}}'))
            )
        )
    assert excinfo.value.details == {"upstream_status": 400}


def test_error_field_not_object_is_ignored() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello",
                client=client_with(error_handler(400, b'{"error": "nope"}')),
            )
        )
    assert excinfo.value.details == {"upstream_status": 400}


def test_unexpected_diagnostic_field_types_are_ignored() -> None:
    body = json.dumps(
        {"error": {"type": 7, "code": ["x"], "message": {"a": 1}}}
    ).encode()
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(400, body))
            )
        )
    assert excinfo.value.details == {"upstream_status": 400}


def test_malicious_oversized_message_is_sanitized() -> None:
    huge = ("A" * 5000) + "\x00\x07" + "gsk_supersecret123"
    body = json.dumps({"error": {"message": huge, "type": "x" * 500}}).encode()
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(400, body))
            )
        )
    err = excinfo.value
    message = err.details["upstream_message"]
    assert isinstance(message, str)
    assert len(message) <= 200
    assert "\x00" not in message and "\x07" not in message
    blob = err.message + json.dumps(err.details) + str(err)
    assert "gsk_supersecret123" not in blob
    assert len(err.details["upstream_type"]) <= 64


def test_forbidden_mapping_preserved_with_diagnostics() -> None:
    body = json.dumps({"error": {"message": "blocked", "type": "permissions_error"}}).encode()
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(403, body))
            )
        )
    err = excinfo.value
    assert err.status_code == 502
    assert err.code == "tts_upstream_auth_error"
    assert err.details["upstream_status"] == 403
    assert err.details["upstream_type"] == "permissions_error"


def test_rate_limited_mapping_preserved_with_diagnostics() -> None:
    body = json.dumps({"error": {"type": "rate_limit_error"}}).encode()
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(error_handler(429, body))
            )
        )
    err = excinfo.value
    assert err.status_code == 503
    assert err.code == "tts_rate_limited"
    assert err.details["upstream_status"] == 429


def test_server_error_mapping_preserved() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(status_handler(500))
            )
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "tts_upstream_error"


# --- secret safety ---------------------------------------------------------- #


def test_secret_not_in_errors() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            tts.synthesize_speech(
                "hello", client=client_with(status_handler(401))
            )
        )
    err = excinfo.value
    blob = err.message + json.dumps(err.details) + str(err)
    assert "test-key-abc" not in blob
