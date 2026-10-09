"""Unit tests for the Groq chat-completions client (Increment 9P).

All upstream calls use ``httpx.MockTransport`` so the default suite makes **no**
real network requests and needs no API key of its own.
"""

import asyncio
import json

import httpx
import pytest

from app.config import settings
from app.errors import ApiError
from app.insights import chat

MESSAGES = [
    {"role": "system", "content": "You are an agricultural advisor."},
    {"role": "user", "content": "How do I water maize?"},
]


def run(coro):
    return asyncio.run(coro)


def client_with(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


def ok_handler(content: str = "Water maize when the topsoil is dry.", captured: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["json"] = json.loads(request.content.decode())
            captured["auth"] = request.headers.get("authorization")
            captured["url"] = str(request.url)
            captured["accept"] = request.headers.get("accept")
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]},
        )

    return handler


def body_handler(status: int, body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body, headers={"content-type": "application/json"}
        )

    return handler


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-abc")


# --- success path ----------------------------------------------------------- #


def test_generate_answer_success_builds_payload() -> None:
    captured: dict = {}
    result = run(
        chat.generate_answer(MESSAGES, client=client_with(ok_handler(captured=captured)))
    )
    assert result.text == "Water maize when the topsoil is dry."
    assert result.model == settings.groq_chat_model

    payload = captured["json"]
    assert payload["model"] == settings.groq_chat_model
    assert payload["messages"] == MESSAGES
    assert payload["temperature"] == 0.2
    assert payload["max_completion_tokens"] == settings.groq_chat_max_tokens
    assert captured["auth"] == "Bearer test-key-abc"
    assert captured["url"] == settings.groq_chat_url
    assert captured["accept"] == "application/json"


def test_explicit_model_override() -> None:
    captured: dict = {}
    result = run(
        chat.generate_answer(
            MESSAGES, model="llama-3.1-8b-instant", client=client_with(ok_handler(captured=captured))
        )
    )
    assert result.model == "llama-3.1-8b-instant"
    assert captured["json"]["model"] == "llama-3.1-8b-instant"


# --- local configuration ---------------------------------------------------- #


def test_missing_messages_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer([], client=client_with(ok_handler())))
    assert excinfo.value.status_code == 500
    assert excinfo.value.code == "insights_configuration_error"


def test_invalid_model_config_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "groq_chat_model", "   ")
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(ok_handler())))
    assert excinfo.value.status_code == 500
    assert excinfo.value.code == "insights_configuration_error"


def test_missing_configuration(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ-API-KEY", raising=False)
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(ok_handler())))
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "insights_not_configured"


# --- response parsing ------------------------------------------------------- #


def test_empty_choices_raises_empty_answer() -> None:
    handler = body_handler(200, b'{"choices": []}')
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(handler)))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_empty_answer"


def test_blank_content_raises_empty_answer() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(ok_handler("   \n  "))))
    assert excinfo.value.code == "insights_empty_answer"


def test_non_string_content_raises_empty_answer() -> None:
    handler = body_handler(200, json.dumps({"choices": [{"message": {"content": 7}}]}).encode())
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(handler)))
    assert excinfo.value.code == "insights_empty_answer"


def test_malformed_json_body_raises_upstream_error() -> None:
    handler = body_handler(200, b"{not json")
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(handler)))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_upstream_error"


def test_answer_control_characters_are_cleaned() -> None:
    raw = "Line one\x00\x07\r\n\r\nLine two\x0b"
    result = run(chat.generate_answer(MESSAGES, client=client_with(ok_handler(raw))))
    assert result.text == "Line one\n\nLine two"
    assert "\x00" not in result.text and "\x07" not in result.text


def test_oversized_answer_rejected() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(
            chat.generate_answer(
                MESSAGES, client=client_with(ok_handler("A" * (chat._MAX_ANSWER_CHARS + 1)))
            )
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_upstream_error"


def test_extract_answer_tolerates_malformed_shapes() -> None:
    assert chat.extract_answer(None) == ""
    assert chat.extract_answer("nope") == ""
    assert chat.extract_answer({"choices": "nope"}) == ""
    assert chat.extract_answer({"choices": ["nope"]}) == ""
    assert chat.extract_answer({"choices": [{"message": None}]}) == ""
    assert chat.extract_answer({"choices": [{"message": {"content": None}}]}) == ""


# --- upstream error mapping ------------------------------------------------- #


def test_upstream_auth_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(401, b"{}"))))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_upstream_auth_error"


def test_forbidden_maps_to_auth_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(403, b"{}"))))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_upstream_auth_error"


def test_rate_limited() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(429, b"{}"))))
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "insights_rate_limited"


def test_upstream_server_error() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(500, b"{}"))))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_upstream_error"


def test_timeout_maps_to_gateway_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow", request=request)

    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(handler)))
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "insights_timeout"


def test_transport_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(handler)))
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "insights_upstream_unavailable"


# --- diagnostics + secret safety -------------------------------------------- #


def test_bad_request_surfaces_safe_diagnostics() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Invalid value for 'model'.",
                "type": "invalid_request_error",
                "code": "invalid_value",
            }
        }
    ).encode()
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(400, body))))
    err = excinfo.value
    assert err.status_code == 502
    assert err.code == "insights_upstream_error"
    assert err.details["upstream_status"] == 400
    assert err.details["upstream_type"] == "invalid_request_error"
    assert err.details["upstream_code"] == "invalid_value"
    assert err.details["upstream_message"] == "Invalid value for 'model'."


def test_malformed_error_body_does_not_break_mapping() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(400, b"{bad"))))
    assert excinfo.value.details == {"upstream_status": 400}


def test_malicious_message_is_sanitized_and_redacted() -> None:
    huge = ("A" * 5000) + "\x00\x07" + "gsk_supersecret123"
    body = json.dumps({"error": {"message": huge, "type": "x" * 500}}).encode()
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(400, body))))
    err = excinfo.value
    message = err.details["upstream_message"]
    assert len(message) <= 200
    assert "\x00" not in message and "\x07" not in message
    blob = err.message + json.dumps(err.details) + str(err)
    assert "gsk_supersecret123" not in blob
    assert len(err.details["upstream_type"]) <= 64


def test_secret_not_in_errors() -> None:
    with pytest.raises(ApiError) as excinfo:
        run(chat.generate_answer(MESSAGES, client=client_with(body_handler(401, b"{}"))))
    err = excinfo.value
    blob = err.message + json.dumps(err.details) + str(err)
    assert "test-key-abc" not in blob


def test_result_carries_no_secret() -> None:
    result = run(chat.generate_answer(MESSAGES, client=client_with(ok_handler())))
    assert not hasattr(result, "api_key")
    assert "test-key-abc" not in repr(result)
