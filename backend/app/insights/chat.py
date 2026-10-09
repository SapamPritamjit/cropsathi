"""Groq chat-completions client for crop Q&A (Increment 9P).

Sends a bounded ``messages`` array to the OpenAI-compatible Groq chat endpoint
(``settings.groq_chat_url``) and returns the model's answer plus the model ID.
Like the speech services, this module:

- uses ``httpx`` directly (no extra dependency);
- reads the API key via ``config.get_groq_api_key()`` and never logs, stores on
  the result, or surfaces it;
- treats upstream bodies/messages as untrusted: control characters are stripped,
  secret-like tokens are redacted, diagnostics are capped, and the returned
  answer is length-bounded;
- accepts an injectable ``httpx.AsyncClient`` so tests use ``MockTransport``.

No answer is fabricated: an empty/unreadable/oversized upstream response raises
``ApiError`` rather than falling back to canned text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from ..config import get_groq_api_key, settings
from ..errors import ApiError

# Defensive sanity cap on the returned answer (well above a normal bounded
# completion). A larger body is treated as an upstream error, never returned.
_MAX_ANSWER_CHARS = 12000

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_DIAGNOSTIC_TOKEN = 64
_MAX_DIAGNOSTIC_MESSAGE = 200
_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]")
_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]+|Bearer\s+\S+)", re.IGNORECASE)


@dataclass(frozen=True)
class ChatAnswer:
    """A successful chat completion. Contains no secrets."""

    text: str
    model: str


def _safe_token(value: object, limit: int = _MAX_DIAGNOSTIC_TOKEN) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _TOKEN_RE.sub("", value).strip()
    return cleaned[:limit] or None


def _safe_message(value: object, limit: int = _MAX_DIAGNOSTIC_MESSAGE) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", _CONTROL_RE.sub(" ", value)).strip()
    if not cleaned:
        return None
    cleaned = _SECRET_RE.sub("[redacted]", cleaned)
    return cleaned[:limit]


def _upstream_diagnostics(response: httpx.Response) -> dict[str, object]:
    """Defensively extract safe diagnostics; never raises."""
    try:
        payload = response.json()
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error")
    if not isinstance(error, dict):
        return {}
    diagnostics: dict[str, object] = {}
    err_type = _safe_token(error.get("type"))
    if err_type:
        diagnostics["upstream_type"] = err_type
    err_code = _safe_token(error.get("code"))
    if err_code:
        diagnostics["upstream_code"] = err_code
    err_message = _safe_message(error.get("message"))
    if err_message:
        diagnostics["upstream_message"] = err_message
    return diagnostics


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    details: dict[str, object] = {"upstream_status": status}
    details.update(_upstream_diagnostics(response))
    if status in (401, 403):
        raise ApiError(
            502,
            "insights_upstream_auth_error",
            "The chat service rejected the server credentials.",
            details,
        )
    if status == 429:
        raise ApiError(
            503,
            "insights_rate_limited",
            "The chat service is rate limited. Please try again shortly.",
            details,
        )
    raise ApiError(
        502,
        "insights_upstream_error",
        "The chat service returned an error.",
        details,
    )


def _clean_answer(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_answer(payload: object) -> str:
    """Extract ``choices[0].message.content`` from an OpenAI-style body.

    Returns an empty string for any malformed shape so the caller can emit a
    stable, non-fabricated error.
    """
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    return _clean_answer(message.get("content"))


async def generate_answer(
    messages: object,
    *,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> ChatAnswer:
    """Ask the configured chat model and return its bounded answer.

    Raises ``ApiError`` for missing configuration, missing credentials, and all
    upstream failures (timeout, rate limit, auth, transport, empty/malformed
    output). Never returns a fabricated answer.
    """
    if not isinstance(messages, list) or not messages:
        raise ApiError(
            500, "insights_configuration_error", "No prompt messages were provided."
        )

    resolved_model = model or settings.groq_chat_model
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise ApiError(
            500, "insights_configuration_error", "The configured chat model is invalid."
        )

    api_key = get_groq_api_key()
    if not api_key:
        raise ApiError(
            503,
            "insights_not_configured",
            "Crop question answering is not configured on the server.",
        )

    payload = {
        "model": resolved_model,
        "messages": messages,
        "temperature": 0.2,
        "max_completion_tokens": settings.groq_chat_max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=settings.groq_chat_timeout_seconds)
    try:
        try:
            response = await client.post(
                settings.groq_chat_url, headers=headers, json=payload
            )
        except httpx.TimeoutException as exc:
            raise ApiError(
                504, "insights_timeout", "The chat service timed out."
            ) from exc
        except httpx.TransportError as exc:
            raise ApiError(
                502,
                "insights_upstream_unavailable",
                "The chat service could not be reached.",
            ) from exc
    finally:
        if own_client:
            await client.aclose()

    _raise_for_status(response)

    try:
        body = response.json()
    except ValueError as exc:
        raise ApiError(
            502,
            "insights_upstream_error",
            "The chat service returned an unreadable response.",
        ) from exc

    text = extract_answer(body)
    if not text:
        raise ApiError(
            502,
            "insights_empty_answer",
            "The chat service returned an empty answer.",
        )
    if len(text) > _MAX_ANSWER_CHARS:
        raise ApiError(
            502,
            "insights_upstream_error",
            "The chat service returned an unexpectedly large response.",
        )

    return ChatAnswer(text=text, model=resolved_model)
