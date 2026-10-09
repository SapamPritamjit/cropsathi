"""Server-side text-to-speech via Groq (Part 6B).

Synthesises audio bytes for a short text string using Groq's
``/openai/v1/audio/speech`` endpoint and returns them to the caller. This module
performs **no playback** and has **no FastAPI routes** — playback belongs to the
frontend, and the HTTP layer is a later increment.

Design notes (verified against Groq docs at implementation time):

- The only currently supported Groq TTS models are ``canopylabs/orpheus-v1-english``
  and ``canopylabs/orpheus-arabic-saudi`` (PlayAI ``playai-tts`` is no longer
  listed). Each model exposes a fixed set of voices.
- ``input`` is capped at 200 characters and the only supported output format is
  ``wav``. Blindly concatenating WAV payloads is not technically safe, so this
  module does **not** chunk or concatenate: over-length text is rejected with a
  stable error and callers must send concise text.
- The API key is read via ``config.get_groq_api_key()`` and is never logged,
  stored on the result, or included in error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from ..config import get_groq_api_key, settings
from ..errors import ApiError

# Per-model voices and output formats, from the current Groq documentation.
SUPPORTED_TTS_MODELS: dict[str, dict[str, object]] = {
    "canopylabs/orpheus-v1-english": {
        "language": "en",
        "voices": frozenset(
            {"autumn", "diana", "hannah", "austin", "daniel", "troy"}
        ),
        "formats": frozenset({"wav"}),
    },
    "canopylabs/orpheus-arabic-saudi": {
        "language": "ar",
        "voices": frozenset(
            {"abdullah", "fahad", "sultan", "lulwa", "noura", "aisha"}
        ),
        "formats": frozenset({"wav"}),
    },
}

# Control characters that must never reach the upstream model.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Diagnostic extraction limits. Upstream error text is treated as untrusted:
# only short, token-like values survive, messages are sanitized and capped, and
# anything resembling a credential is redacted before it can be surfaced.
_MAX_DIAGNOSTIC_TOKEN = 64
_MAX_DIAGNOSTIC_MESSAGE = 200
_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]")
_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9]+|Bearer\s+\S+)", re.IGNORECASE)


@dataclass(frozen=True)
class SynthesizedAudio:
    """Result of a successful synthesis. Contains no secrets."""

    data: bytes
    content_type: str
    model: str
    voice: str
    format: str


def normalize_text(text: object) -> str:
    """Collapse whitespace and strip control characters from ``text``.

    Returns an empty string for ``None``/whitespace-only input so the caller can
    decide how to report it. Raises ``ApiError`` for non-string input.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ApiError(400, "tts_invalid_text", "Text to synthesize must be a string.")
    cleaned = _CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _validate_model_config(model: str, voice: str, audio_format: str) -> None:
    spec = SUPPORTED_TTS_MODELS.get(model)
    if spec is None:
        raise ApiError(
            500,
            "tts_configuration_error",
            "The configured text-to-speech model is not supported.",
        )
    if audio_format not in spec["formats"]:
        raise ApiError(
            500,
            "tts_configuration_error",
            "The configured text-to-speech audio format is not supported.",
        )
    if voice not in spec["voices"]:
        raise ApiError(
            500,
            "tts_configuration_error",
            "The configured text-to-speech voice is not supported for the model.",
        )


def _safe_token(value: object, limit: int = _MAX_DIAGNOSTIC_TOKEN) -> str | None:
    """Return a short, token-like diagnostic value, or ``None`` if unusable."""
    if not isinstance(value, str):
        return None
    cleaned = _TOKEN_RE.sub("", value).strip()
    if not cleaned:
        return None
    return cleaned[:limit]


def _safe_message(value: object, limit: int = _MAX_DIAGNOSTIC_MESSAGE) -> str | None:
    """Sanitize an upstream message: strip controls, collapse whitespace,
    redact credential-like tokens, and cap the length."""
    if not isinstance(value, str):
        return None
    cleaned = _CONTROL_RE.sub(" ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    cleaned = _SECRET_RE.sub("[redacted]", cleaned)
    return cleaned[:limit]


def _upstream_diagnostics(response: httpx.Response) -> dict[str, object]:
    """Defensively extract safe diagnostics from a Groq error response.

    Recognizes the documented ``{"error": {"type", "code", "message"}}`` shape.
    Never raises: malformed JSON, non-object payloads, wrong field types and
    oversized/malicious messages all degrade to an empty (or partial) result so
    that the caller's stable error mapping always proceeds.
    """
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
            "tts_upstream_auth_error",
            "The text-to-speech service rejected the server credentials.",
            details,
        )
    if status == 429:
        raise ApiError(
            503,
            "tts_rate_limited",
            "The text-to-speech service is rate limited. Please try again shortly.",
            details,
        )
    raise ApiError(
        502,
        "tts_upstream_error",
        "The text-to-speech service returned an error.",
        details,
    )


async def synthesize_speech(
    text: object,
    *,
    voice: str | None = None,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> SynthesizedAudio:
    """Synthesize ``text`` to audio bytes via Groq.

    ``client`` may be injected for testing; when omitted an ``AsyncClient`` is
    created and closed internally. Raises ``ApiError`` for empty/oversized text,
    missing configuration, and all upstream failures.
    """
    normalized = normalize_text(text)
    if not normalized:
        raise ApiError(400, "tts_empty_text", "Text to synthesize is empty.")

    resolved_model = model or settings.groq_tts_model
    resolved_voice = voice or settings.groq_tts_voice
    resolved_format = settings.groq_tts_response_format
    _validate_model_config(resolved_model, resolved_voice, resolved_format)

    max_chars = settings.groq_tts_max_input_chars
    if len(normalized) > max_chars:
        raise ApiError(
            400,
            "tts_text_too_long",
            "Text is too long for the text-to-speech model.",
            {"max_characters": max_chars, "characters": len(normalized)},
        )

    api_key = get_groq_api_key()
    if not api_key:
        raise ApiError(
            503,
            "tts_not_configured",
            "Text-to-speech is not configured on the server.",
        )

    payload = {
        "model": resolved_model,
        "input": normalized,
        "voice": resolved_voice,
        "response_format": resolved_format,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=settings.groq_tts_timeout_seconds)
    try:
        try:
            response = await client.post(
                settings.groq_tts_url, headers=headers, json=payload
            )
        except httpx.TimeoutException as exc:
            raise ApiError(
                504, "tts_timeout", "The text-to-speech service timed out."
            ) from exc
        except httpx.TransportError as exc:
            raise ApiError(
                502,
                "tts_upstream_unavailable",
                "The text-to-speech service could not be reached.",
            ) from exc
    finally:
        if own_client:
            await client.aclose()

    _raise_for_status(response)

    content = response.content or b""
    if not content:
        raise ApiError(
            502, "tts_empty_audio", "The text-to-speech service returned no audio."
        )
    if len(content) > settings.groq_tts_max_audio_bytes:
        raise ApiError(
            502,
            "tts_audio_too_large",
            "The text-to-speech service returned an unexpectedly large response.",
        )

    content_type = response.headers.get("content-type", "").split(";")[0].strip()
    if not content_type.startswith("audio/"):
        content_type = "audio/wav"

    return SynthesizedAudio(
        data=content,
        content_type=content_type,
        model=resolved_model,
        voice=resolved_voice,
        format=resolved_format,
    )
