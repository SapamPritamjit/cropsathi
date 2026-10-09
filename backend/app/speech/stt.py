"""Server-side speech-to-text via Groq Whisper (Part 6D).

Transcribes a short uploaded audio clip using Groq's OpenAI-compatible
``/openai/v1/audio/transcriptions`` endpoint and returns the recognized text plus
the upstream-reported language (when present). No playback, no routes, no
response generation.

Verified against the current Groq documentation:

- Models: ``whisper-large-v3-turbo`` and ``whisper-large-v3`` (both listed as
  Multilingual). This module does not enumerate languages; the upstream model
  decides support, and ``language`` is an optional ISO-639-1 hint.
- Accepted upload formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm.
- Upload ceiling: 25 MB (free tier) / 100 MB (dev tier). The application applies
  a bounded read and a configurable (default 25 MB) limit before any upstream
  call, so uploads are never loaded unbounded into memory.

The API key is read via ``config.get_groq_api_key()`` and is never logged,
stored on the result, or included in error messages. Upstream response bodies
and stack traces are never surfaced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

import httpx

from ..config import get_groq_api_key, settings
from ..errors import ApiError

SUPPORTED_STT_MODELS: frozenset[str] = frozenset(
    {"whisper-large-v3-turbo", "whisper-large-v3"}
)

# Documented Groq upload formats.
ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
)

SUPPORTED_RESPONSE_FORMATS: frozenset[str] = frozenset({"json", "verbose_json", "text"})

_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")


@dataclass(frozen=True)
class Transcription:
    """Result of a successful transcription. Contains no secrets."""

    text: str
    language: str | None


def _extension(filename: str | None) -> str:
    if not filename:
        return ""
    # Handle both separators so a path in the filename never bypasses the check.
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = PureWindowsPath(name).name
    dot = name.rfind(".")
    return name[dot:].lower() if dot != -1 else ""


def validate_audio_upload(filename: str | None, content_type: str | None) -> None:
    """Reject uploads that are not a documented Groq audio type (``415``)."""
    if _extension(filename) in ALLOWED_AUDIO_EXTENSIONS:
        return
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype.startswith("audio/") or ctype in {"video/mp4", "video/webm"}:
        return
    raise ApiError(
        415,
        "stt_unsupported_media_type",
        "Unsupported audio type. Upload one of: "
        + ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        + ".",
        {"allowed_extensions": sorted(ALLOWED_AUDIO_EXTENSIONS)},
    )


def normalize_language(code: object) -> str | None:
    """Validate an optional ISO-639-1 language hint. ``None``/blank -> ``None``."""
    if code is None:
        return None
    if not isinstance(code, str):
        raise ApiError(
            400, "stt_invalid_language", "Language must be an ISO-639-1 code."
        )
    cleaned = code.strip().lower()
    if not cleaned:
        return None
    if not _LANGUAGE_RE.match(cleaned):
        raise ApiError(
            400,
            "stt_invalid_language",
            "Language must be an ISO-639-1 two-letter code (for example 'en').",
        )
    return cleaned


def _validate_model_config(model: str, response_format: str) -> None:
    if model not in SUPPORTED_STT_MODELS:
        raise ApiError(
            500,
            "stt_configuration_error",
            "The configured speech-to-text model is not supported.",
        )
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise ApiError(
            500,
            "stt_configuration_error",
            "The configured speech-to-text response format is not supported.",
        )


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    if status in (401, 403):
        raise ApiError(
            502,
            "stt_upstream_auth_error",
            "The transcription service rejected the server credentials.",
            {"upstream_status": status},
        )
    if status == 429:
        raise ApiError(
            503,
            "stt_rate_limited",
            "The transcription service is rate limited. Please try again shortly.",
            {"upstream_status": status},
        )
    raise ApiError(
        502,
        "stt_upstream_error",
        "The transcription service returned an error.",
        {"upstream_status": status},
    )


async def transcribe_audio(
    data: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    language: object = None,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Transcription:
    """Transcribe ``data`` via Groq Whisper and return text + optional language."""
    if not data:
        raise ApiError(400, "stt_empty_audio", "The uploaded audio is empty.")
    if len(data) > settings.stt_max_upload_bytes:
        raise ApiError(
            413,
            "stt_audio_too_large",
            "The uploaded audio exceeds the size limit.",
            {"max_upload_bytes": settings.stt_max_upload_bytes},
        )
    validate_audio_upload(filename, content_type)
    resolved_language = normalize_language(language)

    resolved_model = model or settings.groq_stt_model
    resolved_format = settings.groq_stt_response_format
    _validate_model_config(resolved_model, resolved_format)

    api_key = get_groq_api_key()
    if not api_key:
        raise ApiError(
            503,
            "stt_not_configured",
            "Speech-to-text is not configured on the server.",
        )

    form: dict[str, str] = {
        "model": resolved_model,
        "response_format": resolved_format,
    }
    if resolved_language:
        form["language"] = resolved_language
    files = {
        "file": (
            filename or "audio",
            data,
            content_type or "application/octet-stream",
        )
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=settings.groq_stt_timeout_seconds)
    try:
        try:
            response = await client.post(
                settings.groq_stt_url, headers=headers, data=form, files=files
            )
        except httpx.TimeoutException as exc:
            raise ApiError(
                504, "stt_timeout", "The transcription service timed out."
            ) from exc
        except httpx.TransportError as exc:
            raise ApiError(
                502,
                "stt_upstream_unavailable",
                "The transcription service could not be reached.",
            ) from exc
    finally:
        if own_client:
            await client.aclose()

    _raise_for_status(response)

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(
            502,
            "stt_upstream_error",
            "The transcription service returned an unreadable response.",
        ) from exc
    if not isinstance(payload, dict):
        raise ApiError(
            502,
            "stt_upstream_error",
            "The transcription service returned an unexpected response.",
        )

    text = payload.get("text")
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        raise ApiError(
            422,
            "stt_empty_transcription",
            "No speech could be transcribed from the audio.",
        )

    reported_language = payload.get("language")
    if not isinstance(reported_language, str) or not reported_language.strip():
        reported_language = None
    else:
        reported_language = reported_language.strip().lower()

    return Transcription(text=text, language=reported_language)
