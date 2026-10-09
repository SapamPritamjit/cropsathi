"""Server-side speech synthesis endpoint (Part 6C-2).

Returns raw WAV bytes from the Groq-backed TTS service. No STT, no response
generation, and no playback — those are separate increments.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Response, UploadFile

from ..config import settings
from ..errors import ApiError
from ..schemas import (
    ErrorResponse,
    PredictionSpeechRequest,
    PredictionSpeechResponse,
    SpeechRequest,
    TranscribeResponse,
)
from ..speech.stt import transcribe_audio, validate_audio_upload
from ..speech.text import build_prediction_speech
from ..speech.tts import normalize_text, synthesize_speech

router = APIRouter(tags=["speech"])


@router.post(
    "/speech/speak",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Synthesized WAV audio."},
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def speak(payload: SpeechRequest) -> Response:
    """Synthesize ``text`` into WAV audio and return the bytes directly."""
    normalized = normalize_text(payload.text)
    if not normalized:
        raise ApiError(400, "tts_empty_text", "Text to synthesize is empty.")

    max_chars = settings.groq_tts_max_input_chars
    if len(normalized) > max_chars:
        raise ApiError(
            413,
            "tts_text_too_long",
            "Text is too long for the text-to-speech model.",
            {"max_characters": max_chars, "characters": len(normalized)},
        )

    audio = await synthesize_speech(normalized)
    return Response(content=audio.data, media_type=audio.content_type)


@router.post(
    "/speech/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def transcribe(
    file: UploadFile = File(..., description="Audio clip (wav/mp3/m4a/ogg/webm/flac)."),
    language: str | None = Form(
        None, description="Optional ISO-639-1 language hint, e.g. 'en'."
    ),
) -> TranscribeResponse:
    """Transcribe an uploaded audio clip to text via Groq Whisper."""
    data = await file.read(settings.stt_max_upload_bytes + 1)
    if not data:
        raise ApiError(400, "stt_empty_audio", "The uploaded audio is empty.")
    if len(data) > settings.stt_max_upload_bytes:
        raise ApiError(
            413,
            "stt_audio_too_large",
            "The uploaded audio exceeds the size limit.",
            {"max_upload_bytes": settings.stt_max_upload_bytes},
        )
    validate_audio_upload(file.filename, file.content_type)

    result = await transcribe_audio(
        data,
        filename=file.filename,
        content_type=file.content_type,
        language=language,
    )
    return TranscribeResponse(text=result.text, language=result.language)


@router.post(
    "/speech/response",
    response_model=PredictionSpeechResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def build_response(payload: PredictionSpeechRequest) -> PredictionSpeechResponse:
    """Convert a prediction result into concise, farmer-friendly text.

    Pure transformation only: no inference, no transcription, and no upstream
    (Groq) calls. Field validation and the bounded, uncertainty-preserving
    wording are provided by the shared ``build_prediction_speech`` builder.
    """
    text = build_prediction_speech(payload)
    return PredictionSpeechResponse(text=text)
