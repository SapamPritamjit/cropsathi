"""Application configuration and model-artifact path discovery.

This module performs NO model loading. It only resolves the on-disk locations of
the existing artifacts so the API can report what is present. Weights are never
copied or moved.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

SUPPORTED_CROPS: tuple[str, ...] = ("maize", "rice", "potato", "tomato")

CROP_DISPLAY_NAMES: dict[str, str] = {
    "maize": "Maize",
    "rice": "Rice",
    "potato": "Potato",
    "tomato": "Tomato",
}

# Baseline ConvNeXt-Small (timm) model directories, keyed by crop.
BASELINE_MODEL_DIRS: dict[str, Path] = {
    crop: PROJECT_ROOT / f"{crop}-disease-convnext" for crop in SUPPORTED_CROPS
}

# Fine-tuned EfficientNet-B0 model directories (only where supplied).
FINE_TUNED_MODEL_DIRS: dict[str, Path] = {
    "maize": PROJECT_ROOT / "fine-tune-model" / "ai4seva_maize_final",
    "rice": PROJECT_ROOT / "fine-tune-model" / "ai4seva_rice_final",
}


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free ``.env`` loader.

    Only sets variables that are not already present in the environment, so the
    real environment always wins. Values are never logged or exposed.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def get_groq_api_key() -> str | None:
    """Return the Groq API key from the environment, or ``None`` if unset.

    The canonical name is ``GROQ_API_KEY``. The legacy hyphenated name
    ``GROQ-API-KEY`` is also accepted so the supplied ``.env`` keeps working.
    The value is never logged or returned by any API endpoint.
    """
    return os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ-API-KEY") or None


def parse_allowed_origins(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list of allowed browser origins.

    Whitespace and quote characters are trimmed, empty entries are dropped, a
    trailing slash is removed (browsers send origins without one), and
    duplicates are collapsed while preserving order. The result is empty when
    nothing is configured, which means **no** origin is allowed — the secure
    default. A wildcard is only honoured if an operator sets it explicitly.
    """
    if not raw:
        return ()
    origins: list[str] = []
    for token in raw.split(","):
        origin = token.strip().strip('"').strip("'").rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)
    return tuple(origins)


class Settings:
    """Runtime settings sourced from environment variables with safe defaults."""

    def __init__(self) -> None:
        self.app_name: str = "AI4Seva API"
        self.version: str = "0.1.0"
        self.environment: str = os.environ.get("APP_ENV", "development")
        self.host: str = os.environ.get("APP_HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("APP_PORT", "8000"))
        # --- CORS: explicit browser origins permitted to call the API ---
        # Empty by default (deny all cross-origin browser requests). Configure
        # with an exact, comma-separated allow-list via CORS_ALLOWED_ORIGINS.
        self.cors_allowed_origins: tuple[str, ...] = parse_allowed_origins(
            os.environ.get("CORS_ALLOWED_ORIGINS")
        )
        # Boolean only — the secret value itself is never stored on Settings.
        self.groq_api_key_configured: bool = get_groq_api_key() is not None
        # --- image upload / prediction limits ---
        self.max_upload_bytes: int = int(
            os.environ.get("IMAGE_MAX_BYTES", str(8 * 1024 * 1024))
        )
        self.min_image_side: int = int(os.environ.get("IMAGE_MIN_SIDE", "32"))
        self.max_image_side: int = int(os.environ.get("IMAGE_MAX_SIDE", "10000"))
        # Max number of distinct models kept resident (LRU). Bounds memory.
        self.model_cache_size: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
        # Fine-tuned EfficientNet-B0 predictions stay OFF until the training
        # preprocessing (resize / normalization) is confirmed. The supplied
        # checkpoints do not record it, so enabling is an explicit opt-in.
        self.enable_fine_tuned_predictions: bool = os.environ.get(
            "ENABLE_FINE_TUNED_PREDICTIONS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # --- Part 6B: Groq text-to-speech (server-side only) ---
        # Groq's only currently-documented TTS models are Orpheus; the input is
        # capped at 200 characters and only "wav" output is supported.
        self.groq_tts_url: str = os.environ.get(
            "GROQ_TTS_URL", "https://api.groq.com/openai/v1/audio/speech"
        )
        self.groq_tts_model: str = os.environ.get(
            "GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"
        )
        self.groq_tts_voice: str = os.environ.get("GROQ_TTS_VOICE", "hannah")
        self.groq_tts_response_format: str = os.environ.get(
            "GROQ_TTS_RESPONSE_FORMAT", "wav"
        )
        self.groq_tts_timeout_seconds: float = float(
            os.environ.get("GROQ_TTS_TIMEOUT_SECONDS", "30")
        )
        # Never exceed the model's documented 200-character ceiling.
        self.groq_tts_max_input_chars: int = max(
            1, min(int(os.environ.get("GROQ_TTS_MAX_INPUT_CHARS", "200")), 200)
        )
        self.groq_tts_max_audio_bytes: int = int(
            os.environ.get("GROQ_TTS_MAX_AUDIO_BYTES", str(10 * 1024 * 1024))
        )
        # --- Part 6D: Groq speech-to-text (server-side only) ---
        # Groq transcripts via Whisper. Uploads are bounded before any upstream
        # call; the default matches Groq's 25 MB free-tier ceiling.
        self.groq_stt_url: str = os.environ.get(
            "GROQ_STT_URL",
            "https://api.groq.com/openai/v1/audio/transcriptions",
        )
        self.groq_stt_model: str = os.environ.get(
            "GROQ_STT_MODEL", "whisper-large-v3-turbo"
        )
        self.groq_stt_response_format: str = os.environ.get(
            "GROQ_STT_RESPONSE_FORMAT", "verbose_json"
        )
        self.groq_stt_timeout_seconds: float = float(
            os.environ.get("GROQ_STT_TIMEOUT_SECONDS", "60")
        )
        # Conservative application ceiling; never above Groq's documented 100 MB.
        self.stt_max_upload_bytes: int = max(
            1,
            min(
                int(os.environ.get("STT_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))),
                100 * 1024 * 1024,
            ),
        )


settings = Settings()
