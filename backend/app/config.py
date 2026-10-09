"""Application configuration and model-artifact path discovery.

This module performs NO model loading. It only resolves the on-disk locations of
the existing artifacts so the API can report what is present. Weights are never
copied or moved.
"""

from __future__ import annotations

import math
import os
import re
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


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    """Parse a boolean environment flag (``1/true/yes/on``), else ``default``."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int(raw: str | None, default: int, name: str) -> int:
    """Parse a strictly positive integer setting.

    Blank/unset -> ``default``. Raises ``ValueError`` for non-integers or values
    that are not strictly positive, so misconfiguration fails loudly instead of
    silently degrading (consistent with the existing ``int(...)`` settings).
    """
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer.") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _parse_positive_float(raw: str | None, default: float, name: str) -> float:
    """Parse a strictly positive, finite float setting (blank/unset -> default)."""
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite number.")
    return value


_REGION_FORMAT = re.compile(r"^[a-z]{2}-[a-z]{2,3}$")


def _parse_region(raw: str | None, default: str, name: str) -> str:
    """Parse a DDGS ``<country>-<language>`` region code (e.g. ``in-en``).

    Blank/unset -> ``default``. Raises ``ValueError`` for anything that is not
    of the ``cc-lang`` form the ddgs engines expect (they split the value on
    ``-``), so a typo fails loudly instead of silently degrading search
    relevance.
    """
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if not _REGION_FORMAT.match(value):
        raise ValueError(
            f"{name} must be a '<country>-<language>' code such as 'in-en'."
        )
    return value


def parse_official_origins(raw: str | None) -> tuple[str, ...]:
    """Parse the official-source allow-list for the scheme search.

    Comma-separated. An entry may be typed as ``manipur:<host>`` (state
    government) or ``goi:<host>`` (Government of India); an untyped entry
    defaults to ``goi`` when classified (see ``app.schemes.sources``).
    Whitespace and quotes are trimmed, entries are lower-cased, empties are
    dropped and duplicates are collapsed while preserving order. The result is
    empty when nothing is configured, which means **nothing** is treated as
    official — the safe default.
    """
    if not raw:
        return ()
    origins: list[str] = []
    for token in raw.split(","):
        origin = token.strip().strip('"').strip("'").lower()
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
        # --- Increment 2: Farmer Scheme Finder search provider ---
        # Disabled by default; no search runs unless explicitly enabled. The
        # official allow-list is empty by default, so NOTHING is classified as
        # official until verified hosts are added. No API secret is required.
        self.scheme_search_enabled: bool = _parse_bool(
            os.environ.get("SCHEME_SEARCH_ENABLED")
        )
        self.scheme_search_timeout_seconds: float = _parse_positive_float(
            os.environ.get("SCHEME_SEARCH_TIMEOUT_SECONDS"),
            8.0,
            "SCHEME_SEARCH_TIMEOUT_SECONDS",
        )
        self.scheme_search_max_results: int = _parse_positive_int(
            os.environ.get("SCHEME_SEARCH_MAX_RESULTS"),
            20,
            "SCHEME_SEARCH_MAX_RESULTS",
        )
        # India-oriented English region for the DDGS provider (e.g. in-en).
        self.scheme_search_region: str = _parse_region(
            os.environ.get("SCHEME_SEARCH_REGION"),
            "in-en",
            "SCHEME_SEARCH_REGION",
        )
        self.scheme_official_origins: tuple[str, ...] = parse_official_origins(
            os.environ.get("SCHEME_OFFICIAL_ORIGINS")
        )
        # --- Increment 8B: curated official-resource catalogue ---
        # Disabled by default. When enabled, a small audited set of official
        # information resources is merged into search results, but only after a
        # successful live search and only for hosts the allow-list above accepts
        # (the catalogue is not a second allow-list).
        self.scheme_catalogue_enabled: bool = _parse_bool(
            os.environ.get("SCHEME_CATALOGUE_ENABLED")
        )
        # --- Increment 9P: Groq crop question answering (server-side only) ---
        # DISABLED by default. When enabled AND the Groq API key is configured,
        # POST /insights/ask asks the configured chat model a crop-related
        # question. The API key is never stored on Settings (only the derived
        # boolean below exists), and no answer is fabricated on any failure.
        self.groq_insights_enabled: bool = _parse_bool(
            os.environ.get("GROQ_INSIGHTS_ENABLED")
        )
        self.groq_chat_url: str = os.environ.get(
            "GROQ_CHAT_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        # Model ID verified against Groq's official docs (Supported Models and
        # API Reference both list llama-3.3-70b-versatile for chat completions).
        self.groq_chat_model: str = os.environ.get(
            "GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"
        )
        self.groq_chat_timeout_seconds: float = _parse_positive_float(
            os.environ.get("GROQ_CHAT_TIMEOUT_SECONDS"),
            20.0,
            "GROQ_CHAT_TIMEOUT_SECONDS",
        )
        # Bounded input (characters) and output (tokens). Both fail loudly when
        # misconfigured, consistent with the scheme-search settings.
        self.groq_chat_max_input_chars: int = _parse_positive_int(
            os.environ.get("GROQ_CHAT_MAX_INPUT_CHARS"),
            500,
            "GROQ_CHAT_MAX_INPUT_CHARS",
        )
        self.groq_chat_max_tokens: int = _parse_positive_int(
            os.environ.get("GROQ_CHAT_MAX_TOKENS"),
            400,
            "GROQ_CHAT_MAX_TOKENS",
        )


settings = Settings()
