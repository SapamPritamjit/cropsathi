"""Consistency tests: settings in code vs. docs, and httpx dependency.

These assert that the documented defaults in ``backend/.env.example`` and the
runtime dependency in ``backend/requirements.txt`` stay in sync with the actual
configuration in ``app/config.py``. No network and no secrets involved.

``config`` loads the developer's root ``.env`` into ``os.environ`` at import, so
the module-level ``settings`` singleton can reflect local overrides. Following
the repository pattern (see ``test_scheme_config.py``), the settings-vs-docs
tests clear the documented variables with ``monkeypatch.delenv`` and build a
fresh ``Settings()`` so the *code* defaults are asserted independently of any
local ``.env`` (which is never read, printed, or modified here).
"""

from app.config import PROJECT_ROOT, Settings

ENV_EXAMPLE = PROJECT_ROOT / "backend" / ".env.example"
REQUIREMENTS = PROJECT_ROOT / "backend" / "requirements.txt"

# Documented + verified defaults (must match app/config.py).
EXPECTED_TTS_DEFAULTS = {
    "GROQ_TTS_URL": "https://api.groq.com/openai/v1/audio/speech",
    "GROQ_TTS_MODEL": "canopylabs/orpheus-v1-english",
    "GROQ_TTS_VOICE": "hannah",
    "GROQ_TTS_RESPONSE_FORMAT": "wav",
    "GROQ_TTS_MAX_INPUT_CHARS": "200",
    "GROQ_TTS_TIMEOUT_SECONDS": "30",
    "GROQ_TTS_MAX_AUDIO_BYTES": "10485760",
}

# Farmer Scheme Finder search foundation (Increment 2; region in Increment 5;
# curated catalogue in Increment 8B).
EXPECTED_SCHEME_DEFAULTS = {
    "SCHEME_SEARCH_ENABLED": "false",
    "SCHEME_SEARCH_TIMEOUT_SECONDS": "8",
    "SCHEME_SEARCH_MAX_RESULTS": "20",
    "SCHEME_SEARCH_REGION": "in-en",
    "SCHEME_OFFICIAL_ORIGINS": "",
    "SCHEME_CATALOGUE_ENABLED": "false",
}

# AI crop question answering (Increment 9P).
EXPECTED_INSIGHTS_DEFAULTS = {
    "GROQ_INSIGHTS_ENABLED": "false",
    "GROQ_CHAT_URL": "https://api.groq.com/openai/v1/chat/completions",
    "GROQ_CHAT_MODEL": "llama-3.3-70b-versatile",
    "GROQ_CHAT_TIMEOUT_SECONDS": "20",
    "GROQ_CHAT_MAX_INPUT_CHARS": "500",
    "GROQ_CHAT_MAX_TOKENS": "400",
}


def _active_env(path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _fresh_settings(monkeypatch, documented_keys):
    """Return a ``Settings`` built with ``documented_keys`` cleared from the env.

    ``monkeypatch.delenv`` only touches this test process's environment; it does
    not read or modify the local ``.env`` file. Clearing the documented keys lets
    ``Settings()`` fall back to the in-code defaults that ``.env.example``
    documents.
    """
    for key in documented_keys:
        monkeypatch.delenv(key, raising=False)
    return Settings()


def test_env_example_documents_all_tts_defaults() -> None:
    env = _active_env(ENV_EXAMPLE)
    for key, expected in EXPECTED_TTS_DEFAULTS.items():
        assert key in env, f"{key} missing from .env.example"
        assert env[key] == expected, f"{key} documented as {env[key]!r}"


def test_settings_match_documented_defaults(monkeypatch) -> None:
    active = _fresh_settings(monkeypatch, EXPECTED_TTS_DEFAULTS)
    assert active.groq_tts_url == EXPECTED_TTS_DEFAULTS["GROQ_TTS_URL"]
    assert active.groq_tts_model == EXPECTED_TTS_DEFAULTS["GROQ_TTS_MODEL"]
    assert active.groq_tts_voice == EXPECTED_TTS_DEFAULTS["GROQ_TTS_VOICE"]
    assert active.groq_tts_response_format == "wav"
    assert active.groq_tts_max_input_chars == 200
    assert active.groq_tts_timeout_seconds == 30.0
    assert active.groq_tts_max_audio_bytes == 10485760


def test_env_example_documents_all_scheme_defaults() -> None:
    env = _active_env(ENV_EXAMPLE)
    for key, expected in EXPECTED_SCHEME_DEFAULTS.items():
        assert key in env, f"{key} missing from .env.example"
        assert env[key] == expected, f"{key} documented as {env[key]!r}"


def test_scheme_settings_match_documented_defaults(monkeypatch) -> None:
    active = _fresh_settings(monkeypatch, EXPECTED_SCHEME_DEFAULTS)
    assert active.scheme_search_enabled is False
    assert active.scheme_search_timeout_seconds == 8.0
    assert active.scheme_search_max_results == 20
    assert active.scheme_search_region == "in-en"
    assert active.scheme_official_origins == ()
    assert active.scheme_catalogue_enabled is False


def test_env_example_documents_all_insights_defaults() -> None:
    env = _active_env(ENV_EXAMPLE)
    for key, expected in EXPECTED_INSIGHTS_DEFAULTS.items():
        assert key in env, f"{key} missing from .env.example"
        assert env[key] == expected, f"{key} documented as {env[key]!r}"


def test_insights_settings_match_documented_defaults(monkeypatch) -> None:
    active = _fresh_settings(monkeypatch, EXPECTED_INSIGHTS_DEFAULTS)
    assert active.groq_insights_enabled is False
    assert active.groq_chat_url == EXPECTED_INSIGHTS_DEFAULTS["GROQ_CHAT_URL"]
    assert active.groq_chat_model == EXPECTED_INSIGHTS_DEFAULTS["GROQ_CHAT_MODEL"]
    assert active.groq_chat_timeout_seconds == 20.0
    assert active.groq_chat_max_input_chars == 500
    assert active.groq_chat_max_tokens == 400


def test_ddgs_declared_as_runtime_dependency() -> None:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    declared = [
        ln.strip()
        for ln in lines
        if ln.strip() == "ddgs" or ln.strip().startswith("ddgs=") or ln.strip().startswith("ddgs==")
    ]
    assert declared, "ddgs must be declared in requirements.txt"
    test_marker = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("# --- tests")
    )
    ddgs_index = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("ddgs")
    )
    assert ddgs_index < test_marker, "ddgs must be a runtime dependency"


def test_env_example_documents_legacy_key_compatibility() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "GROQ-API-KEY" in text
    assert "get_groq_api_key" in text


def test_env_example_has_no_real_secret() -> None:
    env = _active_env(ENV_EXAMPLE)
    assert env["GROQ_API_KEY"] == "your-groq-api-key-here"


def test_httpx_declared_once_as_runtime_dependency() -> None:
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    httpx_lines = [ln.strip() for ln in lines if ln.strip().startswith("httpx")]
    assert httpx_lines == ["httpx==0.28.1"], httpx_lines

    test_marker = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("# --- tests")
    )
    httpx_index = next(
        i for i, ln in enumerate(lines) if ln.strip().startswith("httpx==")
    )
    assert httpx_index < test_marker, "httpx must be a runtime dependency"
