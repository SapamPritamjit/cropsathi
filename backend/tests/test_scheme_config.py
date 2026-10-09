"""Configuration tests for the Farmer Scheme Finder search provider (Increment 2).

No network, no secrets. ``Settings`` is re-instantiated against a controlled
environment so defaults and validation can be asserted directly.
"""

import pytest

from app.config import Settings, parse_official_origins

SCHEME_VARS = (
    "SCHEME_SEARCH_ENABLED",
    "SCHEME_SEARCH_TIMEOUT_SECONDS",
    "SCHEME_SEARCH_MAX_RESULTS",
    "SCHEME_SEARCH_REGION",
    "SCHEME_OFFICIAL_ORIGINS",
    "SCHEME_CATALOGUE_ENABLED",
)


def _clear(monkeypatch) -> None:
    for key in SCHEME_VARS:
        monkeypatch.delenv(key, raising=False)


def test_search_disabled_by_default(monkeypatch) -> None:
    _clear(monkeypatch)
    settings = Settings()
    assert settings.scheme_search_enabled is False
    assert settings.scheme_search_timeout_seconds == 8.0
    assert settings.scheme_search_max_results == 20
    assert settings.scheme_search_region == "in-en"
    assert settings.scheme_official_origins == ()
    assert settings.scheme_catalogue_enabled is False


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_catalogue_flag_recognised(monkeypatch, flag: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_CATALOGUE_ENABLED", flag)
    assert Settings().scheme_catalogue_enabled is True


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_catalogue_flag_disabled_recognised(monkeypatch, flag: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_CATALOGUE_ENABLED", flag)
    assert Settings().scheme_catalogue_enabled is False


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_enabled_flag_recognised(monkeypatch, flag: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_ENABLED", flag)
    assert Settings().scheme_search_enabled is True


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_disabled_flag_recognised(monkeypatch, flag: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_ENABLED", flag)
    assert Settings().scheme_search_enabled is False


def test_custom_configuration_is_read(monkeypatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_ENABLED", "true")
    monkeypatch.setenv("SCHEME_SEARCH_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("SCHEME_SEARCH_MAX_RESULTS", "5")
    monkeypatch.setenv(
        "SCHEME_OFFICIAL_ORIGINS",
        "manipur:manipur.gov.in, goi:india.gov.in",
    )
    settings = Settings()
    assert settings.scheme_search_enabled is True
    assert settings.scheme_search_timeout_seconds == 3.5
    assert settings.scheme_search_max_results == 5
    assert settings.scheme_official_origins == (
        "manipur:manipur.gov.in",
        "goi:india.gov.in",
    )


@pytest.mark.parametrize("value", ["0", "-1", "abc", "inf", "nan"])
def test_invalid_timeout_rejected(monkeypatch, value: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        Settings()


@pytest.mark.parametrize("value", ["0", "-3", "abc", "1.5"])
def test_invalid_max_results_rejected(monkeypatch, value: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_MAX_RESULTS", value)
    with pytest.raises(ValueError):
        Settings()


def test_empty_official_origins_is_empty_tuple(monkeypatch) -> None:
    _clear(monkeypatch)
    assert Settings().scheme_official_origins == ()


def test_parse_official_origins_normalizes_and_dedupes() -> None:
    parsed = parse_official_origins(" GoI:India.gov.in , manipur:manipur.gov.in ,goi:india.gov.in")
    assert parsed == ("goi:india.gov.in", "manipur:manipur.gov.in")


def test_parse_official_origins_blank_is_empty() -> None:
    assert parse_official_origins(None) == ()
    assert parse_official_origins("   ") == ()


def test_region_defaults_to_india_english(monkeypatch) -> None:
    _clear(monkeypatch)
    assert Settings().scheme_search_region == "in-en"


@pytest.mark.parametrize("value", ["in-en", "IN-EN", "hi-in", "us-en"])
def test_valid_region_accepted_and_normalised(monkeypatch, value: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_REGION", value)
    assert Settings().scheme_search_region == value.lower()


@pytest.mark.parametrize(
    "value", ["inen", "in", "in-en-extra", "12-en", "en_uk", "in en", "-", "en-"]
)
def test_invalid_region_rejected(monkeypatch, value: str) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SCHEME_SEARCH_REGION", value)
    with pytest.raises(ValueError):
        Settings()
