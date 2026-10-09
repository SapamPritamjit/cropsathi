"""Pure service tests for ``GET /schemes/search`` orchestration (Increment 4).

No network, no ddgs import, no FastAPI: exercises validation, query building and
snippet sanitation directly. ``resolve_search_query`` raises the same
:class:`ApiError` codes the endpoint surfaces.
"""

import pytest

from app.config import settings
from app.errors import ApiError
from app.schemes.service import (
    DEFAULT_LIMIT,
    DISCLAIMER,
    MAX_DISTRICT_LENGTH,
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_SNIPPET_LENGTH,
    MIN_LIMIT,
    build_search_query,
    resolve_search_query,
    sanitize_snippet,
)


def _enable(monkeypatch, *, max_results: int = 20) -> None:
    monkeypatch.setattr(settings, "scheme_search_enabled", True)
    monkeypatch.setattr(settings, "scheme_search_max_results", max_results)


# --- query building --------------------------------------------------------- #


def test_build_query_always_includes_manipur_context() -> None:
    assert build_search_query(None, None, None, None).startswith("Manipur")
    assert "Manipur" in build_search_query("pest control", "rice", None, "pest_disease")


def test_build_query_uses_crop_district_and_need() -> None:
    query = build_search_query(None, "rice", "Imphal East", "insurance")
    assert "Manipur" in query
    assert "rice" in query
    assert "crop insurance" in query
    assert "Imphal East" in query


def test_build_query_is_length_bounded() -> None:
    query = build_search_query("a" * MAX_QUERY_LENGTH, "tomato", "d" * 50, "general")
    assert len(query) <= MAX_SEARCH_QUERY_LENGTH


def test_build_query_neutralizes_url_schemes() -> None:
    query = build_search_query("http://evil.example/x", None, None, None)
    assert "://" not in query


# --- snippet sanitation ----------------------------------------------------- #


def test_sanitize_snippet_collapses_whitespace_and_controls() -> None:
    assert sanitize_snippet("  a\n\nb\t c \r") == "a b c"


def test_sanitize_snippet_is_length_limited() -> None:
    out = sanitize_snippet("x" * (MAX_SNIPPET_LENGTH + 200))
    assert len(out) == MAX_SNIPPET_LENGTH


def test_sanitize_snippet_non_string_is_empty() -> None:
    assert sanitize_snippet(None) == ""
    assert sanitize_snippet(123) == ""


# --- disabled feature ------------------------------------------------------- #


def test_disabled_feature_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(settings, "scheme_search_enabled", False)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(q="seeds")
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "scheme_search_disabled"


# --- meaningful input required ---------------------------------------------- #


def test_no_meaningful_input_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    for kwargs in ({}, {"district": "Imphal East"}, {"language": "en"}):
        with pytest.raises(ApiError) as excinfo:
            resolve_search_query(**kwargs)
        assert excinfo.value.status_code == 400
        assert excinfo.value.code == "scheme_search_empty_query"


def test_blank_query_without_other_input_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(q="   ")
    assert excinfo.value.code == "scheme_search_empty_query"


def test_overlong_query_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(q="a" * (MAX_QUERY_LENGTH + 1))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "scheme_search_query_too_long"
    assert excinfo.value.details == {"max_characters": MAX_QUERY_LENGTH}


def test_blank_query_ignored_when_crop_present(monkeypatch) -> None:
    _enable(monkeypatch)
    resolved = resolve_search_query(q="   ", crop="rice")
    assert resolved.q is None
    assert resolved.crop == "rice"


# --- enumerated inputs ------------------------------------------------------ #


def test_invalid_crop_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(crop="wheat")
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "scheme_search_invalid_crop"


def test_invalid_need_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(need="subsidy")
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "scheme_search_invalid_need"


def test_valid_crop_and_need_are_normalised(monkeypatch) -> None:
    _enable(monkeypatch)
    resolved = resolve_search_query(crop="RICE", need="Insurance")
    assert resolved.crop == "rice"
    assert resolved.need == "insurance"


# --- district policy (no verified list -> bounded safe string) -------------- #


def test_district_accepted_without_a_fixed_list(monkeypatch) -> None:
    _enable(monkeypatch)
    resolved = resolve_search_query(crop="rice", district="Somewhere New")
    assert resolved.district == "Somewhere New"


def test_overlong_district_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(crop="rice", district="x" * (MAX_DISTRICT_LENGTH + 1))
    assert excinfo.value.code == "scheme_search_invalid_district"


def test_district_only_does_not_trigger_search(monkeypatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(district="Imphal East")
    assert excinfo.value.code == "scheme_search_empty_query"


# --- limit ------------------------------------------------------------------ #


def test_limit_defaults_to_five(monkeypatch) -> None:
    _enable(monkeypatch)
    assert resolve_search_query(q="x").limit == DEFAULT_LIMIT == 5


@pytest.mark.parametrize("bad", [0, -1, MAX_LIMIT + 1])
def test_limit_out_of_bounds_rejected(monkeypatch, bad: int) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(q="x", limit=bad)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "scheme_search_invalid_limit"


def test_limit_within_bounds_respected(monkeypatch) -> None:
    _enable(monkeypatch)
    assert resolve_search_query(q="x", limit=MIN_LIMIT).limit == MIN_LIMIT
    assert resolve_search_query(q="x", limit=MAX_LIMIT).limit == MAX_LIMIT


def test_limit_cannot_exceed_configured_maximum(monkeypatch) -> None:
    _enable(monkeypatch, max_results=3)
    assert resolve_search_query(q="x", limit=MAX_LIMIT).limit == 3
    assert resolve_search_query(q="x").limit == 3


# --- language --------------------------------------------------------------- #


@pytest.mark.parametrize("valid", ["en", "hi", "EN", "en-IN"])
def test_valid_language_accepted(monkeypatch, valid: str) -> None:
    _enable(monkeypatch)
    assert resolve_search_query(q="x", language=valid).language == valid.lower()


def test_default_language_is_en(monkeypatch) -> None:
    _enable(monkeypatch)
    assert resolve_search_query(q="x").language == "en"


@pytest.mark.parametrize("bad", ["e", "english", "123", "e n", "xx_yy"])
def test_invalid_language_rejected(monkeypatch, bad: str) -> None:
    _enable(monkeypatch)
    with pytest.raises(ApiError) as excinfo:
        resolve_search_query(q="x", language=bad)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "scheme_search_invalid_language"


# --- constants -------------------------------------------------------------- #


def test_disclaimer_is_stable_and_informational() -> None:
    assert "eligibility" in DISCLAIMER
    assert "official department" in DISCLAIMER
