"""Offline tests for the curated official-resource catalogue (Increment 8B).

No network, no ddgs, no FastAPI: the catalogue is validated directly and merged
through the pure ``run_scheme_search`` orchestration with an injected fake
provider. These tests assert that curated records never overstate verification,
that the feature is off by default, and that provider failures are never masked
by catalogue-only results.
"""

import asyncio
import dataclasses
import socket
import time
import urllib.request

import pytest

from app.config import settings
from app.errors import ApiError
from app.schemes import (
    OFFICIAL_GOI,
    OFFICIAL_MANIPUR,
    OTHER,
    SEED_CATALOGUE,
    CatalogueEntry,
    CatalogueError,
    OfficialSourceRegistry,
    SCHEME_STATUS_ACTIVE,
    SCHEME_STATUS_INFORMATION_PAGE,
    SCHEME_STATUS_UNKNOWN,
    catalogue_candidates,
    hostname_from_url,
    validate_catalogue,
)
from app.schemes.service import run_scheme_search


def _entry(**overrides) -> CatalogueEntry:
    base = dict(
        id="entry",
        title="Example Government Page",
        url="https://e.gov.in/",
        department="Example Department",
        source_type=OFFICIAL_GOI,
        verified=True,
        scheme_status=SCHEME_STATUS_UNKNOWN,
        description="An example.",
        evidence_note="note",
        last_verified="2026-10-09",
    )
    base.update(overrides)
    return CatalogueEntry(**base)


def _seed_registry() -> OfficialSourceRegistry:
    specs = []
    for entry in SEED_CATALOGUE:
        prefix = "manipur" if entry.source_type == OFFICIAL_MANIPUR else "goi"
        specs.append(f"{prefix}:{hostname_from_url(entry.url)}")
    return OfficialSourceRegistry.from_specs(specs)


class _ListProvider:
    def __init__(self, items):
        self.items = list(items)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int):
        self.calls.append((query, limit))
        return list(self.items)


class _BoomProvider:
    def search(self, query: str, limit: int):
        raise RuntimeError("upstream secret detail")


class _SlowProvider:
    def search(self, query: str, limit: int):
        time.sleep(0.3)
        return []


def _enable(
    monkeypatch,
    *,
    max_results: int = 20,
    origins=(),
    catalogue: bool = False,
) -> None:
    monkeypatch.setattr(settings, "scheme_search_enabled", True)
    monkeypatch.setattr(settings, "scheme_search_max_results", max_results)
    monkeypatch.setattr(settings, "scheme_official_origins", origins)
    monkeypatch.setattr(settings, "scheme_catalogue_enabled", catalogue)


def _run(**kwargs):
    return asyncio.run(run_scheme_search(**kwargs))


def _titles(payload) -> list[str]:
    return [item["title"] for item in payload["results"]]


# --- structural integrity --------------------------------------------------- #


def test_seed_entries_are_immutable() -> None:
    entry = SEED_CATALOGUE[0]
    assert isinstance(entry.crops, tuple)
    assert isinstance(entry.needs, tuple)
    assert isinstance(entry.districts, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.id = "changed"  # type: ignore[misc]


def test_seed_catalogue_passes_structure_validation() -> None:
    assert len(validate_catalogue(SEED_CATALOGUE)) == len(SEED_CATALOGUE)


def test_seed_catalogue_passes_registry_validation() -> None:
    registry = _seed_registry()
    assert len(validate_catalogue(SEED_CATALOGUE, registry)) == len(SEED_CATALOGUE)


def test_duplicate_ids_rejected() -> None:
    entries = (_entry(id="dup"), _entry(id="dup", url="https://other.gov.in/"))
    with pytest.raises(CatalogueError):
        validate_catalogue(entries)


def test_non_https_url_rejected() -> None:
    with pytest.raises(CatalogueError):
        validate_catalogue((_entry(url="http://e.gov.in/"),))


def test_duplicate_canonical_urls_rejected() -> None:
    entries = (
        _entry(id="a", url="https://e.gov.in/x"),
        _entry(id="b", url="https://E.gov.in/x#fragment"),
    )
    with pytest.raises(CatalogueError):
        validate_catalogue(entries)


def test_verified_host_not_accepted_by_registry_rejected() -> None:
    entry = _entry(verified=True, url="https://not-listed.gov.in/")
    with pytest.raises(CatalogueError):
        validate_catalogue((entry,), OfficialSourceRegistry())


def test_invalid_source_type_rejected() -> None:
    with pytest.raises(CatalogueError):
        validate_catalogue((_entry(source_type="blog"),))


def test_invalid_scheme_status_rejected() -> None:
    with pytest.raises(CatalogueError):
        validate_catalogue((_entry(scheme_status="active-ish"),))


def test_unknown_crop_or_need_rejected() -> None:
    with pytest.raises(CatalogueError):
        validate_catalogue((_entry(crops=("wheat",)),))
    with pytest.raises(CatalogueError):
        validate_catalogue((_entry(needs=("subsidy",)),))


# --- no overclaiming -------------------------------------------------------- #


def test_unapproved_host_not_promoted_by_title_or_description() -> None:
    entry = _entry(
        id="fake",
        title="Government of Manipur Official Farmer Scheme",
        description="Official department scheme, apply now",
        verified=True,
        source_type=OFFICIAL_MANIPUR,
        url="https://fake-manipur.example/",
    )
    out = catalogue_candidates(OfficialSourceRegistry(), (entry,))
    assert out[0].verified is False
    assert out[0].source_type == OTHER


def test_seed_scheme_status_is_unknown() -> None:
    assert all(entry.scheme_status == SCHEME_STATUS_UNKNOWN for entry in SEED_CATALOGUE)


def test_status_vocabulary_values() -> None:
    assert SCHEME_STATUS_INFORMATION_PAGE == "information_page"
    assert SCHEME_STATUS_ACTIVE == "active"
    assert SCHEME_STATUS_UNKNOWN == "unknown"


def test_catalogue_candidates_classify_via_registry() -> None:
    out = catalogue_candidates(_seed_registry())
    assert len(out) == len(SEED_CATALOGUE)
    assert all(candidate.verified for candidate in out)
    assert {candidate.source_type for candidate in out} == {
        OFFICIAL_MANIPUR,
        OFFICIAL_GOI,
    }


# --- configuration ---------------------------------------------------------- #


def test_catalogue_flag_defaults_disabled() -> None:
    assert settings.scheme_catalogue_enabled is False


# --- merge behaviour -------------------------------------------------------- #


def test_flag_off_does_not_merge_seed(monkeypatch) -> None:
    _enable(monkeypatch, catalogue=False)
    provider = _ListProvider([_item("Live Result", "https://example.com/x")])
    payload = _run(q="seeds", provider=provider, limit=20)
    assert _titles(payload) == ["Live Result"]
    assert payload["result_count"] == 1


def test_flag_on_merges_after_successful_search(monkeypatch) -> None:
    _enable(monkeypatch, catalogue=True)
    provider = _ListProvider([_item("Live Result", "https://example.com/x")])
    payload = _run(q="seeds", provider=provider, limit=20)
    titles = _titles(payload)
    assert "Live Result" in titles
    assert SEED_CATALOGUE[0].title in titles
    assert payload["result_count"] == len(titles)


def test_flag_on_with_empty_live_returns_catalogue(monkeypatch) -> None:
    _enable(monkeypatch, catalogue=True)
    provider = _ListProvider([])
    payload = _run(q="seeds", provider=provider, limit=20)
    assert payload["result_count"] >= 1
    assert SEED_CATALOGUE[0].title in _titles(payload)


def test_provider_failure_does_not_return_catalogue(monkeypatch) -> None:
    _enable(monkeypatch, catalogue=True)
    with pytest.raises(ApiError) as excinfo:
        _run(q="seeds", provider=_BoomProvider(), limit=20)
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "scheme_search_upstream_error"


def test_provider_timeout_does_not_return_catalogue(monkeypatch) -> None:
    _enable(monkeypatch, catalogue=True)
    monkeypatch.setattr(settings, "scheme_search_timeout_seconds", 0.01)
    with pytest.raises(ApiError) as excinfo:
        _run(q="seeds", provider=_SlowProvider(), limit=20)
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "scheme_search_timeout"


def test_canonical_collision_keeps_curated_provenance(monkeypatch) -> None:
    _enable(
        monkeypatch,
        catalogue=True,
        origins=("manipur:agrimanipur.mn.gov.in",),
    )
    provider = _ListProvider(
        [_item("Live Title", "https://agrimanipur.mn.gov.in/")]
    )
    payload = _run(q="seeds", provider=provider, limit=20)

    matches = [
        item
        for item in payload["results"]
        if item["url"] == "https://agrimanipur.mn.gov.in/"
    ]
    assert len(matches) == 1
    assert matches[0]["title"] == SEED_CATALOGUE[0].title
    assert matches[0]["verified"] is True
    assert matches[0]["source_type"] == OFFICIAL_MANIPUR


def test_counts_match_and_response_schema_unchanged(monkeypatch) -> None:
    _enable(monkeypatch, catalogue=True)
    provider = _ListProvider([_item("Live", "https://example.com/x")])
    payload = _run(q="seeds", provider=provider, limit=20)

    assert payload["result_count"] == len(payload["results"])
    assert payload["official_count"] == sum(
        1 for item in payload["results"] if item["verified"]
    )
    assert set(payload.keys()) == {
        "query",
        "results",
        "result_count",
        "official_count",
        "retrieved_at",
        "disclaimer",
        "language",
    }
    assert set(payload["results"][0].keys()) == {
        "title",
        "url",
        "snippet",
        "source_domain",
        "source_type",
        "verified",
    }


def test_catalogue_candidates_make_no_network_calls(monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("the catalogue attempted a network call")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    out = catalogue_candidates(OfficialSourceRegistry())
    assert len(out) == len(SEED_CATALOGUE)


def _item(title: str, url: str, body: str = "A snippet"):
    from app.schemes import SearchCandidate

    return SearchCandidate(title=title, url=url, snippet=body)
