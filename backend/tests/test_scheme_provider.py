"""Provider-foundation tests for the Farmer Scheme Finder (Increment 2).

All providers are fake/injected. No ``ddgs`` import, no outbound request and no
real search is ever performed.
"""

import asyncio
import sys
import time
import types

import pytest

from app.config import settings
from app.errors import ApiError
from app.schemes import provider as scheme_provider
from app.schemes.models import ClassifiedCandidate, SearchCandidate
from app.schemes.sources import OFFICIAL_GOI, OTHER


def run(coro):
    return asyncio.run(coro)


class RecordingProvider:
    """Fake provider that records calls and returns preset items."""

    def __init__(self, items):
        self.items = items
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int):
        self.calls.append((query, limit))
        return list(self.items)


def _enable(monkeypatch, *, timeout: float = 5.0) -> None:
    monkeypatch.setattr(settings, "scheme_search_enabled", True)
    monkeypatch.setattr(settings, "scheme_search_timeout_seconds", timeout)
    monkeypatch.setattr(settings, "scheme_search_max_results", 20)
    monkeypatch.setattr(settings, "scheme_official_origins", ())


# --- disabled / validation -------------------------------------------------- #


def test_disabled_search_never_invokes_provider(monkeypatch) -> None:
    monkeypatch.setattr(settings, "scheme_search_enabled", False)
    provider = RecordingProvider([SearchCandidate("t", "https://a.example", "")])
    with pytest.raises(ApiError) as excinfo:
        run(scheme_provider.search_schemes("seeds", provider=provider))
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "scheme_search_disabled"
    assert provider.calls == []


def test_blank_query_rejected_without_calling_provider(monkeypatch) -> None:
    _enable(monkeypatch)
    provider = RecordingProvider([])
    with pytest.raises(ApiError) as excinfo:
        run(scheme_provider.search_schemes("   ", provider=provider))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "scheme_search_empty_query"
    assert provider.calls == []


def test_non_integer_limit_rejected(monkeypatch) -> None:
    _enable(monkeypatch)
    provider = RecordingProvider([])
    with pytest.raises(ApiError) as excinfo:
        run(scheme_provider.search_schemes("q", "5", provider=provider))
    assert excinfo.value.code == "scheme_search_invalid_limit"
    assert provider.calls == []


def test_limit_clamped_to_configured_ceiling(monkeypatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setattr(settings, "scheme_search_max_results", 3)
    provider = RecordingProvider([])
    run(scheme_provider.search_schemes("q", 99, provider=provider))
    assert provider.calls == [("q", 3)]


# --- normalization ---------------------------------------------------------- #


def test_fake_provider_returns_normalized_classified_candidates(monkeypatch) -> None:
    _enable(monkeypatch)
    provider = RecordingProvider(
        [
            {"title": "A", "href": "https://a.example/x", "body": "snippet a"},
            SearchCandidate("B", "https://b.example", "snippet b"),
        ]
    )
    out = run(scheme_provider.search_schemes("schemes", provider=provider))
    assert [c.title for c in out] == ["A", "B"]
    assert all(isinstance(c, ClassifiedCandidate) for c in out)
    assert out[0].url == "https://a.example/x"
    assert out[0].snippet == "snippet a"
    assert out[0].source_type == OTHER
    assert out[0].verified is False
    assert provider.calls == [("schemes", settings.scheme_search_max_results)]


def test_classification_uses_configured_allowlist(monkeypatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setattr(settings, "scheme_official_origins", ("goi:india.gov.in",))
    provider = RecordingProvider(
        [
            {"title": "GoI", "href": "https://india.gov.in/scheme"},
            {"title": "Other", "href": "https://example.com/scheme"},
        ]
    )
    out = run(scheme_provider.search_schemes("q", provider=provider))
    assert out[0].source_type == OFFICIAL_GOI
    assert out[0].verified is True
    assert out[1].source_type == OTHER
    assert out[1].verified is False


def test_malformed_items_are_dropped() -> None:
    items = [
        {"title": None, "href": "https://a.example"},
        {"title": "ok", "href": None},
        {"title": 123, "href": "https://a.example"},
        {"title": "ok", "href": "ftp://a.example"},
        {"title": "ok", "href": "javascript:alert(1)"},
        "not a mapping",
        42,
        {"title": "   ", "href": "https://a.example"},
        {"title": "good", "href": "https://good.example", "body": "kept"},
    ]
    out = scheme_provider.normalize_candidates(items)
    assert [c.title for c in out] == ["good"]
    assert out[0].url == "https://good.example"


def test_non_sequence_result_is_empty() -> None:
    assert scheme_provider.normalize_candidates(None) == []
    assert scheme_provider.normalize_candidates({"title": "x"}) == []


# --- failures --------------------------------------------------------------- #


def test_provider_timeout_maps_to_504(monkeypatch) -> None:
    _enable(monkeypatch, timeout=0.01)

    class SlowProvider:
        def search(self, query: str, limit: int):
            time.sleep(0.3)
            return []

    with pytest.raises(ApiError) as excinfo:
        run(scheme_provider.search_schemes("q", provider=SlowProvider()))
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "scheme_search_timeout"


def test_provider_exception_is_sanitized(monkeypatch) -> None:
    _enable(monkeypatch)

    class BoomProvider:
        def search(self, query: str, limit: int):
            raise RuntimeError("secret-stacktrace-detail")

    with pytest.raises(ApiError) as excinfo:
        run(scheme_provider.search_schemes("q", provider=BoomProvider()))
    err = excinfo.value
    assert err.status_code == 502
    assert err.code == "scheme_search_upstream_error"
    blob = err.message + str(err.details)
    assert "secret-stacktrace-detail" not in blob
    assert "Traceback" not in blob


def test_api_error_from_provider_is_preserved(monkeypatch) -> None:
    _enable(monkeypatch)

    class NotConfigured:
        def search(self, query: str, limit: int):
            raise ApiError(503, "scheme_search_not_configured", "safe message")

    with pytest.raises(ApiError) as excinfo:
        run(scheme_provider.search_schemes("q", provider=NotConfigured()))
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "scheme_search_not_configured"


# --- lazy / offline guarantees --------------------------------------------- #


def test_provider_module_is_importable_without_ddgs() -> None:
    assert "ddgs" not in __import__("sys").modules


def test_default_provider_construction_is_lazy() -> None:
    provider = scheme_provider.get_default_provider()
    assert provider is not None
    assert "ddgs" not in __import__("sys").modules


# --- DDGS call wiring: region + limit, with a fake module (no real ddgs) ----- #


def _install_fake_ddgs(monkeypatch, recorder: dict, items=None):
    """Inject a fake ``ddgs`` module whose DDGS.text records its arguments."""
    fake = types.ModuleType("ddgs")

    class FakeDDGS:
        def text(self, query, **kwargs):
            recorder["query"] = query
            recorder["kwargs"] = kwargs
            if items is not None:
                return list(items)
            return [{"title": "T", "href": "https://a.example/x", "body": "b"}]

    fake.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    return fake


def test_provider_passes_configured_region_and_limit(monkeypatch) -> None:
    monkeypatch.setattr(scheme_provider.settings, "scheme_search_region", "in-en")
    recorder: dict = {}
    _install_fake_ddgs(monkeypatch, recorder)

    out = scheme_provider.DdgsSearchProvider().search("manipur seeds", 5)

    assert recorder["query"] == "manipur seeds"
    # Region is configured; the only other DDGS option is the result limit.
    assert recorder["kwargs"] == {"region": "in-en", "max_results": 5}
    assert [c.title for c in out] == ["T"]


def test_provider_region_is_configurable_at_call_time(monkeypatch) -> None:
    monkeypatch.setattr(scheme_provider.settings, "scheme_search_region", "hi-in")
    recorder: dict = {}
    _install_fake_ddgs(monkeypatch, recorder)

    scheme_provider.DdgsSearchProvider().search("q", 3)

    assert recorder["kwargs"]["region"] == "hi-in"
    assert recorder["kwargs"]["max_results"] == 3


# --- DDGS outcome normalization (fake ddgs + fake ddgs.exceptions) ----------- #


def _install_fake_ddgs_raising(monkeypatch, *, kind: str, message: str):
    """Inject a fake ``ddgs`` whose ``DDGS.text`` raises a chosen exception.

    ``kind`` selects the class hierarchy ddgs 9.16.0 actually uses, so the
    provider's empty/timeout/failure discrimination is exercised faithfully
    without importing the real package or touching the network.
    """
    exc_mod = types.ModuleType("ddgs.exceptions")

    class DDGSException(Exception):
        ...

    class TimeoutException(DDGSException):
        ...

    class RatelimitException(DDGSException):
        ...

    exc_mod.DDGSException = DDGSException
    exc_mod.TimeoutException = TimeoutException
    exc_mod.RatelimitException = RatelimitException

    classes = {
        "ddgs": DDGSException,
        "timeout": TimeoutException,
        "ratelimit": RatelimitException,
        "runtime": RuntimeError,
    }

    class FakeDDGS:
        def text(self, query, **kwargs):
            raise classes[kind](message)

    fake = types.ModuleType("ddgs")
    fake.DDGS = FakeDDGS
    fake.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exc_mod)
    return fake


def test_ddgs_succeeded_with_no_results_is_empty_list(monkeypatch) -> None:
    _install_fake_ddgs_raising(monkeypatch, kind="ddgs", message="No results found.")
    assert scheme_provider.DdgsSearchProvider().search("q", 3) == []


@pytest.mark.parametrize(
    "message",
    ["No results", "No results found", "engine exploded", "rate limited"],
)
def test_ddgs_non_sentinel_exceptions_are_not_empty(monkeypatch, message: str) -> None:
    _install_fake_ddgs_raising(monkeypatch, kind="ddgs", message=message)
    with pytest.raises(ApiError) as excinfo:
        scheme_provider.DdgsSearchProvider().search("q", 3)
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "scheme_search_upstream_error"


def test_ddgs_timeout_exception_maps_to_504(monkeypatch) -> None:
    _install_fake_ddgs_raising(monkeypatch, kind="timeout", message="request timed out")
    with pytest.raises(ApiError) as excinfo:
        scheme_provider.DdgsSearchProvider().search("q", 3)
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "scheme_search_timeout"


def test_ddgs_engine_failure_is_sanitized_502(monkeypatch) -> None:
    _install_fake_ddgs_raising(monkeypatch, kind="ddgs", message="secret-upstream-html")
    with pytest.raises(ApiError) as excinfo:
        scheme_provider.DdgsSearchProvider().search("q", 3)
    err = excinfo.value
    assert err.status_code == 502
    assert err.code == "scheme_search_upstream_error"
    assert "secret-upstream-html" not in (err.message + str(err.details))


def test_ddgs_ratelimit_is_upstream_error(monkeypatch) -> None:
    _install_fake_ddgs_raising(monkeypatch, kind="ratelimit", message="rate limited")
    with pytest.raises(ApiError) as excinfo:
        scheme_provider.DdgsSearchProvider().search("q", 3)
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "scheme_search_upstream_error"


def test_ddgs_unexpected_runtime_error_is_sanitized_502(monkeypatch) -> None:
    _install_fake_ddgs_raising(monkeypatch, kind="runtime", message="boom-secret")
    with pytest.raises(ApiError) as excinfo:
        scheme_provider.DdgsSearchProvider().search("q", 3)
    assert excinfo.value.code == "scheme_search_upstream_error"
    assert "boom-secret" not in (excinfo.value.message + str(excinfo.value.details))


def test_search_schemes_returns_all_classified_without_pre_slicing(monkeypatch) -> None:
    _enable(monkeypatch)
    items = [
        SearchCandidate(f"T{i}", f"https://e{i}.example/x", "") for i in range(4)
    ]
    provider = RecordingProvider(items)
    out = run(scheme_provider.search_schemes("q", 2, provider=provider))
    # The provider is asked for 2, but the function no longer truncates: the
    # service applies the final limit after dedup/ranking.
    assert provider.calls == [("q", 2)]
    assert [c.title for c in out] == ["T0", "T1", "T2", "T3"]
