"""HTTP tests for ``GET /schemes/search`` (Increment 4).

A fake, in-process provider is injected via ``get_default_provider``; the real
DDGS provider and the network are never used. Timeouts/failures are simulated by
the fake. The route never fetches a result URL, which is asserted directly.
"""

import socket
import sys
import time
import types
import urllib.request
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.schemes import provider as scheme_provider

ENDPOINT = "/schemes/search"
client = TestClient(app)


class FakeProvider:
    """Records calls and returns preset raw items (DDGS-shaped mappings)."""

    def __init__(self, items=None, *, exc=None, delay=0.0):
        self.items = items if items is not None else []
        self.exc = exc
        self.delay = delay
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int):
        self.calls.append((query, limit))
        if self.delay:
            time.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return list(self.items)


def _install(
    monkeypatch,
    provider,
    *,
    enabled: bool = True,
    max_results: int = 20,
    timeout: float = 8.0,
    origins=(),
) -> None:
    monkeypatch.setattr(settings, "scheme_search_enabled", enabled)
    monkeypatch.setattr(settings, "scheme_search_max_results", max_results)
    monkeypatch.setattr(settings, "scheme_search_timeout_seconds", timeout)
    monkeypatch.setattr(settings, "scheme_official_origins", origins)
    monkeypatch.setattr(scheme_provider, "get_default_provider", lambda: provider)


def _item(title="Result", url="https://example.com/scheme", body="A snippet"):
    return {"title": title, "href": url, "body": body}


# --- success paths ---------------------------------------------------------- #


def test_search_with_q_returns_results(monkeypatch) -> None:
    provider = FakeProvider([_item(title="Seed Scheme")])
    _install(monkeypatch, provider)

    response = client.get(ENDPOINT, params={"q": "seeds"})
    assert response.status_code == 200
    body = response.json()

    assert body["result_count"] == 1
    assert body["results"][0]["title"] == "Seed Scheme"
    assert body["results"][0]["url"] == "https://example.com/scheme"
    assert body["query"] == {"q": "seeds", "crop": None, "district": None, "need": None}

    sent_query = provider.calls[0][0]
    assert sent_query.startswith("Manipur")
    assert "seeds" in sent_query


def test_search_with_crop_district_and_need(monkeypatch) -> None:
    provider = FakeProvider([_item()])
    _install(monkeypatch, provider)

    response = client.get(
        ENDPOINT,
        params={"crop": "rice", "district": "Imphal East", "need": "insurance"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == {
        "q": None,
        "crop": "rice",
        "district": "Imphal East",
        "need": "insurance",
    }

    sent_query = provider.calls[0][0]
    assert "Manipur" in sent_query
    assert "rice" in sent_query
    assert "Imphal East" in sent_query
    assert "crop insurance" in sent_query


def test_empty_results_return_200_with_zero_counts(monkeypatch) -> None:
    provider = FakeProvider([])
    _install(monkeypatch, provider)

    response = client.get(ENDPOINT, params={"q": "nothing here"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["result_count"] == 0
    assert body["official_count"] == 0


# --- input validation through HTTP ------------------------------------------ #


def test_missing_meaningful_input_is_envelope_400(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT)
    assert response.status_code == 400
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert body["error"]["code"] == "scheme_search_empty_query"


def test_blank_query_rejected(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"q": "   "})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "scheme_search_empty_query"


def test_overlong_query_rejected(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"q": "a" * 201})
    assert response.status_code == 400
    err = response.json()["error"]
    assert err["code"] == "scheme_search_query_too_long"
    assert err["details"] == {"max_characters": 200}


def test_invalid_crop_rejected(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"crop": "wheat"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "scheme_search_invalid_crop"


def test_invalid_need_rejected(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"need": "subsidy"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "scheme_search_invalid_need"


def test_unknown_district_is_accepted(monkeypatch) -> None:
    provider = FakeProvider([_item()])
    _install(monkeypatch, provider)
    response = client.get(ENDPOINT, params={"crop": "rice", "district": "Somewhere"})
    assert response.status_code == 200
    assert "Somewhere" in provider.calls[0][0]


def test_overlong_district_rejected(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"crop": "rice", "district": "x" * 61})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "scheme_search_invalid_district"


@pytest.mark.parametrize("bad", ["e", "english", "123"])
def test_invalid_language_rejected(monkeypatch, bad: str) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"q": "x", "language": bad})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "scheme_search_invalid_language"


def test_valid_language_returned_normalised(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"q": "x", "language": "HI"})
    assert response.status_code == 200
    assert response.json()["language"] == "hi"


def test_non_integer_limit_is_validation_error(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"q": "x", "limit": "abc"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --- limit handling --------------------------------------------------------- #


def test_limit_defaults_to_five(monkeypatch) -> None:
    provider = FakeProvider([])
    _install(monkeypatch, provider)
    client.get(ENDPOINT, params={"q": "x"})
    assert provider.calls[0][1] == 5


@pytest.mark.parametrize("bad", ["0", "25"])
def test_limit_out_of_bounds_rejected(monkeypatch, bad: str) -> None:
    _install(monkeypatch, FakeProvider())
    response = client.get(ENDPOINT, params={"q": "x", "limit": bad})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "scheme_search_invalid_limit"


def test_limit_capped_by_configured_maximum(monkeypatch) -> None:
    provider = FakeProvider([])
    _install(monkeypatch, provider, max_results=3)
    response = client.get(ENDPOINT, params={"q": "x", "limit": 20})
    assert response.status_code == 200
    assert provider.calls[0][1] == 3


# --- classification / counts / ordering ------------------------------------- #


def test_official_and_unverified_classification_preserved(monkeypatch) -> None:
    provider = FakeProvider(
        [
            _item(title="GoI", url="https://india.gov.in/scheme"),
            _item(title="Other", url="https://example.com/scheme"),
        ]
    )
    _install(monkeypatch, provider, origins=("goi:india.gov.in",))

    response = client.get(ENDPOINT, params={"q": "scheme"})
    assert response.status_code == 200
    body = response.json()
    assert [r["title"] for r in body["results"]] == ["GoI", "Other"]

    first, second = body["results"]
    assert first["verified"] is True
    assert first["source_type"] == "official_goi"
    assert first["source_domain"] == "india.gov.in"
    assert second["verified"] is False
    assert second["source_type"] == "other"

    assert body["result_count"] == 2
    assert body["official_count"] == 1


def test_empty_allowlist_marks_everything_unverified(monkeypatch) -> None:
    provider = FakeProvider([_item(title="Maybe", url="https://india.gov.in/x")])
    _install(monkeypatch, provider, origins=())

    response = client.get(ENDPOINT, params={"q": "x"})
    body = response.json()
    assert body["results"][0]["verified"] is False
    assert body["official_count"] == 0


# --- response metadata ------------------------------------------------------ #


def test_retrieved_at_is_utc_iso8601(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider([_item()]))
    response = client.get(ENDPOINT, params={"q": "x"})
    body = response.json()
    parsed = datetime.fromisoformat(body["retrieved_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert "eligibility" in body["disclaimer"]


def test_snippet_is_sanitized_and_length_limited(monkeypatch) -> None:
    long_body = "line1\n\nline2\t" + "x" * 500
    provider = FakeProvider([{**_item(body=long_body)}])
    _install(monkeypatch, provider)

    response = client.get(ENDPOINT, params={"q": "x"})
    snippet = response.json()["results"][0]["snippet"]
    assert "\n" not in snippet and "\t" not in snippet
    assert len(snippet) <= 300


# --- disabled / provider failures ------------------------------------------- #


def test_disabled_feature_uses_envelope_503(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider(), enabled=False)
    response = client.get(ENDPOINT, params={"q": "seeds"})
    assert response.status_code == 503
    body = response.json()
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert body["error"]["code"] == "scheme_search_disabled"


def test_disabled_feature_short_circuits_before_validation(monkeypatch) -> None:
    _install(monkeypatch, FakeProvider(), enabled=False)
    response = client.get(ENDPOINT)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "scheme_search_disabled"


def test_provider_failure_is_sanitized_502(monkeypatch) -> None:
    provider = FakeProvider(exc=RuntimeError("secret-stacktrace-detail"))
    _install(monkeypatch, provider)

    response = client.get(ENDPOINT, params={"q": "x"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "scheme_search_upstream_error"
    assert "secret-stacktrace-detail" not in response.text
    assert "Traceback" not in response.text


def test_provider_timeout_returns_504(monkeypatch) -> None:
    provider = FakeProvider([], delay=0.3)
    _install(monkeypatch, provider, timeout=0.01)

    response = client.get(ENDPOINT, params={"q": "x"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "scheme_search_timeout"


# --- Increment 7: empty handling, ranking, dedup, counts -------------------- #


def _install_fake_ddgs_empty(monkeypatch) -> None:
    """Fake ddgs whose successful-empty sentinel is raised (no real package)."""
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

    class FakeDDGS:
        def text(self, query, **kwargs):
            raise DDGSException("No results found.")

    fake = types.ModuleType("ddgs")
    fake.DDGS = FakeDDGS
    fake.exceptions = exc_mod
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exc_mod)


def test_ddgs_successful_empty_search_returns_200_empty(monkeypatch) -> None:
    _install(monkeypatch, scheme_provider.DdgsSearchProvider())
    _install_fake_ddgs_empty(monkeypatch)

    response = client.get(ENDPOINT, params={"q": "nothing matches"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["result_count"] == 0
    assert body["official_count"] == 0


def test_official_manipur_precedes_goi_and_other(monkeypatch) -> None:
    provider = FakeProvider(
        [
            _item(title="Other", url="https://example.com/x"),
            _item(title="GoI", url="https://india.gov.in/y"),
            _item(title="Manipur Agri", url="https://agrimanipur.mn.gov.in/z"),
        ]
    )
    _install(
        monkeypatch,
        provider,
        origins=("manipur:agrimanipur.mn.gov.in", "goi:india.gov.in"),
    )

    body = client.get(ENDPOINT, params={"q": "scheme"}).json()
    assert [r["title"] for r in body["results"]] == ["Manipur Agri", "GoI", "Other"]
    assert body["official_count"] == 2
    assert body["result_count"] == 3


def test_crop_and_need_affect_route_ordering(monkeypatch) -> None:
    provider = FakeProvider(
        [
            _item(title="Generic", url="https://e.com/g", body="farming support"),
            _item(
                title="Rice insurance",
                url="https://e.com/r",
                body="crop insurance for paddy rice",
            ),
        ]
    )
    _install(monkeypatch, provider)

    body = client.get(
        ENDPOINT, params={"crop": "rice", "need": "insurance"}
    ).json()
    assert [r["title"] for r in body["results"]] == ["Rice insurance", "Generic"]


def test_duplicate_urls_collapsed_and_counts_match_final_list(monkeypatch) -> None:
    provider = FakeProvider(
        [
            _item(title="A", url="https://e.com/scheme"),
            _item(title="A dup", url="https://e.com/scheme?utm_source=x#frag"),
            _item(title="B", url="https://e.com/other"),
        ]
    )
    _install(monkeypatch, provider)

    body = client.get(ENDPOINT, params={"q": "x"}).json()
    assert body["result_count"] == len(body["results"]) == 2
    assert [r["title"] for r in body["results"]] == ["A", "B"]


def test_limit_is_applied_after_dedup(monkeypatch) -> None:
    provider = FakeProvider(
        [
            _item(title="A", url="https://e.com/1"),
            _item(title="A dup", url="https://e.com/1?utm_source=x"),
            _item(title="B", url="https://e.com/2"),
            _item(title="C", url="https://e.com/3"),
        ]
    )
    _install(monkeypatch, provider)

    body = client.get(ENDPOINT, params={"q": "x", "limit": 2}).json()
    assert body["result_count"] == 2
    assert [r["title"] for r in body["results"]] == ["A", "B"]


def test_lookalike_host_is_not_promoted_to_verified(monkeypatch) -> None:
    provider = FakeProvider(
        [_item(title="Fake gov", url="https://manipur.gov.in.evil.com/x")]
    )
    _install(monkeypatch, provider, origins=("manipur:manipur.gov.in",))

    body = client.get(ENDPOINT, params={"q": "x"}).json()
    assert body["results"][0]["verified"] is False
    assert body["official_count"] == 0


# --- no network egress ------------------------------------------------------ #


def test_result_urls_are_never_fetched(monkeypatch) -> None:
    provider = FakeProvider([_item(url="https://example.com/never-fetch")])
    _install(monkeypatch, provider)

    def _boom(*_args, **_kwargs):
        raise AssertionError("the backend attempted an outbound network call")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    response = client.get(ENDPOINT, params={"q": "x"})
    assert response.status_code == 200
    assert response.json()["results"][0]["url"] == "https://example.com/never-fetch"
    assert len(provider.calls) == 1


# --- existing routes unchanged ---------------------------------------------- #


def test_existing_routes_still_work() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/crops").status_code == 200
    speech = client.post(
        "/speech/response",
        json={"display_name": "Maize", "predicted_label": "Smut"},
    )
    assert speech.status_code == 200
    assert "text" in speech.json()
