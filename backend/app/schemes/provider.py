"""Replaceable search-provider abstraction for the Farmer Scheme Finder.

No FastAPI route lives here (the ``/schemes/search`` endpoint is a later
increment). The default DDGS-backed provider is imported lazily inside its
``search`` method, so importing this module — and running the test suite — never
requires the ``ddgs`` package and never performs any network access.

Safety properties:

- searches run only when ``settings.scheme_search_enabled`` is true;
- the DDGS region comes from ``settings.scheme_search_region`` (default ``in-en``)
  and is the only DDGS option set beyond the query and result limit;
- blocking provider work is bounded by ``settings.scheme_search_timeout_seconds``;
- a *successful* search with zero matches (ddgs's ``DDGSException("No results
  found.")`` sentinel) is converted to an empty list, while genuine upstream
  failures and timeouts stay errors;
- provider failures surface as stable :class:`ApiError` codes with generic
  messages — upstream exception text, HTML and credentials are never exposed;
- result URLs are never fetched, so no SSRF surface is introduced.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Protocol

from ..config import settings
from ..errors import ApiError
from .models import ClassifiedCandidate, SearchCandidate
from .sources import OfficialSourceRegistry, classify_candidate, hostname_from_url

# Keys accepted from a raw DDGS result mapping. Verified against ddgs 9.16.0,
# whose text() returns dicts with "title", "href" and "body"; the extra aliases
# are retained as defensive fallbacks for other backends. This mapping is
# isolated so it can be adjusted without touching the rest of the application.
_TITLE_KEYS = ("title", "text")
_URL_KEYS = ("href", "url")
_SNIPPET_KEYS = ("body", "snippet", "description")

# The exact message ddgs 9.16.0's ``DDGS().text()`` raises when every engine
# completed successfully but produced no matches (``ddgs/ddgs.py``,
# ``_search_sync``: ``raise DDGSException(err or "No results found.")``). When an
# engine actually fails, ``err`` is that engine's own message instead — so only
# this exact sentinel means "verified empty", never a real upstream failure.
_NO_RESULTS_MESSAGE = "No results found."


def _ddgs_exception_types() -> (
    tuple[type[BaseException], type[BaseException], type[BaseException]] | None
):
    """Return ddgs's ``(DDGSException, RatelimitException, TimeoutException)``.

    Imported lazily so importing this module never requires ``ddgs``; returns
    ``None`` when the package (or its exceptions module) is unavailable, in which
    case no exception can be recognized as an "empty result" signal.
    """
    try:
        from ddgs.exceptions import (  # type: ignore[import-not-found]
            DDGSException,
            RatelimitException,
            TimeoutException,
        )
    except Exception:
        return None
    return DDGSException, RatelimitException, TimeoutException


class SearchProvider(Protocol):
    """A source of normalized search candidates.

    Implementations may block; callers wrap them in a thread with a timeout.
    """

    def search(self, query: str, limit: int) -> list[SearchCandidate]:
        ...


def _first_str(raw: Mapping, keys: Sequence[str]) -> str | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_candidate(raw: object) -> SearchCandidate | None:
    """Coerce a raw provider item into a :class:`SearchCandidate`, or ``None``.

    Requires a non-empty string title and an http/https URL. Missing, malformed
    or non-string values are dropped rather than surfaced to callers.
    """
    if isinstance(raw, SearchCandidate):
        title, url = raw.title, raw.url
        snippet = raw.snippet
    elif isinstance(raw, Mapping):
        title = _first_str(raw, _TITLE_KEYS)
        url = _first_str(raw, _URL_KEYS)
        snippet = _first_str(raw, _SNIPPET_KEYS) or ""
    else:
        return None

    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    if hostname_from_url(url) is None:
        return None

    clean_snippet = snippet.strip() if isinstance(snippet, str) else ""
    return SearchCandidate(
        title=title.strip(), url=url.strip(), snippet=clean_snippet
    )


def normalize_candidates(items: object) -> list[SearchCandidate]:
    """Normalize a provider result collection, dropping unusable entries."""
    if not isinstance(items, (list, tuple)):
        return []
    normalized: list[SearchCandidate] = []
    for item in items:
        candidate = normalize_candidate(item)
        if candidate is not None:
            normalized.append(candidate)
    return normalized


class DdgsSearchProvider:
    """Default provider backed by the ``ddgs`` package (lazy import).

    The call below matches the installed and verified ddgs 9.16.0 API —
    ``from ddgs import DDGS`` and ``DDGS().text(query, region=..., max_results=n)``
    — which returns a ``list[dict]`` whose items carry ``title``/``href``/``body``
    keys. The region is ``settings.scheme_search_region`` (India-English by
    default) and is the only DDGS option set beyond the query/limit. The package
    is declared in ``requirements.txt`` but imported lazily, so importing this
    module and running the suite never require it or touch the network. If a
    future ddgs release changes the API, only this method changes.
    """

    def search(self, query: str, limit: int) -> list[SearchCandidate]:
        try:
            from ddgs import DDGS  # type: ignore[import-not-found]
        except Exception as exc:  # ImportError or packaging issues
            raise ApiError(
                503,
                "scheme_search_not_configured",
                "The scheme search provider is not available on the server.",
            ) from exc

        try:
            raw_items = list(
                DDGS().text(
                    query,
                    region=settings.scheme_search_region,
                    max_results=limit,
                )
            )
        except Exception as exc:
            types_ = _ddgs_exception_types()
            if types_ is not None:
                ddgs_exc, ratelimit_exc, timeout_exc = types_
                if isinstance(exc, timeout_exc):
                    # ddgs timed out internally; keep the existing timeout error.
                    raise ApiError(
                        504,
                        "scheme_search_timeout",
                        "The scheme search timed out.",
                    ) from exc
                if (
                    isinstance(exc, ddgs_exc)
                    and not isinstance(exc, ratelimit_exc)
                    and str(exc).strip() == _NO_RESULTS_MESSAGE
                ):
                    # Verified "search succeeded, zero matches" -> empty result.
                    return []
            raise ApiError(
                502,
                "scheme_search_upstream_error",
                "The scheme search provider failed.",
            ) from exc
        return normalize_candidates(raw_items)


_default_provider: SearchProvider | None = None


def get_default_provider() -> SearchProvider:
    """Return the process-wide default provider (constructed lazily)."""
    global _default_provider
    if _default_provider is None:
        _default_provider = DdgsSearchProvider()
    return _default_provider


async def search_schemes(
    query: object,
    limit: int | None = None,
    *,
    provider: SearchProvider | None = None,
    registry: OfficialSourceRegistry | None = None,
) -> list[ClassifiedCandidate]:
    """Run a bounded scheme search and return classified candidates.

    Returns every classified candidate the (bounded) provider call yielded, in
    provider order; the caller applies relevance ranking, deduplication and the
    final result limit. Disabled feature -> ``503 scheme_search_disabled`` (the
    provider is never invoked). Blank query -> ``400
    scheme_search_empty_query``. A provider that exceeds the timeout -> ``504
    scheme_search_timeout``; any other provider failure -> ``502
    scheme_search_upstream_error`` (details are not leaked).
    """
    if not settings.scheme_search_enabled:
        raise ApiError(
            503,
            "scheme_search_disabled",
            "Farmer scheme search is disabled on the server.",
        )

    text = query.strip() if isinstance(query, str) else ""
    if not text:
        raise ApiError(
            400, "scheme_search_empty_query", "A search query is required."
        )

    ceiling = settings.scheme_search_max_results
    resolved_limit = ceiling if limit is None else limit
    if isinstance(resolved_limit, bool) or not isinstance(resolved_limit, int):
        raise ApiError(
            400, "scheme_search_invalid_limit", "Result limit must be an integer."
        )
    resolved_limit = max(1, min(resolved_limit, ceiling))

    active = provider if provider is not None else get_default_provider()
    try:
        raw_items = await asyncio.wait_for(
            asyncio.to_thread(active.search, text, resolved_limit),
            timeout=settings.scheme_search_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ApiError(
            504, "scheme_search_timeout", "The scheme search timed out."
        ) from exc
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            502,
            "scheme_search_upstream_error",
            "The scheme search provider failed.",
        ) from exc

    active_registry = (
        registry
        if registry is not None
        else OfficialSourceRegistry.from_specs(settings.scheme_official_origins)
    )
    candidates = normalize_candidates(raw_items)
    return [classify_candidate(candidate, active_registry) for candidate in candidates]
