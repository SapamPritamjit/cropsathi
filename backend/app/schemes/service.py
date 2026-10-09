"""Orchestration for the Farmer Scheme Finder search endpoint (Increment 4).

Pure input-validation / query-building helpers plus one async entry point,
:func:`run_scheme_search`, that:

- validates and bounds caller input (raising stable :class:`ApiError` codes);
- builds a single, length-bounded search string with Manipur context;
- delegates to the existing provider abstraction (never imports ``ddgs`` here);
- applies deterministic official-first ranking and conservative deduplication
  (``app.schemes.ranking``) and then the final result limit;
- assembles a typed, count-consistent response payload.

No result URL is ever fetched, no eligibility is inferred, and upstream failures
are surfaced as sanitized error codes only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import SUPPORTED_CROPS, settings
from ..errors import ApiError
from . import provider as scheme_provider
from .models import ClassifiedCandidate
from .ranking import canonical_url_key, rank_and_limit
from .sources import OfficialSourceRegistry

# --- public bounds ---------------------------------------------------------- #

SUPPORTED_NEEDS: tuple[str, ...] = (
    "seeds",
    "fertilizer",
    "credit",
    "insurance",
    "irrigation",
    "machinery",
    "market",
    "training",
    "pest_disease",
    "general",
)

DEFAULT_LIMIT = 5
MIN_LIMIT = 1
MAX_LIMIT = 20
MAX_QUERY_LENGTH = 200
MAX_DISTRICT_LENGTH = 60
MAX_SEARCH_QUERY_LENGTH = 240
MAX_SNIPPET_LENGTH = 300
MANIPUR_CONTEXT = "Manipur"

DISCLAIMER = (
    "Search results are informational. Confirm scheme availability and "
    "eligibility with the responsible official department."
)

# Human-readable expansions so a structured "need" reads naturally in the query.
_NEED_PHRASES: dict[str, str] = {
    "seeds": "seeds and seed subsidy",
    "fertilizer": "fertilizer and soil health",
    "credit": "agricultural credit and loans",
    "insurance": "crop insurance",
    "irrigation": "irrigation",
    "machinery": "farm machinery",
    "market": "market and procurement",
    "training": "farmer training",
    "pest_disease": "pest and disease management",
    "general": "farmer welfare",
}

_WHITESPACE = re.compile(r"\s+")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://")
_LANGUAGE = re.compile(r"^[a-z]{2}(?:-[a-z0-9]{2,8})?$")


@dataclass(frozen=True)
class ResolvedSearch:
    """Validated, bounded search inputs plus the built provider query string."""

    q: str | None
    crop: str | None
    district: str | None
    need: str | None
    limit: int
    language: str
    text: str


def _collapse(text: str) -> str:
    """Collapse all whitespace/control characters and trim the result."""
    return _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", text)).strip()


def _neutralize_urls(text: str) -> str:
    """Remove URL schemes so free text can never act as a fetch instruction."""
    return _URL_SCHEME.sub(" ", text)


def sanitize_snippet(text: object) -> str:
    """Return a single-line, control-character-free, length-bounded snippet."""
    if not isinstance(text, str):
        return ""
    cleaned = _collapse(text)
    if len(cleaned) > MAX_SNIPPET_LENGTH:
        cleaned = cleaned[:MAX_SNIPPET_LENGTH].rstrip()
    return cleaned


def build_search_query(
    q: str | None,
    crop: str | None,
    district: str | None,
    need: str | None,
) -> str:
    """Build a single bounded query string that always carries Manipur context.

    Every fragment is already validated/bounded by the caller; the joined result
    is additionally capped at ``MAX_SEARCH_QUERY_LENGTH`` so no combination can
    grow without limit.
    """
    parts: list[str] = [MANIPUR_CONTEXT]
    if crop:
        parts.append(f"{crop} crop")
    parts.append("agriculture scheme")
    if need:
        parts.append(_NEED_PHRASES.get(need, need))
    if district:
        parts.append(f"{district} district")
    if q:
        parts.append(_neutralize_urls(q))

    joined = _WHITESPACE.sub(" ", " ".join(parts)).strip()
    if len(joined) > MAX_SEARCH_QUERY_LENGTH:
        joined = joined[:MAX_SEARCH_QUERY_LENGTH].rstrip()
    return joined


def resolve_search_query(
    *,
    q: object = None,
    crop: object = None,
    district: object = None,
    need: object = None,
    limit: object = None,
    language: object = None,
) -> ResolvedSearch:
    """Validate and bound all inputs, raising :class:`ApiError` on misuse.

    The disabled-feature check runs first so a disabled deployment always
    answers ``503 scheme_search_disabled`` regardless of input validity.
    """
    if not settings.scheme_search_enabled:
        raise ApiError(
            503,
            "scheme_search_disabled",
            "Farmer scheme search is disabled on the server.",
        )

    q_text: str | None = None
    if isinstance(q, str) and q.strip():
        collapsed = _collapse(q)
        if len(collapsed) > MAX_QUERY_LENGTH:
            raise ApiError(
                400,
                "scheme_search_query_too_long",
                "The search query is too long.",
                {"max_characters": MAX_QUERY_LENGTH},
            )
        q_text = _neutralize_urls(collapsed).strip() or None

    crop_key: str | None = None
    if isinstance(crop, str) and crop.strip():
        candidate = _collapse(crop).lower()
        if candidate not in SUPPORTED_CROPS:
            raise ApiError(
                400,
                "scheme_search_invalid_crop",
                f"Unsupported crop '{_collapse(crop)}'. Supported crops: "
                + ", ".join(SUPPORTED_CROPS)
                + ".",
            )
        crop_key = candidate

    need_value: str | None = None
    if isinstance(need, str) and need.strip():
        candidate = _collapse(need).lower()
        if candidate not in SUPPORTED_NEEDS:
            raise ApiError(
                400,
                "scheme_search_invalid_need",
                f"Unsupported need '{_collapse(need)}'. Supported needs: "
                + ", ".join(SUPPORTED_NEEDS)
                + ".",
            )
        need_value = candidate

    # No verified Manipur district list exists in this repository, so a district
    # is accepted as a bounded, safe string rather than checked against one.
    district_value: str | None = None
    if isinstance(district, str) and district.strip():
        collapsed = _collapse(district)
        if len(collapsed) > MAX_DISTRICT_LENGTH:
            raise ApiError(
                400,
                "scheme_search_invalid_district",
                "The district value is too long.",
                {"max_characters": MAX_DISTRICT_LENGTH},
            )
        district_value = _neutralize_urls(collapsed).strip() or None

    if q_text is None and crop_key is None and need_value is None:
        raise ApiError(
            400,
            "scheme_search_empty_query",
            "Provide at least one of 'q', 'crop' or 'need'.",
        )

    language_value = "en"
    if isinstance(language, str) and language.strip():
        language_value = _collapse(language).lower()
    if not _LANGUAGE.match(language_value):
        raise ApiError(
            400,
            "scheme_search_invalid_language",
            "Language must be an ISO-639-1 code such as 'en' or 'hi'.",
        )

    if limit is None:
        resolved_limit = DEFAULT_LIMIT
    else:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ApiError(
                400,
                "scheme_search_invalid_limit",
                "Result limit must be an integer.",
            )
        if limit < MIN_LIMIT or limit > MAX_LIMIT:
            raise ApiError(
                400,
                "scheme_search_invalid_limit",
                f"Result limit must be between {MIN_LIMIT} and {MAX_LIMIT}.",
                {"min": MIN_LIMIT, "max": MAX_LIMIT},
            )
        resolved_limit = limit

    # Never request more than the operator-configured ceiling.
    resolved_limit = max(MIN_LIMIT, min(resolved_limit, settings.scheme_search_max_results))

    return ResolvedSearch(
        q=q_text,
        crop=crop_key,
        district=district_value,
        need=need_value,
        limit=resolved_limit,
        language=language_value,
        text=build_search_query(q_text, crop_key, district_value, need_value),
    )


def _candidate_to_result(candidate: ClassifiedCandidate) -> dict:
    return {
        "title": candidate.title,
        "url": candidate.url,
        "snippet": sanitize_snippet(candidate.snippet),
        "source_domain": candidate.source_host,
        "source_type": candidate.source_type,
        "verified": candidate.verified,
    }


def _merge_curated(
    curated: list[ClassifiedCandidate],
    live: list[ClassifiedCandidate],
) -> list[ClassifiedCandidate]:
    """Put curated entries first, filling an empty curated snippet from a live match.

    Curated entries go first so that, when a curated URL also appears in the live
    results, deterministic deduplication keeps the curated record (its audited
    title/provenance). Only an empty curated snippet is borrowed from the live
    match; nothing else about the curated entry is replaced.
    """
    if not curated:
        return list(live)

    snippet_by_key: dict[str, str] = {}
    for candidate in live:
        key = canonical_url_key(candidate.url)
        if key is not None and candidate.snippet:
            snippet_by_key.setdefault(key, candidate.snippet)

    merged: list[ClassifiedCandidate] = []
    for candidate in curated:
        if not candidate.snippet:
            key = canonical_url_key(candidate.url)
            borrowed = snippet_by_key.get(key) if key is not None else None
            if borrowed:
                candidate = ClassifiedCandidate(
                    title=candidate.title,
                    url=candidate.url,
                    snippet=borrowed,
                    source_host=candidate.source_host,
                    source_type=candidate.source_type,
                    verified=candidate.verified,
                )
        merged.append(candidate)

    return merged + list(live)


async def run_scheme_search(
    *,
    q: object = None,
    crop: object = None,
    district: object = None,
    need: object = None,
    limit: object = None,
    language: object = None,
    provider: object = None,
    registry: object = None,
) -> dict:
    """Validate inputs, run the bounded provider search and assemble the payload.

    Provider/registry are injectable for tests; production callers omit them so
    the process-wide default (and the configured allow-list) is used.
    """
    resolved = resolve_search_query(
        q=q,
        crop=crop,
        district=district,
        need=need,
        limit=limit,
        language=language,
    )

    active_registry = (
        registry
        if registry is not None
        else OfficialSourceRegistry.from_specs(settings.scheme_official_origins)
    )

    # The provider call runs first so upstream errors/timeouts surface as-is; the
    # curated catalogue is only ever merged after a successful live search.
    candidates = await scheme_provider.search_schemes(
        resolved.text,
        resolved.limit,
        provider=provider,
        registry=active_registry,
    )

    if settings.scheme_catalogue_enabled:
        # Lazy import keeps the catalogue optional and avoids any import cycle
        # (the catalogue builds on this module's validated need vocabulary).
        from . import catalogue as scheme_catalogue

        curated = scheme_catalogue.catalogue_candidates(active_registry)
        candidates = _merge_curated(curated, candidates)

    # Official-first, relevance-ordered, deduplicated, then limited. Counts below
    # are derived from this final list, never from the raw provider output.
    ranked = rank_and_limit(
        candidates,
        query=resolved.q,
        crop=resolved.crop,
        need=resolved.need,
        limit=resolved.limit,
    )

    results = [_candidate_to_result(candidate) for candidate in ranked]
    official_count = sum(1 for item in results if item["verified"])

    return {
        "query": {
            "q": resolved.q,
            "crop": resolved.crop,
            "district": resolved.district,
            "need": resolved.need,
        },
        "results": results,
        "result_count": len(results),
        "official_count": official_count,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "language": resolved.language,
    }
