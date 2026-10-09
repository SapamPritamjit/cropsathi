"""Farmer Scheme Finder search foundation.

Holds the provider abstraction, normalized data types, official-source
classification (Increment 2), the validated search orchestration /
query-building used by ``GET /schemes/search`` (Increment 4) and deterministic
official-first ranking / deduplication (Increment 7). The FastAPI route itself
lives in ``app.routers.schemes``.
"""

from __future__ import annotations

from .models import ClassifiedCandidate, SearchCandidate
from .provider import (
    DdgsSearchProvider,
    SearchProvider,
    get_default_provider,
    normalize_candidate,
    normalize_candidates,
    search_schemes,
)
from .ranking import (
    canonical_url_key,
    deduplicate,
    rank_and_limit,
    relevance_score,
    tier_of,
)
from .service import (
    DEFAULT_LIMIT,
    DISCLAIMER,
    MAX_LIMIT,
    MIN_LIMIT,
    SUPPORTED_NEEDS,
    ResolvedSearch,
    build_search_query,
    resolve_search_query,
    run_scheme_search,
    sanitize_snippet,
)
from .sources import (
    OFFICIAL_GOI,
    OFFICIAL_MANIPUR,
    OTHER,
    OfficialSourceRegistry,
    SourceClassification,
    classify_candidate,
    hostname_from_url,
    normalize_host,
)
from .catalogue import (
    DEFAULT_SCHEME_STATUS,
    SCHEME_STATUSES,
    SCHEME_STATUS_ACTIVE,
    SCHEME_STATUS_INFORMATION_PAGE,
    SCHEME_STATUS_UNKNOWN,
    SEED_CATALOGUE,
    CatalogueEntry,
    CatalogueError,
    catalogue_candidates,
    validate_catalogue,
)

__all__ = [
    "ClassifiedCandidate",
    "SearchCandidate",
    "DdgsSearchProvider",
    "SearchProvider",
    "get_default_provider",
    "normalize_candidate",
    "normalize_candidates",
    "search_schemes",
    "DEFAULT_LIMIT",
    "DISCLAIMER",
    "MAX_LIMIT",
    "MIN_LIMIT",
    "SUPPORTED_NEEDS",
    "ResolvedSearch",
    "build_search_query",
    "resolve_search_query",
    "run_scheme_search",
    "sanitize_snippet",
    "canonical_url_key",
    "deduplicate",
    "rank_and_limit",
    "relevance_score",
    "tier_of",
    "OFFICIAL_GOI",
    "OFFICIAL_MANIPUR",
    "OTHER",
    "OfficialSourceRegistry",
    "SourceClassification",
    "classify_candidate",
    "hostname_from_url",
    "normalize_host",
    "DEFAULT_SCHEME_STATUS",
    "SCHEME_STATUSES",
    "SCHEME_STATUS_ACTIVE",
    "SCHEME_STATUS_INFORMATION_PAGE",
    "SCHEME_STATUS_UNKNOWN",
    "SEED_CATALOGUE",
    "CatalogueEntry",
    "CatalogueError",
    "catalogue_candidates",
    "validate_catalogue",
]
