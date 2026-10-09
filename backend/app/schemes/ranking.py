"""Deterministic relevance ranking and conservative deduplication (Increment 7).

Pure, offline, side-effect-free helpers applied to already-classified
candidates. Nothing here fetches a URL, inspects HTML, or infers eligibility,
availability or benefit amounts — it only reorders and de-duplicates the search
results the provider already returned.

Ranking is two-level and fully deterministic:

1. **Source tier** (from the existing :class:`OfficialSourceRegistry`
   classification): verified ``official_manipur`` first, then verified
   ``official_goi``, then ``other``/unverified. A title or snippet is never used
   to promote a result to "official".
2. **Relevance score** within a tier, from the *structured* search inputs:
   Manipur context, the selected crop, the selected support need and the query
   terms. Higher scores sort first; ties keep provider order (``sorted`` is
   stable).

Deduplication is conservative: URLs are canonicalized by lower-casing the host,
dropping fragments, removing only clearly-tracking query parameters, sorting the
remaining (meaningful) parameters and normalizing trailing slashes. Distinct URLs
are never merged merely because their titles look similar; a second result is
dropped only when the canonical URL repeats, or when the same host carries the
same normalized title (exact/near-exact duplicate of the same result). After
deduplication the requested/configured limit is applied.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import ClassifiedCandidate
from .sources import OFFICIAL_GOI, OFFICIAL_MANIPUR, OTHER

# --- ranking weights (documented, deterministic) ---------------------------- #

_TIER: dict[str, int] = {OFFICIAL_MANIPUR: 0, OFFICIAL_GOI: 1, OTHER: 2}

W_MANIPUR = 1
W_CROP = 3
W_NEED = 3
W_QUERY_TERM = 1

_MIN_QUERY_TERM_LENGTH = 3

# Only pure grammar words are ignored; domain words (crop, scheme, government,
# …) are intentionally kept so the query's own vocabulary still scores.
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "that", "this", "these",
        "those", "are", "was", "were", "will", "shall", "can", "could", "would",
        "should", "your", "you", "our", "its", "his", "her", "their", "a", "an",
        "of", "to", "in", "on", "at", "by", "or", "as", "is", "be", "it", "we",
        "not", "but", "all", "any", "how", "what", "who", "when", "where",
    }
)

# --- canonicalization ------------------------------------------------------- #

# Explicitly harmless tracking parameters. "Meaningful" parameters (id, q, page,
# lang, …) are always preserved; only these known trackers are stripped.
_TRACKING_PARAMS = frozenset(
    {
        "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "ref_src",
        "mkt_tok", "_hsenc", "_hsmi", "yclid", "dclid", "msclkid",
    }
)
_TRACKING_PREFIXES = ("utm_",)

_WORD = re.compile(r"[a-z0-9]+")


def _is_tracking_param(key: str) -> bool:
    lowered = key.lower()
    return lowered in _TRACKING_PARAMS or lowered.startswith(_TRACKING_PREFIXES)


def canonical_url_key(url: object) -> str | None:
    """Return a canonical dedup key for an http/https URL, else ``None``.

    Host casing, a fragment, tracking parameters and a trailing slash are
    normalized away; meaningful query parameters are preserved (and sorted so
    parameter order alone never creates a duplicate).
    """
    if not isinstance(url, str):
        return None
    text = url.strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        return None
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"

    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    ]
    pairs.sort()
    return urlunsplit((scheme, netloc, path, urlencode(pairs), ""))


def _normalize_title(title: object) -> str:
    """Lower-case, punctuation-free, single-spaced title for near-exact matching."""
    if not isinstance(title, str):
        return ""
    return " ".join(_WORD.findall(title.lower()))


def _title_host_key(candidate: ClassifiedCandidate) -> tuple[str, str] | None:
    host = (candidate.source_host or "").lower()
    title = _normalize_title(candidate.title)
    if not host or not title:
        return None
    return title, host


def deduplicate(
    candidates: Iterable[ClassifiedCandidate],
) -> list[ClassifiedCandidate]:
    """Drop repeated results, keeping the first occurrence in input order.

    A result is dropped only when its canonical URL was already seen, or when an
    already-kept result shares both the same host and the same normalized title.
    Different hosts with identical titles are always kept.
    """
    seen_urls: set[str] = set()
    seen_title_host: set[tuple[str, str]] = set()
    kept: list[ClassifiedCandidate] = []
    for candidate in candidates:
        key = canonical_url_key(candidate.url)
        if key is None:
            key = candidate.url.strip().lower()
        if key in seen_urls:
            continue
        title_host = _title_host_key(candidate)
        if title_host is not None and title_host in seen_title_host:
            continue
        seen_urls.add(key)
        if title_host is not None:
            seen_title_host.add(title_host)
        kept.append(candidate)
    return kept


# --- scoring ---------------------------------------------------------------- #


def _tokens(text: object) -> set[str]:
    if not isinstance(text, str):
        return set()
    return set(_WORD.findall(text.lower()))


def _candidate_tokens(candidate: ClassifiedCandidate) -> set[str]:
    return _tokens(f"{candidate.title} {candidate.snippet} {candidate.url}")


def _query_terms(query: object) -> set[str]:
    return {
        term
        for term in _tokens(query)
        if len(term) >= _MIN_QUERY_TERM_LENGTH and term not in _STOPWORDS
    }


def _need_tokens(need: object) -> set[str]:
    if not isinstance(need, str):
        return set()
    return {token for token in need.lower().replace("_", " ").split() if token}


def relevance_score(
    candidate: ClassifiedCandidate,
    *,
    query: object = None,
    crop: object = None,
    need: object = None,
) -> int:
    """Score one candidate against the structured search inputs (higher = better).

    Deterministic, integer-only, and based solely on token presence in the
    candidate's title/snippet/URL — it makes no claim about eligibility,
    availability or benefits.
    """
    tokens = _candidate_tokens(candidate)
    score = 0
    if "manipur" in tokens:
        score += W_MANIPUR
    if isinstance(crop, str) and crop.strip() and crop.strip().lower() in tokens:
        score += W_CROP
    need_terms = _need_tokens(need)
    if need_terms and need_terms & tokens:
        score += W_NEED
    terms = _query_terms(query)
    if terms:
        score += W_QUERY_TERM * len(terms & tokens)
    return score


def tier_of(source_type: str) -> int:
    """Return the ranking tier for a source type (unknown types rank last)."""
    return _TIER.get(source_type, 2)


def rank_and_limit(
    candidates: Sequence[ClassifiedCandidate],
    *,
    query: object = None,
    crop: object = None,
    need: object = None,
    limit: int | None = None,
) -> list[ClassifiedCandidate]:
    """Deduplicate, then order official-first by relevance, then cap to ``limit``.

    Sorting is stable, so candidates with equal tier and score keep provider
    order. ``limit`` is applied last, after deduplication.
    """
    unique = deduplicate(candidates)
    ordered = sorted(
        unique,
        key=lambda candidate: (
            tier_of(candidate.source_type),
            -relevance_score(candidate, query=query, crop=crop, need=need),
        ),
    )
    if limit is None:
        return ordered
    return ordered[: max(0, limit)]
