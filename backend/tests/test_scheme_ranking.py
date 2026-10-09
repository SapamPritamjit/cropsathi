"""Pure ranking + deduplication tests for the Farmer Scheme Finder (Increment 7).

Offline and deterministic: builds :class:`ClassifiedCandidate` objects directly
and exercises ``app.schemes.ranking``. No provider, no network, no ddgs.
"""

import pytest

from app.schemes.models import ClassifiedCandidate
from app.schemes.ranking import (
    canonical_url_key,
    deduplicate,
    rank_and_limit,
    relevance_score,
    tier_of,
)
from app.schemes.sources import (
    OFFICIAL_GOI,
    OFFICIAL_MANIPUR,
    OTHER,
    hostname_from_url,
)


def C(
    url: str,
    title: str = "T",
    snippet: str = "",
    source_type: str = OTHER,
    verified: bool = False,
    host: str | None = None,
) -> ClassifiedCandidate:
    return ClassifiedCandidate(
        title=title,
        url=url,
        snippet=snippet,
        source_host=host if host is not None else hostname_from_url(url),
        source_type=source_type,
        verified=verified,
    )


def titles(candidates) -> list[str]:
    return [c.title for c in candidates]


# --- canonicalization ------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [None, "", "   ", "ftp://example.com/x", "javascript:alert(1)", 42],
)
def test_canonical_url_key_rejects_unusable_urls(url) -> None:
    assert canonical_url_key(url) is None


def test_canonical_url_key_normalizes_host_case_and_scheme() -> None:
    assert canonical_url_key("HTTPS://Example.COM/Path") == canonical_url_key(
        "https://example.com/Path"
    )


def test_canonical_url_key_drops_fragment() -> None:
    assert canonical_url_key("https://e.com/p#section") == canonical_url_key(
        "https://e.com/p"
    )


def test_canonical_url_key_drops_tracking_params_only() -> None:
    base = canonical_url_key("https://e.com/scheme")
    assert canonical_url_key("https://e.com/scheme?utm_source=x&utm_medium=y") == base
    assert canonical_url_key("https://e.com/scheme?gclid=abc") == base
    assert canonical_url_key("https://e.com/scheme?fbclid=abc#frag") == base


def test_canonical_url_key_preserves_meaningful_params() -> None:
    a = canonical_url_key("https://e.com/scheme?id=1")
    b = canonical_url_key("https://e.com/scheme?id=2")
    assert a != b
    assert "id=1" in a and "id=2" in b


def test_canonical_url_key_sorts_params_so_order_does_not_matter() -> None:
    assert canonical_url_key("https://e.com/s?a=1&b=2") == canonical_url_key(
        "https://e.com/s?b=2&a=1"
    )


def test_canonical_url_key_normalizes_trailing_slash() -> None:
    assert canonical_url_key("https://e.com/path/") == canonical_url_key(
        "https://e.com/path"
    )
    # The root path is preserved.
    assert canonical_url_key("https://e.com/") is not None


# --- deduplication ---------------------------------------------------------- #


def test_dedup_removes_fragment_and_tracking_variants() -> None:
    items = [
        C("https://e.com/scheme", title="Scheme"),
        C("https://e.com/scheme#top", title="Scheme again"),
        C("https://e.com/scheme?utm_source=x", title="Scheme tracked"),
    ]
    assert titles(deduplicate(items)) == ["Scheme"]


def test_dedup_preserves_distinct_meaningful_params() -> None:
    items = [
        C("https://e.com/s?id=1", title="One"),
        C("https://e.com/s?id=2", title="Two"),
    ]
    assert titles(deduplicate(items)) == ["One", "Two"]


def test_dedup_preserves_duplicate_titles_across_different_hosts() -> None:
    items = [
        C("https://a.example/scheme", title="Farmer Scheme"),
        C("https://b.example/scheme", title="Farmer Scheme"),
    ]
    assert titles(deduplicate(items)) == ["Farmer Scheme", "Farmer Scheme"]


def test_dedup_drops_same_title_same_host_near_duplicates() -> None:
    items = [
        C("https://e.com/scheme-a", title="Farmer Scheme"),
        C("https://e.com/scheme-b", title="farmer   scheme!!!"),
    ]
    assert titles(deduplicate(items)) == ["Farmer Scheme"]


def test_dedup_is_order_preserving_and_stable() -> None:
    items = [
        C("https://e.com/1", title="One"),
        C("https://e.com/2", title="Two"),
        C("https://e.com/1", title="One duplicate"),
    ]
    assert titles(deduplicate(items)) == ["One", "Two"]


# --- tiering / official-first ----------------------------------------------- #


def test_tier_order_manipur_then_goi_then_other() -> None:
    assert tier_of(OFFICIAL_MANIPUR) < tier_of(OFFICIAL_GOI) < tier_of(OTHER)
    assert tier_of("something_else") == tier_of(OTHER)


def test_official_manipur_precedes_official_goi() -> None:
    items = [
        C("https://india.gov.in/x", "GoI", source_type=OFFICIAL_GOI, verified=True),
        C(
            "https://agrimanipur.mn.gov.in/x",
            "Manipur",
            source_type=OFFICIAL_MANIPUR,
            verified=True,
        ),
    ]
    assert titles(rank_and_limit(items)) == ["Manipur", "GoI"]


def test_official_goi_precedes_unverified() -> None:
    items = [
        C("https://example.com/x", "Other"),
        C("https://india.gov.in/x", "GoI", source_type=OFFICIAL_GOI, verified=True),
    ]
    assert titles(rank_and_limit(items)) == ["GoI", "Other"]


def test_tier_beats_relevance_score() -> None:
    # A verified official with a zero score still outranks a highly relevant
    # unverified result: classification, not text, decides "official".
    items = [
        C(
            "https://example.com/x",
            "rice crop insurance Manipur rice",
            snippet="rice insurance",
        ),
        C(
            "https://agrimanipur.mn.gov.in/x",
            "Unrelated",
            source_type=OFFICIAL_MANIPUR,
            verified=True,
        ),
    ]
    assert titles(rank_and_limit(items, crop="rice", need="insurance")) == [
        "Unrelated",
        "rice crop insurance Manipur rice",
    ]


def test_unapproved_source_is_never_promoted() -> None:
    items = [
        C(
            "https://fake-gov.example/manipur",
            "Government of Manipur farmer scheme",
            snippet="official department",
            source_type=OTHER,
            verified=False,
        )
    ]
    out = rank_and_limit(items)
    assert out[0].verified is False
    assert out[0].source_type == OTHER


# --- relevance scoring / ordering ------------------------------------------- #


def test_relevance_score_is_deterministic_and_integer() -> None:
    c = C("https://e.com/x", "Rice scheme", snippet="Manipur crop insurance")
    a = relevance_score(c, query="organic seeds", crop="rice", need="insurance")
    b = relevance_score(c, query="organic seeds", crop="rice", need="insurance")
    assert isinstance(a, int)
    assert a == b
    # Manipur (+1) + crop rice (+3) + need insurance (+3); query terms absent.
    assert a == 7


def test_crop_affects_ordering_deterministically() -> None:
    items = [
        C("https://e.com/generic", "Generic farming support"),
        C("https://e.com/rice", "Rice cultivation support", snippet="paddy rice"),
    ]
    out = rank_and_limit(items, crop="rice")
    assert titles(out) == ["Rice cultivation support", "Generic farming support"]


def test_support_need_affects_ordering_deterministically() -> None:
    items = [
        C("https://e.com/a", "Untargeted support"),
        C("https://e.com/b", "Crop insurance for paddy"),
    ]
    out = rank_and_limit(items, need="insurance")
    assert titles(out) == ["Crop insurance for paddy", "Untargeted support"]


def test_query_terms_affect_ordering() -> None:
    items = [
        C("https://e.com/a", "Random page"),
        C("https://e.com/b", "Organic certification help"),
    ]
    out = rank_and_limit(items, query="organic certification")
    assert titles(out) == ["Organic certification help", "Random page"]


def test_manipur_context_affects_ordering() -> None:
    items = [
        C("https://e.com/a", "Statewide programme"),
        C("https://e.com/b", "Manipur programme", snippet="for Manipur farmers"),
    ]
    out = rank_and_limit(items)
    assert titles(out) == ["Manipur programme", "Statewide programme"]


def test_equal_scores_keep_provider_order() -> None:
    items = [
        C("https://e.com/1", "Alpha"),
        C("https://e.com/2", "Beta"),
        C("https://e.com/3", "Gamma"),
    ]
    assert titles(rank_and_limit(items)) == ["Alpha", "Beta", "Gamma"]


# --- limit after dedup ------------------------------------------------------ #


def test_limit_is_applied_after_deduplication() -> None:
    items = [
        C("https://e.com/1", "One"),
        C("https://e.com/1?utm_source=x", "One duplicate"),
        C("https://e.com/2", "Two"),
        C("https://e.com/3", "Three"),
    ]
    out = rank_and_limit(items, limit=2)
    assert titles(out) == ["One", "Two"]


def test_limit_zero_and_none() -> None:
    items = [C("https://e.com/1", "One"), C("https://e.com/2", "Two")]
    assert rank_and_limit(items, limit=0) == []
    assert titles(rank_and_limit(items, limit=None)) == ["One", "Two"]
