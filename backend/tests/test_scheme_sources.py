"""Official-source classification tests (Increment 2).

Pure/offline: classification is a function of the URL host and the explicit
allow-list only. No search, no network.
"""

from app.schemes.models import SearchCandidate
from app.schemes.sources import (
    OFFICIAL_GOI,
    OFFICIAL_MANIPUR,
    OTHER,
    OfficialSourceRegistry,
    classify_candidate,
    hostname_from_url,
    normalize_host,
)


def test_manipur_only_when_explicitly_allowlisted() -> None:
    registry = OfficialSourceRegistry(manipur=["manipur.gov.in"])
    result = registry.classify("https://manipur.gov.in/schemes")
    assert result.source_type == OFFICIAL_MANIPUR
    assert result.verified is True

    empty = OfficialSourceRegistry()
    unclassified = empty.classify("https://manipur.gov.in/schemes")
    assert unclassified.source_type == OTHER
    assert unclassified.verified is False


def test_allowlisted_goi_classified_correctly() -> None:
    registry = OfficialSourceRegistry(goi=["india.gov.in", "pmkisan.gov.in"])
    assert registry.classify("https://india.gov.in/").source_type == OFFICIAL_GOI
    assert registry.classify("https://pmkisan.gov.in/").source_type == OFFICIAL_GOI
    assert registry.classify("https://pmkisan.gov.in/").verified is True


def test_exact_host_and_dot_boundary_subdomain_matching() -> None:
    registry = OfficialSourceRegistry(goi=["india.gov.in"])
    assert registry.classify("https://india.gov.in").verified is True
    assert registry.classify("http://india.gov.in").verified is True
    assert registry.classify("https://www.india.gov.in").verified is True
    assert registry.classify("https://a.b.india.gov.in").verified is True


def test_lookalike_hosts_are_rejected() -> None:
    registry = OfficialSourceRegistry(manipur=["manipur.gov.in"])
    for bad in (
        "https://manipur.gov.in.example.com",
        "https://evilmanipur.gov.in",
        "https://notmanipur.gov.in",
        "https://manipur.gov.in.evil.com",
        "https://xmanipur.gov.in",
    ):
        result = registry.classify(bad)
        assert result.verified is False, bad
        assert result.source_type == OTHER, bad


def test_goi_lookalikes_are_rejected() -> None:
    registry = OfficialSourceRegistry(goi=["india.gov.in"])
    assert registry.classify("https://india.gov.in.evil.com").verified is False
    assert registry.classify("https://notindia.gov.in").verified is False
    assert registry.classify("https://fakeindia.gov.in").verified is False


def test_only_http_https_urls_can_be_verified() -> None:
    registry = OfficialSourceRegistry(goi=["india.gov.in"])
    for bad in (
        "javascript:alert(1)",
        "ftp://india.gov.in",
        "file://india.gov.in",
        "data:text/html,india.gov.in",
        "not a url",
        "http://",
        "",
        "https://",
    ):
        result = registry.classify(bad)
        assert result.verified is False, bad
        assert result.source_type == OTHER, bad


def test_http_and_https_both_handled() -> None:
    registry = OfficialSourceRegistry(goi=["india.gov.in"])
    assert registry.classify("http://india.gov.in").source_type == OFFICIAL_GOI
    assert registry.classify("https://india.gov.in").source_type == OFFICIAL_GOI


def test_title_government_wording_does_not_make_official() -> None:
    registry = OfficialSourceRegistry()
    candidate = SearchCandidate(
        title="Government of Manipur Scheme Portal",
        url="https://example.com/gov-scheme",
        snippet="Official government assistance for farmers.",
    )
    classified = classify_candidate(candidate, registry)
    assert classified.source_type == OTHER
    assert classified.verified is False
    assert classified.source_host == "example.com"


def test_from_specs_manipur_explicit_and_goi_default() -> None:
    registry = OfficialSourceRegistry.from_specs(
        ["manipur:manipur.gov.in", "india.gov.in", "goi:pmkisan.gov.in"]
    )
    assert registry.classify("https://manipur.gov.in").source_type == OFFICIAL_MANIPUR
    assert registry.classify("https://india.gov.in").source_type == OFFICIAL_GOI
    assert registry.classify("https://pmkisan.gov.in").source_type == OFFICIAL_GOI


def test_registry_ignores_invalid_allowlist_entries() -> None:
    registry = OfficialSourceRegistry(
        manipur=["", "  ", "https://manipur.gov.in/path"],
        goi=["not a host/with space"],
    )
    assert registry.classify("https://manipur.gov.in").source_type == OFFICIAL_MANIPUR
    assert registry.is_empty is False


def test_normalize_host_handles_urls_and_bare_hosts() -> None:
    assert normalize_host("https://Manipur.Gov.In/Path") == "manipur.gov.in"
    assert normalize_host("manipur.gov.in") == "manipur.gov.in"
    assert normalize_host("ftp://manipur.gov.in") is None
    assert normalize_host(None) is None
    assert normalize_host(123) is None


def test_hostname_from_url_rejects_non_http() -> None:
    assert hostname_from_url("https://Example.COM/a") == "example.com"
    assert hostname_from_url("javascript:alert(1)") is None
    assert hostname_from_url("ftp://example.com") is None
    assert hostname_from_url(None) is None


# --- Increment 5: documented, evidence-checked verified origins ------------- #

VERIFIED_MANIPUR = ("agrimanipur.mn.gov.in", "manipur.gov.in")
VERIFIED_GOI = (
    "agriwelfare.gov.in",
    "pmkisan.gov.in",
    "pmfby.gov.in",
    "india.gov.in",
)


def test_verified_manipur_domains_classify_as_official() -> None:
    registry = OfficialSourceRegistry.from_specs(
        [f"manipur:{host}" for host in VERIFIED_MANIPUR]
    )
    for host in VERIFIED_MANIPUR:
        result = registry.classify(f"https://{host}/scheme")
        assert result.source_type == OFFICIAL_MANIPUR, host
        assert result.verified is True, host
    assert (
        registry.classify("https://agrimanipur.mn.gov.in/contact-us/").host
        == "agrimanipur.mn.gov.in"
    )


def test_verified_goi_domains_classify_as_official() -> None:
    registry = OfficialSourceRegistry.from_specs(
        [f"goi:{host}" for host in VERIFIED_GOI]
    )
    for host in VERIFIED_GOI:
        result = registry.classify(f"https://{host}/x")
        assert result.source_type == OFFICIAL_GOI, host
        assert result.verified is True, host


def test_verified_origins_are_not_active_by_default() -> None:
    # The safe default: an empty allow-list means nothing is treated as official.
    registry = OfficialSourceRegistry.from_specs([])
    assert registry.is_empty is True
    for host in VERIFIED_MANIPUR + VERIFIED_GOI:
        assert registry.classify(f"https://{host}/").verified is False, host


def test_agrimanipur_lookalikes_are_rejected() -> None:
    registry = OfficialSourceRegistry(manipur=["agrimanipur.mn.gov.in"])
    for bad in (
        "https://agrimanipur.mn.gov.in.evil.com",
        "https://agrimanipur-mn.gov.in",
        "https://agrimanipur.mn.gov.in.evil.org",
        "https://fakeagrimanipur.mn.gov.in",
        "https://agrimanipur.mn.gov.co",
        "https://agrimanipur.mn.gov.in.sg",
    ):
        assert registry.classify(bad).verified is False, bad


def test_no_broad_suffix_matching_between_verified_hosts() -> None:
    # Approving manipur.gov.in must NOT approve unrelated gov.in hosts, and
    # approving one GoI department host must not approve its siblings.
    manipur = OfficialSourceRegistry(manipur=["manipur.gov.in"])
    assert manipur.classify("https://dept.manipur.gov.in/").verified is True
    assert manipur.classify("https://randomdept.gov.in/").verified is False

    goi = OfficialSourceRegistry(goi=["agriwelfare.gov.in"])
    assert goi.classify("https://agriwelfare.gov.in/").verified is True
    assert goi.classify("https://pmkisan.gov.in/").verified is False
