"""Curated catalogue of official agricultural information resources (Increment 8B).

A small, offline, hand-curated set of official information resources relevant to
farmers in Manipur. It exists so a handful of high-confidence official pages can
appear in ``GET /schemes/search`` results even when live search is thin — without
ever asserting more than the recorded evidence supports.

Design rules enforced here (see the Increment 8A audit):

- The catalogue is **not** a second official-domain allow-list. Runtime
  verification/source-type always comes from the existing
  :class:`~app.schemes.sources.OfficialSourceRegistry` via
  :func:`catalogue_candidates`; an entry is only marked verified when the *same*
  registry used for live results accepts its host.
- ``CatalogueEntry.verified`` records that the **domain identity** was reviewed;
  it is not a claim that a scheme is active, that an application is open, or
  that any statement on the page is correct.
- ``scheme_status`` defaults to ``unknown``. Seeded resources are official
  information pages, not confirmed active schemes.
- Crop/need/district applicability is left empty unless specifically evidenced;
  all seeded entries are general information resources.
- No benefit amounts, deadlines, required documents or eligibility rules are
  invented, and reachability is never claimed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from ..config import SUPPORTED_CROPS
from .models import ClassifiedCandidate, SearchCandidate
from .ranking import canonical_url_key
from .sources import (
    OFFICIAL_GOI,
    OFFICIAL_MANIPUR,
    OTHER,
    OfficialSourceRegistry,
    classify_candidate,
    hostname_from_url,
)
from .service import SUPPORTED_NEEDS

# --- scheme status vocabulary ----------------------------------------------- #

SCHEME_STATUS_INFORMATION_PAGE = "information_page"
SCHEME_STATUS_ACTIVE = "active"
SCHEME_STATUS_UNKNOWN = "unknown"
SCHEME_STATUSES: tuple[str, ...] = (
    SCHEME_STATUS_INFORMATION_PAGE,
    SCHEME_STATUS_ACTIVE,
    SCHEME_STATUS_UNKNOWN,
)
DEFAULT_SCHEME_STATUS = SCHEME_STATUS_UNKNOWN

_SOURCE_TYPES: tuple[str, ...] = (OFFICIAL_MANIPUR, OFFICIAL_GOI, OTHER)


class CatalogueError(Exception):
    """Raised when the curated catalogue violates an integrity rule."""


@dataclass(frozen=True)
class CatalogueEntry:
    """One curated official information resource (immutable).

    ``verified`` is the *audit* domain-verification flag, never the runtime one.
    ``crops``/``needs``/``districts`` are tuples (immutable) and empty unless
    specifically evidenced.
    """

    id: str
    title: str
    url: str
    department: str
    source_type: str
    verified: bool
    scheme_status: str = DEFAULT_SCHEME_STATUS
    crops: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    districts: tuple[str, ...] = ()
    description: str = ""
    evidence_note: str = ""
    last_verified: str | None = None


# --- validation ------------------------------------------------------------- #


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogueError(message)


def _validate_structure(
    entries: Iterable[CatalogueEntry],
    *,
    supported_crops: Sequence[str] = SUPPORTED_CROPS,
    supported_needs: Sequence[str] = SUPPORTED_NEEDS,
) -> None:
    """Deterministic, registry-independent integrity checks.

    Raises :class:`CatalogueError` on the first violation — invalid entries are
    never silently promoted or dropped.
    """
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for entry in entries:
        _require(isinstance(entry, CatalogueEntry), "catalogue entry must be a CatalogueEntry")
        _require(
            isinstance(entry.id, str) and entry.id.strip() != "",
            "catalogue entry needs a non-empty stable id",
        )
        _require(entry.id not in seen_ids, f"duplicate catalogue id: {entry.id!r}")
        seen_ids.add(entry.id)

        _require(
            isinstance(entry.title, str) and entry.title.strip() != "",
            f"{entry.id}: title is required",
        )
        _require(
            isinstance(entry.department, str) and entry.department.strip() != "",
            f"{entry.id}: department is required",
        )

        host = hostname_from_url(entry.url)
        _require(host is not None, f"{entry.id}: URL must be an http(s) URL with a host")
        try:
            scheme = urlsplit(entry.url).scheme.lower()
        except ValueError:
            scheme = ""
        _require(scheme == "https", f"{entry.id}: URL must use HTTPS")

        _require(
            entry.source_type in _SOURCE_TYPES,
            f"{entry.id}: invalid source_type {entry.source_type!r}",
        )
        _require(
            isinstance(entry.verified, bool),
            f"{entry.id}: verified must be a bool",
        )
        _require(
            entry.scheme_status in SCHEME_STATUSES,
            f"{entry.id}: invalid scheme_status {entry.scheme_status!r}",
        )
        _require(
            all(crop in supported_crops for crop in entry.crops),
            f"{entry.id}: unknown crop in {entry.crops!r}",
        )
        _require(
            all(need in supported_needs for need in entry.needs),
            f"{entry.id}: unknown support need in {entry.needs!r}",
        )
        _require(
            all(isinstance(d, str) and d.strip() for d in entry.districts),
            f"{entry.id}: districts must be non-empty strings",
        )

        key = canonical_url_key(entry.url)
        _require(key is not None, f"{entry.id}: URL could not be canonicalized")
        _require(
            key not in seen_urls,
            f"{entry.id}: duplicate canonical URL {key!r}",
        )
        seen_urls.add(key)


def validate_catalogue(
    entries: Iterable[CatalogueEntry],
    registry: OfficialSourceRegistry | None = None,
) -> tuple[CatalogueEntry, ...]:
    """Validate a catalogue, returning it unchanged, or raise :class:`CatalogueError`.

    Structural rules always run. When ``registry`` is supplied, every entry that
    declares ``verified=True`` must have its host accepted by that **same**
    official-source registry used for live results — otherwise the entry is
    rejected rather than quietly downgraded.
    """
    items = tuple(entries)
    _validate_structure(items)
    if registry is not None:
        for entry in items:
            if entry.verified and not registry.classify(entry.url).verified:
                raise CatalogueError(
                    f"{entry.id}: declares verified=True but host "
                    f"{hostname_from_url(entry.url)!r} is not accepted by the "
                    "official-source registry"
                )
    return items


# --- seed data -------------------------------------------------------------- #

# Domain identity was first reviewed in the Increment 8A audit and re-checked
# against public sources in the Increment 8C review (both on 2026-10-09). 8C also
# attempted a live fetch of each exact destination URL; results are recorded
# per entry below. Verification covers the DOMAIN only — never scheme activity,
# eligibility, benefits, deadlines or applicability to Manipur farmers, and a
# failed fetch is never treated as proof that a resource is unavailable.
_REVIEW_8C_DATE = "2026-10-09"

SEED_CATALOGUE: tuple[CatalogueEntry, ...] = (
    CatalogueEntry(
        id="manipur-agri-dept",
        title="Department of Agriculture, Government of Manipur",
        url="https://agrimanipur.mn.gov.in/",
        department="Department of Agriculture, Government of Manipur",
        source_type=OFFICIAL_MANIPUR,
        verified=True,
        scheme_status=SCHEME_STATUS_UNKNOWN,
        description=(
            "Official website of the Manipur Department of Agriculture; a state "
            "agriculture information hub."
        ),
        evidence_note=(
            "8C review 2026-10-09: domain identity corroborated by search-indexed "
            "pages that self-identify as the 'Official Website of Department of "
            "Agriculture | Government of Manipur' with footer 'Department of "
            "Agriculture Govt. of Manipur' (e.g. /about-us/, /agri-director/, "
            "WCAG 2.0, updated by NIELIT Imphal). A direct request to "
            "https://agrimanipur.mn.gov.in/ did NOT complete from the review "
            "environment (transport error), so reachability is unconfirmed. "
            "Scheme activity, eligibility, benefits, documents and deadlines were "
            "NOT verified."
        ),
        last_verified=_REVIEW_8C_DATE,
    ),
    CatalogueEntry(
        id="manipur-state-portal",
        title="Government of Manipur — State Portal",
        url="https://manipur.gov.in/",
        department="Government of Manipur",
        source_type=OFFICIAL_MANIPUR,
        verified=True,
        scheme_status=SCHEME_STATUS_UNKNOWN,
        description=(
            "Official Government of Manipur state portal (general state "
            "information, including departmental pages)."
        ),
        evidence_note=(
            "8C review 2026-10-09: fetched live at https://manipur.gov.in/; "
            "self-identifies as the Government of Manipur and carries footer "
            "'© Government of Manipur. All rights reserved.', with current "
            "(Oct 2026) notifications and Budget 2026-27. It is a general state "
            "information portal, not a specific scheme page; scheme eligibility, "
            "benefits and deadlines were NOT verified."
        ),
        last_verified=_REVIEW_8C_DATE,
    ),
    CatalogueEntry(
        id="dafw",
        title="Department of Agriculture & Farmers Welfare (Government of India)",
        url="https://agriwelfare.gov.in/",
        department=(
            "Department of Agriculture & Farmers Welfare, Ministry of "
            "Agriculture & Farmers Welfare, Government of India"
        ),
        source_type=OFFICIAL_GOI,
        verified=True,
        scheme_status=SCHEME_STATUS_UNKNOWN,
        description=(
            "Official website of the Government of India Department of "
            "Agriculture & Farmers Welfare."
        ),
        evidence_note=(
            "8C review 2026-10-09: fetched live at https://agriwelfare.gov.in/; "
            "footer states 'Website Content Owned & Managed by Department of "
            "Agriculture & Farmers Welfare | MoA & FW | Government of India', "
            "page last updated 08 Oct 2026. Official departmental portal that "
            "links to the official PM-KISAN (pmkisan.gov.in) and PMFBY "
            "(pmfby.gov.in) sites. No scheme-specific eligibility or benefits "
            "asserted for this entry."
        ),
        last_verified=_REVIEW_8C_DATE,
    ),
    CatalogueEntry(
        id="pm-kisan",
        title="PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
        url="https://pmkisan.gov.in/",
        department=(
            "Department of Agriculture & Farmers Welfare, Government of India"
        ),
        source_type=OFFICIAL_GOI,
        verified=True,
        scheme_status=SCHEME_STATUS_UNKNOWN,
        description=(
            "Official Government of India PM-KISAN scheme information site. "
            "Current status and applicability to Manipur farmers were not verified."
        ),
        evidence_note=(
            "8C review 2026-10-09: identified as the official PM-KISAN portal "
            "(Department of Agriculture and Farmers Welfare, Government of India) "
            "via india.gov.in and via a live link from agriwelfare.gov.in. A "
            "direct request to https://pmkisan.gov.in/ did NOT complete from the "
            "review environment (transport error), so reachability is unconfirmed. "
            "Indexed official-portal content references the 23rd installment "
            "(20 Jun 2026) and mandatory eKYC (central-scheme activity), but "
            "Manipur-specific eligibility and applicability were NOT verified."
        ),
        last_verified=_REVIEW_8C_DATE,
    ),
    CatalogueEntry(
        id="pmfby",
        title="Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        url="https://pmfby.gov.in/",
        department=(
            "Department of Agriculture & Farmers Welfare, Government of India"
        ),
        source_type=OFFICIAL_GOI,
        verified=True,
        scheme_status=SCHEME_STATUS_UNKNOWN,
        description=(
            "Official Government of India PMFBY crop-insurance scheme "
            "information site. Current status and applicability were not verified."
        ),
        evidence_note=(
            "8C review 2026-10-09: fetched https://pmfby.gov.in/ (title 'Pradhan "
            "Mantri Fasal Bima Yojana - Crop Insurance | PMFBY'); "
            "agriwelfare.gov.in's Crop Insurance Division page (live) describes "
            "PMFBY as a flagship scheme of the Ministry of Agriculture & Farmers "
            "Welfare. The portal is the official crop-insurance site, but it is "
            "largely JavaScript so scheme details were not extracted; "
            "Manipur-specific eligibility and applicability were NOT verified."
        ),
        last_verified=_REVIEW_8C_DATE,
    ),
)

# Fail fast on any malformed seed data at import time (structure only; the
# registry cross-check is an explicit, registry-supplied validation).
_validate_structure(SEED_CATALOGUE)


def catalogue_candidates(
    registry: OfficialSourceRegistry,
    entries: Iterable[CatalogueEntry] | None = None,
) -> list[ClassifiedCandidate]:
    """Convert catalogue entries into classified candidates for merging.

    Source type and ``verified`` come **only** from ``registry`` (the same
    registry used for live results), so a curated record is never marked official
    unless its host is accepted there. The entry's own ``verified``/``source_type``
    are audit metadata, not runtime authority.
    """
    source = SEED_CATALOGUE if entries is None else tuple(entries)
    candidates: list[ClassifiedCandidate] = []
    for entry in source:
        candidate = SearchCandidate(
            title=entry.title, url=entry.url, snippet=entry.description
        )
        candidates.append(classify_candidate(candidate, registry))
    return candidates
