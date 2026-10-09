"""Official-source classification for scheme search results.

Pure, offline, and independent of any live search. A host is treated as
official **only** when it is explicitly allow-listed and matches an allowed host
either exactly or as a dot-boundary subdomain. Look-alike hosts such as
``manipur.gov.in.example.com`` are never accepted, and a source is never marked
verified based on its title or snippet text.

If the allow-list is empty (the default) every result is ``other`` and
``verified=False`` — the safe default. No government domain is assumed here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple
from urllib.parse import urlsplit

from .models import ClassifiedCandidate, SearchCandidate

OFFICIAL_MANIPUR = "official_manipur"
OFFICIAL_GOI = "official_goi"
OTHER = "other"

_MANIPUR_PREFIX = "manipur:"
_GOI_PREFIX = "goi:"


class SourceClassification(NamedTuple):
    """Result of classifying a single URL against the official allow-list."""

    host: str | None
    source_type: str
    verified: bool


def hostname_from_url(url: object) -> str | None:
    """Return the lower-cased hostname of an http/https URL, else ``None``.

    Non-http(s) schemes (e.g. ``javascript:``, ``ftp:``, ``file:``), empty
    strings and malformed input all return ``None`` so they can never become a
    verified official result.
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
    if parts.scheme.lower() not in {"http", "https"}:
        return None
    host = (parts.hostname or "").strip().lower().rstrip(".")
    return host or None


def normalize_host(value: object) -> str | None:
    """Normalize a configured allow-list entry into a bare hostname.

    Accepts either a bare hostname (``manipur.gov.in``) or a full http/https URL
    (``https://manipur.gov.in/path``). Returns ``None`` for anything unusable.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if "://" in text:
        return hostname_from_url(text)
    host = text.split("/", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if ":" in host:
        host = host.split(":", 1)[0]
    host = host.strip().lower().rstrip(".")
    return host or None


def _matches(host: str, allowed: str) -> bool:
    """Exact host or dot-boundary subdomain match (never a loose substring)."""
    return host == allowed or host.endswith("." + allowed)


class OfficialSourceRegistry:
    """Immutable allow-list mapping official hosts to a source type."""

    def __init__(
        self,
        manipur: Iterable[str] = (),
        goi: Iterable[str] = (),
    ) -> None:
        self._manipur = self._clean(manipur)
        self._goi = self._clean(goi)

    @staticmethod
    def _clean(values: Iterable[str]) -> frozenset[str]:
        hosts: set[str] = set()
        for value in values:
            host = normalize_host(value)
            if host:
                hosts.add(host)
        return frozenset(hosts)

    @classmethod
    def from_specs(cls, specs: Iterable[str]) -> "OfficialSourceRegistry":
        """Build a registry from typed ``manipur:<host>`` / ``goi:<host>`` specs.

        An untyped entry defaults to ``goi`` (Government of India). Manipur hosts
        must be explicitly typed, so they are never inferred.
        """
        manipur: list[str] = []
        goi: list[str] = []
        for spec in specs:
            token = (spec or "").strip().lower()
            if not token:
                continue
            if token.startswith(_MANIPUR_PREFIX):
                manipur.append(token[len(_MANIPUR_PREFIX):])
            elif token.startswith(_GOI_PREFIX):
                goi.append(token[len(_GOI_PREFIX):])
            else:
                goi.append(token)
        return cls(manipur=manipur, goi=goi)

    @property
    def is_empty(self) -> bool:
        return not self._manipur and not self._goi

    def classify(self, url: object) -> SourceClassification:
        """Classify ``url`` by host only. Title/snippet are never consulted."""
        host = hostname_from_url(url)
        if host is None:
            return SourceClassification(None, OTHER, False)
        if any(_matches(host, allowed) for allowed in self._manipur):
            return SourceClassification(host, OFFICIAL_MANIPUR, True)
        if any(_matches(host, allowed) for allowed in self._goi):
            return SourceClassification(host, OFFICIAL_GOI, True)
        return SourceClassification(host, OTHER, False)


def classify_candidate(
    candidate: SearchCandidate, registry: OfficialSourceRegistry
) -> ClassifiedCandidate:
    """Attach source-classification fields to a normalized candidate."""
    result = registry.classify(candidate.url)
    return ClassifiedCandidate(
        title=candidate.title,
        url=candidate.url,
        snippet=candidate.snippet,
        source_host=result.host,
        source_type=result.source_type,
        verified=result.verified,
    )
