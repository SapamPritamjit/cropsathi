"""Normalized data types for the Farmer Scheme Finder search provider.

These types are deliberately independent of any specific search backend, so the
DDGS implementation can be replaced without touching the rest of the app.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchCandidate:
    """A normalized search result returned by a :class:`SearchProvider`.

    ``url`` is an http/https URL; ``snippet`` may be empty.
    """

    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class ClassifiedCandidate:
    """A :class:`SearchCandidate` annotated with source-classification fields.

    ``source_type`` is one of ``official_manipur`` / ``official_goi`` / ``other``
    and ``verified`` is true only for explicitly allow-listed official hosts.
    """

    title: str
    url: str
    snippet: str
    source_host: str | None
    source_type: str
    verified: bool
