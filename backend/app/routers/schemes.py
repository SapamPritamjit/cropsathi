"""Farmer Scheme Finder search endpoint (Increment 4).

``GET /schemes/search`` maps structured, bounded inputs to a single Manipur-
focused search string and returns the provider's classified results. All
validation, query-building and response assembly live in
``app.schemes.service``; this module only wires HTTP to that service.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..schemas import ErrorResponse, SchemeSearchResponse
from ..schemes.service import run_scheme_search

router = APIRouter(tags=["schemes"])


@router.get(
    "/schemes/search",
    response_model=SchemeSearchResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def search_schemes(
    q: str | None = Query(
        None, description="Free-text query (trimmed, max 200 characters)."
    ),
    crop: str | None = Query(
        None, description="One of: maize, rice, potato, tomato."
    ),
    district: str | None = Query(
        None,
        description="Manipur district name (bounded free text; no fixed list).",
    ),
    need: str | None = Query(
        None,
        description=(
            "Support need: seeds, fertilizer, credit, insurance, irrigation, "
            "machinery, market, training, pest_disease, general."
        ),
    ),
    limit: int | None = Query(
        None, description="Maximum number of results (1-20; default 5)."
    ),
    language: str | None = Query(
        None, description="ISO-639-1 language code (default 'en')."
    ),
) -> SchemeSearchResponse:
    """Search official farmer schemes. At least one of ``q``, ``crop`` or
    ``need`` is required; results are informational, not proof of eligibility."""
    payload = await run_scheme_search(
        q=q,
        crop=crop,
        district=district,
        need=need,
        limit=limit,
        language=language,
    )
    return SchemeSearchResponse(**payload)
