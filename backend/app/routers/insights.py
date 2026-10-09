"""Server-side AI crop question answering endpoint (Increment 9P).

``POST /insights/ask`` validates a crop question (plus optional latest-
prediction context), builds a bounded prompt, and returns the configured chat
model's answer. It is disabled unless ``GROQ_INSIGHTS_ENABLED`` is true and a
Groq API key is configured; every failure surfaces through the shared error
envelope. No answer is fabricated.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from ..config import settings
from ..errors import ApiError
from ..insights.chat import generate_answer
from ..insights.prompt import (
    build_messages,
    normalize_context,
    normalize_crop,
    normalize_question,
    normalize_topic,
)
from ..schemas import ErrorResponse, InsightAskRequest, InsightAskResponse

router = APIRouter(tags=["insights"])

DISCLAIMER = (
    "AI-generated crop guidance from a language model. It is informational and "
    "crop-related only, is not a confirmed diagnosis, and does not replace advice "
    "from your local agricultural extension officer."
)


@router.post(
    "/insights/ask",
    response_model=InsightAskResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ask(payload: InsightAskRequest) -> InsightAskResponse:
    """Answer a crop question with the configured chat model."""
    if not settings.groq_insights_enabled:
        raise ApiError(
            503,
            "insights_disabled",
            "Crop question answering is disabled on the server.",
        )

    crop = normalize_crop(payload.crop)
    question = normalize_question(payload.question)
    topic = normalize_topic(payload.topic)
    context = normalize_context(payload.context)

    messages = build_messages(crop, question, topic, context)
    result = await generate_answer(messages)

    return InsightAskResponse(
        crop=crop,
        topic=topic,
        answer=result.text,
        model=result.model,
        generated_at=datetime.now(timezone.utc).isoformat(),
        disclaimer=DISCLAIMER,
    )
