"""Pure prompt/context builder and validators for crop Q&A (Increment 9P).

This module is deterministic and does **no** I/O, network, or Groq calls. It:

- normalizes and validates the crop, question, topic and optional latest-
  prediction context, raising ``ApiError`` with stable codes;
- builds the bounded system/user messages sent to the chat model.

Responsible-AI rules encoded in the system prompt: the crop is named, model
output is framed as an *uncertain model prediction* (never a confirmed
diagnosis), scheme eligibility is not invented, no specific chemical
pesticides/dosages/cures are prescribed, and consulting a local agricultural
extension officer is encouraged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from ..config import CROP_DISPLAY_NAMES, SUPPORTED_CROPS, settings
from ..errors import ApiError

# Topics accepted by ``POST /insights/ask``. The first six mirror the front-end
# topic selector; ``general`` is the neutral default when no topic is supplied.
SUPPORTED_TOPICS: tuple[str, ...] = (
    "general",
    "disease_symptoms",
    "disease_prevention",
    "crop_care",
    "irrigation",
    "nutrients",
    "prediction_interpretation",
)

DEFAULT_TOPIC = "general"

TOPIC_LABELS: dict[str, str] = {
    "general": "general crop questions",
    "disease_symptoms": "disease symptoms",
    "disease_prevention": "disease prevention",
    "crop_care": "crop care",
    "irrigation": "irrigation",
    "nutrients": "nutrients",
    "prediction_interpretation": "interpreting a prediction",
}

# Strip ASCII control characters except tab/newline/CR (answers/questions may
# legitimately contain line breaks). Mirrors the speech modules' sanitization.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_MAX_LABEL_CHARS = 200
_MAX_FAMILY_CHARS = 40

_SYSTEM_PROMPT = (
    "You are an agricultural advisor helping smallholder farmers in Manipur, "
    "India. You are answering questions about {crop} ({crop_key}). "
    "Give practical, crop-related guidance only, in a short paragraph or a few "
    "concise bullet points. "
    "An image-model result is only an uncertain prediction, never a confirmed "
    "diagnosis; never state or imply that a disease is confirmed, and never "
    "guarantee a cause or outcome. "
    "Do not invent government scheme eligibility, benefits, or deadlines. "
    "Do not prescribe specific chemical pesticides, exact dosages, or guaranteed "
    "cures; describe non-chemical and general good-practice options instead. "
    "Encourage consulting a local agricultural extension officer for "
    "confirmation and treatment. "
    "If the question is not about crops or agriculture, briefly say you can only "
    "help with crop-related questions."
)


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", _CONTROL_RE.sub(" ", value)).strip()


def normalize_crop(raw: Any) -> str:
    """Return a supported crop key (lower-cased), else ``400``."""
    crop = raw.strip().lower() if isinstance(raw, str) else ""
    if crop not in SUPPORTED_CROPS:
        raise ApiError(
            400,
            "unsupported_crop",
            "Unsupported crop '" + (crop or "") + "'. Supported crops: "
            + ", ".join(SUPPORTED_CROPS)
            + ".",
            {"supported_crops": list(SUPPORTED_CROPS)},
        )
    return crop


def normalize_question(raw: Any) -> str:
    """Return the trimmed question, enforcing the configured length bound."""
    question = _clean(raw)
    if not question:
        raise ApiError(
            400, "insight_empty_question", "A non-empty question is required."
        )
    limit = settings.groq_chat_max_input_chars
    if len(question) > limit:
        raise ApiError(
            400,
            "insight_question_too_long",
            "The question is too long.",
            {"max_characters": limit, "characters": len(question)},
        )
    return question


def normalize_topic(raw: Any) -> str:
    """Return an allow-listed topic, defaulting to ``general`` when absent."""
    if raw is None:
        return DEFAULT_TOPIC
    if not isinstance(raw, str):
        raise ApiError(
            400,
            "insight_invalid_topic",
            "The topic must be a string.",
            {"supported_topics": list(SUPPORTED_TOPICS)},
        )
    topic = raw.strip().lower()
    if not topic:
        return DEFAULT_TOPIC
    if topic not in SUPPORTED_TOPICS:
        raise ApiError(
            400,
            "insight_invalid_topic",
            "Unsupported topic '" + topic + "'.",
            {"supported_topics": list(SUPPORTED_TOPICS)},
        )
    return topic


def normalize_context(raw: Any) -> dict[str, Any] | None:
    """Validate optional latest-prediction context.

    Accepts only ``predicted_label`` (bounded text), ``confidence`` (finite
    number in ``[0, 1]``) and ``family`` (bounded text); unknown fields are
    ignored. Returns ``None`` when nothing usable is present.
    """
    if raw is None:
        return None
    if hasattr(raw, "model_dump") and callable(raw.model_dump):
        raw = raw.model_dump()
    if not isinstance(raw, Mapping):
        raise ApiError(
            400, "insight_invalid_context", "The prediction context must be an object."
        )

    context: dict[str, Any] = {}
    label = _clean(raw.get("predicted_label"))[:_MAX_LABEL_CHARS]
    if label:
        context["predicted_label"] = label

    family = _clean(raw.get("family"))[:_MAX_FAMILY_CHARS]
    if family:
        context["family"] = family

    if "confidence" in raw and raw.get("confidence") is not None:
        value = raw.get("confidence")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiError(
                400,
                "insight_invalid_context",
                "Prediction confidence must be a number between 0 and 1.",
            )
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ApiError(
                400,
                "insight_invalid_context",
                "Prediction confidence must be a number between 0 and 1.",
            )
        context["confidence"] = value

    return context or None


def build_system_prompt(crop_key: str, context: Mapping[str, Any] | None = None) -> str:
    """Build the system instruction, naming the crop and framing context safely."""
    crop = CROP_DISPLAY_NAMES.get(crop_key, crop_key)
    prompt = _SYSTEM_PROMPT.format(crop=crop, crop_key=crop_key)

    if context:
        fragments: list[str] = []
        label = _clean(context.get("predicted_label"))
        if label:
            fragments.append("the class '" + label + "'")
        confidence = context.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            fragments.append(f"a model confidence of {float(confidence) * 100:.1f}%")
        if fragments:
            prompt += (
                " The farmer's most recent image-model result reported "
                + " with ".join(fragments)
                + ". Treat this only as an uncertain model prediction, not a "
                "confirmed diagnosis."
            )
    return prompt


def build_user_message(question: str, topic: str) -> str:
    """Build the user message from the question and its topic label."""
    label = TOPIC_LABELS.get(topic, topic)
    return "Topic: " + label + ".\nQuestion: " + question


def build_messages(
    crop_key: str,
    question: str,
    topic: str,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the bounded ``messages`` array for the chat-completions request."""
    return [
        {"role": "system", "content": build_system_prompt(crop_key, context)},
        {"role": "user", "content": build_user_message(question, topic)},
    ]
