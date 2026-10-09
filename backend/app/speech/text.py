"""Deterministic farmer-friendly text builder for a crop prediction (Part 6E).

Turns an existing prediction result into a single, concise sentence suitable for
the 200-character Groq TTS limit. This module:

- is pure and deterministic (no I/O, no network, no Groq calls, no routes);
- uses only fields the prediction API actually returns
  (``display_name``/``crop``, ``predicted_label``, optional ``confidence``);
- never fabricates class labels, probabilities, causes, treatments, dosages or
  recovery claims, and never implies a confirmed diagnosis.

Callers pass the ``/predict`` payload (a mapping) or a ``PredictResponse``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..config import settings
from ..errors import ApiError

# Uncertainty language that must survive every shortening variant.
_UNCERTAINTY = "not a confirmed diagnosis"


def _as_mapping(prediction: Any) -> Mapping[str, Any]:
    if hasattr(prediction, "model_dump") and callable(prediction.model_dump):
        prediction = prediction.model_dump()
    if not isinstance(prediction, Mapping):
        raise ApiError(
            400, "invalid_prediction", "Prediction result must be an object."
        )
    return prediction


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _crop_name(prediction: Mapping[str, Any]) -> str:
    name = _clean_text(prediction.get("display_name")) or _clean_text(
        prediction.get("crop")
    )
    if not name:
        raise ApiError(
            400, "response_missing_crop", "Prediction result has no crop name."
        )
    return name


def _class_label(prediction: Mapping[str, Any]) -> str:
    label = _clean_text(prediction.get("predicted_label"))
    if not label:
        raise ApiError(
            400,
            "response_missing_class",
            "Prediction result has no predicted class label.",
        )
    return label


def _confidence_percent(prediction: Mapping[str, Any]) -> str | None:
    if "confidence" not in prediction or prediction.get("confidence") is None:
        return None
    value = prediction["confidence"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(
            400,
            "response_invalid_confidence",
            "Prediction confidence must be a number between 0 and 1.",
        )
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ApiError(
            400,
            "response_invalid_confidence",
            "Prediction confidence must be a number between 0 and 1.",
        )
    return f"{value * 100:.1f}%"


def build_prediction_speech(prediction: Any) -> str:
    """Build a concise, farmer-friendly sentence from a prediction result.

    Raises ``ApiError`` (400) for missing/malformed fields and when even the
    shortest safe wording cannot fit the configured TTS character limit.
    """
    data = _as_mapping(prediction)
    crop = _crop_name(data)
    label = _class_label(data)
    percent = _confidence_percent(data)

    # Ordered richest -> shortest. The uncertainty clause is present in all.
    variants: list[str] = []
    if percent is not None:
        variants.append(
            f"{crop} leaf: the model predicts {label} "
            f"(model confidence {percent}). This is a prediction, {_UNCERTAINTY}."
        )
        variants.append(
            f"{crop} leaf: model predicts {label} "
            f"(confidence {percent}). This is a prediction, {_UNCERTAINTY}."
        )
    variants.append(
        f"{crop} leaf: model predicts {label}. "
        f"This is a prediction, {_UNCERTAINTY}."
    )
    variants.append(f"{crop}: {label}. This is a prediction, {_UNCERTAINTY}.")
    variants.append(f"{crop} {label}. Prediction only, {_UNCERTAINTY}.")

    limit = settings.groq_tts_max_input_chars
    for text in variants:
        if len(text) <= limit:
            return text

    raise ApiError(
        400,
        "response_text_too_long",
        "The prediction could not be summarized within the text-to-speech limit.",
        {"max_characters": limit},
    )
