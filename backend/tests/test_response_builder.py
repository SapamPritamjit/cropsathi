"""Deterministic tests for the farmer-friendly response builder (Part 6E).

Pure functions only — no network, no routes, no Groq.
"""

import pytest

from app.config import settings
from app.errors import ApiError
from app.schemas import PredictionModelMeta, PredictResponse
from app.speech.text import build_prediction_speech

PREDICTION = {
    "crop": "maize",
    "display_name": "Maize",
    "predicted_index": 1,
    "predicted_label": "Northern Leaf Blight",
    "confidence": 0.929,
    "model": {
        "crop": "maize",
        "family": "baseline",
        "implementation": "timm:convnext_small",
        "num_classes": 11,
        "input_size": [3, 224, 224],
        "weights": "model.safetensors",
        "label_source": "config.json",
    },
    "disclaimer": "Confidence is the model's softmax probability.",
}


def test_valid_prediction_is_grounded_in_crop_and_class() -> None:
    text = build_prediction_speech(PREDICTION)
    assert "Maize" in text
    assert "Northern Leaf Blight" in text
    assert text.endswith("diagnosis.")


def test_confidence_included_and_described_correctly() -> None:
    text = build_prediction_speech(PREDICTION)
    assert "model confidence" in text
    assert "92.9%" in text
    assert "not a confirmed diagnosis" in text


def test_confidence_omitted_when_absent() -> None:
    prediction = {k: v for k, v in PREDICTION.items() if k != "confidence"}
    text = build_prediction_speech(prediction)
    assert "%" not in text
    assert "confidence" not in text
    assert "not a confirmed diagnosis" in text


def test_confidence_none_is_treated_as_absent() -> None:
    text = build_prediction_speech({**PREDICTION, "confidence": None})
    assert "confidence" not in text


def test_output_within_configured_limit() -> None:
    text = build_prediction_speech(PREDICTION)
    assert len(text) <= settings.groq_tts_max_input_chars
    assert len(text) <= 200


def test_output_is_deterministic() -> None:
    assert build_prediction_speech(PREDICTION) == build_prediction_speech(PREDICTION)


def test_accepts_predict_response_model() -> None:
    model = PredictResponse(
        crop="rice",
        display_name="Rice",
        model=PredictionModelMeta(
            crop="rice",
            family="baseline",
            implementation="timm:convnext_small",
            num_classes=14,
            input_size=[3, 224, 224],
            weights="model.safetensors",
            label_source="config.json",
        ),
        predicted_index=3,
        predicted_label="Neck Blast",
        confidence=0.94,
        probabilities=[],
        disclaimer="softmax probability, not a guarantee.",
    )
    text = build_prediction_speech(model)
    assert "Rice" in text
    assert "Neck Blast" in text
    assert "94.0%" in text
    assert len(text) <= 200


def test_long_labels_are_shortened_deterministically_without_midword_cut() -> None:
    prediction = {
        **PREDICTION,
        "display_name": "Maize",
        "predicted_label": "Northern Corn Leaf Blight With A Very Long Descriptive Name",
    }
    text = build_prediction_speech(prediction)
    assert len(text) <= 200
    assert "not a confirmed diagnosis" in text
    # The full label must survive — the builder shortens by choosing a terser
    # clause, never by cutting a word.
    assert "Northern Corn Leaf Blight With A Very Long Descriptive Name" in text


# --- safe failure ----------------------------------------------------------- #


def test_missing_crop_raises() -> None:
    prediction = {k: v for k, v in PREDICTION.items() if k not in ("crop", "display_name")}
    with pytest.raises(ApiError) as excinfo:
        build_prediction_speech(prediction)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "response_missing_crop"


def test_missing_class_raises() -> None:
    with pytest.raises(ApiError) as excinfo:
        build_prediction_speech({**PREDICTION, "predicted_label": "   "})
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "response_missing_class"


@pytest.mark.parametrize("bad", ["high", 1.5, -0.1, True, float("nan"), float("inf")])
def test_malformed_confidence_raises(bad) -> None:
    with pytest.raises(ApiError) as excinfo:
        build_prediction_speech({**PREDICTION, "confidence": bad})
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "response_invalid_confidence"


@pytest.mark.parametrize("bad", [None, ["x"], "text", 42])
def test_malformed_prediction_object_raises(bad) -> None:
    with pytest.raises(ApiError) as excinfo:
        build_prediction_speech(bad)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "invalid_prediction"


def test_unsalvageably_long_input_raises_instead_of_truncating() -> None:
    prediction = {
        **PREDICTION,
        "display_name": "a" * 120,
        "predicted_label": "b" * 120,
    }
    with pytest.raises(ApiError) as excinfo:
        build_prediction_speech(prediction)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "response_text_too_long"


# --- safety phrasing -------------------------------------------------------- #


def test_no_invented_or_unsafe_phrasing() -> None:
    text = build_prediction_speech(PREDICTION).lower()
    for forbidden in (
        "sir",
        "treatment",
        "pesticide",
        "dose",
        "dosage",
        "cure",
        "guaranteed",
        "i diagnosed",
        "we diagnosed",
    ):
        assert forbidden not in text
    # uncertainty language present, certainty framing absent
    assert "not a confirmed diagnosis" in text
    assert "is confirmed" not in text
