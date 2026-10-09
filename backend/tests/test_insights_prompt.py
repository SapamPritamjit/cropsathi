"""Unit tests for the pure crop Q&A prompt builder (Increment 9P).

No network and no secrets: ``app.insights.prompt`` is deterministic and performs
no I/O. These tests cover input normalization, validation error codes, and the
responsible-AI constraints baked into the system prompt.
"""

import pytest

from app.config import settings
from app.errors import ApiError
from app.insights import prompt


# --- normalize_crop --------------------------------------------------------- #


@pytest.mark.parametrize("crop", ["maize", "rice", "potato", "tomato"])
def test_normalize_crop_accepts_supported(crop: str) -> None:
    assert prompt.normalize_crop(crop) == crop


def test_normalize_crop_is_case_and_space_insensitive() -> None:
    assert prompt.normalize_crop("  Maize ") == "maize"
    assert prompt.normalize_crop("TOMATO") == "tomato"


@pytest.mark.parametrize("bad", ["wheat", "", "   ", None, 7, ["maize"]])
def test_normalize_crop_rejects_unsupported(bad) -> None:
    with pytest.raises(ApiError) as excinfo:
        prompt.normalize_crop(bad)
    err = excinfo.value
    assert err.status_code == 400
    assert err.code == "unsupported_crop"
    assert err.details["supported_crops"] == list(("maize", "rice", "potato", "tomato"))


# --- normalize_question ----------------------------------------------------- #


def test_normalize_question_collapses_whitespace_and_controls() -> None:
    assert prompt.normalize_question("  How\tdo I\n\nwater maize?  ") == (
        "How do I water maize?"
    )
    assert prompt.normalize_question("a\x00b\x07c") == "a b c"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 5])
def test_normalize_question_rejects_empty(bad) -> None:
    with pytest.raises(ApiError) as excinfo:
        prompt.normalize_question(bad)
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "insight_empty_question"


def test_normalize_question_rejects_too_long() -> None:
    limit = settings.groq_chat_max_input_chars
    with pytest.raises(ApiError) as excinfo:
        prompt.normalize_question("a" * (limit + 1))
    err = excinfo.value
    assert err.status_code == 400
    assert err.code == "insight_question_too_long"
    assert err.details == {"max_characters": limit, "characters": limit + 1}


def test_normalize_question_allows_exact_limit() -> None:
    limit = settings.groq_chat_max_input_chars
    assert prompt.normalize_question("a" * limit) == "a" * limit


# --- normalize_topic -------------------------------------------------------- #


def test_normalize_topic_defaults_when_absent() -> None:
    assert prompt.normalize_topic(None) == prompt.DEFAULT_TOPIC
    assert prompt.normalize_topic("   ") == prompt.DEFAULT_TOPIC


@pytest.mark.parametrize("topic", prompt.SUPPORTED_TOPICS)
def test_normalize_topic_accepts_allow_list(topic: str) -> None:
    assert prompt.normalize_topic(topic) == topic


def test_normalize_topic_is_case_insensitive() -> None:
    assert prompt.normalize_topic("Disease_Symptoms") == "disease_symptoms"


@pytest.mark.parametrize("bad", ["weather", "treatment", 3, ["x"]])
def test_normalize_topic_rejects_unknown(bad) -> None:
    with pytest.raises(ApiError) as excinfo:
        prompt.normalize_topic(bad)
    err = excinfo.value
    assert err.status_code == 400
    assert err.code == "insight_invalid_topic"
    assert err.details["supported_topics"] == list(prompt.SUPPORTED_TOPICS)


# --- normalize_context ------------------------------------------------------ #


def test_normalize_context_none_and_empty_return_none() -> None:
    assert prompt.normalize_context(None) is None
    assert prompt.normalize_context({}) is None
    assert prompt.normalize_context({"unrelated": "x"}) is None


def test_normalize_context_accepts_valid_fields() -> None:
    context = prompt.normalize_context(
        {"predicted_label": "Northern Leaf Blight", "confidence": 0.929, "family": "baseline"}
    )
    assert context == {
        "predicted_label": "Northern Leaf Blight",
        "confidence": 0.929,
        "family": "baseline",
    }


def test_normalize_context_accepts_pydantic_like_object() -> None:
    class _Ctx:
        def model_dump(self):
            return {"predicted_label": "Smut", "confidence": 0.5}

    assert prompt.normalize_context(_Ctx()) == {
        "predicted_label": "Smut",
        "confidence": 0.5,
    }


def test_normalize_context_bounds_label_and_family_length() -> None:
    context = prompt.normalize_context(
        {"predicted_label": "x" * 500, "family": "y" * 500}
    )
    assert len(context["predicted_label"]) <= 200
    assert len(context["family"]) <= 40


@pytest.mark.parametrize("bad", [1.5, -0.1, True, "high", {"x": 1}, float("inf")])
def test_normalize_context_rejects_bad_confidence(bad) -> None:
    with pytest.raises(ApiError) as excinfo:
        prompt.normalize_context({"predicted_label": "Smut", "confidence": bad})
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "insight_invalid_context"


def test_normalize_context_rejects_non_mapping() -> None:
    with pytest.raises(ApiError) as excinfo:
        prompt.normalize_context("not an object")
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "insight_invalid_context"


def test_normalize_context_confidence_bounds_are_inclusive() -> None:
    assert prompt.normalize_context({"confidence": 0.0}) == {"confidence": 0.0}
    assert prompt.normalize_context({"confidence": 1.0}) == {"confidence": 1.0}


# --- build_messages --------------------------------------------------------- #


def test_build_messages_shape_and_content() -> None:
    messages = prompt.build_messages("maize", "How do I water maize?", "irrigation")
    assert len(messages) == 2
    assert [m["role"] for m in messages] == ["system", "user"]
    system = messages[0]["content"]
    assert "Maize" in system and "maize" in system
    user = messages[1]["content"]
    assert "How do I water maize?" in user
    assert "irrigation" in user


def test_system_prompt_encodes_responsible_ai_rules() -> None:
    system = prompt.build_system_prompt("rice").lower()
    assert "uncertain prediction" in system
    assert "never a confirmed diagnosis" in system
    assert "extension officer" in system
    assert "do not invent government scheme" in system
    assert "do not prescribe specific chemical pesticides" in system


def test_system_prompt_includes_prediction_context_as_uncertain() -> None:
    context = {"predicted_label": "Late Blight", "confidence": 0.75}
    system = prompt.build_system_prompt("potato", context)
    assert "Late Blight" in system
    assert "75.0%" in system
    assert "uncertain model prediction" in system


def test_system_prompt_omits_context_wording_when_absent() -> None:
    system = prompt.build_system_prompt("tomato", None)
    assert "most recent image-model result" not in system


def test_unknown_topic_label_falls_back_to_raw_topic() -> None:
    user = prompt.build_user_message("q", "not_a_topic")
    assert "not_a_topic" in user
    assert "general" not in user
