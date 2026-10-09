"""Route tests for ``POST /insights/ask`` (Increment 9P).

No network and no secrets leak: the chat service is monkeypatched so the
endpoint never reaches Groq, and the API key is a harmless test value.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.insights.chat import ChatAnswer, generate_answer as real_generate_answer
from app.main import app
from app.routers import insights as insights_router

client = TestClient(app)

VALID_BODY = {
    "crop": "maize",
    "question": "How do I water maize?",
    "topic": "irrigation",
    "context": {"predicted_label": "Northern Leaf Blight", "confidence": 0.9},
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    async def _boom(*_args, **_kwargs):
        raise AssertionError("the endpoint must not reach the chat service here")

    monkeypatch.setattr(insights_router, "generate_answer", _boom)


def _enable(monkeypatch) -> None:
    monkeypatch.setattr(insights_router.settings, "groq_insights_enabled", True)


# --- disabled / configuration ----------------------------------------------- #


def test_disabled_by_default() -> None:
    response = client.post("/insights/ask", json=VALID_BODY)
    assert response.status_code == 503
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert body["error"]["code"] == "insights_disabled"


def test_enabled_but_no_key_is_not_configured(monkeypatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setattr(insights_router, "generate_answer", real_generate_answer)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ-API-KEY", raising=False)
    response = client.post("/insights/ask", json=VALID_BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "insights_not_configured"


# --- success ---------------------------------------------------------------- #


def test_success_returns_answer(monkeypatch) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-abc")

    async def fake_generate(messages, **_kwargs):
        assert messages[0]["role"] == "system"
        return ChatAnswer(text="Water when the topsoil is dry.", model="llama-3.3-70b-versatile")

    monkeypatch.setattr(insights_router, "generate_answer", fake_generate)

    response = client.post("/insights/ask", json=VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["crop"] == "maize"
    assert body["topic"] == "irrigation"
    assert body["answer"] == "Water when the topsoil is dry."
    assert body["model"] == "llama-3.3-70b-versatile"
    assert body["generated_at"]
    assert "not a confirmed diagnosis" in body["disclaimer"]
    assert "test-key-abc" not in response.text


def test_topic_defaults_to_general_when_omitted(monkeypatch) -> None:
    _enable(monkeypatch)

    async def fake_generate(messages, **_kwargs):
        return ChatAnswer(text="ok", model="m")

    monkeypatch.setattr(insights_router, "generate_answer", fake_generate)
    response = client.post("/insights/ask", json={"crop": "rice", "question": "hello?"})
    assert response.status_code == 200
    assert response.json()["topic"] == "general"


def test_extra_fields_are_ignored(monkeypatch) -> None:
    _enable(monkeypatch)

    async def fake_generate(messages, **_kwargs):
        return ChatAnswer(text="ok", model="m")

    monkeypatch.setattr(insights_router, "generate_answer", fake_generate)
    body = dict(VALID_BODY)
    body["unexpected"] = "ignored"
    body["context"] = {"predicted_label": "Smut", "confidence": 0.5, "extra": 1}
    response = client.post("/insights/ask", json=body)
    assert response.status_code == 200


# --- validation (before any upstream call) ---------------------------------- #


def test_unsupported_crop_is_400(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post("/insights/ask", json={"crop": "wheat", "question": "hi"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_crop"


def test_empty_question_is_400(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post("/insights/ask", json={"crop": "maize", "question": "   "})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "insight_empty_question"


def test_question_too_long_is_400(monkeypatch) -> None:
    _enable(monkeypatch)
    question = "a" * (settings.groq_chat_max_input_chars + 1)
    response = client.post("/insights/ask", json={"crop": "maize", "question": question})
    assert response.status_code == 400
    err = response.json()["error"]
    assert err["code"] == "insight_question_too_long"
    assert err["details"]["max_characters"] == settings.groq_chat_max_input_chars


def test_invalid_topic_is_400(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post(
        "/insights/ask", json={"crop": "maize", "question": "hi", "topic": "weather"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "insight_invalid_topic"


def test_invalid_context_is_400(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post(
        "/insights/ask",
        json={"crop": "maize", "question": "hi", "context": {"confidence": 2}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "insight_invalid_context"


# --- request validation / envelope ------------------------------------------ #


def test_missing_crop_is_validation_error(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post("/insights/ask", json={"question": "hi"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_non_object_body_is_validation_error(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post("/insights/ask", json=["nope"])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_error_envelope_has_no_traceback(monkeypatch) -> None:
    _enable(monkeypatch)
    response = client.post("/insights/ask", json={"crop": "wheat", "question": "hi"})
    assert response.status_code == 400
    assert "traceback" not in response.text.lower()
