"""Tests for environment-driven CORS (Increment 7A).

The allow-list comes from ``CORS_ALLOWED_ORIGINS`` via ``Settings``. Fresh apps
are built through ``create_app()`` so both an explicit allow-list and the
deny-by-default configuration can be exercised. No network access is made.
"""

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import parse_allowed_origins, settings

ALLOWED_ORIGIN = "http://localhost:5173"
DISALLOWED_ORIGIN = "https://evil.example"


@pytest.fixture
def allowed_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "cors_allowed_origins", (ALLOWED_ORIGIN,))
    return TestClient(main_module.create_app())


@pytest.fixture
def deny_all_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "cors_allowed_origins", ())
    return TestClient(main_module.create_app())


# --- parsing ---------------------------------------------------------------- #


def test_parse_empty_and_whitespace_yields_no_origins() -> None:
    assert parse_allowed_origins(None) == ()
    assert parse_allowed_origins("") == ()
    assert parse_allowed_origins("   ") == ()
    assert parse_allowed_origins(" , , ") == ()


def test_parse_trims_dedupes_and_drops_trailing_slash() -> None:
    raw = "  http://localhost:5173 , https://app.lovable.app/ ,http://localhost:5173,"
    assert parse_allowed_origins(raw) == (
        "http://localhost:5173",
        "https://app.lovable.app",
    )


def test_default_settings_never_allow_all_origins() -> None:
    assert isinstance(settings.cors_allowed_origins, tuple)
    assert "*" not in settings.cors_allowed_origins


# --- simple requests -------------------------------------------------------- #


def test_allowed_origin_gets_acao_on_simple_request(allowed_client: TestClient) -> None:
    response = allowed_client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "access-control-allow-credentials" not in response.headers


def test_disallowed_origin_has_no_acao(allowed_client: TestClient) -> None:
    response = allowed_client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_request_without_origin_is_unchanged() -> None:
    response = TestClient(main_module.app).get("/health")
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


# --- preflight -------------------------------------------------------------- #


def test_preflight_allowed_origin_and_method_succeeds(allowed_client: TestClient) -> None:
    response = allowed_client.options(
        "/predict",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]


def test_preflight_disallowed_origin_is_rejected(allowed_client: TestClient) -> None:
    response = allowed_client.options(
        "/predict",
        headers={
            "Origin": DISALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_preflight_allowed_origin_disallowed_method_is_rejected(
    allowed_client: TestClient,
) -> None:
    # Starlette echoes the allowed origin but still fails the preflight (400)
    # and never advertises the un-permitted method.
    response = allowed_client.options(
        "/predict",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert response.status_code == 400
    assert "DELETE" not in response.headers["access-control-allow-methods"]


# --- deny-by-default configuration ------------------------------------------ #


def test_deny_all_lacks_acao_and_rejects_preflight(deny_all_client: TestClient) -> None:
    simple = deny_all_client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert simple.status_code == 200
    assert "access-control-allow-origin" not in simple.headers

    preflight = deny_all_client.options(
        "/predict",
        headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers


# --- existing contracts unchanged ------------------------------------------- #


def test_existing_responses_and_error_envelopes_unchanged(
    allowed_client: TestClient,
) -> None:
    health = allowed_client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert health.status_code == 200
    assert set(health.json().keys()) == {"status", "app", "version", "environment"}

    error = allowed_client.post(
        "/speech/speak",
        json={"text": ""},
        headers={"Origin": ALLOWED_ORIGIN},
    )
    assert error.status_code == 400
    assert set(error.json().keys()) == {"error"}
    assert error.json()["error"]["code"] == "tts_empty_text"
    assert error.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
