from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "AI4Seva API"
    assert body["version"]
    assert body["environment"]


def test_health_exposes_no_secrets() -> None:
    body = client.get("/health").json()
    assert set(body.keys()) == {"status", "app", "version", "environment"}
