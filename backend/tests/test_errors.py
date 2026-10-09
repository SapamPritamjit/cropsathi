import asyncio
import json

from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.main import app, validation_exception_handler

client = TestClient(app)


def test_unknown_route_uses_error_envelope() -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "404"
    assert isinstance(body["error"]["message"], str)


def test_method_not_allowed_uses_error_envelope() -> None:
    response = client.post("/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "405"


def test_validation_error_handler_shape() -> None:
    exc = RequestValidationError(
        errors=[{"loc": ("query", "x"), "msg": "invalid", "type": "value_error"}]
    )
    response = asyncio.run(validation_exception_handler(None, exc))
    assert response.status_code == 422
    body = json.loads(response.body)
    assert body["error"]["code"] == "validation_error"
    assert isinstance(body["error"]["details"], list)
