from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_no_secret_material_in_api_responses() -> None:
    for path in ("/health", "/crops"):
        text = client.get(path).text.lower()
        assert "groq" not in text
        assert "api_key" not in text
        assert "apikey" not in text


def test_settings_hold_only_a_boolean_for_groq() -> None:
    assert isinstance(settings.groq_api_key_configured, bool)
    assert not hasattr(settings, "groq_api_key")
