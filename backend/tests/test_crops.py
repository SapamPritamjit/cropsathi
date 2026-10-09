from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _crops_by_key() -> dict:
    response = client.get("/crops")
    assert response.status_code == 200
    return {c["crop"]: c for c in response.json()["crops"]}


def test_lists_exactly_four_crops() -> None:
    response = client.get("/crops")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 4
    assert {c["crop"] for c in body["crops"]} == {"maize", "rice", "potato", "tomato"}


def test_baseline_present_for_every_crop() -> None:
    crops = _crops_by_key()
    for key, crop in crops.items():
        assert crop["baseline"]["present"] is True, key
        assert crop["baseline"]["metadata_readable"] is True, key
        assert crop["baseline"]["num_classes"] > 0, key


def test_baseline_class_counts_match_config() -> None:
    crops = _crops_by_key()
    assert crops["maize"]["baseline"]["num_classes"] == 11
    assert crops["potato"]["baseline"]["num_classes"] == 3
    assert crops["rice"]["baseline"]["num_classes"] == 14
    assert crops["tomato"]["baseline"]["num_classes"] == 13


def test_fine_tuned_availability_matches_artifacts() -> None:
    crops = _crops_by_key()
    assert crops["maize"]["fine_tuned"]["present"] is True
    assert crops["rice"]["fine_tuned"]["present"] is True
    assert crops["potato"]["fine_tuned"]["present"] is False
    assert crops["tomato"]["fine_tuned"]["present"] is False


def test_fine_tuned_marked_unverified() -> None:
    crops = _crops_by_key()
    for key in ("maize", "rice"):
        assert crops[key]["fine_tuned"]["verified"] is False, key


def test_fine_tuned_class_counts_from_metadata() -> None:
    crops = _crops_by_key()
    assert len(crops["maize"]["fine_tuned"]["class_names"]) == 9
    assert len(crops["rice"]["fine_tuned"]["class_names"]) == 13
