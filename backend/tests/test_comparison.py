"""Fast, mostly-mocked tests for the inventory + comparison API.

The fine-tuned checkpoint metrics reader is replaced with a fake so these tests
load no model weights. Provider metrics are parsed from the real README files
(cheap). Real-checkpoint metric assertions live in ``test_comparison_smoke.py``.
"""

import pytest
from fastapi.testclient import TestClient

from app import comparison
from app.main import app

client = TestClient(app)

TRAINING_DISCLAIMER = "Training-reported; not independently reproduced."
PROVIDER_DISCLAIMER = "Provider-reported; not locally reproduced."

_FAKE_FT = {
    "metrics": {
        "available": True,
        "provenance": "training-reported",
        "disclaimer": TRAINING_DISCLAIMER,
        "items": [
            {
                "name": "test_accuracy",
                "value": 0.9,
                "split": "test",
                "provenance": "training-reported",
                "source": "checkpoint",
            },
            {
                "name": "macro_f1",
                "value": 0.88,
                "split": "unknown",
                "provenance": "training-reported",
                "source": "checkpoint",
            },
        ],
        "test_samples": None,
        "notes": ["fake"],
    },
    "training_info": {"best_epoch": 9},
}


@pytest.fixture(autouse=True)
def _fake_ft_metrics(monkeypatch):
    monkeypatch.setattr(comparison, "read_fine_tuned_metrics", lambda crop: _FAKE_FT)


def models_by_key() -> dict:
    body = client.get("/models").json()
    return {(m["crop"], m["family"]): m for m in body["models"]}


def test_models_lists_six_artifacts() -> None:
    body = client.get("/models").json()
    assert body["count"] == 6
    assert {(m["crop"], m["family"]) for m in body["models"]} == {
        ("maize", "baseline"),
        ("maize", "fine_tuned"),
        ("rice", "baseline"),
        ("rice", "fine_tuned"),
        ("potato", "baseline"),
        ("tomato", "baseline"),
    }


def test_baseline_metadata_correct() -> None:
    models = models_by_key()
    expected = {"maize": 11, "rice": 14, "potato": 3, "tomato": 13}
    for crop, n in expected.items():
        entry = models[(crop, "baseline")]
        assert entry["architecture"] == "convnext_small"
        assert entry["implementation"] == "timm:convnext_small"
        assert entry["num_classes"] == n
        assert entry["display_role"] == "Baseline model"
        assert len(entry["labels"]) == n
        assert entry["preprocessing_verified"] is True
        assert entry["live_inference_enabled"] is True
        assert entry["live_inference_blocked_reason"] is None


def test_fine_tuned_metadata_correct() -> None:
    models = models_by_key()
    assert models[("maize", "fine_tuned")]["architecture"] == "efficientnet_b0"
    assert models[("maize", "fine_tuned")]["num_classes"] == 9
    assert models[("maize", "fine_tuned")]["implementation"] == "torchvision:efficientnet_b0"
    assert models[("maize", "fine_tuned")]["display_role"] == "Fine-tuned checkpoint"
    assert models[("rice", "fine_tuned")]["num_classes"] == 13


def test_fine_tuned_live_inference_blocked() -> None:
    models = models_by_key()
    for crop in ("maize", "rice"):
        entry = models[(crop, "fine_tuned")]
        assert entry["checkpoint_present"] is True
        assert entry["checkpoint_loads"] is True
        assert entry["verification_status"] == "verified"
        assert entry["preprocessing_verified"] is False
        assert entry["live_inference_enabled"] is False
        assert entry["live_inference_blocked_reason"] == "preprocessing_unverified"


def test_baseline_metric_provenance() -> None:
    baseline = models_by_key()[("maize", "baseline")]
    metrics = baseline["metrics"]
    assert metrics["available"] is True
    assert metrics["provenance"] == "provider-reported"
    assert metrics["disclaimer"] == PROVIDER_DISCLAIMER
    splits = {item["name"]: item["split"] for item in metrics["items"]}
    assert splits == {"test_accuracy": "test", "macro_f1": "test"}
    assert metrics["test_samples"] == 1281


def test_fine_tuned_metric_provenance_and_unknown_split() -> None:
    ft = models_by_key()[("maize", "fine_tuned")]
    metrics = ft["metrics"]
    assert metrics["provenance"] == "training-reported"
    assert metrics["disclaimer"] == TRAINING_DISCLAIMER
    splits = {item["name"]: item["split"] for item in metrics["items"]}
    assert splits["test_accuracy"] == "test"
    assert splits["macro_f1"] == "unknown"


def test_comparison_maize_class_sets() -> None:
    body = client.get("/comparison/maize").json()
    class_sets = body["class_sets"]
    assert class_sets["counts"] == {
        "baseline": 11,
        "fine_tuned": 9,
        "shared": 9,
        "baseline_only": 2,
        "fine_tuned_only": 0,
    }
    assert {x["key"] for x in class_sets["baseline_only"]} == {
        "holcus_leaf_spot",
        "smut",
    }
    assert class_sets["direct_class_alignment_possible"] is True
    # alignment is by label name across the two naming conventions
    shared = {x["key"] for x in class_sets["shared"]}
    assert "common_rust" in shared and "northern_leaf_blight" in shared


def test_comparison_rice_class_sets() -> None:
    class_sets = client.get("/comparison/rice").json()["class_sets"]
    assert class_sets["counts"]["baseline"] == 14
    assert class_sets["counts"]["fine_tuned"] == 13
    assert class_sets["counts"]["shared"] == 13
    assert {x["key"] for x in class_sets["baseline_only"]} == {"neck_blast"}
    assert class_sets["counts"]["fine_tuned_only"] == 0


def test_comparison_potato_and_tomato_have_no_fine_tuned() -> None:
    for crop in ("potato", "tomato"):
        body = client.get(f"/comparison/{crop}").json()
        assert body["fine_tuned"] is None
        assert body["class_sets"]["fine_tuned_labels"] is None
        assert body["class_sets"]["direct_class_alignment_possible"] is False
        assert body["class_sets"]["counts"]["fine_tuned"] == 0
        assert body["model_comparison"]["live_prediction_comparison_possible"] is False
        assert (
            body["model_comparison"]["live_prediction_comparison_blocked_reason"]
            == "no_fine_tuned_model"
        )


def test_live_comparison_blocked_for_maize() -> None:
    mc = client.get("/comparison/maize").json()["model_comparison"]
    assert mc["live_prediction_comparison_possible"] is False
    assert mc["live_prediction_comparison_blocked_reason"] == "preprocessing_unverified"
    assert mc["live_prediction_endpoint"] == "POST /predict"


def test_unsupported_crop_uses_error_envelope() -> None:
    response = client.get("/comparison/wheat")
    assert response.status_code == 400
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "unsupported_crop"
    assert "D:\\" not in body["error"]["message"]


def test_responses_leak_no_paths_or_secrets() -> None:
    for path in ("/models", "/comparison/maize", "/comparison/potato"):
        text = client.get(path).text
        assert "D:\\" not in text
        assert "Traceback" not in text
        lowered = text.lower()
        assert "groq" not in lowered
        assert "api_key" not in lowered
        assert "apikey" not in lowered


def test_existing_error_envelope_unchanged() -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "404"
