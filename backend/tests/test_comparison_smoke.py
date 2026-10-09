"""Real-checkpoint tests for comparison metrics (marked ``smoke``).

These actually read the fine-tuned checkpoints and the provider README metrics.
"""

import pytest
from fastapi.testclient import TestClient

from app.comparison import read_fine_tuned_metrics, read_provider_metrics
from app.main import app

pytestmark = pytest.mark.smoke

TRAINING_DISCLAIMER = "Training-reported; not independently reproduced."


def metrics_by_name(crop: str) -> dict:
    return {item["name"]: item for item in read_fine_tuned_metrics(crop)["metrics"]["items"]}


def test_maize_fine_tuned_real_metrics() -> None:
    read = read_fine_tuned_metrics("maize")
    items = {i["name"]: i for i in read["metrics"]["items"]}
    assert items["test_accuracy"]["value"] == pytest.approx(0.936, abs=1e-9)
    assert items["test_accuracy"]["split"] == "test"
    assert items["val_accuracy"]["split"] == "val"
    assert items["macro_f1"]["value"] == pytest.approx(0.9359, abs=1e-9)
    assert items["macro_f1"]["split"] == "unknown"
    assert read["metrics"]["disclaimer"] == TRAINING_DISCLAIMER
    assert read["metrics"]["provenance"] == "training-reported"


def test_rice_fine_tuned_real_metrics() -> None:
    read = read_fine_tuned_metrics("rice")
    items = {i["name"]: i for i in read["metrics"]["items"]}
    assert items["test_accuracy"]["value"] == pytest.approx(0.9001663893510815, abs=1e-9)
    assert items["test_accuracy"]["split"] == "test"
    assert items["test_macro_f1"]["value"] == pytest.approx(0.8584, abs=1e-9)
    assert items["test_macro_f1"]["split"] == "test"
    assert items["test_weighted_f1"]["value"] == pytest.approx(0.9042, abs=1e-9)
    assert items["val_accuracy"]["split"] == "val"
    assert items["val_loss"]["split"] == "val"
    assert read["training_info"]["best_epoch"] == 9


@pytest.mark.parametrize(
    "crop,accuracy,macro_f1,samples",
    [
        ("maize", 0.929, 0.936, 1281),
        ("rice", 0.935, 0.940, 1579),
        ("potato", 0.960, 0.971, 247),
        ("tomato", 0.934, 0.932, 1796),
    ],
)
def test_provider_metrics_parsed_from_readme(
    crop: str, accuracy: float, macro_f1: float, samples: int
) -> None:
    metrics = read_provider_metrics(crop)
    items = {i["name"]: i["value"] for i in metrics["items"]}
    assert metrics["available"] is True
    assert metrics["provenance"] == "provider-reported"
    assert items["test_accuracy"] == pytest.approx(accuracy, abs=1e-9)
    assert items["macro_f1"] == pytest.approx(macro_f1, abs=1e-9)
    assert metrics["test_samples"] == samples


def test_models_endpoint_reports_real_fine_tuned_metrics() -> None:
    client = TestClient(app)
    body = client.get("/models").json()
    ft = {(m["crop"], m["family"]): m for m in body["models"]}
    maize_ft = ft[("maize", "fine_tuned")]
    assert maize_ft["metrics"]["available"] is True
    assert maize_ft["metrics"]["disclaimer"] == TRAINING_DISCLAIMER
    assert maize_ft["live_inference_enabled"] is False
