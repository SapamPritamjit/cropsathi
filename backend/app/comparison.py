"""Model inventory + comparison data built ONLY from verified artifacts.

This module performs **no inference** and exposes **no routes**. It assembles
deterministic comparison metadata from:

- baseline ``config.json`` (architecture, class labels, verified preprocessing),
- provider ``README.md`` metrics (explicitly labelled provider-reported),
- fine-tuned ``classes.json`` + checkpoint-embedded metrics (training-reported),
- ``verification_report.json`` (checkpoint load / verification status).

Integrity rules enforced here:
- class sets are aligned by *label name*, never by index;
- a metric's dataset split is reported exactly as recorded, or ``unknown``;
- a missing class is never turned into a zero-probability prediction;
- "checkpoint exists", "checkpoint loads", "preprocessing verified" and
  "live inference enabled" are kept as separate, explicit flags.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import (
    BASELINE_MODEL_DIRS,
    CROP_DISPLAY_NAMES,
    FINE_TUNED_MODEL_DIRS,
    PROJECT_ROOT,
    SUPPORTED_CROPS,
    settings,
)
from .model_metadata import read_baseline_metadata, read_fine_tuned_metadata

PROVIDER_PROVENANCE = "provider-reported"
TRAINING_PROVENANCE = "training-reported"
PROVIDER_DISCLAIMER = "Provider-reported; not locally reproduced."
TRAINING_DISCLAIMER = "Training-reported; not independently reproduced."
COMPARISON_DISCLAIMER = (
    "Training-reported and provider-reported metrics were produced under "
    "different evaluation protocols and class sets and must not be treated as "
    "directly comparable or as current live-prediction performance. Probability "
    "is the model's softmax output, not a diagnostic certainty."
)

# Fine-tuned checkpoint metric key -> dataset split. ``macro_f1`` (maize) is
# stored without a split prefix, so its split is reported as unknown.
_FT_METRIC_SPLITS: dict[str, str] = {
    "val_accuracy": "val",
    "val_loss": "val",
    "test_accuracy": "test",
    "test_macro_f1": "test",
    "test_weighted_f1": "test",
    "macro_f1": "unknown",
}
_FT_METRIC_ORDER = list(_FT_METRIC_SPLITS)

_README_OVERALL_RE = re.compile(
    r"\*\*([\d.]+)\s*%\*\*\s*test accuracy,\s*macro-F1\s*([\d.]+),"
    r"\s*on\s*([\d,]+)\s*held-out photos"
)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def canonical_label(label: str) -> str:
    """Normalise a class label to a comparison key (case/space/punct agnostic)."""
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Verification report lookup
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _verification_index() -> dict[tuple[str, str], dict[str, Any]]:
    data = _read_json(PROJECT_ROOT / "backend" / "verification_report.json")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(data, dict):
        for entry in data.get("models", []):
            if isinstance(entry, dict):
                index[(entry.get("crop"), entry.get("family"))] = entry
    return index


def _report_status(crop: str, family: str) -> str | None:
    entry = _verification_index().get((crop, family))
    return entry.get("status") if entry else None


def _report_loads(crop: str, family: str) -> bool | None:
    entry = _verification_index().get((crop, family))
    if not entry:
        return None
    value = entry.get("load_ok")
    return bool(value) if isinstance(value, bool) else None


# --------------------------------------------------------------------------- #
# Preprocessing verification
# --------------------------------------------------------------------------- #
def baseline_preprocessing_verified(crop: str) -> bool:
    """True when the baseline config records the full eval transform."""
    cfg = _read_json(BASELINE_MODEL_DIRS[crop] / "config.json") or {}
    pretrained = cfg.get("pretrained_cfg") or {}
    return bool(
        isinstance(pretrained.get("input_size"), list)
        and len(pretrained["input_size"]) == 3
        and isinstance(pretrained.get("mean"), list)
        and len(pretrained["mean"]) == 3
        and isinstance(pretrained.get("std"), list)
        and len(pretrained["std"]) == 3
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _empty_metrics(provenance: str, disclaimer: str, note: str) -> dict[str, Any]:
    return {
        "available": False,
        "provenance": provenance,
        "disclaimer": disclaimer,
        "items": [],
        "test_samples": None,
        "notes": [note],
    }


def read_provider_metrics(crop: str) -> dict[str, Any]:
    """Parse the provider's overall test metrics from the baseline README."""
    try:
        text = (BASELINE_MODEL_DIRS[crop] / "README.md").read_text(encoding="utf-8")
    except OSError:
        return _empty_metrics(
            PROVIDER_PROVENANCE,
            PROVIDER_DISCLAIMER,
            "Provider README not readable; metrics unavailable.",
        )

    match = _README_OVERALL_RE.search(text)
    if not match:
        return _empty_metrics(
            PROVIDER_PROVENANCE,
            PROVIDER_DISCLAIMER,
            "Provider README did not contain a parseable overall metric line.",
        )

    accuracy = float(match.group(1)) / 100.0
    macro_f1 = float(match.group(2))
    test_samples = int(match.group(3).replace(",", ""))
    return {
        "available": True,
        "provenance": PROVIDER_PROVENANCE,
        "disclaimer": PROVIDER_DISCLAIMER,
        "items": [
            {
                "name": "test_accuracy",
                "value": accuracy,
                "split": "test",
                "provenance": PROVIDER_PROVENANCE,
                "source": "provider README",
            },
            {
                "name": "macro_f1",
                "value": macro_f1,
                "split": "test",
                "provenance": PROVIDER_PROVENANCE,
                "source": "provider README",
            },
        ],
        "test_samples": test_samples,
        "notes": [
            "Values come from the provider's own README and were not reproduced here.",
            "The provider notes test photos come from the same datasets used for "
            "training, so real field photos may score lower.",
        ],
    }


@lru_cache(maxsize=8)
def read_fine_tuned_metrics(crop: str) -> dict[str, Any]:
    """Read training-reported metrics from the fine-tuned checkpoint.

    Returns ``{"metrics": <ModelMetrics-shaped dict>, "training_info": {...}}``.
    """
    empty = {
        "metrics": _empty_metrics(
            TRAINING_PROVENANCE,
            TRAINING_DISCLAIMER,
            "Fine-tuned metrics could not be read from the checkpoint.",
        ),
        "training_info": {},
    }
    model_dir = FINE_TUNED_MODEL_DIRS.get(crop)
    if model_dir is None or not model_dir.is_dir():
        return empty
    checkpoints = sorted(model_dir.glob("*.pth"))
    if not checkpoints:
        return empty

    try:
        import torch

        checkpoint = torch.load(
            checkpoints[0], map_location="cpu", weights_only=True
        )
    except Exception:  # pragma: no cover - reported, not hidden
        return empty

    items = [
        {
            "name": key,
            "value": float(checkpoint[key]),
            "split": _FT_METRIC_SPLITS[key],
            "provenance": TRAINING_PROVENANCE,
            "source": "checkpoint",
        }
        for key in _FT_METRIC_ORDER
        if key in checkpoint
    ]

    notes = [
        "Values are embedded in the supplied checkpoint and are training-reported.",
    ]
    if "macro_f1" in checkpoint:
        notes.append(
            "The maize checkpoint stores 'macro_f1' without a split prefix; its "
            "dataset split is not recorded and is reported as 'unknown'."
        )

    training_info: dict[str, Any] = {}
    if "best_epoch" in checkpoint:
        training_info["best_epoch"] = int(checkpoint["best_epoch"])

    return {
        "metrics": {
            "available": bool(items),
            "provenance": TRAINING_PROVENANCE,
            "disclaimer": TRAINING_DISCLAIMER,
            "items": items,
            "test_samples": None,
            "notes": notes,
        },
        "training_info": training_info,
    }


# --------------------------------------------------------------------------- #
# Model inventory entries
# --------------------------------------------------------------------------- #
def _live_inference_enabled(
    family: str, present: bool, preprocessing_verified: bool, loads: bool | None
) -> bool:
    if not present or loads is False:
        return False
    if family == "baseline":
        return True
    # Fine-tuned: every precondition must hold, and preprocessing must be verified.
    return bool(
        settings.enable_fine_tuned_predictions and preprocessing_verified
    )


def _blocked_reason(
    family: str,
    present: bool,
    preprocessing_verified: bool,
    loads: bool | None,
    live: bool,
) -> str | None:
    if live:
        return None
    if not present:
        return "checkpoint_absent"
    if loads is False:
        return "checkpoint_failed_verification"
    if family == "fine_tuned":
        if not preprocessing_verified:
            return "preprocessing_unverified"
        if not settings.enable_fine_tuned_predictions:
            return "fine_tuned_predictions_disabled"
    return "not_enabled"


def build_model_entry(crop: str, family: str) -> dict[str, Any]:
    """Build one model-inventory entry (no inference)."""
    if family == "baseline":
        meta = read_baseline_metadata(crop)
        present = bool(meta.get("present"))
        architecture = meta.get("architecture")
        labels = meta.get("label_names")
        num_classes = meta.get("num_classes")
        input_size = meta.get("input_size")
        implementation = f"timm:{architecture}" if architecture else None
        preprocessing_verified = present and baseline_preprocessing_verified(crop)
        metrics = read_provider_metrics(crop)
        training_info: dict[str, Any] = {}
        display_role = "Baseline model"
    else:
        meta = read_fine_tuned_metadata(crop)
        present = bool(meta.get("present"))
        architecture = "efficientnet_b0" if present else None
        labels = meta.get("class_names") if present else None
        num_classes = meta.get("num_classes") if present else None
        input_size = [3, 224, 224] if present else None
        implementation = "torchvision:efficientnet_b0" if present else None
        # Fine-tuned preprocessing is NOT recorded in the supplied artifacts.
        preprocessing_verified = False
        read = read_fine_tuned_metrics(crop)
        metrics = read["metrics"]
        training_info = read["training_info"]
        display_role = "Fine-tuned checkpoint"

    status = _report_status(crop, family)
    if status is None:
        status = "verified" if present else "absent"
    loads = _report_loads(crop, family)
    if loads is None and family == "baseline":
        loads = present and bool(meta.get("metadata_readable"))

    live = _live_inference_enabled(family, present, preprocessing_verified, loads)
    reason = _blocked_reason(family, present, preprocessing_verified, loads, live)

    return {
        "crop": crop,
        "display_name": CROP_DISPLAY_NAMES[crop],
        "family": family,
        "display_role": display_role,
        "architecture": architecture,
        "implementation": implementation,
        "num_classes": num_classes,
        "labels": labels,
        "input_size": input_size,
        "checkpoint_present": present,
        "checkpoint_loads": loads,
        "verification_status": status,
        "preprocessing_verified": preprocessing_verified,
        "live_inference_enabled": live,
        "live_inference_blocked_reason": reason,
        "metrics": metrics,
        "training_info": training_info,
    }


def fine_tuned_available(crop: str) -> bool:
    return bool(read_fine_tuned_metadata(crop).get("present"))


def build_models() -> list[dict[str, Any]]:
    """Inventory of every available artifact (4 baselines + fine-tuned where present)."""
    entries: list[dict[str, Any]] = []
    for crop in SUPPORTED_CROPS:
        entries.append(build_model_entry(crop, "baseline"))
        if crop in FINE_TUNED_MODEL_DIRS and fine_tuned_available(crop):
            entries.append(build_model_entry(crop, "fine_tuned"))
    return entries


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _class_sets(baseline: dict[str, Any], fine_tuned: dict[str, Any] | None) -> dict[str, Any]:
    baseline_labels = baseline.get("labels") or []
    if fine_tuned is None:
        return {
            "baseline_labels": list(baseline_labels),
            "fine_tuned_labels": None,
            "shared": [],
            "baseline_only": [],
            "fine_tuned_only": [],
            "counts": {
                "baseline": len(baseline_labels),
                "fine_tuned": 0,
                "shared": 0,
                "baseline_only": 0,
                "fine_tuned_only": 0,
            },
            "direct_class_alignment_possible": False,
        }

    fine_labels = fine_tuned.get("labels") or []
    baseline_by_key = {canonical_label(l): l for l in baseline_labels}
    fine_by_key = {canonical_label(l): l for l in fine_labels}

    shared_keys = sorted(set(baseline_by_key) & set(fine_by_key))
    baseline_only_keys = sorted(set(baseline_by_key) - set(fine_by_key))
    fine_only_keys = sorted(set(fine_by_key) - set(baseline_by_key))

    return {
        "baseline_labels": list(baseline_labels),
        "fine_tuned_labels": list(fine_labels),
        "shared": [
            {
                "key": key,
                "baseline_label": baseline_by_key[key],
                "fine_tuned_label": fine_by_key[key],
            }
            for key in shared_keys
        ],
        "baseline_only": [
            {"key": key, "label": baseline_by_key[key]} for key in baseline_only_keys
        ],
        "fine_tuned_only": [
            {"key": key, "label": fine_by_key[key]} for key in fine_only_keys
        ],
        "counts": {
            "baseline": len(baseline_labels),
            "fine_tuned": len(fine_labels),
            "shared": len(shared_keys),
            "baseline_only": len(baseline_only_keys),
            "fine_tuned_only": len(fine_only_keys),
        },
        "direct_class_alignment_possible": bool(shared_keys),
    }


def build_comparison(crop: str) -> dict[str, Any]:
    """Build the comparison payload for one crop."""
    baseline = build_model_entry(crop, "baseline")
    fine_tuned = (
        build_model_entry(crop, "fine_tuned") if fine_tuned_available(crop) else None
    )

    if fine_tuned is None:
        live_possible = False
        blocked_reason = "no_fine_tuned_model"
    elif fine_tuned["live_inference_enabled"] and baseline["live_inference_enabled"]:
        live_possible = True
        blocked_reason = None
    else:
        live_possible = False
        blocked_reason = fine_tuned["live_inference_blocked_reason"]

    return {
        "crop": crop,
        "display_name": CROP_DISPLAY_NAMES[crop],
        "baseline": baseline,
        "fine_tuned": fine_tuned,
        "class_sets": _class_sets(baseline, fine_tuned),
        "model_comparison": {
            "live_prediction_comparison_possible": live_possible,
            "live_prediction_comparison_blocked_reason": blocked_reason,
            "live_prediction_endpoint": "POST /predict",
            "note": (
                "Live predictions are served only by verified baseline models via "
                "POST /predict. Fine-tuned inference is blocked pending "
                "preprocessing verification, so no cross-model live comparison is "
                "produced and no probabilities are fabricated for unsupported classes."
            ),
        },
        "disclaimer": COMPARISON_DISCLAIMER,
    }
