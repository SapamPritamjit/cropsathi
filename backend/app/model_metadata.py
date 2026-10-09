"""Reads supplied model metadata WITHOUT loading any checkpoint.

Only small JSON metadata files are read. Whether a checkpoint is actually
loadable, its true architecture/preprocessing, and its verified metrics are out
of scope until Part 3, so fine-tuned entries are explicitly marked
``verified: False``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BASELINE_MODEL_DIRS, FINE_TUNED_MODEL_DIRS

BASELINE_SOURCE = (
    "Local timm ConvNeXt-Small artifacts "
    "(provider: Hugging Face ixrbhii/*-disease-convnext)"
)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_baseline_metadata(crop: str) -> dict[str, Any]:
    """Return baseline (ConvNeXt-Small) info for ``crop`` from ``config.json``."""
    model_dir = BASELINE_MODEL_DIRS[crop]
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    data = _read_json(config_path)

    if not isinstance(data, dict):
        return {
            "present": weights_path.is_file(),
            "metadata_readable": False,
            "source": BASELINE_SOURCE,
        }

    pretrained_cfg = data.get("pretrained_cfg") or {}
    input_size = pretrained_cfg.get("input_size")
    if not isinstance(input_size, list):
        input_size = None

    label_names = data.get("label_names")
    if not isinstance(label_names, list):
        label_names = None

    return {
        "present": weights_path.is_file(),
        "metadata_readable": True,
        "architecture": data.get("architecture"),
        "num_classes": data.get("num_classes"),
        "label_names": label_names,
        "input_size": input_size,
        "source": BASELINE_SOURCE,
    }


def read_fine_tuned_metadata(crop: str) -> dict[str, Any]:
    """Return fine-tuned (EfficientNet-B0) info for ``crop`` if supplied.

    Class names come from the supplied ``classes.json`` metadata file. The
    checkpoint itself is NOT verified here.
    """
    model_dir = FINE_TUNED_MODEL_DIRS.get(crop)
    if model_dir is None or not model_dir.is_dir():
        return {"present": False}

    weights_files = sorted(p.name for p in model_dir.glob("*.pth"))
    class_names = _read_json(model_dir / "classes.json")
    if not isinstance(class_names, list):
        class_names = None

    return {
        "present": bool(weights_files),
        "verified": False,
        "num_classes": len(class_names) if class_names else None,
        "class_names": class_names,
        "weights_files": weights_files,
        "note": (
            "Class names read from supplied classes.json. Checkpoint architecture, "
            "preprocessing and class-index order are not yet verified."
        ),
    }
