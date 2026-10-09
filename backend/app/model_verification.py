"""Model inventory and verification for the AI4Seva artifacts.

This module loads each checkpoint to verify it, but performs **no image
inference** and exposes **no API routes**. ML libraries (torch/timm/torchvision/
safetensors) are imported lazily inside the verify functions so metadata-only
endpoints keep working even if the ML runtime is absent.

Verified facts (Part 3):
- Baseline: timm ``convnext_small`` checkpoints (safetensors, fp16).
- Fine-tuned: **torchvision** ``efficientnet_b0`` checkpoints (``.pth``, fp32).
  The stored ``architecture`` string says ``efficientnet_b0``, but the state-dict
  key layout determines the implementation — it is NOT timm EfficientNet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    BASELINE_MODEL_DIRS,
    FINE_TUNED_MODEL_DIRS,
    PROJECT_ROOT,
    SUPPORTED_CROPS,
)

BASELINE_CROPS: list[str] = list(SUPPORTED_CROPS)
FINE_TUNED_CROPS: list[str] = ["maize", "rice"]


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def runtime_versions() -> dict[str, str | None]:
    """Best-effort version report for the ML runtime (used in the report)."""
    versions: dict[str, str | None] = {}
    for module_name, label in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("safetensors", "safetensors"),
        ("PIL", "pillow"),
    ):
        try:
            module = __import__(module_name)
            versions[label] = getattr(module, "__version__", None)
        except Exception:  # pragma: no cover - runtime may be absent
            versions[label] = None
    return versions


def verify_baseline(crop: str) -> dict[str, Any]:
    """Load and verify a baseline ConvNeXt-Small artifact."""
    model_dir = BASELINE_MODEL_DIRS[crop]
    weights_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    result: dict[str, Any] = {
        "crop": crop,
        "family": "baseline",
        "path": _rel(weights_path),
        "exists": weights_path.is_file(),
    }
    if not weights_path.is_file():
        result["status"] = "missing"
        return result

    cfg = _read_json(config_path)
    if not isinstance(cfg, dict):
        result["status"] = "metadata_unreadable"
        return result

    pretrained = cfg.get("pretrained_cfg") or {}
    result.update(
        {
            "declared_architecture": cfg.get("architecture"),
            "declared_num_classes": cfg.get("num_classes"),
            "label_names": cfg.get("label_names"),
            "input_size": pretrained.get("input_size"),
            "test_input_size": pretrained.get("test_input_size"),
            "mean": pretrained.get("mean"),
            "std": pretrained.get("std"),
        }
    )

    try:
        import timm
        import torch
        from safetensors.torch import load_file

        arch = cfg.get("architecture")
        declared = cfg.get("num_classes")
        model = timm.create_model(arch, num_classes=declared, pretrained=False)
        state_dict = load_file(str(weights_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()
        size = (pretrained.get("input_size") or [3, 224, 224])[-1]
        with torch.no_grad():
            output = model(torch.zeros(1, 3, size, size))

        head = tuple(model.head.fc.weight.shape)
        result.update(
            {
                "implementation": f"timm:{arch}",
                "checkpoint_keys": len(state_dict),
                "head_weight_shape": list(head),
                "actual_num_classes": head[0],
                "output_shape": list(output.shape),
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
                "load_ok": not missing and not unexpected,
            }
        )
        ok = (
            not missing
            and not unexpected
            and head[0] == declared
            and output.shape[1] == declared
        )
        result["status"] = "verified" if ok else "mismatch"
    except Exception as exc:  # pragma: no cover - reported, not hidden
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def verify_fine_tuned(crop: str) -> dict[str, Any]:
    """Load and verify a fine-tuned EfficientNet-B0 artifact."""
    model_dir = FINE_TUNED_MODEL_DIRS.get(crop)
    result: dict[str, Any] = {
        "crop": crop,
        "family": "fine_tuned",
        "exists": bool(model_dir and model_dir.is_dir()),
    }
    if not model_dir or not model_dir.is_dir():
        result["status"] = "absent"
        return result

    checkpoints = sorted(model_dir.glob("*.pth"))
    if not checkpoints:
        result["status"] = "missing"
        return result
    weights_path = checkpoints[0]
    result["path"] = _rel(weights_path)
    classes_json = _read_json(model_dir / "classes.json")
    result["classes_json"] = classes_json

    try:
        import torch
        from torchvision.models import efficientnet_b0

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            result["status"] = "unexpected_format"
            result["checkpoint_top_level_keys"] = (
                list(ckpt.keys()) if isinstance(ckpt, dict) else None
            )
            return result

        state_dict = ckpt["model_state_dict"]
        keys = list(state_dict.keys())
        if any(k.startswith("features.") for k in keys):
            implementation = "torchvision:efficientnet_b0"
        elif any(k.startswith("conv_stem") for k in keys):
            implementation = "timm:efficientnet_b0"
        else:
            implementation = f"unknown (first keys: {keys[:3]})"

        declared = ckpt.get("num_classes")
        model = efficientnet_b0(weights=None, num_classes=declared)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()
        size = ckpt.get("input_size", 224)
        with torch.no_grad():
            output = model(torch.zeros(1, 3, size, size))

        head = tuple(model.classifier[1].weight.shape)
        ckpt_classes = ckpt.get("classes")
        classes_match = ckpt_classes == classes_json
        class_count_matches = (
            isinstance(ckpt_classes, list)
            and len(ckpt_classes) == declared == len(classes_json or [])
        )
        result.update(
            {
                "declared_architecture": ckpt.get("architecture"),
                "implementation": implementation,
                "checkpoint_keys": len(state_dict),
                "declared_num_classes": declared,
                "actual_num_classes": head[0],
                "head_weight_shape": list(head),
                "output_shape": list(output.shape),
                "input_size": [3, size, size],
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
                "load_ok": not missing and not unexpected,
                "classes_match_json": classes_match,
                "class_count_matches": class_count_matches,
                "preprocessing_verified": False,
                "preprocessing_note": (
                    "Checkpoint stores input_size only. torchvision EfficientNet-B0 "
                    "default preprocessing (ImageNet mean/std, resize+center-crop) is "
                    "assumed, not verified."
                ),
                "test_accuracy": ckpt.get("test_accuracy"),
                "macro_f1": ckpt.get("macro_f1", ckpt.get("test_macro_f1")),
                "val_accuracy": ckpt.get("val_accuracy"),
            }
        )
        ok = (
            not missing
            and not unexpected
            and head[0] == declared
            and output.shape[1] == declared
            and classes_match
            and class_count_matches
        )
        result["status"] = "verified" if ok else "mismatch"
    except Exception as exc:  # pragma: no cover - reported, not hidden
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def verify_all() -> list[dict[str, Any]]:
    """Verify the four baselines and the two fine-tuned checkpoints."""
    results = [verify_baseline(crop) for crop in BASELINE_CROPS]
    results += [verify_fine_tuned(crop) for crop in FINE_TUNED_CROPS]
    return results


def build_report() -> dict[str, Any]:
    """Assemble the full verification report (all six artifacts)."""
    models = verify_all()
    by_status: dict[str, int] = {}
    for entry in models:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_versions(),
        "summary": {"total": len(models), "by_status": by_status},
        "models": models,
    }
