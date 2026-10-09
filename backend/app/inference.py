"""Architecture-independent inference interface + model registry.

Two adapter families are implemented behind one interface:

- **baseline** — timm ConvNeXt-Small (all four crops), active.
- **fine_tuned** — torchvision EfficientNet-B0 (maize, rice), active *internally*
  but gated in the public API because the training preprocessing is not recorded
  in the supplied artifacts (see ``preprocessing.build_efficientnet_assumed_transform``).

Design:
- Adapters expose ``predict_proba(image) -> np.ndarray`` and metadata.
- ``get_baseline_adapter`` / ``get_fine_tuned_adapter`` lazily build and cache
  adapters in a bounded LRU.
- ``predict`` turns a probability vector into the public response payload.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    BASELINE_MODEL_DIRS,
    CROP_DISPLAY_NAMES,
    FINE_TUNED_MODEL_DIRS,
    PROJECT_ROOT,
    SUPPORTED_CROPS,
    settings,
)
from .errors import ApiError
from .preprocessing import build_efficientnet_assumed_transform, build_transform

DISCLAIMER = (
    "Confidence is the model's softmax probability for the predicted class; "
    "it is not a guarantee that the diagnosis is correct."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _is_torchvision_efficientnet_key(key: str) -> bool:
    """True for a torchvision EfficientNet-B0 parameter name (not a timm head)."""
    return key.startswith("features.") or key.startswith("classifier.")


class ClassifierAdapter(ABC):
    """Common interface for crop-disease classifiers."""

    crop: str
    family: str
    implementation: str
    labels: list[str]
    num_classes: int
    input_size: list[int]
    weights: str

    @abstractmethod
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """Return a length-``num_classes`` probability vector (sums to 1)."""


class BaselineConvNeXtAdapter(ClassifierAdapter):
    """timm ConvNeXt-Small baseline loaded from a safetensors checkpoint."""

    def __init__(self, crop: str) -> None:
        import timm
        import torch  # noqa: F401  (imported for availability/side effects)
        from safetensors.torch import load_file

        model_dir = BASELINE_MODEL_DIRS[crop]
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"missing baseline artifact for {crop}")

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        pretrained_cfg = cfg.get("pretrained_cfg") or {}

        self.crop = crop
        self.family = "baseline"
        self.implementation = f"timm:{cfg['architecture']}"
        self.labels = list(cfg["label_names"])
        self.num_classes = int(cfg["num_classes"])
        self.input_size = list(pretrained_cfg.get("input_size") or [3, 224, 224])
        self.weights = _rel(weights_path)
        self.label_source = _rel(config_path)
        self.transform = build_transform(pretrained_cfg)

        if len(self.labels) != self.num_classes:
            raise ValueError("label count does not match num_classes")

        self._torch = torch
        model = timm.create_model(
            cfg["architecture"], num_classes=self.num_classes, pretrained=False
        )
        state_dict = load_file(str(weights_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError("checkpoint does not match architecture")
        model.eval()
        self.model = model

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)
        with self._torch.inference_mode():
            logits = self.model(tensor)
            probs = self._torch.softmax(logits, dim=1)
        return probs[0].cpu().numpy()


class TorchvisionEfficientNetAdapter(ClassifierAdapter):
    """Fine-tuned torchvision EfficientNet-B0 loaded from a ``.pth`` checkpoint.

    The checkpoint carries its own class list, so labels are model-specific and
    never reuse the baseline ordering. The training preprocessing is **not**
    recorded in the supplied artifacts; ``build_efficientnet_assumed_transform``
    encodes an ImageNet-style assumption. ``preprocessing_verified`` is therefore
    ``False`` and the public API keeps fine-tuned predictions gated until the
    training pipeline is confirmed.
    """

    def __init__(self, crop: str) -> None:
        import torch
        from torchvision.models import efficientnet_b0

        model_dir = FINE_TUNED_MODEL_DIRS.get(crop)
        if model_dir is None or not model_dir.is_dir():
            raise FileNotFoundError(f"no fine-tuned artifact for {crop}")
        checkpoint_paths = sorted(model_dir.glob("*.pth"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"no fine-tuned checkpoint for {crop}")
        checkpoint_path = checkpoint_paths[0]

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("unexpected checkpoint format")
        state_dict = checkpoint["model_state_dict"]
        keys = list(state_dict)
        if not keys or not all(_is_torchvision_efficientnet_key(k) for k in keys):
            raise ValueError(
                "checkpoint is not a torchvision EfficientNet-B0 state dict"
            )
        if checkpoint.get("architecture") not in (None, "efficientnet_b0"):
            raise ValueError("checkpoint architecture is not efficientnet_b0")

        num_classes = int(checkpoint["num_classes"])
        classes_json = json.loads(
            (model_dir / "classes.json").read_text(encoding="utf-8")
        )
        labels = list(checkpoint.get("classes") or classes_json)
        if len(labels) != num_classes:
            raise ValueError("label count does not match num_classes")
        if list(classes_json) != labels:
            raise ValueError("checkpoint classes disagree with classes.json")

        input_size = int(checkpoint.get("input_size") or 224)

        self.crop = crop
        self.family = "fine_tuned"
        self.implementation = "torchvision:efficientnet_b0"
        self.labels = labels
        self.num_classes = num_classes
        self.input_size = [3, input_size, input_size]
        self.weights = _rel(checkpoint_path)
        self.label_source = _rel(model_dir / "classes.json")
        self.transform = build_efficientnet_assumed_transform()
        self.preprocessing_verified = False
        self.preprocessing_note = (
            "Training preprocessing (resize/normalization) is not recorded in the "
            "supplied artifacts; an ImageNet-style transform is assumed."
        )
        # Checkpoint-embedded, training self-reported metrics (not re-evaluated).
        self.checkpoint_metrics = {
            key: checkpoint[key]
            for key in (
                "val_accuracy",
                "test_accuracy",
                "macro_f1",
                "test_macro_f1",
                "weighted_f1",
                "test_weighted_f1",
            )
            if key in checkpoint
        }

        self._torch = torch
        model = efficientnet_b0(weights=None, num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError("checkpoint does not match architecture")
        model.eval()
        self.model = model

    def predict_proba(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image).unsqueeze(0)
        with self._torch.inference_mode():
            logits = self.model(tensor)
            probs = self._torch.softmax(logits, dim=1)
        return probs[0].cpu().numpy()


@lru_cache(maxsize=settings.model_cache_size)
def _load_baseline(crop: str) -> ClassifierAdapter:
    return BaselineConvNeXtAdapter(crop)


@lru_cache(maxsize=settings.model_cache_size)
def _load_fine_tuned(crop: str) -> ClassifierAdapter:
    return TorchvisionEfficientNetAdapter(crop)


def get_baseline_adapter(crop: str) -> ClassifierAdapter:
    """Return a cached baseline adapter, mapping load failures to ApiError."""
    try:
        return _load_baseline(crop)
    except ApiError:
        raise
    except FileNotFoundError:
        raise ApiError(
            503,
            "model_unavailable",
            f"The baseline model for '{crop}' is not available.",
        )
    except Exception:
        raise ApiError(
            503,
            "model_load_failed",
            f"The baseline model for '{crop}' could not be loaded.",
        )


def get_fine_tuned_adapter(crop: str) -> ClassifierAdapter:
    """Return a cached fine-tuned adapter, mapping load failures to ApiError."""
    try:
        return _load_fine_tuned(crop)
    except ApiError:
        raise
    except FileNotFoundError:
        raise ApiError(
            503,
            "model_unavailable",
            f"The fine-tuned model for '{crop}' is not available.",
        )
    except Exception:
        raise ApiError(
            503,
            "model_load_failed",
            f"The fine-tuned model for '{crop}' could not be loaded.",
        )


def clear_cache() -> None:
    """Clear the adapter caches (used by tests)."""
    _load_baseline.cache_clear()
    _load_fine_tuned.cache_clear()


def _resolve_adapter(crop: str, family: str) -> ClassifierAdapter:
    if family == "baseline":
        return get_baseline_adapter(crop)
    if family == "fine_tuned":
        return get_fine_tuned_adapter(crop)
    raise ApiError(400, "unsupported_family", f"Unsupported model family '{family}'.")


def predict(
    crop: str, image: Image.Image, *, family: str = "baseline"
) -> dict[str, Any]:
    """Run inference and build the public prediction payload."""
    if crop not in SUPPORTED_CROPS:
        raise ApiError(
            400,
            "unsupported_crop",
            f"Unsupported crop '{crop}'. Supported crops: "
            + ", ".join(SUPPORTED_CROPS)
            + ".",
        )

    adapter = _resolve_adapter(crop, family)

    try:
        probs = np.asarray(adapter.predict_proba(image), dtype=np.float64).reshape(-1)
    except ApiError:
        raise
    except Exception:
        raise ApiError(
            500, "inference_failed", "The model failed to produce a prediction."
        )

    if probs.shape[0] != adapter.num_classes:
        raise ApiError(
            500, "inference_failed", "The model produced an invalid output."
        )
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        raise ApiError(
            500, "inference_failed", "The model produced an invalid output."
        )
    probs = probs / probs.sum()

    order = np.argsort(probs)[::-1]
    best = int(order[0])
    probabilities = [
        {
            "index": int(i),
            "label": adapter.labels[int(i)],
            "probability": float(probs[int(i)]),
        }
        for i in order
    ]

    return {
        "crop": crop,
        "display_name": CROP_DISPLAY_NAMES[crop],
        "model": {
            "crop": crop,
            "family": adapter.family,
            "implementation": adapter.implementation,
            "num_classes": adapter.num_classes,
            "input_size": adapter.input_size,
            "weights": adapter.weights,
            "label_source": getattr(adapter, "label_source", adapter.weights),
        },
        "predicted_index": best,
        "predicted_label": adapter.labels[best],
        "confidence": float(probs[best]),
        "probabilities": probabilities,
        "disclaimer": DISCLAIMER,
    }
