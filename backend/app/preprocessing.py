"""Build image transforms from a model's own verified metadata.

For the baselines the relevant values come from the supplied ``config.json``
(``pretrained_cfg``): ``input_size``, ``crop_pct``, ``interpolation``, ``mean``,
``std``. The pipeline mirrors timm's eval transform: resize the shorter side to
``round(size / crop_pct)`` then centre-crop to ``size``, then ToTensor and
Normalize. With the supplied ``crop_pct == 1.0`` this is a plain square resize.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

_INTERPOLATION: dict[str, transforms.InterpolationMode] = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
    "nearest-exact": transforms.InterpolationMode.NEAREST_EXACT,
    "lanczos": transforms.InterpolationMode.LANCZOS,
}

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def build_transform(pretrained_cfg: dict[str, Any]) -> transforms.Compose:
    """Return the eval transform described by ``pretrained_cfg``."""
    input_size = pretrained_cfg.get("input_size") or [3, 224, 224]
    size = int(input_size[-1])

    crop_pct = pretrained_cfg.get("crop_pct")
    crop_pct = float(crop_pct) if crop_pct else 1.0
    resize_size = int(round(size / crop_pct)) if crop_pct else size

    interp_name = str(pretrained_cfg.get("interpolation", "bilinear")).lower()
    interpolation = _INTERPOLATION.get(
        interp_name, transforms.InterpolationMode.BILINEAR
    )

    mean = pretrained_cfg.get("mean") or _DEFAULT_MEAN
    std = pretrained_cfg.get("std") or _DEFAULT_STD

    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# --- Fine-tuned EfficientNet-B0: ASSUMED, NOT VERIFIED -----------------------
# The supplied fine-tuned checkpoints record only architecture / classes /
# input_size (224). They do NOT record the training resize/interpolation or the
# normalization mean/std, and no training script or notebook ships with the
# repository. The config below encodes the *torchvision ImageNet* convention
# (resize shorter side to 256 -> centre-crop 224 -> ImageNet normalisation).
# This is an assumption, not a verified fact; the public API keeps fine-tuned
# predictions disabled until the real training pipeline is confirmed.
EFFICIENTNET_ASSUMED_CFG: dict[str, Any] = {
    "input_size": [3, 224, 224],
    "crop_pct": 224 / 256,
    "interpolation": "bilinear",
    "mean": _DEFAULT_MEAN,
    "std": _DEFAULT_STD,
}


def build_efficientnet_assumed_transform() -> transforms.Compose:
    """Return the *assumed* EfficientNet-B0 eval transform (see note above)."""
    return build_transform(EFFICIENTNET_ASSUMED_CFG)
