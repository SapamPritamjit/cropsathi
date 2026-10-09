---
license: cc-by-4.0
library_name: timm
pipeline_tag: image-classification
tags:
- agriculture
- plant-disease
- convnext
- image-classification
- timm
---

# Maize leaf disease classifier

Tells which of 11 classes a **maize** leaf photo shows: Common Rust, Fall Armyworm, Grasshopper Damage, Gray Leaf Spot, Healthy, Holcus Leaf Spot, Leaf Beetle, Lethal Necrosis, Streak Virus, Northern Leaf Blight, Smut.

ConvNeXt-Small (`convnext_small`, from `convnext_small.fb_in22k_ft_in1k`), 224 × 224 RGB, 99 MB (half precision).
**92.9 %** test accuracy, macro-F1 0.936, on 1,281 held-out photos. Test photos come from the same datasets the model learned from, so real field photos score lower.

| Class | Recall | F1 | Test photos |
|---|---|---|---|
| Common Rust | 96.0% | 0.980 | 150 |
| Fall Armyworm | 100.0% | 0.955 | 42 |
| Grasshopper Damage | 99.0% | 0.985 | 100 |
| Gray Leaf Spot | 80.0% | 0.787 | 150 |
| Healthy | 96.7% | 0.964 | 150 |
| Holcus Leaf Spot | 100.0% | 0.966 | 70 |
| Leaf Beetle | 97.8% | 0.986 | 139 |
| Lethal Necrosis | 97.3% | 0.983 | 150 |
| Streak Virus | 90.7% | 0.913 | 150 |
| Northern Leaf Blight | 81.3% | 0.830 | 150 |
| Smut | 100.0% | 0.952 | 30 |

## Use

```python
import timm, torch
from PIL import Image

model = timm.create_model("hf-hub:ixrbhii/maize-disease-convnext", pretrained=True).eval()
cfg = timm.data.resolve_data_config({}, model=model)
x = timm.data.create_transform(**cfg)(Image.open("leaf.jpg").convert("RGB")).unsqueeze(0)
p = model(x).softmax(-1)[0]
labels = model.pretrained_cfg["label_names"]
for i in p.argsort(descending=True)[:3]:
    print(labels[i], f"{100 * p[i]:.1f} %")
```

The training images were resized to 224 × 224 without cropping; timm's transform above (resize + 224 centre crop) gives the
same result for square photos. `label_descriptions` in config.json maps each name to the dataset's original label.
Best results: one leaf filling most of the photo, in daylight and in focus. Below about 45 % confidence, ask for a better photo.

**In an app:** [LULC Fetch](https://github.com/agritechixr/lulc-fetch) (Agri ▸ Diagnose crop disease) recognises the crop first
and then runs the right disease model, with a photo-quality check and a disease map from geotagged photos.

## Related

- All 44 models (2 crop detectors, 42 crops) as separate repositories: [the collection](https://huggingface.co/collections/ixrbhii/multi-crop-disease-models-42-crops-6ac0a3291f2153fa716e2993)
- All 44 models in one repository: [ixrbhii/multicrop-disease-models](https://huggingface.co/ixrbhii/multicrop-disease-models)
- Symptoms, treatment and pests for each disease: [ixrbhii/crop-disease-qa](https://huggingface.co/datasets/ixrbhii/crop-disease-qa)

## Training data and licence

Trained on [LeafNet](https://huggingface.co/datasets/enalis/LeafNet) (CC-BY-4.0) and the authors' own field photos. Released under CC-BY-4.0: please credit "Multi-crop disease models, IXR (agritechixr)" and the
training datasets. A diagnosis supports, but doesn't replace, a local agriculture expert.
