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

# Rice leaf disease classifier

Tells which of 14 classes a **rice** leaf photo shows: Bacterial Blight, Bacterial Leaf Streak, Bacterial Panicle Blight, Blast, Brown Spot, Dead Heart, Downy Mildew, Healthy, Hispa, Leaf Scald, Narrow Brown Leaf Spot, Neck Blast, Sheath Blight, Tungro.

ConvNeXt-Small (`convnext_small`, from `convnext_small.fb_in22k_ft_in1k`), 224 × 224 RGB, 99 MB (half precision).
**93.5 %** test accuracy, macro-F1 0.940, on 1,579 held-out photos. Test photos come from the same datasets the model learned from, so real field photos score lower.

| Class | Recall | F1 | Test photos |
|---|---|---|---|
| Bacterial Blight | 88.0% | 0.926 | 150 |
| Bacterial Leaf Streak | 98.2% | 0.957 | 56 |
| Bacterial Panicle Blight | 100.0% | 0.962 | 50 |
| Blast | 85.3% | 0.880 | 150 |
| Brown Spot | 90.7% | 0.935 | 150 |
| Dead Heart | 98.0% | 0.990 | 150 |
| Downy Mildew | 94.5% | 0.887 | 91 |
| Healthy | 89.3% | 0.896 | 150 |
| Hispa | 89.3% | 0.879 | 150 |
| Leaf Scald | 98.2% | 0.946 | 54 |
| Narrow Brown Leaf Spot | 100.0% | 0.959 | 35 |
| Neck Blast | 100.0% | 1.000 | 144 |
| Sheath Blight | 99.0% | 0.990 | 99 |
| Tungro | 96.7% | 0.948 | 150 |

## Use

```python
import timm, torch
from PIL import Image

model = timm.create_model("hf-hub:ixrbhii/rice-disease-convnext", pretrained=True).eval()
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
