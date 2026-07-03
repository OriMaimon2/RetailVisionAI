# Synthetic CCTV Supermarket — Multi-Label Person Classifier

End-to-end research pipeline that **generates synthetic CCTV supermarket footage with
diffusion models**, detects people with **pretrained YOLOv8**, crops + letterboxes person
regions, and trains a **multi-label classifier** (10 sigmoid outputs) on the crops.

## Label space (strict, 10 classes, multi-label)

```
right_hand_visible   right_hand_in_pocket   right_hand_in_bag
left_hand_visible    left_hand_in_pocket    left_hand_in_bag
object_in_hand       interacting_with_shelf
hand_occluded_generic                       both_hands_not_visible
```

Each sample can activate several labels simultaneously — the model uses
**sigmoid outputs + `BCEWithLogitsLoss`**, never softmax. The canonical list lives in
[`labels.py`](labels.py); every module imports it from there.

## Pipeline

```
diffusion (SDXL/SD1.5 | OpenAI | Gemini | mock)
    └─> dataset/synthetic_raw/          ≥100 images per label + metadata
augmentation (albumentations + CCTV effects: fisheye, timestamp overlay)
    └─> dataset/augmented/
YOLOv8 person detection ─> box expand 20% ─> crop ─> letterbox 224×224
    └─> dataset/cropped/ + dataset/annotations.json
train  (ResNet18 / EfficientNet-B0, BCEWithLogitsLoss, pos_weight, fixed seeds)
    └─> model/checkpoints/best.pt + model/training_log.json
infer  (frame -> person boxes -> per-person label probabilities)
```

## Quickstart

```bash
pip install -r requirements.txt

# Fast end-to-end smoke test (no GPU needed — procedural mock backend):
python main_pipeline.py --stage all --backend mock --images-per-label 10 --epochs 3

# Full run with local Stable Diffusion (GPU recommended):
python main_pipeline.py --stage all
```

Individual stages: `--stage generate | augment | crop | train | evaluate | infer`.
Each stage is also runnable standalone, e.g. `python diffusion/dataset_generator.py`,
`python model/train.py`, `python model/inference.py --image path/to/frame.png`.

### Backends

| backend | requirement | notes |
|---|---|---|
| `sd` (default) | GPU + `diffusers` | SDXL, auto-fallback to SD 1.5; `offline: true` for local-files-only |
| `openai` | `OPENAI_API_KEY` + `pip install openai` | Images API fallback |
| `gemini` | `GEMINI_API_KEY` + `pip install google-genai` | Imagen fallback |
| `mock` | nothing | procedural CCTV-style placeholders for smoke tests / CI |

## Configuration

- [`configs/diffusion.yaml`](configs/diffusion.yaml) — backend, model, images per label (100), seeds, augmentation.
- [`configs/training.yaml`](configs/training.yaml) — backbone, epochs, lr, YOLO/crop settings, threshold.

## Data format

`dataset/annotations.json` (produced by the crop stage) — one entry per training sample:

```json
{
  "image_path": "dataset/cropped/right_hand_in_bag_0001_crop.jpg",
  "labels": {"right_hand_in_bag": 1, "left_hand_visible": 1, "...": 0},
  "source_image": "dataset/synthetic_raw/right_hand_in_bag/right_hand_in_bag_0001.png",
  "box": [412, 180, 690, 720],
  "det_conf": 0.87,
  "fallback_full_image": false
}
```

Generation metadata (prompt, seed, backend, diversity attributes) is logged per image in
`dataset/synthetic_annotations.json`.

## Reproducibility

Single `seed` per config fixes python/numpy/torch RNGs, derives deterministic per-image
diffusion seeds, per-variant augmentation seeds, the train/val split and DataLoader
workers. Horizontal-flip augmentation automatically **swaps left/right hand labels**.

## Google Colab

[`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) runs the entire
pipeline (8 sections: install → load diffusion → generate → augment → detect → crop →
train → inference demo). Defaults to a small demo scale; set `IMAGES_PER_LABEL = 100`
for the full dataset.

## Optional extensions included

- **ByteTrack tracking** — `yolo/track_people.py` (`PersonTracker`, `track_video`).
- **Temporal smoothing** — `TemporalLabelSmoother` (per-track probability averaging).

## Repository layout

```
project/
├── labels.py                  # canonical 10-label space (STRICT)
├── common.py                  # paths, config, seeding, logging
├── main_pipeline.py           # stage orchestrator
├── diffusion/                 # prompt generation + SD / OpenAI / Gemini / mock backends
├── augmentation/              # albumentations pipeline + fisheye/timestamp CCTV effects
├── yolo/                      # YOLOv8 detection + ByteTrack tracking
├── preprocessing/             # letterbox + 20%-expanded person cropping
├── model/                     # classifier, training, inference
├── configs/                   # diffusion.yaml, training.yaml
├── dataset/                   # synthetic_raw/ augmented/ cropped/ annotations.json
└── notebooks/colab_pipeline.ipynb
```
