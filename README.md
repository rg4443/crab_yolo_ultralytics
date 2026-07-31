![Live green crab detection demo](assets/detection_demo.gif)

A YOLO11-based computer vision pipeline for detecting **invasive European green crabs** (_Carcinus maenas_) underwater, built for the **MATE ROV Global Competition** by Slugbotics.

The repo covers the full loop: raw image collection -> labeling -> augmentation -> dataset versioning -> training -> real-time multi-camera inference on the ROV.

## Why this exists

European green crabs are one of the most damaging invasive species in North American and European coastal estuaries, outcompeting native species and destroying eelgrass beds. MATE ROV mission tasks this pipeline was built for require a submersible vehicle to visually survey a simulated seafloor, distinguish the invasive green crab from visually similar native species (e.g. native rock crab), and report counts/locations, which is exactly what this repo trains a model to do, then runs live on the ROV's cameras.

## Pipeline overview

```
Raw stills (v1_original_images/)
        │
        ▼
rename_images_to_be_labeled.py   →  normalizes filenames for batch labeling
        │
        ▼
rgb.py                           →  forces every image to RGB JPEG (labelImg requires this)
        │
        ▼
Manual labeling (labelImg)        →  YOLO-format .txt boxes per image
        │
        ▼
assign_txt.py                    →  auto-generates whole-image labels for single-subject
                                     reference images (e.g. bulk native rock crab shots),
                                     skipping manual box-drawing where every image is one class
        │
        ▼
generate.py (Albumentations)     →  augments + builds a versioned dataset_v{N}/
                                     (images/ + labels/, train + val splits)
        │
        ▼
train.py                         →  trains YOLO11n on dataset_v{N}/data.yaml
        │
        ▼
inference.py  /  camera_threading.py   →  real-time detection on ROV camera feed(s)
```

## Repo structure

| Path                                                                   | Purpose                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v1_original_images/`                                                  | Raw, unlabeled source images                                                                                                                                                                                                                                                     |
| `dataset/`, `dataset_v2/`, `dataset_v3/`, `dataset_v5/`, `dataset_v6/` | Successive dataset builds (see [Dataset progression](#dataset-progression))                                                                                                                                                                                                      |
| `runs/detect/`                                                         | Ultralytics training run outputs (weights, logs, metrics per run)                                                                                                                                                                                                                |
| `rename_images_to_be_labeled.py`                                       | Renames a batch of raw images to a clean, sequential naming scheme before labeling                                                                                                                                                                                               |
| `rgb.py`                                                               | Converts arbitrary image formats/color modes to RGB `.jpg` so `labelImg` can open them                                                                                                                                                                                           |
| `assign_txt.py`                                                        | Bulk-generates full-frame YOLO labels for images that are already single-subject (avoids re-drawing the same box hundreds of times)                                                                                                                                              |
| `generate.py`                                                          | Albumentations pipeline: rotation, flips, crop, brightness/contrast, color jitter, CLAHE, Gaussian noise, motion blur, coarse dropout. Builds the numbered `dataset_v{N}/` folder from raw `train/`/`val/` folders                                                               |
| `train.py`                                                             | Trains `yolo11n` on a chosen dataset version                                                                                                                                                                                                                                     |
| `inference.py`                                                         | Single-webcam real-time detection + live crab count overlay                                                                                                                                                                                                                      |
| `camera_threading.py`                                                  | Multi-camera (4-feed) threaded capture with per-stream watchdog/auto-recovery, latency telemetry logging, and a stitched mosaic view for the ROV's full camera array (this is a prototype for the ROV's onboard multi-camera view, it is now deprciated from what was deployed.) |
| `multiple_cameras.py`                                                  | Earlier, single-threaded version of the multi-camera viewer reading directly from the ROV's UDP video streams kept as the predecessor to `camera_threading.py`                                                                                                                   |
| `yolo11n.pt`, `yolov8n.pt`                                             | Pretrained base weights used as training starting points                                                                                                                                                                                                                         |

## Dataset progression

The dataset went through six iterations before the model was usable on real ROV footage:

- **`dataset` (v1):** First labeled set. The model badly overfit, it learned to associate _anything white_ in frame (light-colored rocks, sediment, glare) with "crab," because the early training images didn't have enough visual diversity in background and lighting.
- **`dataset_v2` → `dataset_v5`:** Iteratively expanded and rebalanced using new labeled data as it came in, with the Albumentations pipeline in `generate.py` (rotation, crop, brightness/contrast/color jitter, noise, blur, dropout) applied to force the model to key on crab shape/texture rather than background color or lighting.
- **`dataset_v6`:** Current production dataset, used by `train.py`. Training data gets 2 augmented variants per source image; validation data is kept unaugmented (1:1) to give a clean read on real generalization.

## Setup

```bash
git clone https://github.com/rg4443/green-crab-detector.git
cd green-crab-detector
pip install -r requirements.txt
```

> Note: `generate.py` uses `albumentations`, which isn't currently pinned in `requirements.txt`, install it separately if `generate.py` fails on import:
>
> ```bash
> pip install albumentations
> ```

## Usage

**1. Prep new raw images for labeling**

```bash
python rename_images_to_be_labeled.py
python rgb.py
```

Then label with `labelImg` (installed via `requirements.txt`), saving boxes in YOLO format.

**2. Bulk-label single-subject reference images** (optional, for images that are one class per frame)

```bash
python assign_txt.py
```

**3. Build an augmented, versioned dataset**

```bash
python generate.py
```

Edit `DATASET_VERSION_NUM` and the `FOLDERS`/`MULTIPLIERS` config at the top of the script to control the output version and augmentation count.

**4. Train**

```bash
python train.py
```

Trains `yolo11n.pt` on `dataset_v6/data.yaml` for 50 epochs at 640×640. Update the `data=` path to point at a different dataset version if needed.

**5. Run inference**

Single camera, quick check:

```bash
python inference.py
```

Full 4-camera ROV rig (threaded, with auto-recovery and telemetry logging):

```bash
python camera_threading.py
```

## Acknowledgments

Built by the 25-26 Slugbotics software team for the MATE ROV Global Competition.
