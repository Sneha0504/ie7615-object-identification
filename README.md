# Discriminative Project (Study Group 9)

Two milestones over the same **73 object classes** (`OBJ001` … `OBJ073`):

- **Milestone 1 — single-object identification.** Given a 224×224 photo, predict which object it shows. A CNN from scratch plus three ImageNet-pretrained backbones (transfer learning), compared on a held-out test split.
- **Milestone 2 — multi-object detection.** Single-object photos are auto-labelled with one tight box each, concatenated into multi-object composites, and a **YOLOv8** detector is trained to report the ID **and location** of every object in an image.

**Repository:** https://github.com/Sneha0504/ie7615-object-identification

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt       # numpy stays < 2 on purpose — see the file's note
```

Then download the data (below) and run the step you want. Device is picked automatically: **CUDA → Apple MPS → CPU**.

## What is in this repository

```
class_identification.py    # M1: image cleaning, 4-model train/compare/tune, prediction, plots
annotate_single_object.py  # M2 step 1: auto-label each single-object photo with one box (YOLOE + DINOv2 + classifier)
generate_dataset.py        # M2 layout engine (make_layout / rectangles) + a standalone 4-class generator
generate_multi_object.py   # M2 step 2: concatenate single-object photos into the YOLO detection dataset
train_yolo_detector.py     # M2 step 3: train / evaluate / predict with YOLOv8
requirements.txt           # pinned dependencies
report_group9.pdf          # project report
results/                   # M1 figures and CSVs (model checkpoints are gitignored)
```

Every script is **configured by the constants at the top of the file** and run with no arguments
(e.g. `python train_yolo_detector.py`); edit `RUN_MODE` / paths there to change behaviour.

## Data (not in git — download separately)

The datasets and model weights are several GB and exceed GitHub's limits, so they are **not committed**.
Download them from the team's shared Google Drive and place them exactly as below:

```
photo/
├── IE7615 Deep Learning for AI/      # raw 73-class photos (images_OBJ001 … images_OBJ073)
├── single_object_labels/             # per-object boxes produced by annotate_single_object.py
└── multi_object/
    └── multi_object_det_10000/       # the YOLO detection dataset (images/, labels/, data.yaml, …)
```

> **Drive:** raw photos — https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS
> The prebuilt `single_object_labels/` and `multi_object_det_10000/` datasets are shared as archives on
> the same Drive; unzip them into `photo/` as shown above. (They cannot be regenerated without the
> Milestone-1 classifier run, so download the prebuilt versions to train the detector directly.)

## Milestone 2 — train the detector (the main deliverable)

The detection dataset is already built, so you only need to train:

```bash
python train_yolo_detector.py          # RUN_MODE="all": train, then evaluate on the test split
```

Edit the constants at the top of `train_yolo_detector.py` to change modes:
- `RUN_MODE="predict"`, `WEIGHTS="<path>/best.pt"`, `PREDICT_IMAGES=["img.jpg", ...]` — detect objects in new
  images; prints each object's **ID + box (x0,y0,x1,y1) + confidence** and saves annotated images.
- `RUN_MODE="eval"` — evaluate an existing `best.pt` on the multi-object test split.

Training writes to `results_yolo/run_<timestamp>/`; the best checkpoint is `weights/best.pt`.
On the first run it downloads `yolov8s.pt` automatically.

### Rebuilding the dataset from scratch (optional)
1. `python class_identification.py` — trains the M1 classifier, writing a run to `results_ie7615/run_<ts>/`.
2. Point `CLEAN_RUN` in `annotate_single_object.py` at that run, then `python annotate_single_object.py` — writes `photo/single_object_labels/`.
3. `python generate_multi_object.py` — writes `photo/multi_object/multi_object_det_10000/`.

## Milestone 1 — single-object classifier

```bash
python class_identification.py          # runs the configured pipeline end-to-end
```

Models (`custom_cnn`, `resnet18`, `efficientnet_b0`, `swin_t`) and switches
(`RUN_CNN_SEARCH`, `RUN_BEST_MODEL_SEARCH`, `EPOCHS_OVERRIDE`, `RESUME_DIR`) are set at the top of the file.
Outputs go to `results_ie7615/run_<timestamp>/` (comparison CSVs, curves, confusion matrices, checkpoints).

### Milestone-1 results

Held-out test set (73 classes, same stratified split for every model), ranked by validation accuracy.

| Model | Val acc | Test acc | Test F1 | Params (M) | Latency (ms) | Best epoch |
|---|---|---|---|---|---|---|
| **Swin-Tiny** | **0.962** | **0.948** | **0.949** | 27.6 | 29.1 | 7 |
| ResNet-18 | 0.945 | 0.936 | 0.936 | 11.2 | 10.3 | 14 |
| Custom CNN | 0.799 | 0.789 | 0.788 | 1.6 | 7.8 | 39 |

See [`report_group9.pdf`](report_group9.pdf) for the full write-up and all figures.
