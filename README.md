# Discriminative Project (Study Group 9)

Two tasks over the same **73 object classes** (`OBJ001` … `OBJ073`):

- **Milestone 1 — single-object identification.** Given a 224×224 photo, predict which object it shows. A CNN from scratch plus three ImageNet-pretrained backbones (transfer learning), compared on a held-out test split.
- **Milestone 2 — multi-object detection.** Single-object photos are auto-labelled with one tight box each, concatenated into multi-object **composites** (2–4 objects) and **grid images** (4–25 objects, 2×2 … 5×5), and a **YOLOv8** detector is trained to report the ID **and location** of every object in an image.

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
class_identification.py     # M1: image cleaning, 4-model train/compare/tune, prediction, plots
annotate_single_object.py   # detection step 1: auto-label each single-object photo with one box (YOLOE + DINOv2 + classifier)
generate_dataset.py         # layout engine (make_layout / rectangles) + a standalone 4-class generator
generate_multi_object.py    # detection step 2a: concatenate photos into free-form composites (2–4 objects)
generate_grid_dataset.py    # detection step 2b: concatenate photos into full grids (2×2 … 5×5, 4–25 objects)
train_yolo_detector.py      # detection step 3: train / evaluate / predict with YOLOv8
kaggle_train_grids.py       # train the grid detector on a Kaggle GPU (see note below)
run_live_demo.py            # detect objects in any image(s); saves annotated outputs + predictions.json
requirements.txt            # pinned dependencies
report_group9.pdf           # Milestone-1 report
milestone_2_report_group9.pdf   # Milestone-2 (detection) report
results/                    # M1 figures and CSVs (model checkpoints are gitignored)
```

Each script is **configured by the constants at the top of the file** and run with no arguments
(e.g. `python train_yolo_detector.py`); edit `RUN_MODE` / paths there to change behaviour.

## Data (not in git — download separately)

The datasets and model weights are several GB and exceed GitHub's limits, so they are **not committed**.
Download them from the team's shared Google Drive and place them as below:

```
photo/
├── IE7615 Deep Learning for AI/       # raw 73-class photos (images_OBJ001 … images_OBJ073)
├── single_object_labels/              # per-object boxes produced by annotate_single_object.py
└── multi_object/
    ├── multi_object_det_10000/        # composite detection dataset (2–4 objects)
    └── multi_object_grid/             # grid detection dataset (2×2 … 5×5)
```

> **Drive (raw photos):** https://drive.google.com/drive/folders/1GKLyP17tm5JC0XENv25GeoGgYynm97LS
> The prebuilt `single_object_labels/` and the detection datasets are shared as archives on the same Drive;
> unzip them into `photo/` as shown. Trained weights (`best.pt`) are also shared there.

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

See [`report_group9.pdf`](report_group9.pdf) for the full Milestone-1 write-up and figures.

## Milestone 2 — multi-object detection

**Train / evaluate** (the dataset is already built, so you can train directly):

```bash
python train_yolo_detector.py          # RUN_MODE="all": train, then evaluate on the test split
```

Edit the constants at the top of `train_yolo_detector.py`:
- `YOLO_DIR` — which detection dataset to use (`multi_object_det_10000` or `multi_object_grid`), `IMGSZ`, `BATCH`.
- `RUN_MODE="eval"`, `WEIGHTS="<path>/best.pt"` — evaluate an existing checkpoint on the test split.
- `RUN_MODE="predict"`, `WEIGHTS=...`, `PREDICT_IMAGES=[...]` — detect objects in specific images.

Training writes to `results_yolo/run_<timestamp>/`; the best checkpoint is `weights/best.pt`. The first run downloads `yolov8s.pt` automatically.

> **GPU note.** YOLOv8 training hits a PyTorch backward-pass bug on Apple **MPS**, so **train on a GPU**.
> `kaggle_train_grids.py` runs the grid training on a free Kaggle GPU (upload the grid dataset as a Kaggle
> Dataset, enable GPU + Internet, paste the file, "Save & Run All"). Inference/evaluation works fine on MPS/CPU.

**Live demo — detect objects in any image(s):**

```bash
# set INPUT_PATH at the top of the file (image / folder / glob), then:
python run_live_demo.py
```

It prints every detected **Object ID + box + confidence** per image and saves annotated copies +
`predictions.json` to `live_demo/Run_X/outputs/` (X auto-increments each run).

### Rebuilding the detection datasets from scratch (optional)
1. `python class_identification.py` — trains the M1 classifier (used to score candidate boxes), writing `results_ie7615/run_<ts>/`.
2. Point `CLEAN_RUN` in `annotate_single_object.py` at that run, then `python annotate_single_object.py` — writes `photo/single_object_labels/`.
3. `python generate_multi_object.py` (composites) and/or `python generate_grid_dataset.py` (grids).

### Milestone-2 results

Deployed YOLOv8s detector, scored on the held-out grid and composite test splits. About **87% of objects** are found with the correct ID and a box overlapping the truth at IoU ≥ 0.5; wrong IDs are rare (~0.6%).

| Test set | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Object recall |
|---|---|---|---|---|---|
| Grid images (4–25 objects) | 0.877 | 0.782 | 0.891 | 0.864 | 87.4% |
| Composites (2–4 objects) | 0.871 | 0.781 | 0.872 | 0.863 | 86.5% |

Per-grid object F1 stays flat (0.87–0.88) from 2×2 to 5×5; only whole-image success falls with object count.

**Live demo:** on the 10 grid images sent to the TA, the detector found **146 / 148 objects**.
See [`milestone_2_report_group9.pdf`](milestone_2_report_group9.pdf) for the full write-up, tables and figures.
