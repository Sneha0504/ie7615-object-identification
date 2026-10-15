"""Live demo — detect the IDs and locations of all objects in any image(s).

Works on any image: the grid images sent to the TA, a single multi-object photo,
a folder of images, or a wildcard. For each image it prints every detected Object ID
with its box and confidence, and saves an annotated copy.

HOW TO USE
    1. Set INPUT_PATH below to an image, a folder, or a glob, e.g.
         INPUT_PATH = "photo/grid_images/*.jpg"
         INPUT_PATH = "some_folder_of_test_images"
         INPUT_PATH = "one_image.jpg"
    2. Run:  python run_live_demo.py
    3. Results are written to  live_demo/Run_X/outputs/  (X auto-increments each run):
         <name>_pred.jpg   annotated image (boxes + Object ID + confidence)
         predictions.json  machine-readable detections for every image
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Raise the NMS time budget so dense grids (5x5 = 25 objects) are never truncated on slow devices.
import ultralytics.utils.nms as _nms
_orig_nms = _nms.non_max_suppression
_nms.non_max_suppression = lambda *a, **k: _orig_nms(*a, **{**k, "max_time_img": 8.0})
from ultralytics import YOLO

# ------------------------------------------------------------------ settings
INPUT_PATH = ""                                   # <-- set me (image / folder / glob)
WEIGHTS = "grid_outputs/grid/weights/best.pt"     # trained detector
IMGSZ = 896                                       # match training; keeps dense-grid objects large
CONF = 0.25                                        # confidence threshold
IOU = 0.5                                           # NMS IoU
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE / "live_demo"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def next_run_dir():
    """live_demo/Run_X/outputs with X = highest existing run number + 1."""
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_")[1]) for p in DEMO_ROOT.glob("Run_*")
            if "_" in p.name and p.name.split("_")[1].isdigit()]
    run = max(nums, default=0) + 1
    out = DEMO_ROOT / f"Run_{run}" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return run, out


def expand_images(spec):
    p = Path(spec).expanduser()
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    if any(c in str(p) for c in "*?["):
        return sorted(Path(f) for f in glob.glob(str(p)) if Path(f).suffix.lower() in IMAGE_EXTS)
    return [p] if p.is_file() and p.suffix.lower() in IMAGE_EXTS else []


def main():
    if not INPUT_PATH:
        raise SystemExit('Set INPUT_PATH at the top of this file first, e.g. '
                         'INPUT_PATH = "photo/grid_images/*.jpg"')
    images = expand_images(INPUT_PATH)
    if not images:
        raise SystemExit(f"No images found at: {INPUT_PATH!r}")
    weights = Path(WEIGHTS)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}. Point WEIGHTS at the trained best.pt.")

    device = pick_device()
    run, out = next_run_dir()
    print(f"Run {run}  |  device {device}  |  {len(images)} image(s)  |  outputs -> {out}", flush=True)

    model = YOLO(str(weights))
    names = model.names
    all_results, total = {}, 0
    for path in images:
        r = model.predict(str(path), imgsz=IMGSZ, conf=CONF, iou=IOU, agnostic_nms=True,
                          device=device, verbose=False)[0]
        dets = [{"id": names[int(c)], "confidence": round(float(s), 3),
                 "box_xyxy": [round(float(v), 1) for v in b]}
                for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        dets.sort(key=lambda d: (d["box_xyxy"][1], d["box_xyxy"][0]))     # top->bottom, left->right
        print(f"\n{path.name}: {len(dets)} object(s)")
        for d in dets:
            x0, y0, x1, y1 = d["box_xyxy"]
            print(f"  {d['id']:<8} conf={d['confidence']:.3f}  box=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
        Image.fromarray(r.plot()[:, :, ::-1]).save(out / f"{path.stem}_pred.jpg", quality=92)
        all_results[path.name] = dets
        total += len(dets)

    (out / "predictions.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nDone. {len(images)} image(s), {total} object(s) detected.")
    print(f"Annotated images + predictions.json -> {out}")


if __name__ == "__main__":
    main()
