# =============================================================================
# YOLOv8 grid multi-object detector — run this in a Kaggle notebook.
# Notebook settings:  Accelerator = GPU (T4 x2 or P100)   |   Internet = ON
# Add Input: the dataset you uploaded from grid_dataset.zip.
# Paste this whole file into one notebook cell and run (or "Save & Run All").
# =============================================================================
import glob
import json
import random
import yaml
from pathlib import Path

import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics==8.4.175"], check=True)
from ultralytics import YOLO

# 1) locate the uploaded dataset — search any depth under /kaggle/input/
hits = glob.glob("/kaggle/input/**/images/train", recursive=True)
assert hits, ("images/train not found. /kaggle/input currently contains: "
              f"{glob.glob('/kaggle/input/*')} — click '+ Add Input' and attach Grid_Dataset, "
              "then re-run (Run > Restart & Run All if needed).")
root = Path(hits[0]).parents[1]       # .../images/train -> dataset root
print("Dataset:", root)

# 2) classes + a data.yaml pointing at the Kaggle path
cls_file = root / "classes.txt"
classes = cls_file.read_text().split() if cls_file.exists() else [f"OBJ{i:03d}" for i in range(1, 74)]
DATA = "/kaggle/working/data.yaml"
yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test",
                "names": {i: c for i, c in enumerate(classes)}}, open(DATA, "w"), sort_keys=False)
print(len(classes), "classes")

# 3) TRAIN. imgsz=896 is the chosen balance; set imgsz=1120 for full "match the largest grid"
#    compliance (then use batch=8). If you hit CUDA out of memory, lower batch to 8.
model = YOLO("yolov8s.pt")
model.train(data=DATA, epochs=50, patience=10, imgsz=896, batch=16, device=0,
            optimizer="auto", cos_lr=True, seed=20261003, plots=True,
            project="/kaggle/working/runs", name="grid")
best = "/kaggle/working/runs/grid/weights/best.pt"

# 4) REQUIRED metrics on the held-out TEST split
m = YOLO(best).val(data=DATA, split="test", imgsz=896, device=0,
                   project="/kaggle/working/runs", name="test_eval")
metrics = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
           "precision": float(m.box.mp), "recall": float(m.box.mr)}
json.dump(metrics, open("/kaggle/working/test_metrics.json", "w"), indent=2)
print("TEST METRICS:", json.dumps(metrics, indent=2))

# 5) example detections for the report (IDs + locations drawn on sample test grids)
imgs = sorted(glob.glob(str(root / "images" / "test" / "*.jpg")))
random.seed(0)
YOLO(best).predict(random.sample(imgs, min(8, len(imgs))), imgsz=896, conf=0.25, save=True,
                   project="/kaggle/working/runs", name="examples")

print("\nDONE. From the notebook Output tab, download:")
print("  best.pt  ->  /kaggle/working/runs/grid/weights/best.pt")
print("  metrics  ->  /kaggle/working/test_metrics.json")
print("  examples ->  /kaggle/working/runs/examples/")
