"""Build grid multi-object composites (2x2 ... 5x5) for YOLOv8 detection.

Reuses the per-object tight boxes already in photo/single_object_labels/boxes.csv
(created by annotate_single_object.py). Each grid cell is one 224x224 single-object
photo placed side by side at the sizes required by Milestone 2:

    grid   objects   image size
    2x2      4        448 x 448
    3x3      9        672 x 672
    4x4     16        896 x 896
    5x4     20       1120 x 896
    5x5     25       1120 x 1120

Each tile's tight box is mapped into the composite as a YOLO label. Tiles for a
composite are drawn ONLY from that composite's split pool (train/val/test), so no
source photo crosses splits. Augmentation: per-tile horizontal flip (box flipped
too) + whole-image brightness/contrast jitter; YOLO adds mosaic/HSV at train time.
"""
import json
import random
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
LABELS = HERE / "photo" / "single_object_labels"
SIZE = 224
PER_GRID = 715                      # per grid size; at 70/15/15 this gives ~2500 train images
GRIDS = {"2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4), "5x4": (4, 5), "5x5": (5, 5)}  # (rows, cols)
SPLIT_FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
FLIP_PROB = 0.5
AUG_PROB = 0.45
JPEG_QUALITY = 92
VIS_PER_GRID = 3
SEED = 20261013
OUT = HERE / "photo" / "multi_object" / "multi_object_grid"


class Deck:
    """Draw-without-replacement over items, reshuffling when empty (class balance)."""
    def __init__(self, items, rng):
        self.items, self.rng, self.q = list(items), rng, deque()

    def draw(self, exclude=()):
        for _ in range(len(self.items) * 2 + 10):
            if not self.q:
                refill = list(self.items)
                self.rng.shuffle(refill)
                self.q.extend(refill)
            item = self.q.popleft()
            if item not in exclude:
                return item
            self.q.append(item)
        raise ValueError("deck cannot satisfy exclude set")


def main():
    rng = random.Random(SEED)
    classes = (LABELS / "classes.txt").read_text(encoding="utf-8").split()
    df = pd.read_csv(LABELS / "boxes.csv", encoding="utf-8-sig")
    obj = df[(~df["is_background"].astype(bool)) & (df["use"].astype(bool))].copy()
    obj = obj.dropna(subset=["x0", "y0", "x1", "y1"])

    pools = {s: defaultdict(list) for s in SPLIT_FRAC}
    for r in obj.to_dict("records"):
        if r["split"] in pools:
            pools[r["split"]][int(r["class_id"])].append(r)

    max_tiles = max(rows * cols for rows, cols in GRIDS.values())
    for s in SPLIT_FRAC:
        avail = [c for c in range(len(classes)) if pools[s][c]]
        if len(avail) < max_tiles:
            raise SystemExit(f"split {s}: only {len(avail)} classes have usable photos, need {max_tiles}")

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images", "labels"):
        for s in SPLIT_FRAC:
            (OUT / sub / s).mkdir(parents=True, exist_ok=True)
    (OUT / "vis").mkdir(exist_ok=True)
    font = ImageFont.load_default(size=14)

    gt_rows, records, seen = [], [], set()
    img_id = 0
    for name, (rows, cols) in GRIDS.items():
        n, W, H = rows * cols, cols * SIZE, rows * SIZE
        vis_saved = 0
        for split, frac in SPLIT_FRAC.items():
            class_deck = Deck([c for c in range(len(classes)) if pools[split][c]], rng)
            src_decks = {c: Deck(range(len(pools[split][c])), rng)
                         for c in range(len(classes)) if pools[split][c]}
            for _ in range(round(PER_GRID * frac)):
                for _attempt in range(200):
                    cids = []
                    for _ in range(n):
                        cids.append(class_deck.draw(exclude=cids))
                    picks = [pools[split][c][src_decks[c].draw()] for c in cids]
                    key = tuple(sorted(p["image"] for p in picks))
                    if key not in seen:
                        seen.add(key)
                        break
                else:
                    raise RuntimeError(f"{name}/{split}: no new combination after 200 tries")

                img_id += 1
                stem = f"grid_{img_id:05d}"
                canvas = Image.new("RGB", (W, H), (255, 255, 255))
                lines, boxes = [], []
                for i, p in enumerate(picks):
                    r_, c_ = divmod(i, cols)
                    ox, oy = c_ * SIZE, r_ * SIZE
                    flip = rng.random() < FLIP_PROB
                    with Image.open(LABELS / p["image"]) as im:
                        im = im.convert("RGB")
                        if im.size != (SIZE, SIZE):
                            im = ImageOps.fit(im, (SIZE, SIZE), Image.LANCZOS)
                        if flip:
                            im = ImageOps.mirror(im)
                        canvas.paste(im, (ox, oy))
                    x0, y0, x1, y1 = (float(p["x0"]), float(p["y0"]), float(p["x1"]), float(p["y1"]))
                    if flip:
                        x0, x1 = SIZE - x1, SIZE - x0
                    gx0, gy0, gx1, gy1 = ox + x0, oy + y0, ox + x1, oy + y1
                    cid = int(p["class_id"])
                    lines.append(f"{cid} {(gx0 + gx1) / 2 / W:.6f} {(gy0 + gy1) / 2 / H:.6f} "
                                 f"{(gx1 - gx0) / W:.6f} {(gy1 - gy0) / H:.6f}")
                    gt_rows.append(dict(image=f"{stem}.jpg", split=split, grid=name, class_id=cid,
                                        class_name=classes[cid], x0=round(gx0, 1), y0=round(gy0, 1),
                                        x1=round(gx1, 1), y1=round(gy1, 1), source=p["image"]))
                    boxes.append((gx0, gy0, gx1, gy1, classes[cid]))

                if rng.random() < AUG_PROB:
                    canvas = ImageEnhance.Brightness(canvas).enhance(round(rng.uniform(0.94, 1.06), 4))
                    canvas = ImageEnhance.Contrast(canvas).enhance(round(rng.uniform(0.94, 1.06), 4))
                canvas.save(OUT / "images" / split / f"{stem}.jpg", quality=JPEG_QUALITY)
                (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                records.append(dict(id=img_id, grid=name, split=split, objects=n, width=W, height=H))

                if vis_saved < VIS_PER_GRID:
                    vis = canvas.copy()
                    d = ImageDraw.Draw(vis)
                    for bx0, by0, bx1, by1, cn in boxes:
                        d.rectangle([bx0, by0, bx1, by1], outline=(230, 40, 40), width=3)
                        d.text((bx0 + 2, by0 + 2), cn, fill=(255, 255, 0), font=font)
                    vis.save(OUT / "vis" / f"{stem}_{name}.jpg", quality=88)
                    vis_saved += 1
        print(f"  {name}: {sum(1 for r in records if r['grid'] == name)} images ({W}x{H}, {n} objects each)",
              flush=True)

    gt = pd.DataFrame(gt_rows)
    gt.to_csv(OUT / "gt_boxes.csv", index=False, encoding="utf-8-sig")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   f"test: images/test\nnames:\n{names}\n", encoding="utf-8")
    (OUT / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    occ = Counter(gt["class_name"])
    summary = {
        "image_count": len(records), "seed": SEED, "object_boxes": len(gt),
        "grids": {name: f"{cols*SIZE}x{rows*SIZE}, {rows*cols} objects" for name, (rows, cols) in GRIDS.items()},
        "images_per_grid": dict(Counter(r["grid"] for r in records)),
        "images_per_split": dict(Counter(r["split"] for r in records)),
        "boxes_per_split": dict(Counter(gt["split"])),
        "class_occurrences_min_max": [int(min(occ.values())), int(max(occ.values()))],
        "distinct_source_photos_used": int(gt["source"].nunique()),
        "recommended_train_imgsz": max(cols * SIZE for _, cols in GRIDS.values()),
    }
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nDone: {OUT}", flush=True)


if __name__ == "__main__":
    main()
