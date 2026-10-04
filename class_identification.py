"""Single-object image ID: train, compare and tune four classifiers.

Models
    custom_cnn       CNN trained from scratch; architecture + hyperparameters tuned with Optuna
    resnet18         ImageNet-pretrained, transfer learning
    efficientnet_b0  ImageNet-pretrained, transfer learning
    swin_t           ImageNet-pretrained (Swin-Tiny), transfer learning

Data
    photo/images_<ObjectID>/   224x224 JPEGs, one folder per class (the ID)
    Files named *_bg_* show only background; they are used for train/val/test by default (--no-include-bg to skip them).
    photo/external/images_<ObjectID>/   optional external test set: other photos of the same
    objects, never trained on. Matched to classes by ObjectID; classes without a folder are
    simply not tested. Enabled with USE_EXTERNAL / --external.

Usage
    python class_identification.py                  # no command -> DEFAULT_COMMAND ("all")
    python class_identification.py tune_cnn         # Optuna search: custom CNN architecture + hparams
    python class_identification.py compare          # train + compare 4 models (CNN uses tuned config)
    python class_identification.py tune             # grid search on the best model
    python class_identification.py all              # tune_cnn -> compare -> tune
    python class_identification.py predict img.jpg  # ID a single image with best_model.pt
"""

import argparse
import itertools
import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported MPS ops fall back to CPU
try:  # macOS framework Python often lacks CA certs; needed to download pretrained weights
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ModuleNotFoundError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---- Defaults used when the matching command-line argument is omitted ----
DEFAULT_COMMAND = "all"             # tune_cnn | compare | tune | all | predict
INCLUDE_BG = True                   # use *_bg_* background images in train/val/test
USE_EXTERNAL = False                # evaluate on EXTERNAL_DIR (see below)
NUM_WORKERS = 0                     # DataLoader worker processes; 0 = main process (fastest on CPU-only; try 2+ with a GPU)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "IE7615 Deep Learning for AI"   # 73 class folders: images_OBJ001 ... images_OBJ073
OUT_DIR = HERE / "results"

# External test set: same layout as DATA_DIR (images_<ObjectID>/), matched by ObjectID
EXTERNAL_DIR = DATA_DIR / "external"

SIZE = 224
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device(name=None):
    """Compute device: explicit name, else CUDA -> Apple MPS -> CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()

# Default training settings per model (lr = head lr; backbone gets lr * backbone_mult)
DEFAULTS = {
    "custom_cnn":      dict(epochs=40, lr=1e-3, backbone_mult=1.0, weight_decay=1e-4, batch_size=32),
    "resnet18":        dict(epochs=15, lr=1e-3, backbone_mult=0.1, weight_decay=1e-4, batch_size=32),
    "efficientnet_b0": dict(epochs=15, lr=1e-3, backbone_mult=0.1, weight_decay=1e-4, batch_size=32),
    "swin_t":          dict(epochs=15, lr=1e-3, backbone_mult=0.1, weight_decay=1e-4, batch_size=32),
}

# Custom CNN architecture used when no Optuna result exists (results/cnn_best_config.json)
CNN_ARCH = dict(n_blocks=5, base_width=32, convs_per_block=1, dropout=0.3)

# Optuna search for the custom CNN (stops at CNN_TRIALS or CNN_TIMEOUT_S, whichever first)
CNN_TRIALS = 25
CNN_TRIAL_EPOCHS = 25
CNN_TIMEOUT_S = 2 * 3600
CNN_CONFIG_FILE = "cnn_best_config.json"

# Hyperparameter grid for the best model (backbone_mult=0 -> frozen backbone / linear probe)
TUNE_GRID = {
    "lr": [3e-4, 1e-3, 3e-3],
    "backbone_mult": [0.0, 0.1, 1.0],
    "weight_decay": [1e-4],                     # add values if CPU time allows (swin_t ~100 s/epoch on CPU)
}
TUNE_EPOCHS = None                  # None -> same epochs as compare (DEFAULTS), so results are comparable

# Chart colors: categorical slots 1-4 (fixed order), neutral ink for text
MODEL_COLORS = {
    "custom_cnn":      "#2a78d6",   # blue
    "resnet18":        "#eb6834",   # orange
    "efficientnet_b0": "#1baf7a",   # aqua green
    "swin_t":          "#eda100",   # amber yellow
}
INK = "#0b0b0b"                     # near-black: titles
INK_MUTED = "#52514e"               # dark warm gray: axes, ticks, value labels
GRID = "#e4e3df"                    # light warm gray: grid lines


# ---------------------------------------------------------------- data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def load_rgb(path):
    """Open, honor EXIF rotation, force RGB, center-crop to SIZE (same as photo_resize.py)."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if img.size != (SIZE, SIZE):
            img = ImageOps.fit(img, (SIZE, SIZE), Image.LANCZOS)
        return img.copy()


def list_images(folder, include_bg=INCLUDE_BG):
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS
                   and (include_bg or "_bg_" not in p.name)), key=natural_key)


def load_dataset(include_bg=INCLUDE_BG):
    class_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("images_"))
    classes = [d.name.removeprefix("images_") for d in class_dirs]
    paths, labels = [], []
    for idx, d in enumerate(class_dirs):
        files = list_images(d, include_bg)
        paths += files
        labels += [idx] * len(files)
        n_bg = sum("_bg_" in f.name for f in files)
        print(f"  {classes[idx]:<8} {len(files):4d} images ({n_bg} background)  ({d.name})")
    return classes, paths, labels


def load_external(classes):
    """Images from EXTERNAL_DIR/images_<ObjectID>/ for every ObjectID that is a trained class."""
    paths, labels = [], []
    if not EXTERNAL_DIR.is_dir():
        print(f"  [warn] external test set not found: {EXTERNAL_DIR}")
        return paths, labels
    for d in sorted(EXTERNAL_DIR.iterdir()):
        if not (d.is_dir() and d.name.startswith("images_")):
            continue
        cid = d.name.removeprefix("images_")
        if cid not in classes:
            print(f"  [skip] external {d.name}: {cid} is not a trained class")
            continue
        files = list_images(d, include_bg=True)
        paths += files
        labels += [classes.index(cid)] * len(files)
    print(f"  external: {len(set(labels))}/{len(classes)} classes have external images")
    return paths, labels


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def dataset_stats(images):
    """Per-channel mean / std of a list of PIL images, pixels scaled to [0, 1]."""
    x = np.stack([np.asarray(im, dtype=np.float32) / 255 for im in images])   # N,H,W,3
    to_tuple = lambda v: tuple(round(float(c), 4) for c in v)
    return to_tuple(x.mean(axis=(0, 1, 2))), to_tuple(x.std(axis=(0, 1, 2)))


def norm_stats(name, train_images=None):
    """Normalization for a model: read from its pretrained weights; models trained from
    scratch (custom_cnn) use the statistics of their own training images."""
    if name in WEIGHTS:
        t = WEIGHTS[name].transforms()
        return tuple(t.mean), tuple(t.std)
    if train_images is None:
        raise ValueError(f"{name}: no pretrained weights, training images needed for mean/std")
    return dataset_stats(train_images)


def make_transforms(mean, std):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_tf, eval_tf


def build_splits(include_bg=INCLUDE_BG, use_external=USE_EXTERNAL):
    print("Loading dataset:")
    classes, paths, labels = load_dataset(include_bg)
    idx = np.arange(len(paths))
    tr, tmp = train_test_split(idx, test_size=VAL_FRAC + TEST_FRAC, stratify=labels, random_state=SEED)
    tmp_labels = [labels[i] for i in tmp]
    va, te = train_test_split(tmp, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
                              stratify=tmp_labels, random_state=SEED)
    images = [load_rgb(p) for p in paths]
    ext_paths, ext_labels = load_external(classes) if use_external else ([], [])
    ext_images = [load_rgb(p) for p in ext_paths]
    print(f"Split: train {len(tr)} / val {len(va)} / test {len(te)} | "
          + (f"external test {len(ext_images)}" if use_external else "external test off"))
    pick = lambda ids: ([images[i] for i in ids], [labels[i] for i in ids])
    return classes, {"train": pick(tr), "val": pick(va), "test": pick(te),
                     "external": (ext_images, ext_labels)}


def make_loaders(splits, batch_size, mean, std):
    train_tf, eval_tf = make_transforms(mean, std)
    g = torch.Generator().manual_seed(SEED)
    # persistent workers start once per loader, not every epoch (process start is slow on Windows)
    opts = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    loaders = {"train": DataLoader(ImageDataset(*splits["train"], train_tf), batch_size,
                                   shuffle=True, generator=g, **opts)}
    for name in ("val", "test", "external"):
        if splits[name][0]:
            loaders[name] = DataLoader(ImageDataset(*splits[name], eval_tf), 64, **opts)
    return loaders


# ---------------------------------------------------------------- models

def conv_block(cin, cout, n_convs=1):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                   nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


class CustomCNN(nn.Module):
    """n_blocks x [conv-BN-ReLU]*convs_per_block + max-pool; filters start at base_width and
    double each block (capped at 512); global average pool, dropout, linear head."""

    def __init__(self, num_classes, n_blocks=5, base_width=32, convs_per_block=1, dropout=0.3):
        super().__init__()
        chans = [3] + [min(base_width * 2 ** i, 512) for i in range(n_blocks)]
        self.features = nn.Sequential(*[conv_block(a, b, convs_per_block)
                                        for a, b in zip(chans, chans[1:])])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(chans[-1], num_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ImageNet-pretrained weights per transfer model (also the source of their mean / std)
WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    "swin_t": models.Swin_T_Weights.DEFAULT,
}


def build_model(name, num_classes, arch=None):
    """Return (model, head_parameters). `arch` only applies to custom_cnn."""
    if name == "custom_cnn":
        m = CustomCNN(num_classes, **(arch or CNN_ARCH))
        return m, list(m.head.parameters())
    if name == "resnet18":
        m = models.resnet18(weights=WEIGHTS[name])
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, list(m.fc.parameters())
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=WEIGHTS[name])
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, list(m.classifier.parameters())
    if name == "swin_t":
        m = models.swin_t(weights=WEIGHTS[name])
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, list(m.head.parameters())
    raise ValueError(name)


def make_optimizer(model, head_params, lr, backbone_mult, weight_decay):
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [{"params": head_params, "lr": lr}]
    if backbone_mult > 0:
        groups.append({"params": backbone, "lr": lr * backbone_mult})
    else:                                       # linear probe: freeze backbone
        for p in backbone:
            p.requires_grad = False
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss, preds, ys = 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss += criterion(out, y).item() * len(y)
        preds += out.argmax(1).tolist()
        ys += y.tolist()
    preds, ys = np.array(preds), np.array(ys)
    return {"loss": loss / len(ys), "acc": float((preds == ys).mean()),
            "f1": float(f1_score(ys, preds, average="macro")), "preds": preds, "labels": ys}


def val_score(acc, loss):
    """Single number for Optuna: val acc, with val loss as a tie-breaker (< one val sample)."""
    return acc - 0.01 * loss


def train_model(name, splits, classes, epochs, lr, backbone_mult, weight_decay, batch_size,
                arch=None, trial=None, tag=None, verbose=True):
    """Train one configuration; keep the epoch with best val acc (ties -> lower val loss).
    With an Optuna `trial`, report the best score each epoch and stop early if pruned."""
    set_seed(SEED)
    mean, std = norm_stats(name, splits["train"][0])
    loaders = make_loaders(splits, batch_size, mean, std)
    model, head = build_model(name, len(classes), arch)
    model.to(DEVICE)
    model.norm_stats = {"mean": mean, "std": std}   # saved with the checkpoint for predict
    opt = make_optimizer(model, head, lr, backbone_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss()

    history, best, best_state = [], None, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if backbone_mult == 0:                  # linear probe: keep frozen BatchNorm stats fixed
            model.eval()
        tl, tc, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        v = evaluate(model, loaders["val"], eval_criterion)
        history.append({"epoch": ep, "train_loss": tl / n, "train_acc": tc / n,
                        "val_loss": v["loss"], "val_acc": v["acc"]})
        if best is None or (v["acc"], -v["loss"]) > (best["val_acc"], -best["val_loss"]):
            best = {"epoch": ep, "val_acc": v["acc"], "val_loss": v["loss"], "val_f1": v["f1"]}
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        if trial is not None:
            trial.report(val_score(best["val_acc"], best["val_loss"]), ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned(f"pruned at epoch {ep}")
        if verbose:
            print(f"  [{tag or name}] ep {ep:2d}/{epochs}  train loss {tl / n:.3f} acc {tc / n:.3f}"
                  f" | val loss {v['loss']:.3f} acc {v['acc']:.3f}  ({time.time() - t0:.0f}s)")
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    result = {"model": name, **best, "train_time_s": round(train_time, 1),
              "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
              "latency_ms": round(measure_latency(model), 1)}
    evals = {}
    for split in ("test", "external"):
        if split in loaders:
            r = evaluate(model, loaders[split], eval_criterion)
            result[f"{split}_acc"], result[f"{split}_f1"] = r["acc"], r["f1"]
            evals[split] = r
    return model, result, pd.DataFrame(history), evals


@torch.no_grad()
def measure_latency(model, runs=20):
    model.eval()
    dev = next(model.parameters()).device
    x = torch.randn(1, 3, SIZE, SIZE, device=dev)
    sync = (torch.cuda.synchronize if dev.type == "cuda" else
            torch.mps.synchronize if dev.type == "mps" else (lambda: None))
    for _ in range(3):
        model(x)
    sync()
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    sync()
    return (time.perf_counter() - t) / runs * 1000


# ---------------------------------------------------------------- plots

def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax in zip(("val_acc", "val_loss"), axes):
        for name, h in histories.items():
            ax.plot(h["epoch"], h[metric], color=MODEL_COLORS.get(name, INK), linewidth=2, label=name)
            ax.plot(h["epoch"], h[metric.replace("val", "train")], color=MODEL_COLORS.get(name, INK),
                    linewidth=1, linestyle="--", alpha=0.6)
        style_axes(ax)
        ax.set_xlabel("epoch", color=INK_MUTED)
        ax.set_title(f"{metric.replace('_', ' ')} (dashed = train)", color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(df, path):
    metrics = [c for c in ("val_acc", "test_acc", "external_acc") if c in df]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(df)
    x = np.arange(len(metrics))
    for i, row in enumerate(df.itertuples()):
        vals = [getattr(row, m) for m in metrics]
        bars = ax.bar(x + (i - (len(df) - 1) / 2) * width, vals, width * 0.92,
                      color=MODEL_COLORS.get(row.model, INK), label=row.model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_MUTED, padding=2)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy", color=INK_MUTED)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=len(df), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Model comparison", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(evals, classes, title, path):
    fig, axes = plt.subplots(1, len(evals), figsize=(4.6 * len(evals), 4.2), squeeze=False)
    for ax, (split, r) in zip(axes[0], evals.items()):
        cm = confusion_matrix(r["labels"], r["preds"], labels=range(len(classes)))
        ax.imshow(cm, cmap="Blues")
        for i, j in itertools.product(range(len(classes)), repeat=2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks(range(len(classes)), classes, fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_xlabel("predicted", color=INK_MUTED)
        ax.set_ylabel("true", color=INK_MUTED)
        ax.set_title(f"{split} acc {r['acc']:.3f}", color=INK, fontsize=10, loc="left")
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tuning(df, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [f"lr={r.lr:g}\nbb×{r.backbone_mult:g}\nwd={r.weight_decay:g}" for r in df.itertuples()]
    bars = ax.bar(range(len(df)), df["val_acc"], color="#2a78d6", width=0.7)
    bars[0].set_color("#eb6834")               # df is sorted; first = selected config
    ax.bar_label(bars, fmt="%.3f", fontsize=7, color=INK_MUTED, padding=2)
    ax.set_xticks(range(len(df)), labels, fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("val accuracy", color=INK_MUTED)
    style_axes(ax)
    ax.set_title("Hyperparameter search (sorted; orange = selected)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cnn_search(df, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    done, pruned = df[df["state"] == "COMPLETE"], df[df["state"] == "PRUNED"]
    ax.scatter(pruned["number"], pruned["value"], s=40, color="#c3c2b7", label="pruned", zorder=3)
    ax.scatter(done["number"], done["value"], s=48, color="#2a78d6", label="complete", zorder=3,
               edgecolors="white", linewidths=1.5)
    ax.plot(done["number"], done["value"].cummax(), color="#eb6834", linewidth=2,
            drawstyle="steps-post", label="best so far")
    style_axes(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("trial", color=INK_MUTED)
    ax.set_ylabel("val acc - 0.01 x val loss", color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Custom CNN: Optuna search", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- commands

def rank(df):
    """Best first: val acc, then val loss (lower), then val F1."""
    return df.sort_values(["val_acc", "val_loss", "val_f1"], ascending=[False, True, False]) \
             .reset_index(drop=True)


def save_checkpoint(model, name, classes, config, path):
    torch.save({"model": name, "classes": classes, "config": config,
                "norm": getattr(model, "norm_stats", None), "state_dict": model.state_dict()}, path)


def load_cnn_config():
    """Tuned custom-CNN config from tune_cnn, else the hand-set defaults."""
    f = OUT_DIR / CNN_CONFIG_FILE
    if f.exists():
        print(f"  custom_cnn: using Optuna config from {f.name}")
        return json.loads(f.read_text())
    print("  custom_cnn: no Optuna config found, using defaults")
    return dict(DEFAULTS["custom_cnn"], arch=CNN_ARCH)


def cmd_tune_cnn(args, splits, classes):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    epochs = args.epochs or CNN_TRIAL_EPOCHS

    def objective(trial):
        arch = dict(
            n_blocks=trial.suggest_int("n_blocks", 4, 6),                      # add / drop a block
            base_width=trial.suggest_categorical("base_width", [16, 32, 48]),  # fewer / more filters
            convs_per_block=trial.suggest_int("convs_per_block", 1, 2),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
        )
        cfg = dict(epochs=epochs, backbone_mult=1.0,
                   lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                   weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                   batch_size=trial.suggest_categorical("batch_size", [16, 32]))
        _, res, _, _ = train_model("custom_cnn", splits, classes, **cfg, arch=arch,
                                   trial=trial, verbose=False)
        for k in ("epoch", "val_acc", "val_loss", "params_M", "train_time_s"):
            trial.set_user_attr(k, res[k])
        return val_score(res["val_acc"], res["val_loss"])

    def log_trial(study, t):
        msg = (f"value {t.value:.4f} (val acc {t.user_attrs['val_acc']:.3f}, "
               f"{t.user_attrs['params_M']}M params, {t.user_attrs['train_time_s']:.0f}s)"
               if t.state == optuna.trial.TrialState.COMPLETE else t.state.name.lower())
        print(f"  [trial {t.number:2d}] {t.params}  -> {msg}")

    print(f"\n=== Optuna search on custom_cnn: up to {args.cnn_trials} trials x {epochs} epochs"
          f" (timeout {args.cnn_timeout}s)")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=epochs // 3))
    study.enqueue_trial(dict(CNN_ARCH, lr=1e-3, weight_decay=1e-4, batch_size=32))  # baseline first
    study.optimize(objective, n_trials=args.cnn_trials, timeout=args.cnn_timeout, callbacks=[log_trial])

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    df.columns = [c.removeprefix("params_").removeprefix("user_attrs_") for c in df.columns]
    df["duration"] = df["duration"].dt.total_seconds().round(1)
    df["value"] = [t.value if t.value is not None                  # pruned -> last reported score
                   else next(reversed(t.intermediate_values.values()), np.nan) for t in study.trials]
    df.to_csv(OUT_DIR / "cnn_optuna.csv", index=False)
    plot_cnn_search(df, OUT_DIR / "cnn_optuna.png")
    try:
        imp = optuna.importance.get_param_importances(study)
        (OUT_DIR / "cnn_param_importance.json").write_text(json.dumps(imp, indent=2))
        print("  param importance:", {k: round(v, 3) for k, v in imp.items()})
    except Exception as e:                      # needs >= 2 complete trials
        print(f"  [skip] param importance: {e}")

    bp = study.best_params
    cfg = dict(epochs=epochs, backbone_mult=1.0, lr=bp["lr"], weight_decay=bp["weight_decay"],
               batch_size=bp["batch_size"],
               arch={k: bp[k] for k in ("n_blocks", "base_width", "convs_per_block", "dropout")})
    (OUT_DIR / CNN_CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"\nBest CNN (trial {study.best_trial.number}, {n_done}/{len(study.trials)} completed): {cfg}")
    return cfg


def cmd_compare(args, splits, classes):
    rows, histories = [], {}
    for name in args.models:
        cfg = load_cnn_config() if name == "custom_cnn" else dict(DEFAULTS[name])
        if args.epochs:
            cfg["epochs"] = args.epochs
        print(f"\n=== {name}  {cfg}")
        model, res, hist, evals = train_model(name, splits, classes, **cfg)
        rows.append(res)
        histories[name] = hist
        hist.to_csv(OUT_DIR / f"history_{name}.csv", index=False)
        plot_confusion(evals, classes, name, OUT_DIR / f"confusion_{name}.png")
        save_checkpoint(model, name, classes, cfg, OUT_DIR / f"{name}.pt")
        print("  ->", {k: v for k, v in res.items() if k != "model"})

    df = rank(pd.DataFrame(rows))
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    plot_curves(histories, OUT_DIR / "curves.png")
    plot_comparison(df, OUT_DIR / "comparison.png")
    print("\nModel comparison (ranked by val acc, then val loss):")
    print(df.to_string(index=False))
    best = df.loc[0, "model"]
    print(f"\nBest model: {best}")
    return best


def cmd_tune(args, splits, classes, model_name):
    if model_name == "custom_cnn":              # already tuned by Optuna in tune_cnn
        ckpt = torch.load(OUT_DIR / "custom_cnn.pt", map_location="cpu")
        torch.save(ckpt, OUT_DIR / "best_model.pt")
        print(f"\nBest model is custom_cnn (Optuna-tuned); saved -> {OUT_DIR / 'best_model.pt'}")
        return
    grid = [dict(zip(TUNE_GRID, v)) for v in itertools.product(*TUNE_GRID.values())]
    baseline = {k: DEFAULTS[model_name][k] for k in TUNE_GRID}
    if baseline not in grid:                    # the compare config is always a candidate
        grid.insert(0, baseline)
    epochs = args.epochs or TUNE_EPOCHS or DEFAULTS[model_name]["epochs"]
    print(f"\n=== Hyperparameter search on {model_name}: {len(grid)} configs x {epochs} epochs")
    rows, best_model, best_key = [], None, None
    for i, hp in enumerate(grid, 1):
        cfg = dict(epochs=epochs, batch_size=DEFAULTS[model_name]["batch_size"], **hp)
        model, res, _, evals = train_model(model_name, splits, classes, **cfg,
                                           tag=f"{i}/{len(grid)}", verbose=False)
        res.update(hp)
        rows.append(res)
        print(f"  [{i:2d}/{len(grid)}] {hp}  val acc {res['val_acc']:.3f} loss {res['val_loss']:.3f}"
              f"  (best ep {res['epoch']}, {res['train_time_s']:.0f}s)")
        key = (res["val_acc"], -res["val_loss"], res["val_f1"])
        if best_key is None or key > best_key:
            best_key, best_model, best_cfg, best_evals = key, model, cfg, evals

    df = rank(pd.DataFrame(rows))
    cols = ["lr", "backbone_mult", "weight_decay", "epoch", "val_acc", "val_loss", "val_f1",
            "test_acc", "test_f1", "external_acc", "external_f1", "train_time_s"]
    df = df[[c for c in cols if c in df]]
    df.to_csv(OUT_DIR / "tuning.csv", index=False)
    plot_tuning(df, OUT_DIR / "tuning.png")
    plot_confusion(best_evals, classes, f"{model_name} (tuned)", OUT_DIR / "confusion_best_tuned.png")
    save_checkpoint(best_model, model_name, classes, best_cfg, OUT_DIR / "best_model.pt")
    print("\nTuning results (ranked):")
    print(df.to_string(index=False))
    summary = {"model": model_name, "config": best_cfg, **df.iloc[0].to_dict()}
    (OUT_DIR / "best_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSelected config: {best_cfg}\nSaved -> {OUT_DIR / 'best_model.pt'}")


def cmd_predict(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model, _ = build_model(ckpt["model"], len(ckpt["classes"]), ckpt["config"].get("arch"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    norm = ckpt.get("norm") or dict(zip(("mean", "std"), norm_stats(ckpt["model"])))
    _, eval_tf = make_transforms(norm["mean"], norm["std"])
    for p in args.images:
        with torch.no_grad():
            prob = model(eval_tf(load_rgb(p)).unsqueeze(0).to(DEVICE)).softmax(1)[0]
        k = int(prob.argmax())
        print(f"{p}: {ckpt['classes'][k]}  (p={prob[k]:.3f})")


def main():
    global NUM_WORKERS, DEVICE, DATA_DIR, EXTERNAL_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default=DEFAULT_COMMAND,
                    choices=["tune_cnn", "compare", "tune", "all", "predict"],
                    help=f"default: {DEFAULT_COMMAND}")
    ap.add_argument("images", nargs="*", type=Path, help="images for 'predict'")
    ap.add_argument("--models", nargs="+", default=list(DEFAULTS), choices=list(DEFAULTS))
    ap.add_argument("--tune-model", choices=list(DEFAULTS),
                    help="model for 'tune' (default: best in results/comparison.csv)")
    ap.add_argument("--epochs", type=int, help="override epochs (quick runs)")
    ap.add_argument("--cnn-trials", type=int, default=CNN_TRIALS, help="Optuna trials for custom_cnn")
    ap.add_argument("--cnn-timeout", type=int, default=CNN_TIMEOUT_S, help="Optuna time budget (s)")
    ap.add_argument("--include-bg", action=argparse.BooleanOptionalAction, default=INCLUDE_BG,
                    help="use *_bg_* background images in train/val/test")
    ap.add_argument("--external", action=argparse.BooleanOptionalAction, default=USE_EXTERNAL,
                    help=f"evaluate on external test set in {EXTERNAL_DIR}")
    ap.add_argument("--checkpoint", type=Path, default=OUT_DIR / "best_model.pt")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, help="DataLoader worker processes")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR,
                    help=f"dataset root with images_<ObjectID>/ folders (default: {DATA_DIR})")
    ap.add_argument("--device", default=None, help="force device: cpu | mps | cuda (default: auto)")
    args = ap.parse_args()
    NUM_WORKERS = args.workers
    DATA_DIR = args.data_dir
    EXTERNAL_DIR = DATA_DIR / "external"
    DEVICE = pick_device(args.device)
    print(f"Device: {DEVICE}  |  Data: {DATA_DIR}")

    if args.command == "predict":
        return cmd_predict(args)

    OUT_DIR.mkdir(exist_ok=True)
    classes, splits = build_splits(args.include_bg, args.external)
    best = None
    if args.command in ("tune_cnn", "all"):
        cmd_tune_cnn(args, splits, classes)
    if args.command in ("compare", "all"):
        best = cmd_compare(args, splits, classes)
    if args.command in ("tune", "all"):
        best = args.tune_model or best or pd.read_csv(OUT_DIR / "comparison.csv").loc[0, "model"]
        cmd_tune(args, splits, classes, best)


if __name__ == "__main__":
    main()
