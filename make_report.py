"""Generate report.pdf: a self-contained project report with the comparison
table and all figures from results/. Run after training + merge:

    python make_report.py
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "report.pdf"
GITHUB = "https://github.com/Sneha0504/ie7615-object-identification"

INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
LIGHT = colors.HexColor("#eef4fb")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=INK, fontSize=15,
                    spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=ACCENT, fontSize=12,
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], textColor=INK, fontSize=10,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("Cap", parent=styles["BodyText"], textColor=MUTED, fontSize=8.5,
                     alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
TITLE = ParagraphStyle("Title", parent=styles["Title"], textColor=INK, fontSize=20,
                       alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=styles["BodyText"], textColor=MUTED, fontSize=11,
                     alignment=TA_CENTER, spaceAfter=2)

story = []


def para(t, s=BODY):
    story.append(Paragraph(t, s))


def fig(path, width_in, caption):
    if not path.exists():
        return
    from PIL import Image as PImage
    w, h = PImage.open(path).size
    width = width_in * inch
    story.append(Image(str(path), width=width, height=width * h / w))
    para(caption, CAP)


# ---- Title block ----
para("Single-Object Image Identification with Deep Learning", TITLE)
para("Comparing a Custom CNN against ImageNet Transfer-Learning Backbones", SUB)
para("IE7615 &mdash; Deep Learning for AI", SUB)
para("Author: Sneha Mishra &nbsp;|&nbsp; Date: October 4, 2026", SUB)
story.append(Spacer(1, 10))

# ---- Abstract ----
para("Abstract", H1)
para(
    "We study single-object image identification across <b>73 object classes</b> "
    "(~100 photographs each) and compare four convolutional / transformer models: a "
    "<b>custom CNN trained from scratch</b> and three <b>ImageNet-pretrained</b> backbones "
    "(ResNet-18, EfficientNet-B0, and Swin-Tiny) adapted by transfer learning. On a held-out "
    "test split, transfer learning dramatically outperforms training from scratch: "
    "<b>Swin-Tiny reaches 94.8% test accuracy</b> and ResNet-18 93.6%, versus 78.9% for the "
    "custom CNN. We analyse the accuracy / model-size / latency trade-off and identify "
    "ResNet-18 as the best efficiency-accuracy balance and Swin-Tiny as the most accurate model.")

# ---- Introduction ----
para("1. Introduction and Problem Statement", H1)
para(
    "The task is to identify which single object, out of 73 distinct objects "
    "(<font face='Courier'>OBJ001</font>&ndash;<font face='Courier'>OBJ073</font>), appears in a "
    "224&times;224 colour photograph. This is a fine-grained, many-class classification problem: "
    "classes are visually similar everyday objects, each represented by only ~100 images taken "
    "under varied backgrounds and viewpoints. The central question is whether a network must be "
    "trained from scratch or whether transfer learning from large-scale ImageNet features gives a "
    "decisive advantage when per-class data is limited. We therefore implement and compare four "
    "architectures spanning classical CNNs and a modern vision transformer.")

# ---- Dataset ----
para("2. Dataset", H1)
para(
    "The dataset contains <b>7,343 images</b> organised into 73 folders "
    "(<font face='Courier'>images_OBJ&lt;ID&gt;/</font>), one per class, averaging ~100 images "
    "per class. 379 of these are background-only shots (filenames containing "
    "<font face='Courier'>_bg_</font>) that are included in all splits to teach the models the "
    "appearance of each object's typical background. All images are RGB and resized to "
    "224&times;224. We use a stratified <b>70% / 15% / 15%</b> train / validation / test split "
    "(random seed 42), giving <b>5,140 / 1,101 / 1,102</b> images respectively. The same split is "
    "used for every model so the comparison is controlled.")

# ---- Methodology ----
para("3. Methodology", H1)

para("3.1 Data pipeline and augmentation", H2)
para(
    "Each image is opened with EXIF-orientation correction, converted to RGB and fit to "
    "224&times;224. Training images are augmented on-the-fly with <b>random resized crop</b> "
    "(scale 0.6&ndash;1.0), <b>horizontal flip</b>, <b>&plusmn;15&deg; rotation</b> and "
    "<b>colour jitter</b> (brightness/contrast/saturation 0.3, hue 0.05); validation and test "
    "images are only tensor-converted and normalised. Pretrained models are normalised with their "
    "own ImageNet mean/standard-deviation statistics, while the from-scratch CNN uses the mean/std "
    "computed from its own training images.")

para("3.2 Models", H2)
para(
    "<b>Custom CNN (from scratch).</b> Five blocks of [Conv&ndash;BatchNorm&ndash;ReLU] followed "
    "by max-pooling; filter counts start at 32 and double each block (capped at 512), ending in a "
    "global-average-pool, dropout (0.3) and a linear classifier (~1.6M parameters).")
para(
    "<b>ResNet-18, EfficientNet-B0, Swin-Tiny (transfer learning).</b> Each is initialised with "
    "ImageNet weights; the final classification layer is replaced with a new 73-way linear head. "
    "The head is trained at the base learning rate and the pretrained backbone is fine-tuned at "
    "0.1&times; that rate, letting the general ImageNet features adapt gently to the new classes.")

para("3.3 Training configuration", H2)
para(
    "All models use the <b>AdamW</b> optimiser with a <b>cosine-annealing</b> learning-rate "
    "schedule, base learning rate 1e-3, weight decay 1e-4, batch size 32, and "
    "<b>cross-entropy loss with 0.1 label smoothing</b>. For each run we keep the checkpoint from "
    "the epoch with the best validation accuracy. Models are evaluated with <b>top-1 accuracy</b> "
    "and <b>macro-averaged F1</b>; inference <b>latency</b> is the mean wall-clock time for a "
    "single 224&times;224 image, measured uniformly on CPU so the numbers are directly comparable.")

story.append(PageBreak())

# ---- Results ----
para("4. Results", H1)

df = pd.read_csv(RES / "comparison.csv")
name_map = {"swin_t": "Swin-Tiny", "resnet18": "ResNet-18",
            "custom_cnn": "Custom CNN", "efficientnet_b0": "EfficientNet-B0"}
header = ["Model", "Val acc", "Test acc", "Test F1", "Params (M)", "Latency (ms)", "Best epoch"]
data = [header]
for _, r in df.iterrows():
    data.append([
        name_map.get(r["model"], r["model"]),
        f"{r['val_acc']:.3f}", f"{r['test_acc']:.3f}", f"{r['test_f1']:.3f}",
        f"{r['params_M']:.1f}", f"{r['latency_ms']:.1f}", f"{int(r['epoch'])}",
    ])
tbl = Table(data, hAlign="CENTER", colWidths=[1.4 * inch] + [0.9 * inch] * 6)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ("BACKGROUND", (0, 1), (-1, 1), LIGHT),       # best row (sorted first)
    ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d8d4")),
    ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(tbl)
para("Table 1. Final comparison on the held-out test set, ranked by validation accuracy "
     "(best row highlighted). Latency is single-image CPU inference.", CAP)

para(
    "Transfer learning is decisively better than training from scratch. <b>Swin-Tiny</b> is the "
    "most accurate (94.8% test accuracy, 0.949 F1), closely followed by <b>ResNet-18</b> (93.6%). "
    "The <b>custom CNN</b> &mdash; despite 40 epochs of training &mdash; reaches only 78.9%, a "
    "~16-point gap that reflects how valuable pretrained features are when each class has only "
    "~100 examples.")

fig(RES / "comparison.png", 6.3, "Figure 1. Validation and test accuracy by model.")
fig(RES / "curves.png", 6.6,
    "Figure 2. Training dynamics: validation (solid) and training (dashed) accuracy and loss per "
    "epoch. The pretrained models converge within a few epochs; the from-scratch CNN needs many "
    "more and plateaus lower.")

story.append(PageBreak())
para("4.1 Confusion matrices", H2)
para(
    "The 73&times;73 confusion matrices below (test split) are near-diagonal for the transfer "
    "models, confirming few systematic confusions; the custom CNN shows noticeably more "
    "off-diagonal mass.")
fig(RES / "confusion_swin_t.png", 3.6, "Figure 3. Swin-Tiny confusion matrix (test).")
fig(RES / "confusion_resnet18.png", 3.6, "Figure 4. ResNet-18 confusion matrix (test).")
fig(RES / "confusion_custom_cnn.png", 3.6, "Figure 5. Custom CNN confusion matrix (test).")

story.append(PageBreak())

# ---- Discussion ----
para("5. Discussion", H1)
para(
    "<b>Accuracy vs. efficiency trade-off.</b> Swin-Tiny gives the highest accuracy but is the "
    "largest model (27.6M parameters) and the slowest (29.1 ms/image). ResNet-18 trails it by only "
    "~1.2 points of accuracy while using <b>2.4&times; fewer parameters</b> and running "
    "<b>~3&times; faster</b> (10.3 ms), making it the best accuracy-per-cost choice for deployment. "
    "The custom CNN is the smallest and fastest (1.6M parameters, 7.8 ms) but its accuracy is not "
    "competitive &mdash; a useful baseline that quantifies the benefit of pretraining.")
para(
    "<b>Why transfer learning wins.</b> With only ~100 images per class across 73 fine-grained "
    "categories, a from-scratch network must learn generic visual primitives (edges, textures, "
    "shapes) and class-specific features simultaneously from scarce data. Pretrained backbones "
    "already encode those primitives, so fine-tuning only has to specialise them &mdash; which is "
    "why they converge in a handful of epochs and generalise far better.")

# ---- Limitations ----
para("6. Experimental Notes and Limitations", H1)
para(
    "<b>EfficientNet-B0.</b> EfficientNet-B0 is fully implemented and part of the released code, "
    "but is omitted from the final results table because of a hardware limitation on the training "
    "machine: it hangs on the Apple-Silicon <font face='Courier'>MPS</font> GPU backend (a known "
    "PyTorch issue) and is prohibitively slow on CPU (~30 min/epoch, as its depth-wise convolutions "
    "are memory-bandwidth bound) under heavy system memory pressure. Partial CPU runs reached "
    "~0.93 validation accuracy within two epochs, consistent with the other transfer models.")
para(
    "<b>Compute environment.</b> Because of the same MPS incompatibilities and memory pressure, "
    "models were trained on a mix of GPU (custom CNN, ResNet-18) and CPU (Swin-Tiny), one at a "
    "time. Training-time figures in Table 1 therefore reflect different devices and are not "
    "directly comparable; <b>accuracy, F1 and the CPU-measured latency are comparable</b>. Epoch "
    "budgets differ by model (40 for the from-scratch CNN, 15 for ResNet-18, 8 for Swin-Tiny); "
    "since the transfer models' validation accuracy plateaus within a few epochs, the comparison "
    "remains fair.")

# ---- Conclusion ----
para("7. Conclusion", H1)
para(
    "On a 73-class, ~100-images-per-class identification task, ImageNet transfer learning is the "
    "clear winner. <b>Swin-Tiny</b> achieves the best accuracy (94.8%), while <b>ResNet-18</b> "
    "offers the best accuracy-to-cost ratio and is the recommended model for practical deployment. "
    "A CNN trained from scratch reaches only 78.9%, quantifying the substantial value of pretrained "
    "features in the low-data regime. All four architectures, the full training / tuning / "
    "evaluation pipeline, and the figures in this report are reproducible from the repository "
    "below.")

story.append(Spacer(1, 16))
link = ParagraphStyle("Link", parent=BODY, alignment=TA_CENTER, fontSize=11, textColor=ACCENT)
story.append(Paragraph(
    f"<b>Code repository:</b> <a href='{GITHUB}'><font color='#2a78d6'>{GITHUB}</font></a>", link))

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                        title="IE7615 Project Report", author="Sneha Mishra")
doc.build(story)
print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
