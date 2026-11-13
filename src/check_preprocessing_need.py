"""Checks whether preprocessing steps are actually needed.
Run:  python src/check_preprocessing_need.py
Prints a table per class and saves outputs/preprocessing_check.png
(needs: pip install opencv-python-headless pandas matplotlib numpy pillow)
"""
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data")
CLASSES = ["COVID-19", "Non-COVID", "Normal"]
N = 300  # images per class
random.seed(0)

df = pd.read_csv(ROOT / "manifest.csv")
df = df[(df.dataset == "lung_seg") & (df.split == "train")]

KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def noise_sigma(img):  # Immerkaer noise estimate (in gray levels)
    h, w = img.shape
    conv = cv2.filter2D(img.astype(np.float32), -1, KERNEL)[1:-1, 1:-1]
    return np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2))


rows = []
for c in CLASSES:
    files = random.sample(list(df[df["class"] == c].image), N)
    for f in files:
        a = np.array(Image.open(ROOT / f).convert("L"))
        border = np.concatenate([a[:10].ravel(), a[-10:].ravel(), a[:, :10].ravel(), a[:, -10:].ravel()])
        rows.append({
            "class": c, "min": a.min(), "max": a.max(), "mean": a.mean(), "std": a.std(),
            "p1": np.percentile(a, 1), "p99": np.percentile(a, 99),
            "pct_black": (a == 0).mean() * 100, "pct_white": (a == 255).mean() * 100,
            "noise": noise_sigma(a), "border_mean": border.mean(),
        })
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.groupby("class").mean().round(1).T)
print("\nOverall: min of mins =", int(res["min"].min()), "| max of maxes =", int(res["max"].max()))

# figure: brightness histograms + sample images
fig, ax = plt.subplots(2, 1 + 0, figsize=(8, 5))
for c in CLASSES:
    ax[0].hist(res[res["class"] == c]["mean"], bins=30, alpha=0.5, label=c)
ax[0].set_title("Mean brightness per image"); ax[0].legend()
for c in CLASSES:
    ax[1].hist(res[res["class"] == c]["noise"], bins=30, alpha=0.5, label=c)
ax[1].set_title("Estimated noise level per image"); ax[1].legend()
plt.tight_layout(); Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/preprocessing_check_hist.png", dpi=110)

fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for r, c in enumerate(CLASSES):
    for k, f in enumerate(random.sample(list(df[df["class"] == c].image), 4)):
        ax[r, k].imshow(Image.open(ROOT / f), cmap="gray", vmin=0, vmax=255)
        ax[r, k].axis("off")
    ax[r, 0].set_title(c, loc="left")
plt.tight_layout()
plt.savefig("outputs/preprocessing_check.png", dpi=110)
print("\nSaved outputs/preprocessing_check_hist.png and outputs/preprocessing_check.png")