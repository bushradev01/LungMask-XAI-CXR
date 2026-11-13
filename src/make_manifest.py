"""Builds data/manifest.csv and outputs/sample_grid.png.
Run:  python src/make_manifest.py
(needs: pip install pandas matplotlib numpy pillow)
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SPLITS = ["train", "val", "test"]
CLASSES = ["COVID-19", "Non-COVID", "Normal"]
LABEL = {c: i for i, c in enumerate(CLASSES)}
random.seed(0)

# ---------- 1) manifest ----------
rows = []
for ds in ["lung_seg", "infection_seg"]:
    for sp in SPLITS:
        for c in CLASSES:
            base = ROOT / ds / sp / c
            for p in sorted((base / "images").iterdir()):
                inf = base / "infection_masks" / p.name
                rows.append({
                    "dataset": ds,
                    "split": sp,
                    "class": c,
                    "label": LABEL[c],
                    "image": (base / "images" / p.name).relative_to(ROOT).as_posix(),
                    "lung_mask": (base / "lung_masks" / p.name).relative_to(ROOT).as_posix(),
                    "infection_mask": inf.relative_to(ROOT).as_posix() if inf.exists() else "",
                })
df = pd.DataFrame(rows)
df.to_csv(ROOT / "manifest.csv", index=False)
print(df.groupby(["dataset", "split", "class"]).size().unstack("class"))

# ---------- 2) is infection_seg a subset of lung_seg? ----------
lung = set(zip(df[df.dataset == "lung_seg"].split, df[df.dataset == "lung_seg"]["class"],
               df[df.dataset == "lung_seg"].image.map(lambda s: Path(s).name)))
inf = set(zip(df[df.dataset == "infection_seg"].split, df[df.dataset == "infection_seg"]["class"],
              df[df.dataset == "infection_seg"].image.map(lambda s: Path(s).name)))
print(f"\ninfection_seg images also present in lung_seg (same split/class/name): "
      f"{len(inf & lung)} of {len(inf)}")

# ---------- 3) lung area per class (sample) ----------
print("\nMean lung area (% of image), 200 train images per class:")
sub = df[(df.dataset == "lung_seg") & (df.split == "train")]
for c in CLASSES:
    files = random.sample(list(sub[sub["class"] == c].lung_mask), 200)
    areas = [(np.array(Image.open(ROOT / f)) > 0).mean() * 100 for f in files]
    print(f"  {c:10s} {np.mean(areas):.1f}%  (std {np.std(areas):.1f})")

# ---------- 4) sample grid: image | lung overlay | infection overlay ----------
inf_df = df[(df.dataset == "infection_seg") & (df.split == "train")]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for r, c in enumerate(CLASSES):
    row = inf_df[inf_df["class"] == c].sample(1, random_state=1).iloc[0]
    img = np.array(Image.open(ROOT / row.image))
    lm = np.array(Image.open(ROOT / row.lung_mask)) > 0
    im_ = np.array(Image.open(ROOT / row.infection_mask)) > 0
    axes[r, 0].imshow(img, cmap="gray")
    axes[r, 0].set_ylabel(c, fontsize=12)
    axes[r, 1].imshow(img, cmap="gray")
    axes[r, 1].imshow(np.ma.masked_where(~lm, lm), cmap="Blues", alpha=0.4)
    axes[r, 2].imshow(img, cmap="gray")
    axes[r, 2].imshow(np.ma.masked_where(~im_, im_), cmap="Reds", alpha=0.5)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_title("X-ray"); axes[0, 1].set_title("+ lung mask"); axes[0, 2].set_title("+ infection mask")
plt.tight_layout()
plt.savefig(OUT / "sample_grid.png", dpi=120)
print("\nSaved data/manifest.csv and outputs/sample_grid.png")