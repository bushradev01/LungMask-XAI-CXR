"""Cleans the dataset WITHOUT deleting any file.
Reads data/manifest.csv, checks every image and mask, writes:
  data/manifest_clean.csv   (same rows + 'issues' and 'keep' columns)
  data/cleaning_report.csv  (only flagged rows)
Run:  python src/clean_data.py     (takes a few minutes)

Checks per row:
  corrupt        image or mask cannot be decoded
  wrong_size     image or mask is not 256x256
  wrong_mode     image is not single-channel grayscale
  blank_image    image has almost no variation (std < 5)
  mask_not_binary lung/infection mask has values other than 0 and 255
  lung_empty     lung mask has no lung pixels  -> excluded
  lung_tiny/lung_huge  lung area < 3% or > 60% of image  (reported, still kept: review by eye)
  duplicate      identical image already seen in the same dataset -> excluded
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data")
SIZE = (256, 256)
EXCLUDE = {"corrupt", "wrong_size", "wrong_mode", "blank_image", "lung_empty", "duplicate"}

df = pd.read_csv(ROOT / "manifest.csv")
df["infection_mask"] = df["infection_mask"].fillna("")
seen = {}  # (dataset, md5) -> first path
issues_col = []

for n, r in enumerate(df.itertuples(index=False)):
    if n % 5000 == 0:
        print(f"{n}/{len(df)}")
    issues = []
    try:
        raw = (ROOT / r.image).read_bytes()
        with Image.open(ROOT / r.image) as im:
            im.load()
            if im.size != SIZE:
                issues.append("wrong_size")
            if im.mode != "L":
                issues.append("wrong_mode")
            a = np.array(im.convert("L"))
        if a.std() < 5:
            issues.append("blank_image")
        h = (r.dataset, hashlib.md5(raw).hexdigest())
        if h in seen:
            issues.append("duplicate")
        else:
            seen[h] = r.image
    except Exception:
        issues.append("corrupt")

    for kind, rel in [("lung", r.lung_mask), ("infection", r.infection_mask)]:
        if rel == "":
            continue
        try:
            with Image.open(ROOT / rel) as m:
                m.load()
                if m.size != SIZE:
                    issues.append("wrong_size")
                arr = np.array(m.convert("L"))
            if not set(np.unique(arr).tolist()) <= {0, 255}:
                issues.append("mask_not_binary")
            if kind == "lung":
                frac = (arr > 0).mean()
                if frac == 0:
                    issues.append("lung_empty")
                elif frac < 0.03:
                    issues.append("lung_tiny")
                elif frac > 0.60:
                    issues.append("lung_huge")
        except Exception:
            issues.append("corrupt")
    issues_col.append(";".join(sorted(set(issues))))

df["issues"] = issues_col
df["keep"] = [not (set(i.split(";")) & EXCLUDE) for i in df["issues"]]
df.to_csv(ROOT / "manifest_clean.csv", index=False)
df[df["issues"] != ""].to_csv(ROOT / "cleaning_report.csv", index=False)

print("\n===== SUMMARY =====")
print("total rows:", len(df), "| kept:", int(df.keep.sum()), "| excluded:", int((~df.keep).sum()))
flat = pd.Series([x for i in df["issues"] if i for x in i.split(";")])
print("\nissue counts:\n", flat.value_counts().to_string() if len(flat) else "none")
print("\nkept images per dataset/split/class:")
print(df[df.keep].groupby(["dataset", "split", "class"]).size().unstack("class"))