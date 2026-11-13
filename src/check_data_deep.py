"""Deeper dataset study. Run AFTER restructure_data.py:  python src/check_data_deep.py
Checks: image/mask filename pairing, image size and mode, mask pixel values,
how many masks are empty, and identical images shared between splits.
"""
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data")
SPLITS = ["train", "val", "test"]
CLASSES = ["COVID-19", "Non-COVID", "Normal"]
DATASETS = {"lung_seg": ["lung_masks"], "infection_seg": ["lung_masks", "infection_masks"]}
random.seed(0)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


for ds, mask_kinds in DATASETS.items():
    print(f"\n=================== {ds} ===================")
    hash_to_splits = defaultdict(set)
    sizes, modes = Counter(), Counter()

    # 1) pairing + hashing
    for sp in SPLITS:
        for c in CLASSES:
            base = ROOT / ds / sp / c
            img_names = {p.name for p in (base / "images").iterdir()}
            for mk in mask_kinds:
                mask_names = {p.name for p in (base / mk).iterdir()}
                if img_names != mask_names:
                    print(f"MISMATCH {sp}/{c}/{mk}: "
                          f"{len(img_names - mask_names)} images without mask, "
                          f"{len(mask_names - img_names)} masks without image")
            for p in (base / "images").iterdir():
                hash_to_splits[md5(p)].add(sp)

    # 2) sizes / modes (sample)
    for sp in SPLITS:
        for c in CLASSES:
            files = sorted((ROOT / ds / sp / c / "images").iterdir())
            for p in random.sample(files, min(30, len(files))):
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
    print("image sizes (sample):", sizes.most_common(5))
    print("image modes (sample):", dict(modes))

    # 3) mask values + empty masks (sample from train)
    for mk in mask_kinds:
        for c in CLASSES:
            files = sorted((ROOT / ds / "train" / c / mk).iterdir())
            vals, empty = set(), 0
            sample = random.sample(files, min(100, len(files)))
            for p in sample:
                arr = np.array(Image.open(p))
                vals.update(np.unique(arr).tolist()[:6])
                empty += int(arr.max() == 0)
            print(f"{mk:15s} {c:10s} unique values (first few): {sorted(vals)[:6]} | "
                  f"empty masks: {empty}/{len(sample)}")

    # 4) exact duplicate images across splits
    shared = sum(1 for s in hash_to_splits.values() if len(s) > 1)
    print(f"identical images found in more than one split: {shared}")

print("\nNote: patient-level overlap cannot be checked, the dataset has no patient IDs.")