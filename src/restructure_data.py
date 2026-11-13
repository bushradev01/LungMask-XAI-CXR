
"""Flatten COVID-QU-Ex into:
    data/lung_seg/{train,val,test}/{COVID-19,Non-COVID,Normal}/{images,lung_masks}
    data/infection_seg/{train,val,test}/{COVID-19,Non-COVID,Normal}/{images,lung_masks,infection_masks}

Dry run by default (only prints). Use:  python src/restructure_data.py --apply
Folders are MOVED (not copied), so it is fast and uses no extra disk space.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path("data")
DATASETS = {
    "Infection Segmentation Data": "infection_seg",
    "Lung Segmentation Data": "lung_seg",
}
SPLITS = {"Train": "train", "Val": "val", "Test": "test"}
SUBS = {"images": "images", "infection masks": "infection_masks", "lung masks": "lung_masks"}


def main(apply: bool, archive: str):
    ARCHIVE = Path(archive)
    moves = []
    for old_ds, new_ds in DATASETS.items():
        inner = ARCHIVE / old_ds / old_ds  # the doubled folder name
        if not inner.exists():
            print(f"!! missing: {inner}")
            continue
        for old_sp, new_sp in SPLITS.items():
            for cls_dir in sorted((inner / old_sp).iterdir()):
                if not cls_dir.is_dir():
                    continue
                for sub in sorted(cls_dir.iterdir()):
                    if sub.is_dir():
                        dst = ROOT / new_ds / new_sp / cls_dir.name / SUBS.get(sub.name, sub.name)
                        moves.append((sub, dst))

    for src, dst in moves:
        print(f"{src}\n   -> {dst}")
        if apply:
            if dst.exists():
                print("   skipped (destination exists)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if not apply:
        print(f"\nDRY RUN: {len(moves)} folders would be moved. Re-run with --apply to do it.")
        return

    # loose files inside archive (e.g. the COVID-QU-Ex metadata file) go to data/
    for f in ARCHIVE.iterdir():
        if f.is_file():
            shutil.move(str(f), str(ROOT / f.name))

    leftover = [p for p in ARCHIVE.rglob("*") if p.is_file()]
    if leftover:
        print(f"Archive NOT deleted, {len(leftover)} files still inside it.")
    else:
        shutil.rmtree(ARCHIVE)
        print("Done. 'archive (7)' removed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--archive", default="data/archive (7)",
                    help="folder that contains the extracted Kaggle download")
    a = ap.parse_args()
    main(a.apply, a.archive)