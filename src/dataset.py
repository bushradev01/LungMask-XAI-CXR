"""Preprocessing + Dataset for COVID-QU-Ex.

Pipeline per image (256x256 grayscale -> model input):
  1. optional CLAHE (contrast enhancement, on the grayscale image)
  2. grayscale -> 3 channels (pretrained models expect RGB)
  3. resize to img_size (default 224)
  4. train only: small rotation + zoom (0.9-1.1) + brightness/contrast jitter
     (+ optional h-flip, off by default). No denoising: noise is ~1/255, not needed.
  5. to tensor + ImageNet normalization

Works locally (layout="local") and on Kaggle (layout="kaggle", original nested folders).

Self-test:  python src/dataset.py   -> saves outputs/preprocess_check.png
(needs: pip install torch torchvision opencv-python-headless pandas pillow matplotlib)
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["COVID-19", "Non-COVID", "Normal"]

_KAGGLE_DS = {"lung_seg": "Lung Segmentation Data", "infection_seg": "Infection Segmentation Data"}
_KAGGLE_SPLIT = {"train": "Train", "val": "Val", "test": "Test"}


def to_kaggle_path(rel: str) -> str:
    """'lung_seg/train/COVID-19/lung_masks/x.png' -> original nested Kaggle path."""
    ds, sp, cls, sub, name = rel.split("/")
    d = _KAGGLE_DS[ds]
    return f"{d}/{d}/{_KAGGLE_SPLIT[sp]}/{cls}/{sub.replace('_', ' ')}/{name}"


class CXRDataset(Dataset):
    def __init__(self, manifest, root, split, dataset="lung_seg", layout="local",
                 train=False, img_size=224, use_clahe=False, hflip=False,
                 return_masks=False):
        assert not (train and return_masks), "masks are for evaluation only (no augmentation)"
        df = pd.read_csv(manifest)
        if "keep" in df.columns:  # manifest_clean.csv: drop flagged images
            df = df[df["keep"]]
        self.df = df[(df.split == split) & (df.dataset == dataset)].reset_index(drop=True)
        self.root = Path(root)
        self.layout = layout
        self.size = img_size
        self.use_clahe = use_clahe
        self.return_masks = return_masks
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        tf = [T.Resize((img_size, img_size))]
        if train:
            tf += [T.RandomAffine(degrees=10, scale=(0.9, 1.1)),
                   T.ColorJitter(brightness=0.2, contrast=0.2)]
            if hflip:  # off by default: flipping moves the heart to the other side
                tf.append(T.RandomHorizontalFlip())
        self.tf = T.Compose(tf + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def _path(self, rel):
        return self.root / (to_kaggle_path(rel) if self.layout == "kaggle" else rel)

    def _load_mask(self, rel):
        if not isinstance(rel, str) or rel == "":
            return torch.zeros(self.size, self.size)
        m = Image.open(self._path(rel)).convert("L").resize((self.size, self.size), Image.NEAREST)
        return torch.from_numpy((np.array(m) > 0).astype(np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        gray = np.array(Image.open(self._path(r.image)).convert("L"))
        if self.use_clahe:
            gray = self.clahe.apply(gray)
        img = self.tf(Image.fromarray(gray).convert("RGB"))
        if not self.return_masks:
            return img, int(r.label)
        lung = self._load_mask(r.lung_mask)
        inf = self._load_mask(r.infection_mask if isinstance(r.infection_mask, str) else "")
        return img, int(r.label), lung, inf


def denorm(x):
    m = torch.tensor(MEAN).view(3, 1, 1); s = torch.tensor(STD).view(3, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path("outputs").mkdir(exist_ok=True)
    kw = dict(manifest="data/manifest_clean.csv", root="data", split="train", dataset="lung_seg")
    plain = CXRDataset(**kw, train=False)
    clahe = CXRDataset(**kw, train=False, use_clahe=True)
    aug = CXRDataset(**kw, train=True)
    evalm = CXRDataset(**{**kw, "split": "val"}, return_masks=True)

    idx = [0, 9000, 16000]  # one from each class region of the train set
    fig, ax = plt.subplots(3, 3, figsize=(9, 9))
    for r, i in enumerate(idx):
        for c, (name, ds) in enumerate([("original", plain), ("CLAHE", clahe), ("augmented", aug)]):
            x, y = ds[i]
            ax[r, c].imshow(denorm(x)); ax[r, c].axis("off")
            ax[r, c].set_title(f"{name} | {CLASSES[y]}", fontsize=9)
    plt.tight_layout(); plt.savefig("outputs/preprocess_check.png", dpi=120)

    x, y = plain[0]
    print("train size:", len(plain), "| val size:", len(evalm))
    print("image tensor:", tuple(x.shape), "dtype", x.dtype,
          f"min {x.min():.2f} max {x.max():.2f}")
    _, _, lung, inf = evalm[0]
    print("mask tensors:", tuple(lung.shape), "lung px", int(lung.sum()), "| infection px", int(inf.sum()))
    print("Saved outputs/preprocess_check.png")