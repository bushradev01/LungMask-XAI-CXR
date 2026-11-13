"""Train a COVID-19 / Non-COVID / Normal classifier on Kaggle.

Needs two Kaggle inputs:
  1) the covidqu dataset (anasmohammedtahir)
  2) your own dataset containing: dataset.py, manifest_clean.csv, train_classifier.py
Notebook settings: GPU on, Internet ON (pretrained weights are downloaded).

Run in a notebook cell:   !pip install -q timm
then in the next cell:    %run /kaggle/input/<your-code-dataset>/train_classifier.py
(Or paste this whole file into one cell.)

To train another model, change MODEL (and BATCH) below, e.g.
    MODEL = "maxvit_tiny_tf_224" ; BATCH = 32
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ======================= CONFIG (edit here) =======================
MODEL = "convnext_tiny"      # "convnext_tiny" | "maxvit_tiny_tf_224" | "convnextv2_tiny"
BATCH = 64                   # use 32 for maxvit_tiny_tf_224
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
USE_CLAHE = False            # try True later as an experiment
TRAIN_FRACTION = 1.0         # 1.0 = all training images; 0.05 = random 5% (same share per class)
FREEZE_BACKBONE = False      # True = train only the final layer
TAG = f"{MODEL}_frac{TRAIN_FRACTION}" + ("_frozen" if FREEZE_BACKBONE else "")
PATIENCE = 3                 # stop if val accuracy does not improve for 3 epochs
IMG = 224
SEED = 42
OUT = Path("/kaggle/working")
# ==================================================================


def find_file(name, base="/kaggle/input", max_depth=4):
    for dp, dn, fn in os.walk(base):
        if name in fn:
            return dp
        if dp[len(base):].count(os.sep) >= max_depth:
            dn[:] = []
    raise FileNotFoundError(f"{name} not found under {base}. Did you add it as an Input?")


def find_dataset_root(base="/kaggle/input", max_depth=4):
    key = "Lung Segmentation Data"
    for dp, dn, _ in os.walk(base):
        if key in dn and os.path.isdir(os.path.join(dp, key, key)):
            return dp
        if dp[len(base):].count(os.sep) >= max_depth:
            dn[:] = []
    raise FileNotFoundError("covidqu dataset not found. Add it as an Input in the notebook.")


code_dir = find_file("dataset.py")
sys.path.insert(0, code_dir)
from dataset import CLASSES, CXRDataset  # noqa: E402

MANIFEST = os.path.join(find_file("manifest_clean.csv"), "manifest_clean.csv")
ROOT = find_dataset_root()
print("code dir :", code_dir)
print("manifest :", MANIFEST)
print("data root:", ROOT)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device   :", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "(NO GPU! turn it on)")

import timm  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             f1_score, roc_auc_score)

kw = dict(manifest=MANIFEST, root=ROOT, dataset="lung_seg", layout="kaggle",
          img_size=IMG, use_clahe=USE_CLAHE)
ds_tr = CXRDataset(split="train", train=True, **kw)
ds_va = CXRDataset(split="val", train=False, **kw)
ds_te = CXRDataset(split="test", train=False, **kw)
if TRAIN_FRACTION < 1.0:  # keep the same share of every class; val and test stay full
    ds_tr.df = pd.concat([g.sample(frac=TRAIN_FRACTION, random_state=SEED)
                          for _, g in ds_tr.df.groupby("label")]).reset_index(drop=True)
print(f"images -> train {len(ds_tr)} | val {len(ds_va)} | test {len(ds_te)}")
_ = ds_tr[0]  # fail early if a path is wrong

dl = dict(num_workers=4, pin_memory=True, persistent_workers=True)
tr_loader = DataLoader(ds_tr, BATCH, shuffle=True, drop_last=True, **dl)
va_loader = DataLoader(ds_va, BATCH, shuffle=False, **dl)
te_loader = DataLoader(ds_te, BATCH, shuffle=False, **dl)

model = timm.create_model(MODEL, pretrained=True, num_classes=len(CLASSES)).to(dev)
if FREEZE_BACKBONE:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True
crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")


@torch.no_grad()
def evaluate(loader):
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x)
            loss_sum += crit(out, y).item() * len(y)
        probs.append(out.float().softmax(1).cpu()); labels.append(y.cpu())
    probs = torch.cat(probs).numpy(); labels = torch.cat(labels).numpy()
    preds = probs.argmax(1)
    return (loss_sum / len(labels), accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro"), probs, labels)


best_acc, bad, history = 0.0, 0, []
ckpt = OUT / f"best_{TAG}.pt"
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    tl, tc, tn = 0.0, 0, 0
    for x, y in tr_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(x); loss = crit(out, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); tn += len(y)
    sched.step()
    vl, va, vf, _, _ = evaluate(va_loader)
    history.append(dict(epoch=ep, train_loss=tl / tn, train_acc=tc / tn,
                        val_loss=vl, val_acc=va, val_f1=vf))
    print(f"ep {ep:02d} | train loss {tl/tn:.3f} acc {tc/tn:.3f} | "
          f"val loss {vl:.3f} acc {va:.3f} f1 {vf:.3f} | {(time.time()-t0)/60:.1f} min")
    if va > best_acc:
        best_acc, bad = va, 0
        torch.save(model.state_dict(), ckpt)
    else:
        bad += 1
        if bad >= PATIENCE:
            print("early stopping"); break

pd.DataFrame(history).to_csv(OUT / f"history_{TAG}.csv", index=False)

# ---------------- final TEST evaluation (run once) ----------------
model.load_state_dict(torch.load(ckpt, map_location=dev))
_, acc, f1, probs, labels = evaluate(te_loader)
auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
print("\n================ TEST RESULTS ================")
print(f"run {TAG} | accuracy {acc:.4f} | macro F1 {f1:.4f} | macro AUC {auc:.4f}")
print(classification_report(labels, probs.argmax(1), target_names=CLASSES, digits=4))
cm = confusion_matrix(labels, probs.argmax(1))
print("confusion matrix (rows = true, cols = predicted):\n", cm)

json.dump(dict(model=MODEL, train_fraction=TRAIN_FRACTION, frozen=FREEZE_BACKBONE, train_images=len(ds_tr), test_acc=acc, test_macro_f1=f1, test_macro_auc=auc,
               best_val_acc=best_acc, clahe=USE_CLAHE, lr=LR, batch=BATCH, epochs_run=len(history)),
          open(OUT / f"results_{TAG}.json", "w"), indent=2)

import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
h = pd.DataFrame(history)
ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
ax[0].set_title("loss"); ax[0].legend()
ax[1].plot(h.epoch, h.train_acc, label="train"); ax[1].plot(h.epoch, h.val_acc, label="val")
ax[1].set_title("accuracy"); ax[1].legend()
ax[2].imshow(cm, cmap="Blues")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(CLASSES); ax[2].set_yticks(range(3)); ax[2].set_yticklabels(CLASSES)
for i in range(3):
    for j in range(3):
        ax[2].text(j, i, cm[i, j], ha="center", va="center")
ax[2].set_title("confusion matrix (test)")
plt.tight_layout(); plt.savefig(OUT / f"curves_{TAG}.png", dpi=110); plt.show()
print("saved to", OUT)
