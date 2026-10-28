"""Pictures of the validation images flagged as same-specimen copies.

Reuses the generic-backbone features cached by part4_leakage.py (NEARDUP/full/{val2022,train2022}.npy):
for each val_2022 image, the most similar same-species train_2022 image and the cosine similarity.
Writes (to NEARDUP/full/gallery/):
  flagged_pairs.csv        every pair with similarity >= 0.93 (the copies counted in the report)
  near_misses_0.90_0.93.csv  pairs just BELOW the cutoff, to judge whether 0.93 is a sensible line
  sheet_<band>.png         8 random pairs per similarity band, each as [validation | training match]
CPU only (a few minutes).  Usage: python loss_ablation/copy_gallery.py
"""

import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/NEARDUP/full"
GAL = f"{OUT}/gallery"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
VAL_JSON = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
LABEL = "scientificNameEncoded"
BANDS = [("sim_0.99_up", 0.99, 1.01), ("sim_0.97_0.99", 0.97, 0.99), ("sim_0.95_0.97", 0.95, 0.97),
         ("sim_0.93_0.95", 0.93, 0.95), ("below_cutoff_0.90_0.93", 0.90, 0.93)]
PAIRS_PER_SHEET, COLS, SIDE = 8, 2, 380


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def fit(path, side):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def main():
    os.makedirs(GAL, exist_ok=True)
    tr, va = load(TRAIN_JSON), load(VAL_JSON)
    ytr = np.array([r[LABEL] for r in tr])
    yva = np.array([r[LABEL] for r in va])
    fva = np.load(f"{OUT}/val2022.npy", mmap_mode="r")
    ftr = np.load(f"{OUT}/train2022.npy", mmap_mode="r")
    n = int(ytr.max()) + 1
    order = np.argsort(ytr, kind="stable")
    starts = np.searchsorted(ytr[order], np.arange(n + 1))
    best = np.full(len(va), -1.0, dtype=np.float32)
    best_j = np.full(len(va), -1, dtype=np.int64)
    for c in np.unique(yva):
        vi = np.where(yva == c)[0]
        tj = order[starts[c]:starts[c + 1]]
        if len(tj) == 0:
            continue
        s = np.asarray(fva[vi], dtype=np.float32) @ np.asarray(ftr[tj], dtype=np.float32).T
        a = s.argmax(axis=1)
        best[vi] = s[np.arange(len(vi)), a]
        best_j[vi] = tj[a]
    print(f"__GAL__ flagged (>= 0.93): {(best >= 0.93).sum():,}", flush=True)

    def name(r):
        return r.get("scientificName") or r.get("species") or str(r[LABEL])

    for fname, lo, hi in (("flagged_pairs.csv", 0.93, 2.0), ("near_misses_0.90_0.93.csv", 0.90, 0.93)):
        idx = np.where((best >= lo) & (best < hi))[0]
        with open(f"{GAL}/{fname}", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["similarity", "species", "val_image", "train_image_match"])
            for i in idx[np.argsort(-best[idx])]:
                w.writerow([f"{best[i]:.4f}", name(va[i]), va[i]["filename"], tr[best_j[i]]["filename"]])
        print(f"__GAL__ wrote {fname}: {len(idx):,} pairs", flush=True)

    rng = np.random.default_rng(0)
    for tag, lo, hi in BANDS:
        idx = np.where((best >= lo) & (best < hi))[0]
        pick = rng.choice(idx, size=min(PAIRS_PER_SHEET, len(idx)), replace=False)
        rows = (len(pick) + COLS - 1) // COLS
        cap = 34
        sheet = Image.new("RGB", (COLS * (2 * SIDE + 12), rows * (SIDE + cap)), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for k, i in enumerate(pick):
            x0, y0 = (k % COLS) * (2 * SIDE + 12), (k // COLS) * (SIDE + cap)
            sheet.paste(fit(va[i]["filename"], SIDE), (x0, y0))
            sheet.paste(fit(tr[best_j[i]]["filename"], SIDE), (x0 + SIDE, y0))
            d.text((x0 + 4, y0 + SIDE + 4), f"validation | training match   similarity {best[i]:.3f}", fill=(0, 0, 0))
            d.text((x0 + 4, y0 + SIDE + 18), name(va[i])[:60], fill=(90, 90, 90))
        sheet.save(f"{GAL}/sheet_{tag}.png")
        print(f"__GAL__ wrote sheet_{tag}.png ({len(pick)} of {len(idx):,} pairs)", flush=True)


if __name__ == "__main__":
    main()
