"""Near-duplicate (leakage) check for the scaling ladder's merged 2021+2022 split.

Question: do the ladder's validation images have near-copies (the same specimen, re-scanned
or re-cropped) in its training set? If so, the very high accuracy on its Herbarium 2021
validation images (0.933, vs ~0.85 on 2022 images) is partly leakage.

Method (15.5k rung):
  1. Embed every n15k train and val image with a GENERIC pretrained backbone
     (SWIN-B ImageNet-22k, 224px; not fine-tuned on herbaria, so "same species" does not
     automatically mean "very similar"). Features are cached to FEAT_DIR.
  2. For each val image, cosine similarity to every train image of the SAME species (a copy
     of a specimen must carry the same label); keep the max and which image it was.
  3. Per val source (Herbarium 2021 / 2022 train / val_2022): distribution of that max
     similarity, share above several thresholds, and where the best match comes from.
  4. Ladder 15.5k DBM accuracy on val images with vs without a near-copy in train.
  5. Example pairs at several similarity levels, for visual inspection.

Usage: python loss_ablation/near_duplicates.py   (1 GPU)
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
FEAT_DIR = f"{OUT}/NEARDUP/n15k"
VAL22 = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
TRAIN22 = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
BACKBONE = "microsoft/swin-base-patch4-window7-224-in22k"
THRESHOLDS = [0.90, 0.95, 0.98, 0.99]
EXAMPLE_BINS = [(0.99, 1.01), (0.97, 0.99), (0.93, 0.97), (0.85, 0.93)]


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.tf(Image.open(self.paths[i]).convert("RGB")), True
        except Exception:
            return torch.zeros(3, 224, 224), False


@torch.no_grad()
def embed(paths, name, model, device):
    """L2-normalised pooled features (fp16), cached as FEAT_DIR/<name>.npy."""
    path = f"{FEAT_DIR}/{name}.npy"
    if os.path.isfile(path):
        f = np.load(path)
        if len(f) == len(paths):
            print(f"__ND__ reusing cached {name} features {f.shape}", flush=True)
            return f
    out = np.zeros((len(paths), model.config.hidden_size), dtype=np.float16)
    bad, done = 0, 0
    dl = DataLoader(Images(paths), batch_size=256, num_workers=16, pin_memory=True)
    for x, ok in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(pixel_values=x.to(device, non_blocking=True)).pooler_output
        f = torch.nn.functional.normalize(f.float(), dim=1)
        out[done:done + len(x)] = f.cpu().numpy().astype(np.float16)
        bad += int((~ok).sum())
        done += len(x)
        if (done // 256) % 500 == 0:
            print(f"__ND__ {name}: {done:,}/{len(paths):,} embedded", flush=True)
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.save(path, out)
    print(f"__ND__ {name}: embedded {done:,} images ({bad} unreadable -> zero vectors)", flush=True)
    return out


def main():
    device = "cuda"
    tr, va = load(f"{LAD}/n15k_train.json"), load(f"{LAD}/n15k_val.json")
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    ftr = embed([r["filepath"] for r in tr], "train", model, device)
    fva = embed([r["filepath"] for r in va], "val", model, device)

    # source of each val image, and of each train image
    v22 = {r["filename"] for r in load(VAL22)}
    t22 = {r["filename"] for r in load(TRAIN22)}

    def source(p):
        return "val_2022" if p in v22 else "train_2022" if p in t22 else (
            "herbarium_2021" if "herbarium-2021" in p else "other")

    vsrc = np.array([source(r["filepath"]) for r in va])
    tsrc = np.array([source(r["filepath"]) for r in tr])

    # nearest same-species train image for every val image
    ytr = np.array([r["scientificNameEncoded"] for r in tr])
    yva = np.array([r["scientificNameEncoded"] for r in va])
    order = np.argsort(ytr, kind="stable")
    starts = np.searchsorted(ytr[order], np.arange(ytr.max() + 2))
    best = np.full(len(va), np.nan, dtype=np.float32)
    best_j = np.full(len(va), -1, dtype=np.int64)
    Ftr = torch.as_tensor(ftr, device=device)
    for c in np.unique(yva):
        vi = np.where(yva == c)[0]
        tj = order[starts[c]:starts[c + 1]]
        if len(tj) == 0:
            continue
        s = torch.as_tensor(fva[vi], device=device).float() @ Ftr[tj].float().T
        v, a = s.max(dim=1)
        best[vi] = v.cpu().numpy()
        best_j[vi] = tj[a.cpu().numpy()]
    print(f"__ND__ nearest same-species train image found for {np.isfinite(best).sum():,} of {len(va):,} val images")

    lp = np.load(f"{OUT}/LADDER_EVAL/n15k_dbm/predictions.npz")
    correct = lp["predictions"] == lp["labels"]
    assert np.array_equal(lp["labels"], yva), "prediction order != n15k_val order"

    res = {"backbone": BACKBONE, "by_source": {}}
    for src in ("herbarium_2021", "train_2022", "val_2022"):
        m = (vsrc == src) & np.isfinite(best)
        b = best[m]
        row = {"n": int(m.sum()),
               "max_sim_quantiles": {q: float(np.quantile(b, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
               "share_above": {t: float(np.mean(b >= t)) for t in THRESHOLDS},
               "best_match_from": {s: float(np.mean(tsrc[best_j[m]] == s))
                                   for s in ("herbarium_2021", "train_2022", "val_2022")},
               "dbm_accuracy": float(np.mean(correct[m]))}
        row["dbm_accuracy_by_near_copy"] = {
            f">= {t}": [float(np.mean(correct[m & (best >= t)])) if np.any(m & (best >= t)) else None,
                        float(np.mean(correct[m & (best < t)])) if np.any(m & (best < t)) else None]
            for t in THRESHOLDS}
        res["by_source"][src] = row
        print(f"__ND__ {src:15s} n={row['n']:7,} median max-sim={row['max_sim_quantiles'][0.5]:.3f} "
              + " ".join(f">={t}:{100*v:.1f}%" for t, v in row["share_above"].items())
              + f" | best match from: " + ", ".join(f"{k} {100*v:.0f}%" for k, v in row["best_match_from"].items())
              + f" | DBM acc {row['dbm_accuracy']:.4f}", flush=True)
        for t, (yes, no) in row["dbm_accuracy_by_near_copy"].items():
            if yes is not None and no is not None:
                print(f"__ND__   {src}: DBM acc with near-copy (sim {t}) {yes:.4f} "
                      f"(n={int(np.sum(m & (best >= float(t[3:]))))}) vs without {no:.4f}", flush=True)

    rng = np.random.default_rng(0)
    examples = {}
    for lo, hi in EXAMPLE_BINS:
        idx = np.where((best >= lo) & (best < hi))[0]
        pick = rng.choice(idx, size=min(4, len(idx)), replace=False) if len(idx) else []
        examples[f"{lo}-{hi}"] = [{"val": va[i]["filepath"], "train": tr[best_j[i]]["filepath"],
                                   "sim": float(best[i]), "val_source": vsrc[i],
                                   "train_source": tsrc[best_j[i]]} for i in pick]
    res["examples"] = examples
    np.savez_compressed(f"{FEAT_DIR}/nearest.npz", best=best, best_j=best_j, val_source=vsrc)
    json.dump(res, open(f"{FEAT_DIR}/near_duplicates_summary.json", "w"), indent=1)
    print(f"__ND__ wrote {FEAT_DIR}/near_duplicates_summary.json")


if __name__ == "__main__":
    main()
