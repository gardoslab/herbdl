"""Same-specimen copies between val_2022 and train_2022, measured on the FULL validation set,
and every saved model re-scored with and without them.

Why: the earlier check (near_duplicates*.py) only covered the 145,344 of 167,955 validation
images that survived the 2021+2022 merge, so its 1.7% was a lower bound. Here every
val_2022 image and every train_2022 image is embedded.

Method (same as near_duplicates.py, so the numbers are comparable):
  1. Embed all val_2022 and train_2022 images with a GENERIC pretrained backbone
     (SWIN-B ImageNet-22k, 224px, not fine-tuned on herbaria). Cached under NEARDUP/full/.
  2. For each val image: cosine similarity to every train_2022 image of the SAME species;
     keep the maximum. (A copy of a specimen carries the same label. Training images of other
     species are not searched, so a copy filed under a different species would be missed.)
  3. A val image "has a copy" if that maximum is >= 0.93 (the earlier visual check: about 0.93
     and above is the same physical specimen; identical scans at >= 0.99).
  4. For every saved prediction set (Part 1 / Part 4 models): accuracy and macro-F1 on all
     validation images, on the images WITHOUT a copy ("clean"), plus accuracy on the copies.

Usage (1 GPU): python loss_ablation/part4_leakage.py
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
FEAT = f"{OUT}/NEARDUP/full"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
VAL_JSON = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
LABEL = "scientificNameEncoded"
BACKBONE = "microsoft/swin-base-patch4-window7-224-in22k"
THRESHOLDS = [0.90, 0.93, 0.95, 0.98, 0.99]
COPY_T = 0.93
RESULT = f"{OUT}/NEARDUP/full/part4_leakage.json"
PRED_DIRS = {  # model -> predictions dir
    "swin_ce": "PHASE3_EVAL/ce", "swin_ce_la": "PHASE3_EVAL/ce_la", "swin_arcface": "PHASE3_EVAL/arcface",
    "swin_subcenter_arcface": "PHASE3_EVAL/subcenter_arcface", "swin_hace": "PHASE3_EVAL/hace",
    "swin_dbm": "PHASE3_EVAL/dbm", "swin_dbm_seed43": "PHASE3_EVAL/dbm_seed43",
    "swin_kappaface": "PHASE3_EVAL/kappaface", "swin_x2softmax": "PHASE3_EVAL/x2softmax",
    "swin_arcface_ecc": "PHASE3_EVAL/arcface_ecc", "swin_arcface_tara": "PHASE3_EVAL/arcface_tara",
    "cnx_ce_la": "PHASE4_EVAL/convnext_ce_la", "cnx_arcface": "PHASE4_EVAL/convnext_arcface",
    "cnx_dbm": "PHASE4_EVAL/convnext_dbm", "dino_ce_la": "PHASE4_EVAL/dinov2_ce_la",
    "dino_arcface": "PHASE4_EVAL/dinov2_arcface", "dino_dbm": "PHASE4_EVAL/dinov2_dbm",
    # llindon's CE checkpoints, re-scored on the FULL validation set (run d341b8fb)
    "cnx_ce_llindon": "PHASE4_EVAL_FULL/convnext_ce", "dino_ce_llindon": "PHASE4_EVAL_FULL/dinov2_ce",
    "coatnet_ce_llindon": "PHASE4_EVAL_FULL/coatnet_ce", "effnet_ce_llindon": "PHASE4_EVAL_FULL/efficientnet_ce",
}


def load(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


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
    """L2-normalised pooled features (fp16), cached as FEAT/<name>.npy."""
    path = f"{FEAT}/{name}.npy"
    if os.path.isfile(path):
        f = np.load(path)
        if len(f) == len(paths):
            print(f"__LEAK__ reusing cached {name} features {f.shape}", flush=True)
            return f
    out = np.zeros((len(paths), model.config.hidden_size), dtype=np.float16)
    bad = done = 0
    for x, ok in DataLoader(Images(paths), batch_size=256, num_workers=16, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(pixel_values=x.to(device, non_blocking=True)).pooler_output
        out[done:done + len(x)] = torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy().astype(np.float16)
        bad += int((~ok).sum())
        done += len(x)
        if (done // 256) % 200 == 0:
            print(f"__LEAK__ {name}: {done:,}/{len(paths):,} embedded", flush=True)
    os.makedirs(FEAT, exist_ok=True)
    np.save(path, out)
    print(f"__LEAK__ {name}: embedded {done:,} images ({bad} unreadable -> zero vectors)", flush=True)
    return out


def macro_f1(y, p, n):
    tp = np.bincount(y[p == y], minlength=n).astype(np.float64)
    nt = np.bincount(y, minlength=n).astype(np.float64)
    npd = np.bincount(p, minlength=n).astype(np.float64)
    keep = (nt + npd) > 0
    return float(np.mean(2 * tp[keep] / (nt[keep] + npd[keep])))


def main():
    device = "cuda"
    tr, va = load(TRAIN_JSON), load(VAL_JSON)
    ytr = np.array([r[LABEL] for r in tr])
    yva = np.array([r[LABEL] for r in va])
    n = int(ytr.max()) + 1
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    fva = embed([r["filename"] for r in va], "val2022", model, device)
    ftr = embed([r["filename"] for r in tr], "train2022", model, device)

    # most similar same-species train_2022 image for every val image
    order = np.argsort(ytr, kind="stable")
    starts = np.searchsorted(ytr[order], np.arange(n + 1))
    best = np.full(len(va), np.nan, dtype=np.float32)
    Ftr = torch.as_tensor(ftr, device=device)
    for c in np.unique(yva):
        vi = np.where(yva == c)[0]
        tj = order[starts[c]:starts[c + 1]]
        if len(tj) == 0:
            continue
        s = torch.as_tensor(fva[vi], device=device).float() @ Ftr[tj].float().T
        best[vi] = s.max(dim=1).values.cpu().numpy()
    has_match = np.isfinite(best)
    print(f"__LEAK__ val images with at least one same-species train_2022 image: {has_match.sum():,} of {len(va):,}")
    np.savez_compressed(f"{FEAT}/val2022_vs_train2022_full.npz", best=best)

    res = {"n_val": len(va), "threshold": COPY_T, "share_above": {}, "models": {}}
    for t in THRESHOLDS:
        m = has_match & (best >= t)
        res["share_above"][str(t)] = {"n": int(m.sum()), "share": float(m.sum() / len(va))}
        print(f"__LEAK__ val images with a same-species train copy >= {t}: {m.sum():,} ({100 * m.sum() / len(va):.2f}%)")
    copy = has_match & (best >= COPY_T)
    clean = ~copy
    # coverage of the earlier (partial) check: val images that survived the merge
    prev = f"{OUT}/NEARDUP/n15k/val2022_vs_train2022.npz"
    if os.path.isfile(prev):
        z = np.load(prev)
        vi, b_old = z["val_index"], z["best"]
        ok = np.isfinite(b_old)
        agree = np.mean((b_old[ok] >= COPY_T) == copy[vi][ok])
        print(f"__LEAK__ earlier partial check: {ok.sum():,} images, {100 * np.mean(b_old[ok] >= COPY_T):.2f}% copies; "
              f"same images in this run: {100 * np.mean(copy[vi][ok]):.2f}%; flag agreement {100 * agree:.2f}%")
        uncovered = np.ones(len(va), bool)
        uncovered[vi] = False
        print(f"__LEAK__ images the earlier check could not cover: {uncovered.sum():,}; copies among them now: "
              f"{copy[uncovered].sum():,} ({100 * copy[uncovered].mean():.2f}%)")
        res["uncovered"] = {"n": int(uncovered.sum()), "copies": int(copy[uncovered].sum()),
                            "share": float(copy[uncovered].mean())}

    for name, d in PRED_DIRS.items():
        path = f"{OUT}/{d}/predictions.npz"
        if not os.path.isfile(path):
            print(f"__LEAK__ MISSING {name}: {path}")
            continue
        z = np.load(path)
        y, p = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
        if not np.array_equal(y, yva):
            print(f"__LEAK__ SKIP {name}: labels are not val_2022 in order ({len(y)} rows)")
            continue
        r = {"acc_all": float(np.mean(p == y)), "f1_all": macro_f1(y, p, n),
             "acc_clean": float(np.mean(p[clean] == y[clean])), "f1_clean": macro_f1(y[clean], p[clean], n),
             "acc_on_copies": float(np.mean(p[copy] == y[copy]))}
        res["models"][name] = r
        print(f"__LEAK__ {name}: all {r['acc_all']:.4f}/{r['f1_all']:.4f} | clean {r['acc_clean']:.4f}/{r['f1_clean']:.4f} "
              f"| on copies {r['acc_on_copies']:.4f}")
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"__LEAK__ wrote {RESULT}")


if __name__ == "__main__":
    main()
