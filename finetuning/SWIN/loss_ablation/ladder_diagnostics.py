"""The 15.5k diagnostics, repeated on the three ladder rungs.

Reads the pooled validation features and labels written by run_ladder_geom.sh
and computes, for every (rung, loss):

  * per-image cosine margins (the boundary_margins.py quantity),
  * the angle between each sample and its positive classifier row, and the
    angle to the class centroid (the DBM paper's compactness metric),
  * the pairwise Fisher criterion with a pooled within-class covariance,
  * accuracy, macro-F1, and the same rarity-band breakdown.

The class count is not 15,501 at the larger rungs, so this script reads it from
the checkpoint's head rather than assuming it. Fisher scores are computed in
row chunks: the full 64k x 64k matrix would be 16 GB.

Features live in LADDER_GEOM/<rung>_<loss>/features.npy; labels come from
LADDER_EVAL/<rung>_<loss>/predictions.npz, written by the same eval pass.
Train counts for the rarity bands come from the rung's own train JSON.

Usage: python loss_ablation/ladder_diagnostics.py [--rungs n15k ...]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary_margins import band_stats, boundary_scores  # noqa: E402
from paper_separability import angle_stats  # noqa: E402

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
FEAT = f"{OUT}/LADDER_GEOM"
PRED = f"{OUT}/LADDER_EVAL"
DATA = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
LABEL = "scientificNameEncoded"
# The ladder adds species with a single training image, so the base bands are
# extended downwards by a "1" band.
BANDS = [(1, 1), (2, 10), (11, 25), (26, 50), (51, 73)]
# cosine_m0 is NormFace (registry ArcFace head with m=0); ce has no features yet
# and is skipped as MISSING.
LOSSES = {"dbm": "registry", "kappaface": "registry", "arcface": "arcface",
          "cosine_m0": "registry", "ce": "registry"}
CHUNK = 2048


def read_directions(rung, loss):
    """(N, k, C) classifier rows from that rung's checkpoint.

    The class count comes from the head itself: the 15.5k, 32k and 64k rungs
    do not share it, and the highest validation label need not be the last
    class. Ladder runs use k=1.
    """
    ckpt = f"{OUT}/SWIN_BASE_LADDER_{rung.upper()}_{loss.upper()}/model.safetensors"
    with safe_open(ckpt, "pt") as f:
        keys = set(f.keys())
        # The ladder runs all go through the loss registry, so the classifier is
        # head.weight even for ArcFace. The native arcface.weight name is kept
        # as a fallback for checkpoints that use the built-in ArcFace path.
        if "head.weight" in keys:
            return f.get_tensor("head.weight").float().unsqueeze(1)
        if "arcface.weight" in keys:
            w = f.get_tensor("arcface.weight").float()
            return w.view(w.shape[0], 1, w.shape[1])
        raise KeyError(f"{ckpt}: no head.weight or arcface.weight")


def macro_f1(y, pred, n_classes):
    """Mean per-class F1 over classes present in labels OR predictions."""
    pred = np.asarray(pred, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    tp = np.bincount(y[pred == y], minlength=n_classes).astype(np.float64)
    n_true = np.bincount(y, minlength=n_classes).astype(np.float64)
    n_pred = np.bincount(pred, minlength=n_classes).astype(np.float64)
    present = (n_true + n_pred) > 0
    return float(np.mean(2 * tp[present] / (n_true[present] + n_pred[present])))


def angle_or_empty(cos, mask, n):
    """angle_stats, but an empty rarity band is reported as empty, not NaN."""
    if not mask.any():
        return {"n_images": 0}
    return angle_stats(cos, mask, n)


def fisher_band(fisher, train_count, lo, hi):
    m = (train_count >= lo) & (train_count <= hi)
    return {"n_species": int(m.sum()),
            "mean": float(fisher[m].mean()) if m.any() else float("nan")}


def fisher_class_scores(mu, sw, device, chunk=CHUNK):
    """Per-class mean pairwise Fisher score, without the N x N matrix."""
    a = torch.linalg.inv(sw)
    b = mu @ a
    diag = (mu * b).sum(1)
    n = len(mu)
    out = torch.empty(n, device=device)
    for lo in range(0, n, chunk):
        bc = b[lo:lo + chunk]
        d = diag[lo:lo + chunk, None] + diag[None, :] - 2.0 * (bc @ mu.T)
        out[lo:lo + chunk] = (d.sum(1) - d.diagonal()) / (n - 1)
    return out.cpu().numpy()


def run_cell(rung, loss, device):
    name = f"{rung}_{loss}"
    fpath = f"{FEAT}/{name}/features.npy"
    ppath = f"{PRED}/{name}/predictions.npz"
    if not (os.path.isfile(fpath) and os.path.isfile(ppath)):
        print(f"__LADDERDIAG__ MISSING {name}", flush=True)
        return None

    feat = np.load(fpath, mmap_mode="r")
    z = np.load(ppath)
    y = z["labels"].astype(np.int64)
    assert len(feat) == len(y), (name, len(feat), len(y))

    dirs = read_directions(rung, loss)
    n_classes = dirs.shape[0]
    if int(y.max()) >= n_classes:
        raise ValueError(f"{name}: label {int(y.max())} outside head ({n_classes})")

    train_count = np.bincount(
        [json.loads(line)[LABEL] for line in open(f"{DATA}/{rung}_train.json")
         if line.strip()], minlength=n_classes)

    # Margins and predictions, from the same cosine rule the model uses.
    s = boundary_scores(np.asarray(feat, dtype=np.float32), y, dirs, device)
    pred = s["predicted"].astype(np.int64)
    tc = train_count[y]

    result = {
        "n_images": int(len(y)),
        "n_classes": int(n_classes),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": macro_f1(y, pred, n_classes),
        "margin": band_stats(np.ones(len(y), bool), y, pred, s["cos_true"],
                             s["cos_best_other"], s["margin"]),
        "margin_bands": {f"{lo}-{hi}": band_stats(
            (tc >= lo) & (tc <= hi), y, pred, s["cos_true"], s["cos_best_other"],
            s["margin"]) for lo, hi in BANDS},
        "intra_classifier": {
            "overall": angle_stats(s["cos_true"], np.ones(len(y), bool), len(y)),
            "bands": {f"{lo}-{hi}": angle_or_empty(
                s["cos_true"], (tc >= lo) & (tc <= hi), len(y))
                for lo, hi in BANDS},
        },
    }

    # Feature statistics for the centroid angle and the Fisher score.
    x = F.normalize(torch.as_tensor(np.asarray(feat, dtype=np.float32),
                                   device=device), dim=1)
    y_t = torch.as_tensor(y, device=device)
    counts = torch.bincount(y_t, minlength=n_classes).float()
    mu = torch.zeros(n_classes, x.size(1), device=device)
    mu.index_add_(0, y_t, x)
    mu /= counts.clamp(min=1)[:, None]

    cos_centroid = (x * F.normalize(mu, dim=1)[y_t]).sum(1).cpu().numpy()
    result["intra_centroid"] = {
        "overall": angle_stats(cos_centroid, np.ones(len(y), bool), len(y)),
        "bands": {f"{lo}-{hi}": angle_or_empty(
            cos_centroid, (tc >= lo) & (tc <= hi), len(y)) for lo, hi in BANDS},
    }

    centered = x - mu[y_t]
    sw = centered.T @ centered / len(y)
    ridge = 1e-3 * torch.trace(sw) / sw.size(0)
    sw = sw + ridge * torch.eye(sw.size(0), device=device)
    fisher = fisher_class_scores(mu, sw, device)
    result["fisher"] = {
        "mean": float(fisher.mean()),
        "median": float(np.median(fisher)),
        "bands": {f"{lo}-{hi}": fisher_band(fisher, train_count, lo, hi)
                  for lo, hi in BANDS},
    }

    del feat, x, mu, centered, sw
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="*", default=["n15k", "n32k", "n64k"])
    ap.add_argument("--losses", nargs="*", default=list(LOSSES))
    ap.add_argument("--out", default="ladder_diagnostics.json",
                    help="file name under LADDER_GEOM/analysis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"__LADDERDIAG__ device={device}", flush=True)

    out = {}
    for rung in args.rungs:
        out[rung] = {}
        for loss in args.losses:
            r = run_cell(rung, loss, device)
            if r is None:
                continue
            out[rung][loss] = r
            print(f"__LADDERDIAG__ {rung} {loss}: acc={r['accuracy']:.4f} "
                  f"f1={r['macro_f1']:.4f} margin={r['margin']['mean_margin']:+.4f} "
                  f"err={r['margin']['frac_error']:.4f} "
                  f"angle={r['intra_classifier']['overall']['mean_deg']:.2f}deg "
                  f"fisher={r['fisher']['mean']:.1f}", flush=True)

    # The scale contrast: DBM and NormFace minus ArcFace on each rung.
    def delta(a, b):
        return {
            "macro_f1": a["macro_f1"] - b["macro_f1"],
            "margin": a["margin"]["mean_margin"] - b["margin"]["mean_margin"],
            "error": a["margin"]["frac_error"] - b["margin"]["frac_error"],
            "angle_deg": a["intra_classifier"]["overall"]["mean_deg"]
                         - b["intra_classifier"]["overall"]["mean_deg"],
            "fisher": a["fisher"]["mean"] - b["fisher"]["mean"],
        }

    for loss in ("dbm", "cosine_m0"):
        out[f"deltas_{loss}_minus_arcface"] = {
            rung: delta(cells[loss], cells["arcface"])
            for rung, cells in out.items()
            if isinstance(cells, dict) and loss in cells and "arcface" in cells}

    os.makedirs(f"{FEAT}/analysis", exist_ok=True)
    path = f"{FEAT}/analysis/{args.out}"
    json.dump(out, open(path, "w"), indent=1)
    print(f"__LADDERDIAG__ wrote {path}", flush=True)


if __name__ == "__main__":
    main()
