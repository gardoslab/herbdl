"""Ladder geometry on validation and training images, every loss (supplement).

For each (rung, loss) and split: compactness angle to the species' weight (overall and by training
images per species), bias-corrected Fisher (geometry_robustness_part2.fisher_variants), the split of
the mean cosine into within-species spread and weight-centre alignment, error and macro-F1; and the
change from training to validation images when both splits exist. Same estimators as the base-data
tables (see fisher-corrected-number.md), so numbers are comparable.

Inputs: validation  LADDER_GEOM/<rung>_<loss>/features.npy + LADDER_EVAL/<rung>_<loss>/predictions.npz
        training    LADDER_GEOM_TRAIN/<rung>_<loss>/{features.npy, predictions.npz}
Classifier rows: head.weight (cosine heads) or head.linear.weight (plain CE; bias ignored).
Output: lines prefixed __LADDERFULL__; LADDER_GEOM/analysis/ladder_geometry_full.json (merged, so
cells can be added in later passes).
Usage: python loss_ablation/analysis/ladder_geometry_full.py --rungs n15k --losses dbm arcface --splits val train
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
import geometry_robustness as g1  # noqa: E402
import geometry_robustness_part2 as g2  # noqa: E402
import ladder_diagnostics as ld  # noqa: E402
from paper_separability import angle_stats  # noqa: E402

OUT = ld.OUT
ROOT = os.path.dirname(OUT)  # .../finetuning/output
PATHS = {"val": (f"{OUT}/LADDER_GEOM", f"{OUT}/LADDER_EVAL"),
         "train": (f"{OUT}/LADDER_GEOM_TRAIN", f"{OUT}/LADDER_GEOM_TRAIN")}
RESULT = f"{OUT}/LADDER_GEOM/analysis/ladder_geometry_full.json"
P = "__LADDERFULL__"


BACKBONES = ("convnext_", "dinov2_")


def split_backbone(loss):
    """('convnext', 'dbm') for 'convnext_dbm'; (None, loss) for the SWIN ladder losses."""
    for p in BACKBONES:
        if loss.startswith(p):
            return p[:-1], loss[len(p):]
    return None, loss


def cell_name(rung, loss):
    """Feature folder name: <rung>_<loss> for SWIN, <backbone>_<rung>_<loss> for losses given as <backbone>_<loss>."""
    bb, l = split_backbone(loss)
    return f"{bb}_{rung}_{l}" if bb else f"{rung}_{loss}"


def classifier(rung, loss, device):
    bb, l = split_backbone(loss)
    if bb:  # ConvNeXt / DINOv2 ladder models (configs_advanced/backbones/generate_ladder_all.py)
        tag = f"LADDER_{rung.upper()}_{l.upper()}" if bb == "convnext" else f"LADDER_DINOV2_{rung.upper()}_{l.upper()}"
        ckpt = f"{ROOT}/BACKBONES/{tag}/model.safetensors"
    else:
        ckpt = f"{OUT}/SWIN_BASE_LADDER_{rung.upper()}_{loss.upper()}/model.safetensors"
    with safe_open(ckpt, "pt") as f:
        keys = set(f.keys())
        key = next(k for k in ("head.weight", "head.linear.weight", "arcface.weight") if k in keys)
        w = f.get_tensor(key).float()
    return F.normalize(w.to(device), dim=1).unsqueeze(1)


def macro_f1(y, p):
    n = int(max(y.max(), p.max())) + 1
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    nt, npred = np.bincount(y, minlength=n), np.bincount(p, minlength=n)
    m = (nt + npred) > 0
    return float(np.mean(2 * tp[m] / (nt[m] + npred[m])))


def cell(rung, loss, split, train_count, device):
    fdir, pdir = PATHS[split]
    name = cell_name(rung, loss)
    if split_backbone(loss)[0]:
        pdir = fdir  # the ConvNeXt feature passes save predictions next to the features
    fp, pp = f"{fdir}/{name}/features.npy", f"{pdir}/{name}/predictions.npz"
    if not (os.path.isfile(fp) and os.path.isfile(pp)):
        print(f"{P} MISSING {name} {split}", flush=True)
        return None
    w = classifier(rung, loss, device)
    n_classes = w.size(0)
    train_count = np.pad(train_count[:n_classes], (0, max(0, n_classes - len(train_count))))
    z = np.load(pp)
    y_np, p_np = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    x, y = g2.load(fp, pp, device)
    n = len(y_np)
    cos_true = torch.empty(n, device=device)
    for s in range(0, n, 65536):
        cos_true[s:s + 65536] = torch.einsum("bc,bkc->bk", x[s:s + 65536], w[y[s:s + 65536]]).max(1).values
    cos_true = cos_true.cpu().numpy()
    tc = train_count[y_np]
    angle = {"overall": angle_stats(cos_true, np.ones(n, bool), n)["mean_deg"]}
    for lo, hi in ld.BANDS:
        m = (tc >= lo) & (tc <= hi)
        angle[f"{lo}-{hi}"] = angle_stats(cos_true, m, n)["mean_deg"] if m.any() else None
    fv, counts, meta, _ = g2.fisher_variants(x, y, n_classes, with_original=False)
    ok = counts >= 2
    fisher = {k: g2.summarise(v, train_count, ld.BANDS, ok) for k, v in fv.items()}
    g1.N = n_classes  # compactness() sizes its per-species arrays from this module global
    comp = g1.compactness(x, y, w)
    cs = {k: g2.summarise(v, train_count, ld.BANDS, ok) for k, v in comp.items() if k in ("cos_w", "R2")}
    cs["align"] = {b: (cs["cos_w"][b] / np.sqrt(cs["R2"][b]) if cs["R2"][b] and cs["R2"][b] > 0 else None)
                   for b in cs["R2"]}
    res = {"n_images": n, "n_classes": n_classes, "error_pct": float(100 * np.mean(p_np != y_np)),
           "macro_f1": macro_f1(y_np, p_np), "angle_deg": angle, "fisher": fisher, "fisher_meta": meta,
           "compactness": cs}
    print(f"{P} {rung}_{loss} {split}: n={n:,} err={res['error_pct']:.2f}% macroF1={res['macro_f1']:.4f} "
          f"angle={angle['overall']:.1f}deg Fisher corrected={fisher['corrected']['all']:.0f} "
          f"(bias {meta['mean_bias']:.0f}) R2={cs['R2']['all']:.3f} align={cs['align']['all']:.3f}", flush=True)
    del x, y, w
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["n15k", "n32k", "n64k"])
    ap.add_argument("--losses", nargs="+", default=["cosine_m0", "dbm", "kappaface", "arcface", "ce"])
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    out = json.load(open(RESULT)) if os.path.isfile(RESULT) else {}
    for rung in args.rungs:
        train_count = None
        for loss in args.losses:
            if train_count is None:
                train_count = np.bincount([json.loads(l)[ld.LABEL] for l in open(f"{ld.DATA}/{rung}_train.json")
                                           if l.strip()])
            entry = out.setdefault(rung, {}).setdefault(loss, {})
            for split in args.splits:
                r = cell(rung, loss, split, train_count, device)
                if r is not None:
                    entry[split] = r
            if "val" in entry and "train" in entry:
                v, t = entry["val"]["fisher"]["corrected"], entry["train"]["fisher"]["corrected"]
                entry["train_to_val_fisher_change_pct"] = {b: (100 * (v[b] / t[b] - 1) if v[b] and t[b] else None)
                                                           for b in v}
                print(f"{P} {rung}_{loss} train->val corrected Fisher "
                      f"{entry['train_to_val_fisher_change_pct']['all']:+.1f}% | angle "
                      f"{entry['train']['angle_deg']['overall']:.1f} -> {entry['val']['angle_deg']['overall']:.1f} deg",
                      flush=True)
            os.makedirs(os.path.dirname(RESULT), exist_ok=True)
            json.dump(out, open(RESULT, "w"), indent=1)
    print(f"{P} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
