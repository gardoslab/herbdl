"""Geometry of the ConvNeXt / DINOv2 models on the 15.5k base data (supplement).

Same measures as the SWIN analysis, so the tables line up:
  compactness   mean angle between an image's L2-normalised feature and its species' classifier
                weight (degrees), overall and by training images per species;
  separability  bias-corrected pairwise Fisher score (geometry_robustness_part2.fisher_variants:
                present species only, covariance / (n - C), minus d_eff (1/n_i + 1/n_j)); the
                uncorrected "original" value is kept for reference;
  split         mean cos to the weight = within-species spread (R^2, unbiased) x weight-centre
                alignment (geometry_robustness.compactness);
plus macro-F1 and error from the saved predictions, and the change from training to validation.
CE and CE+LA use a linear head (with bias); their "weight" is the normalised row of that linear
layer and the bias is ignored, so their angles are not directly comparable with the cosine heads.

Inputs: PHASE4_GEOM{,_TRAIN}/<name>/{features.npy, predictions.npz} (configs_advanced/backbones/geom)
and BACKBONES/<name>/model.safetensors. Output: lines prefixed __BACKGEOM__ and
MARGIN_DIAG/backbone_geometry.json (merged with any earlier results, so models can be added later).
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
import paper_measures as pm  # noqa: E402
from paper_separability import angle_stats  # noqa: E402

ROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output"
GEOM = {"val": f"{ROOT}/SWIN/PHASE4_GEOM", "train": f"{ROOT}/SWIN/PHASE4_GEOM_TRAIN"}
RESULT = f"{ROOT}/SWIN/MARGIN_DIAG/backbone_geometry.json"
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
P = "__BACKGEOM__"


def classifier(name, device):
    """(N, 1, C) unit-norm classifier rows: head.weight (cosine heads) or head.linear.weight (CE)."""
    with safe_open(f"{ROOT}/BACKBONES/{name}/model.safetensors", "pt") as f:
        keys = set(f.keys())
        key = "head.weight" if "head.weight" in keys else "head.linear.weight"
        w = f.get_tensor(key).float()
    return F.normalize(w.to(device), dim=1).unsqueeze(1), key


def macro_f1(y, p):
    n = int(max(y.max(), p.max())) + 1
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    nt, npred = np.bincount(y, minlength=n), np.bincount(p, minlength=n)
    m = (nt + npred) > 0
    return float(np.mean(2 * tp[m] / (nt[m] + npred[m])))


def measure(name, split, train_count, device):
    fp, pp = f"{GEOM[split]}/{name}/features.npy", f"{GEOM[split]}/{name}/predictions.npz"
    if not (os.path.isfile(fp) and os.path.isfile(pp)):
        print(f"{P} MISSING {name} {split}", flush=True)
        return None
    z = np.load(pp)
    y_np, p_np = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    x, y = g2.load(fp, pp, device)
    w, key = classifier(name, device)
    n = len(y_np)

    cos_true = torch.empty(n, device=device)
    for s in range(0, n, 65536):
        cos_true[s:s + 65536] = torch.einsum("bc,bkc->bk", x[s:s + 65536], w[y[s:s + 65536]]).max(1).values
    cos_true = cos_true.cpu().numpy()
    tc = train_count[y_np]
    angle = {"overall": angle_stats(cos_true, np.ones(n, bool), n)["mean_deg"]}
    for lo, hi in BANDS:
        m = (tc >= lo) & (tc <= hi)
        angle[f"{lo}-{hi}"] = angle_stats(cos_true, m, n)["mean_deg"] if m.any() else None

    fv, counts, meta, _ = g2.fisher_variants(x, y, pm.N)
    ok = counts >= 2
    fisher = {k: g2.summarise(v, train_count, BANDS, ok) for k, v in fv.items() if k != "original"}
    fisher["original_all_species_mean"] = float(np.mean(fv["original"]))
    comp = g1.compactness(x, y, w)
    comp_s = {k: g2.summarise(v, train_count, BANDS, ok) for k, v in comp.items() if k in ("cos_w", "R2")}
    comp_s["align"] = {b: (comp_s["cos_w"][b] / np.sqrt(comp_s["R2"][b]) if comp_s["R2"][b] and comp_s["R2"][b] > 0
                           else None) for b in comp_s["R2"]}
    res = {"n_images": n, "feature_dim": int(x.size(1)), "classifier_key": key,
           "error_pct": float(100 * np.mean(p_np != y_np)), "macro_f1": macro_f1(y_np, p_np),
           "angle_deg": angle, "fisher": fisher, "fisher_meta": meta, "compactness": comp_s}
    print(f"{P} {name} {split}: n={n:,} dim={x.size(1)} err={res['error_pct']:.2f}% macroF1={res['macro_f1']:.4f} "
          f"angle={angle['overall']:.1f}deg Fisher orig={fisher['original_all_species_mean']:.0f} "
          f"corrected={fisher['corrected']['all']:.0f} (d_eff={meta['d_eff']:.0f}) R2={comp_s['R2']['all']:.3f} "
          f"align={comp_s['align']['all']:.3f}", flush=True)
    del x, y, w
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    train_count = np.bincount([json.loads(l)[pm.LABEL] for l in open(pm.TRAIN_JSON) if l.strip()], minlength=pm.N)
    out = json.load(open(RESULT)) if os.path.isfile(RESULT) else {}
    for name in args.models:
        r = {s: measure(name, s, train_count, device) for s in ("val", "train")}
        if r["val"] and r["train"]:
            v, t = r["val"]["fisher"]["corrected"], r["train"]["fisher"]["corrected"]
            r["train_to_val_fisher_change_pct"] = {b: (100 * (v[b] / t[b] - 1) if v[b] and t[b] else None) for b in v}
            print(f"{P} {name} train->val corrected Fisher {r['train_to_val_fisher_change_pct']['all']:+.1f}% | angle "
                  f"{r['train']['angle_deg']['overall']:.1f} -> {r['val']['angle_deg']['overall']:.1f} deg", flush=True)
        out[name] = r
        os.makedirs(os.path.dirname(RESULT), exist_ok=True)
        json.dump(out, open(RESULT, "w"), indent=1)
    print(f"{P} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
