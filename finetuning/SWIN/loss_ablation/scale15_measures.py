"""Scale control (s=15): macro-F1, top-1, DBM-paper angle and Fisher score on the 15.5k validation set.

Reads PHASE3_GEOM/<name>/{features.npy, predictions.npz} for the s=15 models and their s=30
counterparts and prints one line per model. Macro-F1 follows the paper (mean over classes present
in labels or predictions). Angle and Fisher use paper_measures.measure() unchanged.
Output: lines prefixed __SCALE15__; MARGIN_DIAG/scale15.json.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_measures as pm  # noqa: E402

pm.RUNS["arcface_s15"] = ("SWIN_BASE_ARCFACE_S15_384", "arcface")
pm.RUNS["cosine_m0_s15"] = ("SWIN_BASE_COSINE_M0_S15_384", "registry")
MODELS = [("cosine_m0", "NormFace s=30"), ("cosine_m0_s15", "NormFace s=15"),
          ("arcface", "ArcFace s=30"), ("arcface_s15", "ArcFace s=15")]
RESULT = f"{pm.OUT}/MARGIN_DIAG/scale15.json"


def macro_f1(y, p):
    n = int(max(y.max(), p.max())) + 1
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    nt, npred = np.bincount(y, minlength=n), np.bincount(p, minlength=n)
    m = (nt + npred) > 0
    return float(np.mean(2 * tp[m] / (nt[m] + npred[m])))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = np.bincount([json.loads(l)[pm.LABEL] for l in open(pm.TRAIN_JSON) if l.strip()], minlength=pm.N)
    out = {}
    for name, label in MODELS:
        ppath = f"{pm.GEOM}/{name}/predictions.npz"
        if not os.path.isfile(ppath):
            print(f"__SCALE15__ {label}: not scored yet", flush=True)
            continue
        z = np.load(ppath)
        y, p = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
        r = pm.measure("val", name, tc, device)
        out[name] = {"top1": float(np.mean(y == p)), "macro_f1": macro_f1(y, p),
                     "angle_deg": r["angle_to_weight"]["overall"]["mean_deg"],
                     "fisher": r["fisher"]["overall"]["mean"]}
        o = out[name]
        print(f"__SCALE15__ {label}: top1={o['top1']:.4f} macroF1={o['macro_f1']:.4f} "
              f"angle={o['angle_deg']:.2f}deg fisher={o['fisher']:.1f}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    if "arcface_s15" in out and "cosine_m0_s15" in out:
        d15 = 100 * (out["cosine_m0_s15"]["macro_f1"] - out["arcface_s15"]["macro_f1"])
        d30 = 100 * (out["cosine_m0"]["macro_f1"] - out["arcface"]["macro_f1"])
        print(f"__SCALE15__ NormFace - ArcFace macro-F1: s=15 {d15:+.2f} points, s=30 {d30:+.2f} points", flush=True)
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(out, open(RESULT, "w"), indent=1)
    print(f"__SCALE15__ wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
