"""Paired bootstrap intervals for the two margin controls in the paper's main table (15.5k base data).

Same procedure as phase3_per_species.py: resample the 167,955 validation images with replacement
(1000 resamples, resample i drawn from rng([SEED, i]) so every pair sees the same resamples), and
recompute macro-F1 (mean over classes present in labels or predictions, as in sklearn 'macro') of
both models on each resample. DBM - ArcFace is included as a check against the reported
[1.70, 2.14]. Measures evaluation-set noise only, not seed-to-seed noise.

Input:  PHASE3_GEOM/<model>/predictions.npz (labels, predictions in val order)
Output: lines prefixed __CI__; <OUT>/MARGIN_DIAG/main_table_ci.json
"""

import json
import os

import numpy as np

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
GEOM = f"{OUT}/PHASE3_GEOM"
RESULT = f"{OUT}/MARGIN_DIAG/main_table_ci.json"
PAIRS = [("dbm", "arcface"), ("cosine_m0", "arcface"), ("arcface_m01", "arcface")]
N_BOOT, SEED = 1000, 0


def per_class_counts(labels, preds, n, w):
    correct = (labels == preds).astype(np.float64)
    tp = np.bincount(labels, weights=w * correct, minlength=n)
    n_true = np.bincount(labels, weights=w, minlength=n)
    n_pred = np.bincount(preds, weights=w, minlength=n)
    return tp, n_true, n_pred


def macro_f1(tp, n_true, n_pred):
    present = (n_true + n_pred) > 0
    return float(np.mean(2 * tp[present] / (n_true[present] + n_pred[present])))


def main():
    preds, y = {}, None
    for name in sorted({m for p in PAIRS for m in p}):
        z = np.load(f"{GEOM}/{name}/predictions.npz")
        yy = z["labels"].astype(np.int64)
        y = yy if y is None else y
        assert np.array_equal(y, yy), f"labels differ for {name}"
        preds[name] = z["predictions"].astype(np.int64)
    n_img = len(y)
    n = int(max(y.max(), *(p.max() for p in preds.values()))) + 1
    ones = np.ones(n_img)
    point = {k: macro_f1(*per_class_counts(y, p, n, ones)) for k, p in preds.items()}
    for k, v in point.items():
        print(f"__CI__ {k}: accuracy={np.mean(preds[k] == y):.4f} macroF1={v:.4f}", flush=True)
    diffs = {f"{a}-{b}": [] for a, b in PAIRS}
    for i in range(N_BOOT):
        idx = np.random.default_rng([SEED, i]).integers(0, n_img, size=n_img)
        w = np.bincount(idx, minlength=n_img).astype(np.float64)
        f = {k: macro_f1(*per_class_counts(y, p, n, w)) for k, p in preds.items()}
        for a, b in PAIRS:
            diffs[f"{a}-{b}"].append(f[a] - f[b])
    res = {"n_val_images": n_img, "macro_f1": point, "pairs": {}}
    for a, b in PAIRS:
        d = np.array(diffs[f"{a}-{b}"]) * 100
        gap = 100 * (point[a] - point[b])
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        res["pairs"][f"{a}-{b}"] = {"gap_points": gap, "ci95_points": ci}
        print(f"__CI__ {a} - {b}: {gap:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]", flush=True)
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"__CI__ wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
