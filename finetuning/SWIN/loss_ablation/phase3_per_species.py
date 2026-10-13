"""Phase 3 per-species analysis of the 8 finished 384px loss models.

Reads the per-image validation predictions written by run_phase3_eval.sh
(PHASE3_EVAL/<model>/predictions.npz: predictions, labels, top5 in val-set order) and
the training manifest (for per-species training-image counts and the genus/family of
each species). numpy only; CPU.

For every model it reports:
  1. accuracy, macro-F1 and top-5 accuracy recomputed from the saved predictions,
     checked against that run's own final all_results.json (they must match);
  2. mean per-species F1 within bands of training-images-per-species (rare -> common);
  3. how often a wrong prediction is still in the right genus / family.
And for the key pairs (KappaFace vs DBM, each loss vs ArcFace): per-species wins/losses
and a paired bootstrap over validation images for the accuracy and macro-F1 gaps.
That bootstrap measures evaluation-set noise only, NOT seed-to-seed training noise --
the seed-43 reruns measure that.

Macro-F1 follows sklearn/evaluate's average='macro': the mean over every class that
appears in the labels OR the predictions (a class that is only ever predicted counts
with F1 = 0). This is what the training script's eval_f1 reports.

Usage: python loss_ablation/phase3_per_species.py   (paths below are the SCC ones)
"""

import json
import os
import sys

import numpy as np

EVAL_ROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_EVAL"
OUT_ROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
VAL_JSON = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
LABEL = "scientificNameEncoded"

# model name -> the training run's own output dir (for its all_results.json)
MODELS = {
    "arcface": "SWIN_BASE_ARCFACE_384",
    "subcenter_arcface": "SWIN_BASE_SUBCENTER_ARCFACE_384_PILOT",
    "hace": "SWIN_BASE_HACE_384_PILOT",
    "dbm": "SWIN_BASE_DBM_384_PILOT",
    "x2softmax": "SWIN_BASE_X2SOFTMAX_384_PILOT",
    "kappaface": "SWIN_BASE_KAPPAFACE_384_PILOT",
    "arcface_ecc": "SWIN_BASE_ARCFACE_ECC_384_PILOT",
    "arcface_tara": "SWIN_BASE_ARCFACE_TARA_384_PILOT",
}
# Bands of training images per species (counts run 2..73 in train_2022.json).
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
PAIRS = [("kappaface", "dbm"), ("kappaface", "arcface"), ("dbm", "arcface"),
         ("subcenter_arcface", "arcface"), ("arcface_tara", "arcface"),
         ("arcface_ecc", "arcface"), ("hace", "arcface")]
N_BOOT = 1000
SEED = 0


def load_rows(path):
    """train/val manifests are one JSON object per line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_class_counts(labels, preds, n_classes, weights=None):
    """tp, true count, predicted count per class (optionally image-weighted)."""
    correct = (labels == preds).astype(np.float64)
    w = np.ones_like(correct) if weights is None else weights
    tp = np.bincount(labels, weights=w * correct, minlength=n_classes)
    n_true = np.bincount(labels, weights=w, minlength=n_classes)
    n_pred = np.bincount(preds, weights=w, minlength=n_classes)
    return tp, n_true, n_pred


def macro_f1(tp, n_true, n_pred):
    present = (n_true + n_pred) > 0
    return float(np.mean(2 * tp[present] / (n_true[present] + n_pred[present])))


def per_species_f1(tp, n_true, n_pred):
    """F1 per class; NaN for classes with no validation images (not scored per species)."""
    f1 = np.full(len(tp), np.nan)
    has = n_true > 0
    f1[has] = 2 * tp[has] / (n_true[has] + n_pred[has])
    return f1


def main():
    train = load_rows(TRAIN_JSON)
    n_classes = max(r[LABEL] for r in train) + 1
    train_count = np.bincount([r[LABEL] for r in train], minlength=n_classes)
    genus_of, family_of = {}, {}
    for r in train:
        genus_of[r[LABEL]] = (r["family"], r["genus"])
        family_of[r[LABEL]] = r["family"]
    genus_ids = {g: i for i, g in enumerate(sorted(set(genus_of.values())))}
    family_ids = {f: i for i, f in enumerate(sorted(set(family_of.values())))}
    cls_genus = np.full(n_classes, -1)
    cls_family = np.full(n_classes, -1)
    for c, g in genus_of.items():
        cls_genus[c] = genus_ids[g]
        cls_family[c] = family_ids[family_of[c]]
    val_labels = np.array([r[LABEL] for r in load_rows(VAL_JSON)])
    print(f"__P3__ classes={n_classes} train_images={len(train)} val_images={len(val_labels)} "
          f"genera={len(genus_ids)} families={len(family_ids)}")

    preds, f1s, results = {}, {}, {}
    for name, run_dir in MODELS.items():
        path = os.path.join(EVAL_ROOT, name, "predictions.npz")
        if not os.path.isfile(path):
            print(f"__P3__ MISSING {name}: {path}")
            continue
        z = np.load(path)
        p, y, top5 = z["predictions"].astype(np.int64), z["labels"].astype(np.int64), z["top5"]
        if not np.array_equal(y, val_labels):
            sys.exit(f"{name}: saved labels are not val_2022.json in order -- refusing to analyze")
        tp, nt, npd = per_class_counts(y, p, n_classes)
        acc, mf1 = float(np.mean(p == y)), macro_f1(tp, nt, npd)
        ref = json.load(open(os.path.join(OUT_ROOT, run_dir, "all_results.json")))
        wrong = p != y
        r = {
            "accuracy": acc, "macro_f1": mf1,
            "top5_accuracy": float(np.mean((top5 == y[:, None]).any(axis=1))),
            "training_run_accuracy": ref["eval_accuracy"], "training_run_macro_f1": ref["eval_f1"],
            "matches_training_run": bool(abs(acc - ref["eval_accuracy"]) < 1e-6
                                         and abs(mf1 - ref["eval_f1"]) < 1e-6),
            "errors_in_right_genus": float(np.mean(cls_genus[p[wrong]] == cls_genus[y[wrong]])),
            "errors_in_right_family": float(np.mean(cls_family[p[wrong]] == cls_family[y[wrong]])),
        }
        f1 = per_species_f1(tp, nt, npd)
        r["bands"] = {}
        for lo, hi in BANDS:
            m = (train_count >= lo) & (train_count <= hi) & ~np.isnan(f1)
            r["bands"][f"{lo}-{hi}"] = {"n_species": int(m.sum()), "mean_species_f1": float(np.mean(f1[m]))}
        preds[name], f1s[name], results[name] = p, f1, r
        print(f"__P3__ {name}: acc={acc:.4f} macroF1={mf1:.4f} top5={r['top5_accuracy']:.4f} "
              f"matches_training_run={r['matches_training_run']} "
              f"errors_same_genus={r['errors_in_right_genus']:.3f} errors_same_family={r['errors_in_right_family']:.3f} | "
              + " ".join(f"band{k}={v['mean_species_f1']:.4f}(n={v['n_species']})" for k, v in r["bands"].items()))

    n = len(val_labels)

    def boot_weights(i):
        # Resample i is regenerated from (SEED, i), so every pair sees the same resamples
        # without holding all N_BOOT weight vectors in memory at once.
        idx = np.random.default_rng([SEED, i]).integers(0, n, size=n)
        return np.bincount(idx, minlength=n).astype(np.float64)

    pairs = {}
    for a, b in PAIRS:
        if a not in preds or b not in preds:
            continue
        d_acc, d_f1 = [], []
        for i in range(N_BOOT):  # same resampled images for both models (paired)
            w = boot_weights(i)
            fa = macro_f1(*per_class_counts(val_labels, preds[a], n_classes, w))
            fb = macro_f1(*per_class_counts(val_labels, preds[b], n_classes, w))
            d_f1.append(fa - fb)
            d_acc.append((np.sum(w * (preds[a] == val_labels)) - np.sum(w * (preds[b] == val_labels))) / n)
        both = ~np.isnan(f1s[a]) & ~np.isnan(f1s[b])
        diff = f1s[a][both] - f1s[b][both]
        band_delta = {k: results[a]["bands"][k]["mean_species_f1"] - results[b]["bands"][k]["mean_species_f1"]
                      for k in results[a]["bands"]}
        pr = {
            "accuracy_gap": results[a]["accuracy"] - results[b]["accuracy"],
            "accuracy_gap_95ci": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
            "macro_f1_gap": results[a]["macro_f1"] - results[b]["macro_f1"],
            "macro_f1_gap_95ci": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
            "species_better": int(np.sum(diff > 0)), "species_worse": int(np.sum(diff < 0)),
            "species_tied": int(np.sum(diff == 0)),
            "band_mean_species_f1_gap": band_delta,
        }
        pairs[f"{a}_vs_{b}"] = pr
        print(f"__P3__ PAIR {a} vs {b}: acc_gap={pr['accuracy_gap']:+.4f} "
              f"[{pr['accuracy_gap_95ci'][0]:+.4f},{pr['accuracy_gap_95ci'][1]:+.4f}] "
              f"macroF1_gap={pr['macro_f1_gap']:+.4f} "
              f"[{pr['macro_f1_gap_95ci'][0]:+.4f},{pr['macro_f1_gap_95ci'][1]:+.4f}] "
              f"species better/worse/tied={pr['species_better']}/{pr['species_worse']}/{pr['species_tied']} | "
              + " ".join(f"band{k}={v:+.4f}" for k, v in band_delta.items()))

    out = os.path.join(EVAL_ROOT, "analysis")
    os.makedirs(out, exist_ok=True)
    json.dump({"models": results, "pairs": pairs, "bands": BANDS, "n_bootstrap": N_BOOT,
               "bootstrap_note": "paired image-level bootstrap of the validation set; "
                                 "evaluation-set noise only, not seed noise"},
              open(os.path.join(out, "per_species_summary.json"), "w"), indent=1)
    # One row per species: training count, val count, F1 under every model.
    names = list(f1s)
    with open(os.path.join(out, "per_species_f1.csv"), "w") as f:
        f.write("species_id,family,genus,train_images,val_images," + ",".join(f"f1_{m}" for m in names) + "\n")
        n_val = np.bincount(val_labels, minlength=n_classes)
        for c in range(n_classes):
            fam, gen = genus_of.get(c, ("", ""))
            vals = ",".join("" if np.isnan(f1s[m][c]) else f"{f1s[m][c]:.4f}" for m in names)
            f.write(f"{c},{fam},{gen},{train_count[c]},{n_val[c]},{vals}\n")
    print(f"__P3__ wrote {out}/per_species_summary.json and per_species_f1.csv")


if __name__ == "__main__":
    main()
