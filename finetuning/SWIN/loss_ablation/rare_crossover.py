"""Is KappaFace better than DBM on rare species and worse on common ones? And can a
simple routing rule keep the best of both?

Inputs: full-validation-set predictions saved by the Phase 3 geometry run
(PHASE3_GEOM/<model>/predictions.npz: predictions, labels, top5; val_2022 order) for two
seeds each of DBM, KappaFace and ArcFace; training-image counts from train_2022.json.

Part 1 -- crossover check. Mean per-species F1 in bands of training images, per seed, and
the KappaFace - DBM gap with a paired bootstrap over species (95% CI). Per-species F1 is
2*TP / (n_true + n_pred), over species with validation images, as in phase3_per_species.py.

Part 2 -- routing test (no retraining, predictions only). Final answer = KappaFace's top-1
if that species has <= T training images, else DBM's top-1. T is picked to maximise
macro-F1 on a random half of the validation images (half A) and scored on the other half
(half B), separately for each seed pair, so the reported gain is not tuned on the data it
is measured on.

Usage: python loss_ablation/rare_crossover.py   (CPU, numpy only)
"""

import json
import os

import numpy as np

GEOM = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_GEOM"
TRAIN = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
OUT = f"{GEOM}/analysis/rare_crossover.json"
BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 73)]
PAIRS = {"seed42": ("dbm", "kappaface", "arcface"),
         "seed43": ("dbm_seed43", "kappaface_seed43", "arcface_seed43")}
THRESHOLDS = [0, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 73]
N_BOOT = 2000


def per_species_f1(y, p, n):
    tp = np.bincount(y[p == y], minlength=n).astype(float)
    nt = np.bincount(y, minlength=n).astype(float)
    npd = np.bincount(p, minlength=n).astype(float)
    f1 = np.full(n, np.nan)
    has = nt > 0
    f1[has] = 2 * tp[has] / (nt[has] + npd[has])
    return f1


def macro_f1(y, p, n):
    tp = np.bincount(y[p == y], minlength=n).astype(float)
    nt = np.bincount(y, minlength=n).astype(float)
    npd = np.bincount(p, minlength=n).astype(float)
    keep = (nt + npd) > 0
    return float(np.mean(2 * tp[keep] / (nt[keep] + npd[keep])))


def band_mask(count, lo, hi):
    return (count >= lo) & (count <= hi)


def boot_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    with open(TRAIN) as f:
        labels = [json.loads(l)["scientificNameEncoded"] for l in f if l.strip()]
    n = max(labels) + 1
    count = np.bincount(labels, minlength=n)
    preds = {}
    y = None
    for names in PAIRS.values():
        for m in names:
            z = np.load(f"{GEOM}/{m}/predictions.npz")
            preds[m] = z["predictions"].astype(np.int64)
            yl = z["labels"].astype(np.int64)
            assert y is None or np.array_equal(y, yl), m
            y = yl
    rng = np.random.default_rng(0)
    res = {"bands": {}, "routing": {}}

    # ---- Part 1: crossover by band -------------------------------------------------
    f1 = {m: per_species_f1(y, p, n) for m, p in preds.items()}
    has = ~np.isnan(f1["dbm"])
    for lo, hi in BANDS:
        key = f"{lo}-{hi}"
        mask = band_mask(count, lo, hi) & has
        row = {"n_species": int(mask.sum())}
        diffs = []
        for seed, (d, k, a) in PAIRS.items():
            dk = f1[k][mask] - f1[d][mask]
            diffs.append(dk)
            row[seed] = {"dbm": float(np.mean(f1[d][mask])), "kappaface": float(np.mean(f1[k][mask])),
                         "arcface": float(np.mean(f1[a][mask])),
                         "kappa_minus_dbm": float(dk.mean()), "ci95": boot_ci(dk, rng)}
        both = (diffs[0] + diffs[1]) / 2
        row["both_seeds"] = {"kappa_minus_dbm": float(both.mean()), "ci95": boot_ci(both, rng)}
        res["bands"][key] = row
        print(f"__RC__ band {key:>5} (n={row['n_species']:5d}) "
              + "  ".join(f"{s}: DBM {row[s]['dbm']:.4f} Kappa {row[s]['kappaface']:.4f} "
                          f"K-D {100*row[s]['kappa_minus_dbm']:+.2f} [{100*row[s]['ci95'][0]:+.2f},{100*row[s]['ci95'][1]:+.2f}]"
                          for s in PAIRS)
              + f"  | both seeds K-D {100*both.mean():+.2f} [{100*row['both_seeds']['ci95'][0]:+.2f},"
                f"{100*row['both_seeds']['ci95'][1]:+.2f}]", flush=True)

    # ---- Part 2: routing -----------------------------------------------------------
    perm = rng.permutation(len(y))
    half_a, half_b = np.sort(perm[: len(y) // 2]), np.sort(perm[len(y) // 2:])
    for seed, (d, k, a) in PAIRS.items():
        pd_, pk = preds[d], preds[k]

        def route(t):
            return np.where(count[pk] <= t, pk, pd_)

        tune = {t: macro_f1(y[half_a], route(t)[half_a], n) for t in THRESHOLDS}
        t_best = max(tune, key=tune.get)
        out = {"threshold_tuned_on_half_A": t_best, "half_A_macro_f1_by_threshold": tune}
        for name, p in (("dbm", pd_), ("kappaface", pk), ("routed", route(t_best))):
            yb, pb = y[half_b], p[half_b]
            fb = per_species_f1(yb, pb, n)
            hb = ~np.isnan(fb)
            out[name] = {"half_B_accuracy": float(np.mean(pb == yb)), "half_B_macro_f1": macro_f1(yb, pb, n),
                         "half_B_band_f1": {f"{lo}-{hi}": float(np.mean(fb[band_mask(count, lo, hi) & hb]))
                                            for lo, hi in BANDS}}
        out["routed_share_of_images_from_kappaface"] = float(np.mean(count[pk[half_b]] <= t_best))
        res["routing"][seed] = out
        print(f"__RC__ routing {seed}: T*={t_best} (tuned on half A; A macro-F1 by T: "
              + ", ".join(f"{t}:{v:.4f}" for t, v in tune.items()) + ")", flush=True)
        for name in ("dbm", "kappaface", "routed"):
            o = out[name]
            print(f"__RC__ routing {seed} half B {name:9s}: acc={o['half_B_accuracy']:.4f} "
                  f"macroF1={o['half_B_macro_f1']:.4f} | "
                  + " ".join(f"{b}={v:.4f}" for b, v in o["half_B_band_f1"].items()), flush=True)
        print(f"__RC__ routing {seed}: {100*out['routed_share_of_images_from_kappaface']:.1f}% of half-B "
              f"images take KappaFace's answer", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"__RC__ wrote {OUT}")


if __name__ == "__main__":
    main()
