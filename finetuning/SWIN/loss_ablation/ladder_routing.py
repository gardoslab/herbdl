"""Routing test on the scaling ladder: does "KappaFace's answer for rare species, DBM's
otherwise" help more as rare species are added?

Same method as rare_crossover.py (the 15.5k routing test on Herbarium 2022), applied to each
ladder rung (n15k / n32k / n64k, merged 2021+2022 data). Inputs: per-image predictions of
the finished DBM and KappaFace ladder models on their rung's full validation set
(LADDER_EVAL/<rung>_{dbm,kappaface}/predictions.npz, <rung>_val order) and training-image
counts per species from <rung>_train.json.

Part 1 -- KappaFace - DBM per-species F1 by band of training images, with a bootstrap over
species (95% CI). Per-species F1 = 2*TP / (n_true + n_pred), species with val images only.
Part 2 -- routing: take KappaFace's prediction if it names a species with <= T training
images, else DBM's. T is picked to maximise macro-F1 on a random half of the val images
(half A) and scored on the other half (half B), so the gain is not tuned on the data it is
measured on. One seed per model; no seed noise estimate at these rungs.

Usage: python loss_ablation/ladder_routing.py   (CPU, numpy only)
"""

import json
import os

import numpy as np

LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
EV = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/LADDER_EVAL"
OUT = f"{EV}/analysis/ladder_routing.json"
BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 100), (101, 10 ** 6)]
THRESHOLDS = [0, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
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


def band_name(lo, hi):
    return f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


def main():
    rng = np.random.default_rng(0)
    res = {}
    for rung in ("n15k", "n32k", "n64k"):
        with open(f"{LAD}/{rung}_train.json") as f:
            labels = [json.loads(l)["scientificNameEncoded"] for l in f if l.strip()]
        n = max(labels) + 1
        count = np.bincount(labels, minlength=n)
        zd = np.load(f"{EV}/{rung}_dbm/predictions.npz")
        zk = np.load(f"{EV}/{rung}_kappaface/predictions.npz")
        y = zd["labels"].astype(np.int64)
        assert np.array_equal(y, zk["labels"]), rung
        pd_, pk = zd["predictions"].astype(np.int64), zk["predictions"].astype(np.int64)
        r = {"n_species": int(n), "val_images": int(len(y)),
             "dbm": {"accuracy": float(np.mean(pd_ == y)), "macro_f1": macro_f1(y, pd_, n)},
             "kappaface": {"accuracy": float(np.mean(pk == y)), "macro_f1": macro_f1(y, pk, n)},
             "bands": {}}
        print(f"__LR__ {rung}: {n:,} species, {len(y):,} val images | DBM acc {r['dbm']['accuracy']:.4f} "
              f"mF1 {r['dbm']['macro_f1']:.4f} | KappaFace acc {r['kappaface']['accuracy']:.4f} "
              f"mF1 {r['kappaface']['macro_f1']:.4f}", flush=True)

        fd, fk = per_species_f1(y, pd_, n), per_species_f1(y, pk, n)
        has = ~np.isnan(fd)
        for lo, hi in BANDS:
            m = has & (count >= lo) & (count <= hi)
            if m.sum() == 0:
                continue
            d = fk[m] - fd[m]
            boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            r["bands"][band_name(lo, hi)] = {"n_species": int(m.sum()), "dbm": float(fd[m].mean()),
                                             "kappaface": float(fk[m].mean()),
                                             "kappa_minus_dbm": float(d.mean()), "ci95": ci}
            print(f"__LR__ {rung} band {band_name(lo, hi):>7} (n={m.sum():6,}): DBM {fd[m].mean():.4f} "
                  f"Kappa {fk[m].mean():.4f}  K-D {100*d.mean():+.2f} [{100*ci[0]:+.2f},{100*ci[1]:+.2f}]",
                  flush=True)

        perm = rng.permutation(len(y))
        a, b = np.sort(perm[: len(y) // 2]), np.sort(perm[len(y) // 2:])

        def route(t):
            return np.where(count[pk] <= t, pk, pd_)

        tune = {t: macro_f1(y[a], route(t)[a], n) for t in THRESHOLDS}
        t_best = max(tune, key=tune.get)
        rt = {"threshold_tuned_on_half_A": t_best, "half_A_macro_f1_by_threshold": tune}
        for name, p in (("dbm", pd_), ("kappaface", pk), ("routed", route(t_best))):
            fb = per_species_f1(y[b], p[b], n)
            hb = ~np.isnan(fb)
            rt[name] = {"half_B_accuracy": float(np.mean(p[b] == y[b])),
                        "half_B_macro_f1": macro_f1(y[b], p[b], n),
                        "half_B_band_f1": {band_name(lo, hi): float(np.mean(fb[hb & (count >= lo) & (count <= hi)]))
                                           for lo, hi in BANDS if np.any(hb & (count >= lo) & (count <= hi))}}
        rt["share_from_kappaface"] = float(np.mean(count[pk[b]] <= t_best))
        r["routing"] = rt
        print(f"__LR__ {rung} routing: T*={t_best}; half-A macro-F1 by T: "
              + ", ".join(f"{t}:{v:.4f}" for t, v in tune.items()), flush=True)
        for name in ("dbm", "kappaface", "routed"):
            o = rt[name]
            print(f"__LR__ {rung} half B {name:9s}: acc={o['half_B_accuracy']:.4f} mF1={o['half_B_macro_f1']:.4f} | "
                  + " ".join(f"{k}={v:.4f}" for k, v in o["half_B_band_f1"].items()), flush=True)
        print(f"__LR__ {rung}: {100*rt['share_from_kappaface']:.1f}% of half-B images take KappaFace's answer",
              flush=True)
        res[rung] = r

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"__LR__ wrote {OUT}")


if __name__ == "__main__":
    main()
