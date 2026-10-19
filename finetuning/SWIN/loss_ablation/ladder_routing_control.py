"""Control for the ladder routing result: does routing need KappaFace, or does any second
model help?

Routing rule (as in ladder_routing.py): take the SECOND model's answer if it names a species
with <= T training images, otherwise DBM's. Here it is run twice per rung -- second model =
KappaFace and second model = ArcFace -- on the SAME fixed half-split of the validation images
(T picked on half A, scored on half B), so the two are directly comparable.

Also reported, on half B, for each second model:
  disagree        share of images where its answer differs from DBM's
  rare_switch     share of images where routing actually switches to it (its answer is rare
                  and differs from DBM's)
  switch_right    among switched images: share where the second model is right
  switch_dbm_right  among switched images: share where DBM was right (what routing gives up)

Usage: python loss_ablation/ladder_routing_control.py   (CPU, numpy only)
"""

import json
import os

import numpy as np

LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
EV = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/LADDER_EVAL"
OUT = f"{EV}/analysis/ladder_routing_control.json"
BANDS = [(2, 5), (6, 10), (11, 25), (26, 100), (101, 10 ** 6)]
THRESHOLDS = [0, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
SECOND = ("kappaface", "arcface")


def macro_f1(y, p, n):
    tp = np.bincount(y[p == y], minlength=n).astype(float)
    nt = np.bincount(y, minlength=n).astype(float)
    npd = np.bincount(p, minlength=n).astype(float)
    keep = (nt + npd) > 0
    return float(np.mean(2 * tp[keep] / (nt[keep] + npd[keep])))


def band_f1(y, p, n, count):
    tp = np.bincount(y[p == y], minlength=n).astype(float)
    nt = np.bincount(y, minlength=n).astype(float)
    npd = np.bincount(p, minlength=n).astype(float)
    has = nt > 0
    f1 = np.full(n, np.nan)
    f1[has] = 2 * tp[has] / (nt[has] + npd[has])
    out = {}
    for lo, hi in BANDS:
        m = has & (count >= lo) & (count <= hi)
        if m.any():
            out[f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"] = float(np.mean(f1[m]))
    return out


def main():
    res = {}
    for rung in ("n15k", "n32k", "n64k"):
        with open(f"{LAD}/{rung}_train.json") as f:
            labels = [json.loads(l)["scientificNameEncoded"] for l in f if l.strip()]
        n = max(labels) + 1
        count = np.bincount(labels, minlength=n)
        P = {m: np.load(f"{EV}/{rung}_{m}/predictions.npz") for m in ("dbm",) + SECOND}
        y = P["dbm"]["labels"].astype(np.int64)
        for m in SECOND:
            assert np.array_equal(y, P[m]["labels"]), (rung, m)
        pd_ = P["dbm"]["predictions"].astype(np.int64)
        perm = np.random.default_rng(1234).permutation(len(y))  # fixed split, same for both
        a, b = np.sort(perm[: len(y) // 2]), np.sort(perm[len(y) // 2:])
        base = {"half_B_accuracy": float(np.mean(pd_[b] == y[b])), "half_B_macro_f1": macro_f1(y[b], pd_[b], n),
                "half_B_band_f1": band_f1(y[b], pd_[b], n, count)}
        r = {"dbm_alone": base}
        print(f"__RC2__ {rung}: DBM alone half B acc={base['half_B_accuracy']:.4f} mF1={base['half_B_macro_f1']:.4f} | "
              + " ".join(f"{k}={v:.4f}" for k, v in base["half_B_band_f1"].items()), flush=True)
        for m in SECOND:
            ps = P[m]["predictions"].astype(np.int64)

            def route(t):
                return np.where(count[ps] <= t, ps, pd_)

            tune = {t: macro_f1(y[a], route(t)[a], n) for t in THRESHOLDS}
            t_best = max(tune, key=tune.get)
            pr = route(t_best)
            sw = b[(count[ps[b]] <= t_best) & (ps[b] != pd_[b])]
            o = {"threshold": t_best,
                 "second_alone_half_B": {"accuracy": float(np.mean(ps[b] == y[b])), "macro_f1": macro_f1(y[b], ps[b], n)},
                 "routed_half_B": {"accuracy": float(np.mean(pr[b] == y[b])), "macro_f1": macro_f1(y[b], pr[b], n),
                                   "band_f1": band_f1(y[b], pr[b], n, count)},
                 "disagree": float(np.mean(ps[b] != pd_[b])), "rare_switch": len(sw) / len(b),
                 "switch_right": float(np.mean(ps[sw] == y[sw])) if len(sw) else None,
                 "switch_dbm_right": float(np.mean(pd_[sw] == y[sw])) if len(sw) else None}
            o["gain"] = {"accuracy": o["routed_half_B"]["accuracy"] - base["half_B_accuracy"],
                         "macro_f1": o["routed_half_B"]["macro_f1"] - base["half_B_macro_f1"]}
            r[m] = o
            print(f"__RC2__ {rung} second={m:9s} T*={t_best:3d} | routed acc={o['routed_half_B']['accuracy']:.4f} "
                  f"mF1={o['routed_half_B']['macro_f1']:.4f} (gain {100*o['gain']['accuracy']:+.2f} / "
                  f"{100*o['gain']['macro_f1']:+.2f}) | alone acc={o['second_alone_half_B']['accuracy']:.4f} "
                  f"mF1={o['second_alone_half_B']['macro_f1']:.4f} | disagree {100*o['disagree']:.1f}% "
                  f"switch {100*o['rare_switch']:.1f}% (second right {100*(o['switch_right'] or 0):.1f}%, "
                  f"DBM right {100*(o['switch_dbm_right'] or 0):.1f}%) | "
                  + " ".join(f"{k}={v:.4f}" for k, v in o["routed_half_B"]["band_f1"].items()), flush=True)
        res[rung] = r
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"__RC2__ wrote {OUT}")


if __name__ == "__main__":
    main()
