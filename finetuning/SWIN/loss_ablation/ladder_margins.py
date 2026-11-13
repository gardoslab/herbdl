"""Margins each loss actually trained with on the ladder, by training-image band.

Question: on Herbarium 2022 KappaFace beats DBM on the rarest species, but on the ladder it
trails DBM in every band. KappaFace's population weight is w_pop = (cos(pi * n / max_n) + 1) / 2.
On Herbarium 2022 max_n = 73, so w_pop falls from ~1 (rare) to ~0 (common). On the ladder max_n
is 1,717-2,532, so w_pop is ~1 for almost every species and the rare/common difference in
KappaFace's margin should mostly disappear. This reads the per-class margin KappaFace ended
training with (head.per_class_margin) and DBM's class margin 0.1 * n_min / n_y, per band.

Inputs: <OUT>/<run_dir>/model.safetensors, ladder training manifests (scientificNameEncoded =
the head's class index), train_2022.json for the base run.
Output: <OUT>/MARGIN_DIAG/ladder_margins.json, lines prefixed __LADMARGIN__.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from band_leads_by_rung import BANDS, LAD, OUT  # noqa: E402
from ladder_routing import band_name  # noqa: E402
from scale_free_margins import read_buffer  # noqa: E402

RESULT = f"{OUT}/MARGIN_DIAG/ladder_margins.json"
PREFIX = "__LADMARGIN__"
# rung -> (training manifest, KappaFace run dir)
RUNGS = {
    "base": ("/projectnb/herbdl/data/kaggle-herbaria/train_2022.json", "SWIN_BASE_KAPPAFACE_384_PILOT"),
    "n15k": (f"{LAD}/n15k_train.json", "SWIN_BASE_LADDER_N15K_KAPPAFACE"),
    "n32k": (f"{LAD}/n32k_train.json", "SWIN_BASE_LADDER_N32K_KAPPAFACE"),
    "n64k": (f"{LAD}/n64k_train.json", "SWIN_BASE_LADDER_N64K_KAPPAFACE"),
}


def main():
    out = {}
    for rung, (manifest, run_dir) in RUNGS.items():
        with open(manifest) as f:
            enc = np.array([json.loads(l)["scientificNameEncoded"] for l in f if l.strip()])
        count = np.bincount(enc)
        kappa = read_buffer(run_dir, "head.per_class_margin")
        if kappa is None:
            print(f"{PREFIX} MISSING {rung}: no head.per_class_margin in {run_dir}", flush=True)
            continue
        kappa = kappa[:len(count)].astype(np.float64)
        present = count > 0
        n_min, n_max = count[present].min(), count.max()
        dbm = 0.1 * n_min / np.maximum(count, 1)
        # population part of KappaFace's weight, to separate it from the concentration part
        w_pop = (np.cos(np.pi * count / n_max) + 1) / 2
        r = {"n_species": int(present.sum()), "n_min": int(n_min), "n_max": int(n_max), "bands": {}}
        print(f"{PREFIX} {rung}: {present.sum():,} species, n_min={n_min}, n_max={n_max} | "
              f"KappaFace mean {np.degrees(kappa[present]).mean():.2f} deg, "
              f"DBM mean {np.degrees(dbm[present]).mean():.2f} deg, ArcFace 17.19 deg", flush=True)
        for lo, hi in BANDS:
            m = present & (count >= lo) & (count <= hi)
            if not m.any():
                continue
            b = {"n_species": int(m.sum()),
                 "kappa_deg_mean": float(np.degrees(kappa[m]).mean()),
                 "kappa_deg_median": float(np.degrees(np.median(kappa[m]))),
                 "w_pop_mean": float(w_pop[m].mean()),
                 "dbm_deg_mean": float(np.degrees(dbm[m]).mean())}
            r["bands"][band_name(lo, hi)] = b
            print(f"{PREFIX} {rung} band {band_name(lo, hi):>7} (n={m.sum():>6,}): "
                  f"KappaFace {b['kappa_deg_mean']:5.2f} deg (median {b['kappa_deg_median']:5.2f}, "
                  f"w_pop {b['w_pop_mean']:.3f}) | DBM {b['dbm_deg_mean']:5.2f} deg", flush=True)
        out[rung] = r
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{PREFIX} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
