"""DBM-paper angle and Fisher score, plus training error, on the TRAINING images of every model.

Uses paper_measures.measure() unchanged (same angle and Fisher code, same ridge) on
PHASE3_GEOM_TRAIN/<model>/{features.npy, predictions.npz}, written by an eval-only forward pass
over train_2022 (671,817 images). Training error = share of training images whose prediction
(over all 15,501 species) is not the label. DBM and ArcFace reuse the train features extracted
for the head swap; their angle/Fisher must reproduce run d87d734a (55.14 / 38.59 deg,
1731.8 / 3977.4).

Output: lines prefixed __TRAINMEAS__; MARGIN_DIAG/train_measures.json.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_measures import LABEL, N, TRAIN, TRAIN_JSON, measure  # noqa: E402
from phase3_geometry import OUT  # noqa: E402

MODELS = ["cosine_m0", "dbm", "arcface_m01", "kappaface", "subcenter_arcface",
          "arcface_tara", "arcface", "arcface_ecc"]
RESULT = f"{OUT}/MARGIN_DIAG/train_measures.json"
PREFIX = "__TRAINMEAS__"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_count = np.bincount([json.loads(l)[LABEL] for l in open(TRAIN_JSON) if l.strip()], minlength=N)
    out = {}
    for m in MODELS:
        r = measure("train", m, train_count, device)
        if r is None:
            continue
        z = np.load(f"{TRAIN}/{m}/predictions.npz")
        r["train_error_pct"] = float(100 * np.mean(z["predictions"] != z["labels"]))
        out[m] = r
        print(f"{PREFIX} {m}: n={r['n_images']:,} train_error={r['train_error_pct']:.2f}% "
              f"angle_mean={r['angle_to_weight']['overall']['mean_deg']:.2f}deg "
              f"fisher_mean={r['fisher']['overall']['mean']:.1f}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(out, open(RESULT, "w"), indent=1)
    print(f"{PREFIX} wrote {RESULT} ({len(out)} models)", flush=True)


if __name__ == "__main__":
    main()
