"""Characterize the long tail of the scaling-ladder rungs.

The ladder manifests were built by build_ladder.py (run 5c9db6fa) from the merged
Herbarium 2021+2022 train.json / val.json. This script reads the resulting
n15k / n32k / n64k JSON files and reports, per rung, how many species have each
number of TRAINING images. It covers every species in the rung, including the
ones with no validation image (the earlier per-band analysis only saw species
that had at least one val image).

Output is plain text for humans plus one JSON object per rung between
__DIST_JSON_BEGIN__ / __DIST_JSON_END__ markers, so the numbers can be lifted
from the run log without re-parsing the manifests.
"""

import json
import os
from collections import Counter

import numpy as np

LADDER = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
RUNGS = ["n15k", "n32k", "n64k"]

# Fixed bins used for the species-by-training-image histogram. The first four are
# single counts so the very bottom of the tail is visible; the rest are the usual
# rarity bands.
BINS = [
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-500", 201, 500),
    ("501-1000", 501, 1000),
    ("1001+", 1001, 10**9),
]

CUMULATIVE_KS = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]


def counts(path):
    """Map scientificNameEncoded -> image count for one manifest file."""
    c = Counter()
    with open(path) as f:
        for line in f:
            if line.strip():
                c[json.loads(line)["scientificNameEncoded"]] += 1
    return c


def gini(values):
    """Gini coefficient of a vector of non-negative counts."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    # Standard formula for sorted data: G = (2*sum(i*x_i) - (n+1)*sum(x)) / (n*sum(x))
    return float((2.0 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * x.sum()) / (n * x.sum()))


def summarize(name, train, val):
    train_per = np.array(list(train.values()))
    val_per = np.array(list(val.values()))
    species = int(train_per.size)
    images = int(train_per.sum())
    q = np.percentile(train_per, [1, 5, 25, 50, 75, 95, 99])
    hist = []
    for label, lo, hi in BINS:
        n = int(np.sum((train_per >= lo) & (train_per <= hi)))
        hist.append({"bin": label, "species": n})
    cumulative = {str(k): int(np.sum(train_per >= k)) for k in CUMULATIVE_KS}
    top1 = max(1, int(round(0.01 * species)))
    top1_share = float(np.sort(train_per)[-top1:].sum() / images)
    no_val = [int(v) for k, v in train.items() if k not in val]
    no_val_hist = []
    for label, lo, hi in BINS:
        n = sum(1 for v in no_val if lo <= v <= hi)
        no_val_hist.append({"bin": label, "species": n})

    print(f"__DIST__ {name}: species={species:,} train_images={images:,} "
          f"min={train_per.min()} p1={q[0]:.0f} p5={q[1]:.0f} p25={q[2]:.0f} "
          f"median={q[3]:.0f} p75={q[4]:.0f} p95={q[5]:.0f} p99={q[6]:.0f} "
          f"max={train_per.max()} mean={train_per.mean():.1f}")
    print(f"__DIST__ {name} species by TRAINING images: "
          + " | ".join(f"{b['bin']}: {b['species']:,}" for b in hist))
    print(f"__DIST__ {name} cumulative species with >= k training images: "
          + " | ".join(f">={k}: {v:,}" for k, v in cumulative.items()))
    print(f"__DIST__ {name} many(>100)={int(np.sum(train_per > 100)):,} "
          f"medium(21-100)={int(np.sum((train_per > 20) & (train_per <= 100))):,} "
          f"few(<=20)={int(np.sum(train_per <= 20)):,}")
    print(f"__DIST__ {name} gini={gini(train_per):.3f} "
          f"imbalance_factor(max/min)={int(train_per.max() / train_per.min())} "
          f"top1pct_species={top1:,} top1pct_share_of_images={top1_share:.3f}")
    print(f"__DIST__ {name} species with a validation image={val_per.size:,}; "
          f"without one={len(no_val):,}")
    print(f"__DIST__ {name} no-val species by TRAINING images: "
          + " | ".join(f"{b['bin']}: {b['species']:,}" for b in no_val_hist))

    return {
        "rung": name,
        "species": species,
        "train_images": images,
        "val_images": int(val_per.sum()),
        "species_with_val": int(val_per.size),
        "species_without_val": len(no_val),
        "percentiles": {
            "min": int(train_per.min()), "p1": float(q[0]), "p5": float(q[1]),
            "p25": float(q[2]), "median": float(q[3]), "p75": float(q[4]),
            "p95": float(q[5]), "p99": float(q[6]), "max": int(train_per.max()),
            "mean": float(train_per.mean()),
        },
        "histogram": hist,
        "cumulative": cumulative,
        "many": int(np.sum(train_per > 100)),
        "medium": int(np.sum((train_per > 20) & (train_per <= 100))),
        "few": int(np.sum(train_per <= 20)),
        "gini": gini(train_per),
        "imbalance_factor": int(train_per.max() / train_per.min()),
        "top1pct_share": top1_share,
        "no_val_histogram": no_val_hist,
    }


def main():
    out = {}
    for rung in RUNGS:
        train = counts(os.path.join(LADDER, f"{rung}_train.json"))
        val = counts(os.path.join(LADDER, f"{rung}_val.json"))
        out[rung] = summarize(rung, train, val)
        print("__DIST_JSON_BEGIN__" + rung)
        print(json.dumps(out[rung], sort_keys=True))
        print("__DIST_JSON_END__" + rung)


if __name__ == "__main__":
    main()
