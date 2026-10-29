"""Plain counts behind the ladder routing result, on the FULL validation set of the 32k and
64k rungs, with the routing cutoff found in run a8745409 (32k: T=10, 64k: T=8).

  by TRUE species band:      images; DBM right; KappaFace right; only DBM right; only Kappa right
  by PREDICTED species band: predictions each model makes into the band, and how many are right
  routing:                   images switched to KappaFace (its answer has <= T training images and
                             differs from DBM's); of those: fixed (Kappa right, DBM wrong),
                             broken (DBM right, Kappa wrong), wrong either way; by true band
"""

import json

import numpy as np

LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
EV = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/LADDER_EVAL"
BANDS = [(2, 5), (6, 10), (11, 25), (26, 100), (101, 10 ** 6)]
T = {"n32k": 10, "n64k": 8}


def name(lo, hi):
    return f"{lo}+" if hi >= 10 ** 6 else f"{lo}-{hi}"


for rung, t in T.items():
    with open(f"{LAD}/{rung}_train.json") as f:
        labels = [json.loads(l)["scientificNameEncoded"] for l in f if l.strip()]
    count = np.bincount(labels)
    zd, zk = np.load(f"{EV}/{rung}_dbm/predictions.npz"), np.load(f"{EV}/{rung}_kappaface/predictions.npz")
    y = zd["labels"].astype(np.int64)
    pd_, pk = zd["predictions"].astype(np.int64), zk["predictions"].astype(np.int64)
    dr, kr = pd_ == y, pk == y
    print(f"__RCNT__ ===== {rung}: {len(y):,} validation images, routing cutoff T={t}")
    print("__RCNT__ BY TRUE SPECIES (images of species in this band)")
    for lo, hi in BANDS:
        m = (count[y] >= lo) & (count[y] <= hi)
        print(f"__RCNT__   {name(lo, hi):>6}: images {m.sum():7,} | DBM right {(dr & m).sum():7,} "
              f"| Kappa right {(kr & m).sum():7,} | only DBM right {(dr & ~kr & m).sum():6,} "
              f"| only Kappa right {(kr & ~dr & m).sum():6,}")
    print("__RCNT__ BY PREDICTED SPECIES (answers naming a species in this band)")
    for lo, hi in BANDS:
        md, mk = (count[pd_] >= lo) & (count[pd_] <= hi), (count[pk] >= lo) & (count[pk] <= hi)
        print(f"__RCNT__   {name(lo, hi):>6}: DBM predicts {md.sum():7,} (right {(dr & md).sum():7,}) "
              f"| Kappa predicts {mk.sum():7,} (right {(kr & mk).sum():7,})")
    sw = (count[pk] <= t) & (pk != pd_)
    fixed, broken, neither = sw & kr, sw & dr, sw & ~kr & ~dr
    print(f"__RCNT__ ROUTING: switched {sw.sum():,} images to KappaFace's answer -> fixed {fixed.sum():,} "
          f"(Kappa right, DBM wrong) | broken {broken.sum():,} (DBM right, Kappa wrong) | "
          f"wrong either way {neither.sum():,} | net {fixed.sum() - broken.sum():+,} correct answers")
    for lo, hi in BANDS:
        m = (count[y] >= lo) & (count[y] <= hi)
        print(f"__RCNT__   switched images whose TRUE species is {name(lo, hi):>6}: {(sw & m).sum():6,} "
              f"-> fixed {(fixed & m).sum():6,} | broken {(broken & m).sum():5,} | wrong either way {(neither & m).sum():5,}")
