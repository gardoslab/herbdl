"""Build nested species-count rungs for the Phase 6 scaling ladder.

Source: the merged Herbarium 2021+2022 manifest (train.json / val.json, 64,113 species,
one JSON object per line with filepath, scientificName "Family Genus species",
scientificNameEncoded). The project's earlier "21K"/"50K" runs all trained on this full
64k set; there is no smaller merged label set, so the rungs are cut here.

Rungs are NESTED, so each larger rung contains every species of the smaller ones:
  n15k  the 15,501 Herbarium-2022 species (matched by exact scientificName)
  n32k  n15k + 16,499 other species drawn uniformly at random (seed 0)
  n64k  every species
Labels are re-encoded 0..N-1 per rung, 2022 species first (ordered by their 2022 id), so
a species keeps the same id in every rung it belongs to. Val keeps only the rung's species.

Caveat: the merged split is NOT the 2022 split. Its val contains Herbarium-2022 *train*
images and vice versa, and 2022 species pick up extra 2021 images. So the n15k rung is the
same species as Phases 2-3 but not the same data; it is re-run as the ladder's own base.
"""

import json
import os
from collections import Counter

import numpy as np

SRC = "/projectnb/herbdl/data/kaggle-herbaria"
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
N_MID, SEED = 32000, 0


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    train, val = load(f"{SRC}/train.json"), load(f"{SRC}/val.json")
    t22 = load(f"{SRC}/train_2022.json")
    # 2022 species in their 2022 id order, matched to merged names by exact string
    id22 = {}
    for r in t22:
        id22[r["scientificName"]] = r["scientificNameEncoded"]
    all_names = sorted({r["scientificName"] for r in train})
    base = sorted((n for n in all_names if n in id22), key=lambda n: id22[n])
    rest = [n for n in all_names if n not in id22]
    print(f"__LADDER__ merged train species={len(all_names):,} 2022 species found={len(base):,} "
          f"(of {len(id22):,}) others={len(rest):,}")
    assert len(base) == len(id22), "some 2022 species are missing from the merged manifest"

    rng = np.random.default_rng(SEED)
    extra = [rest[i] for i in sorted(rng.choice(len(rest), N_MID - len(base), replace=False))]
    extra_set = set(extra)
    rungs = {
        "n15k": base,
        "n32k": base + extra,
        "n64k": base + extra + [n for n in rest if n not in extra_set],
    }
    os.makedirs(OUT, exist_ok=True)
    for rung, names in rungs.items():
        enc = {n: i for i, n in enumerate(names)}
        summary = {}
        for split, rows in (("train", train), ("val", val)):
            kept = []
            for r in rows:
                i = enc.get(r["scientificName"])
                if i is None:
                    continue
                fam, gen, *sp = r["scientificName"].split(" ")
                kept.append({"filepath": r["filepath"], "scientificName": r["scientificName"],
                             "family": fam, "genus": gen, "species": " ".join(sp),
                             "scientificNameEncoded": i,
                             "merged64kEncoded": r["scientificNameEncoded"]})
            with open(f"{OUT}/{rung}_{split}.json", "w") as f:
                for r in kept:
                    f.write(json.dumps(r) + "\n")
            per = np.array(list(Counter(r["scientificNameEncoded"] for r in kept).values()))
            n22 = sum("herbarium-2022" in r["filepath"] for r in kept)
            summary[split] = dict(images=len(kept), species=len(per), median_per_species=float(np.median(per)),
                                  min_per_species=int(per.min()), max_per_species=int(per.max()),
                                  images_from_2022=n22)
            print(f"__LADDER__ {rung} {split}: images={len(kept):,} species={len(per):,} "
                  f"per-species min/median/max={per.min()}/{np.median(per):.0f}/{per.max()} "
                  f"from-2022={n22:,} ({n22 / len(kept):.1%})")
        summary["n_species"] = len(names)
        summary["steps_per_epoch_at_batch_256"] = summary["train"]["images"] / 256
        json.dump(summary, open(f"{OUT}/{rung}_summary.json", "w"), indent=1)
    # nesting check: a species' id is the same in every rung that contains it
    for small, big in (("n15k", "n32k"), ("n32k", "n64k")):
        assert rungs[big][:len(rungs[small])] == rungs[small]
    print(f"__LADDER__ wrote rungs {list(rungs)} to {OUT}; nesting verified")


if __name__ == "__main__":
    main()
