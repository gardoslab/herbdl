"""The rare-species images KappaFace gets right and DBM gets wrong (64k ladder rung).

Selection: validation images whose TRUE species has 2-5 training images, KappaFace's top-1
is right and DBM's is wrong (469 images; see routing_counts.py). Writes to OUT:
  kappaface_wins_64k.csv      one row per image: path, true species, DBM's guess, their
                              training-image counts, and whether the guess is the same genus/family
  summary.json                concentration: distinct species/genera/families, top families
                              compared with the whole 2-5 group, rarity of DBM's guesses
  sheet_*.png                 picture sheets for a seeded random sample: validation image |
                              a training image of the true species | a training image of the
                              species DBM guessed
"""

import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
EV = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/LADDER_EVAL"
OUT = f"{EV}/analysis/kappaface_wins_64k"
ROWS_PER_SHEET, N_SHEETS = 8, 4


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def thumb(path, size=320):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, va = load(f"{LAD}/n64k_train.json"), load(f"{LAD}/n64k_val.json")
    n = max(r["scientificNameEncoded"] for r in tr) + 1
    count = np.bincount([r["scientificNameEncoded"] for r in tr], minlength=n)
    info, example = {}, {}
    for r in tr:  # species id -> (family, genus, name) and its first training image
        c = r["scientificNameEncoded"]
        if c not in info:
            info[c] = (r["family"], r["genus"], r["scientificName"])
            example[c] = r["filepath"]
    zd, zk = np.load(f"{EV}/n64k_dbm/predictions.npz"), np.load(f"{EV}/n64k_kappaface/predictions.npz")
    y = zd["labels"].astype(np.int64)
    pd_, pk = zd["predictions"].astype(np.int64), zk["predictions"].astype(np.int64)
    band = (count[y] >= 2) & (count[y] <= 5)
    win = np.where(band & (pk == y) & (pd_ != y))[0]
    print(f"__KW__ 2-5 group: {band.sum():,} images; KappaFace right & DBM wrong: {len(win):,}")

    rows = []
    for i in win:
        t, g = int(y[i]), int(pd_[i])
        rows.append({"val_image": va[i]["filepath"], "true_species": info[t][2], "true_train_images": int(count[t]),
                     "dbm_guess": info[g][2], "dbm_guess_train_images": int(count[g]),
                     "same_genus": info[t][:2] == info[g][:2], "same_family": info[t][0] == info[g][0]})
    with open(f"{OUT}/kappaface_wins_64k.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fam_win = Counter(info[int(y[i])][0] for i in win)
    fam_band = Counter(info[int(c)][0] for c in y[band])
    gc = np.array([r["dbm_guess_train_images"] for r in rows])
    s = {"images": len(win), "band_images": int(band.sum()),
         "distinct_true_species": len({r["true_species"] for r in rows}),
         "species_in_band_with_val_images": int(len(np.unique(y[band]))),
         "max_wins_from_one_species": max(Counter(r["true_species"] for r in rows).values()),
         "distinct_families": len(fam_win),
         "top_families": [{"family": f, "wins": c, "share_of_wins": c / len(win),
                           "share_of_2to5_group": fam_band[f] / band.sum()} for f, c in fam_win.most_common(10)],
         "dbm_guess_same_genus": float(np.mean([r["same_genus"] for r in rows])),
         "dbm_guess_same_family": float(np.mean([r["same_family"] for r in rows])),
         "dbm_guess_train_images_median": float(np.median(gc)),
         "dbm_guess_train_images_share_over_100": float(np.mean(gc > 100))}
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1)
    print(f"__KW__ distinct true species {s['distinct_true_species']} (of {s['species_in_band_with_val_images']} "
          f"in the 2-5 group with val images); max wins from one species {s['max_wins_from_one_species']}; "
          f"families {s['distinct_families']}")
    print(f"__KW__ DBM's wrong guess: same genus {100*s['dbm_guess_same_genus']:.0f}%, same family "
          f"{100*s['dbm_guess_same_family']:.0f}%; its training images: median {s['dbm_guess_train_images_median']:.0f}, "
          f"over 100 for {100*s['dbm_guess_train_images_share_over_100']:.0f}%")
    for t in s["top_families"]:
        print(f"__KW__   {t['family']:22s} wins {t['wins']:3d} ({100*t['share_of_wins']:.1f}% of wins vs "
              f"{100*t['share_of_2to5_group']:.1f}% of the 2-5 group)")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(len(rows), ROWS_PER_SHEET * N_SHEETS), replace=False)
    for sheet in range(N_SHEETS):
        idx = pick[sheet * ROWS_PER_SHEET:(sheet + 1) * ROWS_PER_SHEET]
        if len(idx) == 0:
            break
        fig, axes = plt.subplots(len(idx), 3, figsize=(7.5, 2.55 * len(idx)), layout="constrained")
        for r, k in enumerate(idx):
            row, i = rows[k], win[k]
            panels = [(row["val_image"], f"VALIDATION IMAGE\ntrue: {row['true_species']}\n"
                                         f"({row['true_train_images']} training images)"),
                      (example[int(y[i])], f"a training image of the\ntrue species\n{row['true_species']}"),
                      (example[int(pd_[i])], f"a training image of DBM's guess\n{row['dbm_guess']}\n"
                                             f"({row['dbm_guess_train_images']} training images)")]
            for c, (p, label) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(thumb(p))
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_xlabel(label, fontsize=6.5)
        fig.savefig(f"{OUT}/sheet_{sheet + 1}.png", dpi=150)
        plt.close(fig)
        print(f"__KW__ wrote {OUT}/sheet_{sheet + 1}.png ({len(idx)} examples)")


if __name__ == "__main__":
    main()
