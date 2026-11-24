"""Who pays for a margin? Which common species lose accuracy, and where do their images go.

A margin helps species with few training images and, in earlier runs, costs accuracy for common
ones; the cost grows with the margin (ArcFace > KappaFace > DBM). Before designing a fix, this asks
where the cost falls. Two hypotheses that predict different patterns:

  territory     A rare species' margin against a common sibling pushes that sibling's template
                away from the rare species' images and shrinks the sibling's region, so the
                common sibling's images are lost *to the rare sibling*. The loss is concentrated in
                common species that have a rare sibling, and the lost images land on it.
  everywhere    Persistent margin pressure or crowding lowers accuracy for all common species alike,
                with no link to rare siblings.

For the 15.5k base data it compares NormFace (cosine head, no margin) with DBM, KappaFace and
ArcFace on the saved validation predictions of sibling_ceiling.py, for species with at least 26
training images ("common"):

  1. the change in per-species F1 (points) of each margin model against NormFace, by the training
     count of the species' rarest genus sibling and by how many siblings have 10 or fewer images,
     with 95% bootstrap intervals over species;
  2. the images NormFace classifies correctly that the margin model gets wrong ("lost"): where they
     go (rare sibling, other sibling, same family, other family) against where NormFace's own errors
     on common species go, and an enrichment ratio for rare siblings: lost images sent to a rare
     sibling divided by the number expected if they were sent to a random sibling of their species.

No images are read; CPU only. The last line is RESULT_JSON.

Usage (from finetuning/SWIN):  python loss_ablation/who_pays.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss_ablation import sibling_margin as sm  # noqa: E402

OUT_ROOT = f"/projectnb/herbdl/workspaces/{os.environ.get('USER', 'unknown')}/herbdl/finetuning/output/SWIN"
PRED_DIR = f"{OUT_ROOT}/sibling_ceiling"  # <model>.npz with labels and pred, from sibling_ceiling.py
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
REF = "normface"
MODELS = ["dbm", "kappaface", "arcface"]
NAMES = {"normface": "NormFace", "dbm": "DBM", "kappaface": "KappaFace", "arcface": "ArcFace m=0.3"}
COMMON_MIN, RARE_MAX, BOOT = 26, 10, 1000
SIB_BANDS = [("no sibling", None, None), ("2-5", 2, 5), ("6-10", 6, 10), ("11-25", 11, 25), ("26-50", 26, 50), ("51+", 51, 10 ** 9)]
RARE_BINS = [("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 10 ** 9)]
DEST = ["rare sibling", "other sibling", "same family, other genus", "other family"]


# ---------------------------------------------------------------- analysis (plain arrays)

def species_f1(labels, pred, C):
    tp = np.bincount(labels[pred == labels], minlength=C)
    n_true, n_pred = np.bincount(labels, minlength=C), np.bincount(pred, minlength=C)
    return np.where(n_true + n_pred > 0, 2 * tp / np.maximum(n_true + n_pred, 1), 0.0), n_true > 0


def sibling_summary(counts, gid):
    """Per species: training count of its rarest genus sibling (inf if none) and number of siblings
    with RARE_MAX or fewer training images."""
    C = len(counts)
    rarest, n_rare = np.full(C, np.inf), np.zeros(C, dtype=np.int64)
    order = np.argsort(gid, kind="stable")
    bounds = np.flatnonzero(np.diff(gid[order])) + 1
    for members in np.split(order, bounds):
        if len(members) < 2:
            continue
        c = counts[members]
        for i, s in enumerate(members):
            others = np.delete(c, i)
            rarest[s], n_rare[s] = others.min(), int((others <= RARE_MAX).sum())
    return rarest, n_rare


def boot_mean(values, rng):
    """Mean and 95% interval of the mean over species resampled with replacement."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return None
    draw = rng.integers(0, len(v), size=(BOOT, len(v)))
    means = v[draw].mean(1)
    return [float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def destination(y, w, counts, gid, fid):
    """Index into DEST of where the wrong prediction w of an image of species y landed."""
    sib = gid[w] == gid[y]
    fam = (fid[w] == fid[y]) & ~sib
    return np.where(sib & (counts[w] <= RARE_MAX), 0, np.where(sib, 1, np.where(fam, 2, 3)))


def analyse(labels, preds, counts, gid, fid, seed=0):
    """preds: dict model -> predictions, with REF among them. Returns the numbers behind every table."""
    rng = np.random.default_rng(seed)
    C = len(counts)
    rarest, n_rare = sibling_summary(counts, gid)
    n_sib = np.bincount(gid, minlength=gid.max() + 1)[gid] - 1
    f1 = {m: species_f1(labels, p, C) for m, p in preds.items()}
    present = f1[REF][1]
    common = present & (counts >= COMMON_MIN)
    res = {"species_common": int(common.sum()), "images_common": int(common[labels].sum()), "models": {}}
    ref = preds[REF]
    for m in MODELS:
        delta = 100 * (f1[m][0] - f1[REF][0])
        out = {"mean_delta_f1_common": boot_mean(delta[common], rng), "by_rarest_sibling": {}, "by_rare_siblings": {}}
        for name, lo, hi in SIB_BANDS:
            sel = common & (np.isinf(rarest) if lo is None else (rarest >= lo) & (rarest <= hi))
            out["by_rarest_sibling"][name] = {"species": int(sel.sum()), "delta": boot_mean(delta[sel], rng)}
        for name, lo, hi in RARE_BINS:
            sel = common & (n_sib > 0) & (n_rare >= lo) & (n_rare <= hi)
            out["by_rare_siblings"][name] = {"species": int(sel.sum()), "delta": boot_mean(delta[sel], rng)}
        # where do the images go?
        img_common = common[labels]
        lost = img_common & (ref == labels) & (preds[m] != labels)
        gained = img_common & (ref != labels) & (preds[m] == labels)
        d_lost = destination(labels[lost], preds[m][lost], counts, gid, fid)
        sib_lost = d_lost <= 1
        # expected number sent to a rare sibling if each lost image picked a random sibling of its species
        rare_frac = np.zeros(C)
        order = np.argsort(gid, kind="stable")
        for members in np.split(order, np.flatnonzero(np.diff(gid[order])) + 1):
            if len(members) > 1:
                r = (counts[members] <= RARE_MAX).astype(float)
                rare_frac[members] = (r.sum() - r) / (len(members) - 1)
        expected = rare_frac[labels[lost]][sib_lost].sum()
        out["lost"] = int(lost.sum())
        out["gained"] = int(gained.sum())
        out["lost_destination"] = {n: int((d_lost == i).sum()) for i, n in enumerate(DEST)}
        out["rare_sibling_enrichment"] = float((d_lost == 0).sum() / expected) if expected > 0 else None
        out["rare_sibling_expected"] = float(expected)
        res["models"][m] = out
    # reference: where NormFace's own errors on common species go
    err = common[labels] & (ref != labels)
    d_err = destination(labels[err], ref[err], counts, gid, fid)
    res["reference_errors"] = {"errors": int(err.sum()), "destination": {n: int((d_err == i).sum()) for i, n in enumerate(DEST)}}
    return res


# ---------------------------------------------------------------- printing

def fmt(b):
    return "-" if b is None else f"{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]"


def print_tables(res):
    print(f"\nCommon species (at least {COMMON_MIN} training images): {res['species_common']:,} species, "
          f"{res['images_common']:,} validation images.\n")
    print("Change in per-species F1 against NormFace (points, mean over species [95% interval]):")
    print("| | " + " | ".join(NAMES[m] for m in MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    print("| all common species | " + " | ".join(fmt(res["models"][m]["mean_delta_f1_common"]) for m in MODELS) + " |")
    print("\nBy training images of the rarest genus sibling:")
    print("| rarest sibling | species | " + " | ".join(NAMES[m] for m in MODELS) + " |")
    print("|---|---|" + "---|" * len(MODELS))
    for name, _, _ in SIB_BANDS:
        n = res["models"][MODELS[0]]["by_rarest_sibling"][name]["species"]
        print(f"| {name} | {n:,} | " + " | ".join(fmt(res["models"][m]["by_rarest_sibling"][name]["delta"]) for m in MODELS) + " |")
    print(f"\nBy the number of siblings with {RARE_MAX} or fewer training images (species with at least one sibling):")
    print("| rare siblings | species | " + " | ".join(NAMES[m] for m in MODELS) + " |")
    print("|---|---|" + "---|" * len(MODELS))
    for name, _, _ in RARE_BINS:
        n = res["models"][MODELS[0]]["by_rare_siblings"][name]["species"]
        print(f"| {name} | {n:,} | " + " | ".join(fmt(res["models"][m]["by_rare_siblings"][name]["delta"]) for m in MODELS) + " |")
    print("\nImages of common species that NormFace gets right and the margin model gets wrong (lost), and where they land:")
    print("| model | lost | gained | " + " | ".join(DEST) + " | rare-sibling enrichment |")
    print("|---|---|---|" + "---|" * len(DEST) + "---|")
    for m in MODELS:
        o = res["models"][m]
        tot = max(sum(o["lost_destination"].values()), 1)
        enr = "-" if o["rare_sibling_enrichment"] is None else f"{o['rare_sibling_enrichment']:.2f}"
        print(f"| {NAMES[m]} | {o['lost']:,} | {o['gained']:,} | " + " | ".join(f"{100 * o['lost_destination'][d] / tot:.1f}%" for d in DEST) + f" | {enr} |")
    r = res["reference_errors"]
    tot = max(sum(r["destination"].values()), 1)
    print(f"| NormFace's own errors | {r['errors']:,} | | " + " | ".join(f"{100 * r['destination'][d] / tot:.1f}%" for d in DEST) + " | |")


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred-dir", default=PRED_DIR)
    ap.add_argument("--train", default=TRAIN_JSON)
    args = ap.parse_args(argv)
    ids, genus, family = [], [], []
    with open(args.train) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(int(r[LABEL])), genus.append(r["genus"]), family.append(r["family"])
    ids = np.array(ids)
    C = int(ids.max()) + 1
    gid, fid = sm.taxonomy_arrays(ids, genus, family, C)
    counts = np.bincount(ids, minlength=C)
    labels, preds = None, {}
    for m in [REF] + MODELS:
        z = np.load(f"{args.pred_dir}/{m}.npz")
        if labels is None:
            labels = z["labels"].astype(np.int64)
        elif not np.array_equal(labels, z["labels"]):
            sys.exit(f"{m}: validation labels differ from {REF}'s")
        preds[m] = z["pred"].astype(np.int64)
        print(f"{m}: {len(labels):,} validation images, accuracy {(preds[m] == labels).mean():.4f}")
    res = analyse(labels, preds, counts, gid, fid)
    print_tables(res)
    print("RESULT_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
