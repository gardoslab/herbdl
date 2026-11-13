"""Long-tail and uniform distribution options for the cleaned image pool.

Pool = master_index.csv (Kaggle 21+22, Harvard, GBIF-F24) + the latest GBIF-F25
index snapshot, minus F25 records already in F24/Harvard and minus repeated
sheets inside F25 (both from the latest overlap snapshot). Only Plantae images
labelled to species are counted; a species is the lower-cased "genus epithet".

Prints, for the natural pool and for "natural tail with a minimum and a cap"
variants, the statistics long-tail papers report: #species, #images, min /
median / max images per species, imbalance factor (max / min), Gini, share of
images in the top 1% of species, and the Many (>100) / Medium (21-100) /
Few (<=20) species split. Then the uniform-k table.

Writes per-species counts (total and per source) to OUT_DIR.
"""
import glob
import os
import time

import numpy as np
import pandas as pd

import common as C

OUT_DIR = "/projectnb/herbdl/workspaces/faridkar/dataset_stats/"
# (label, minimum images per species, cap on images per species; None = no cap)
VARIANTS = [
    ("natural pool, every species", 1, None),
    ("natural, min 3 (Herbarium 2021 rule)", 3, None),
    ("natural, min 10", 10, None),
    ("natural, min 20", 20, None),
    ("natural, min 20, cap 1000", 20, 1000),
    ("natural, min 20, cap 500", 20, 500),
    ("natural, min 50, cap 1000", 50, 1000),
    ("natural, min 100, cap 1000", 100, 1000),
]


def gini(x):
    """Gini coefficient of images-per-species (0 = perfectly even, 1 = all in one species)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def describe(label, c):
    c = np.sort(c.to_numpy())[::-1]
    n, tot = len(c), int(c.sum())
    top1 = c[: max(1, n // 100)].sum() / tot
    many, med, few = c > 100, (c > 20) & (c <= 100), c <= 20
    print(f"\n### {label}")
    print(f"species {n} | images {tot} | min {c.min()} | median {int(np.median(c))} | mean {c.mean():.1f} "
          f"| max {c.max()} | imbalance factor {c.max() / c.min():.1f} | gini {gini(c):.3f} "
          f"| top-1% species hold {top1:.1%} of images")
    print(f"Many(>100): {many.sum()} species, {c[many].sum() / tot:.1%} of images | "
          f"Medium(21-100): {med.sum()} species, {c[med].sum() / tot:.1%} | "
          f"Few(<=20): {few.sum()} species, {c[few].sum() / tot:.1%} | "
          f"<=5 images: {(c <= 5).sum()} species")
    return dict(variant=label, species=n, images=tot, min=int(c.min()), median=int(np.median(c)),
                max=int(c.max()), imbalance_factor=round(c.max() / c.min(), 1), gini=round(gini(c), 3),
                top1pct_share=round(top1, 4), many=int(many.sum()), medium=int(med.sum()), few=int(few.sum()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d")

    # Sources already in master_index.csv.
    base = C.master_species_counts().unstack("src", fill_value=0)

    # GBIF-F25 snapshot minus overlap / repeats.
    idx_path = C.latest_index()
    ov_path = sorted(glob.glob(C.F25_DIR + "overlap_snapshot_*.csv"))[-1]
    print("F25 index:", idx_path, "\noverlap:", ov_path)
    f25 = pd.read_csv(idx_path, usecols=["gbif_id", "species_binomial", "kingdom"], dtype=C.STR)
    f25["gbif_id"] = pd.to_numeric(f25.gbif_id).astype(np.int64)
    ov = pd.read_csv(ov_path, usecols=["gbif_id", "in_f24", "in_harvard", "f25_keep"])
    drop = ov.gbif_id[ov.in_f24 | ov.in_harvard | ~ov.f25_keep]
    n0 = len(f25)
    f25 = f25[~f25.gbif_id.isin(drop) & (f25.kingdom == "Plantae") & f25.species_binomial.notna()]
    print(f"F25 images: {n0} in snapshot, {len(f25)} kept (species-labelled, not overlap, not repeat)")
    f25c = f25.species_binomial.str.lower().str.strip().value_counts().rename("GBIF-F25")

    pool = base.join(f25c, how="outer").fillna(0).astype(int)
    pool["total"] = pool.sum(axis=1)
    pool = pool.sort_values("total", ascending=False)
    pool.index.name = "species_key"
    out = OUT_DIR + f"species_counts_pool_{stamp}.csv"
    pool.to_csv(out)
    print(f"wrote {out}: {len(pool)} species")
    print("images per source:", pool.drop(columns="total").sum().to_dict(), "total", int(pool.total.sum()))
    print("species per source (>=1 image):", (pool.drop(columns="total") > 0).sum().to_dict())

    # Long-tail variants: keep species with >= min images; cap heads by subsampling to `cap`.
    rows = []
    for label, lo, cap in VARIANTS:
        c = pool.total[pool.total >= lo]
        if cap:
            c = c.clip(upper=cap)
        rows.append(describe(label, c))
    pd.DataFrame(rows).to_csv(OUT_DIR + f"longtail_variants_{stamp}.csv", index=False)

    # Each source on its own, for comparison with the published datasets.
    for src in pool.columns.drop("total"):
        c = pool[src][pool[src] > 0]
        describe(f"single source: {src}", c)

    C.uniform_table("uniform options (cleaned pool)", pool.total)


if __name__ == "__main__":
    main()
