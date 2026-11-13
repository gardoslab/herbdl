"""Record-level overlap of the GBIF-F25 index with GBIF-F24 and Harvard.

GBIF-F25 is a download of all imaged vascular-plant records on GBIF, so it
should re-contain specimens already in GBIF-F24 (NY, COLO, MO) and Harvard. A
F25 record counts as "already have it" when any of these match a record that
is in master_index.csv for that source:

  gbif   - same gbifID
  occ    - same occurrenceID (the publisher's global record id)
  cat    - same institutionCode + catalogNumber (the physical sheet's barcode)

It also counts duplicates *inside* F25: one sheet published under several
gbifIDs (e.g. by the herbarium and by an aggregator).

Pixel-level near-duplicates are out of scope (a later image-hash job).

Output: /projectnb/herbdl/data/GBIF-F25/overlap_snapshot_<YYYYMMDD>.csv with one
row per F25 record that matched F24/Harvard or has an in-F25 duplicate.
"""
import os
import time

import numpy as np
import pandas as pd

import common as C

REC_COLS = ["gbifID", "institutionCode", "occurrenceID", "catalogNumber"]


def norm_keys(df):
    """Normalised occurrenceID and institution+catalogue keys (<NA> when empty)."""
    occ = df.occurrenceID.str.strip().str.lower()
    occ = occ.where(occ.str.len() > 0)
    cat = df.catalogNumber.str.upper().str.replace(r"\s+", "", regex=True)
    inst = df.institutionCode.str.upper().str.strip()
    catkey = (inst + "|" + cat).where(cat.str.len() > 0)
    return occ, catkey


def source_records(src_name, occ_path, master_ids):
    """Occurrence keys for the records of one source that are in master_index.csv."""
    t0 = time.time()
    tsv = os.path.join(C.scratch(), f"{src_name}_occ_cols.tsv")
    names = C.cut_columns(occ_path, REC_COLS, tsv)
    parts = []
    for ch in C.read_tsv_chunks(tsv, names):
        ids = pd.to_numeric(ch.gbifID, errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        keep = np.isin(ids, master_ids)
        ch = ch[keep].copy()
        ch["gbif_id"] = ids[keep]
        parts.append(ch)
    rec = pd.concat(parts, ignore_index=True).drop_duplicates("gbif_id")
    rec["occ"], rec["catkey"] = norm_keys(rec)
    print(f"{src_name}: {len(master_ids)} gbifIDs in master_index, {len(rec)} found in its occurrence.txt "
          f"({len(rec) / max(len(master_ids), 1):.1%}); with occurrenceID {rec.occ.notna().mean():.1%}, "
          f"with inst+catalogNumber {rec.catkey.notna().mean():.1%}; top institutions "
          f"{rec.institutionCode.value_counts().head(6).to_dict()} ({time.time() - t0:.0f}s)")
    return rec


def main():
    stamp = time.strftime("%Y%m%d")
    out = C.F25_DIR + f"overlap_snapshot_{stamp}.csv"
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite existing {out}")

    # 1. F25 index -> one row per record (gbifID) plus image counts.
    idx_path = C.latest_index()
    t0 = time.time()
    f25 = pd.read_csv(idx_path, dtype=C.STR,
                      usecols=["gbif_id", "creator", "catalogNumber", "occurrenceID",
                               "species_binomial", "kingdom", "basisOfRecord"])
    f25["gbif_id"] = pd.to_numeric(f25.gbif_id).astype(np.int64)
    print(f"F25 index {idx_path}: {len(f25)} images ({time.time() - t0:.0f}s)")
    n_img = f25.groupby("gbif_id").size().rename("n_images")
    rec = f25.drop_duplicates("gbif_id").set_index("gbif_id").join(n_img).reset_index()
    rec = rec.rename(columns={"creator": "institutionCode"})
    rec["occ"], rec["catkey"] = norm_keys(rec)
    del f25
    print(f"F25 records: {len(rec)}; with occurrenceID {rec.occ.notna().mean():.1%}, "
          f"with inst+catalogNumber {rec.catkey.notna().mean():.1%}")

    # 2. F24 / Harvard records that we actually hold (their master_index rows; image_id == gbifID).
    mi = pd.read_csv(C.MASTER_INDEX, usecols=["image_id", "filepath"], dtype=C.STR)
    mi_src = mi.filepath.str.split("/").str[4]
    for src_name, occ_path, tag in (("GBIF-F24", C.F24_OCC, "f24"),
                                    ("harvard-herbaria", C.HARVARD_OCC, "harvard")):
        ids = pd.to_numeric(mi.image_id[mi_src == src_name], errors="coerce").dropna()
        ids = np.unique(ids.astype(np.int64).to_numpy())
        src = source_records(src_name, occ_path, ids)
        by_gbif = rec.gbif_id.isin(src.gbif_id)
        by_occ = rec.occ.isin(src.occ.dropna()) & rec.occ.notna()
        by_cat = rec.catkey.isin(src.catkey.dropna()) & rec.catkey.notna()
        rec[f"in_{tag}"] = by_gbif | by_occ | by_cat
        rec[f"{tag}_by"] = np.select([by_gbif, by_occ, by_cat], ["gbif", "occ", "cat"], default="")
        m = rec[f"in_{tag}"]
        print(f"F25 records matching {src_name}: {int(m.sum())} records / {int(rec.n_images[m].sum())} images "
              f"(by gbif {int(by_gbif.sum())}, by occurrenceID {int(by_occ.sum())}, "
              f"by inst+catalog {int(by_cat.sum())}); covers {m.sum() / max(len(src), 1):.1%} of {src_name} records")
        print(f"  matched F25 institutions: {rec.institutionCode[m].value_counts().head(8).to_dict()}")
        del src
    del mi, mi_src

    # 3. Duplicates inside F25: same sheet (inst+catalog, or occurrenceID) under several gbifIDs.
    grp = rec.groupby("catkey").gbif_id.transform("size").where(rec.catkey.notna(), 1)
    ogrp = rec.groupby("occ").gbif_id.transform("size").where(rec.occ.notna(), 1)
    rec["f25_dup_size"] = np.maximum(grp.fillna(1), ogrp.fillna(1)).astype(int)
    first_cat = ~rec.catkey.duplicated() | rec.catkey.isna()
    first_occ = ~rec.occ.duplicated() | rec.occ.isna()
    rec["f25_keep"] = first_cat & first_occ
    dup = rec.f25_dup_size > 1
    print(f"\nwithin-F25: {int(dup.sum())} records share a sheet with another F25 record; "
          f"dropping repeats removes {int((~rec.f25_keep).sum())} records / "
          f"{int(rec.n_images[~rec.f25_keep].sum())} images")
    print(f"  top institutions among repeats: {rec.institutionCode[~rec.f25_keep].value_counts().head(8).to_dict()}")

    flagged = rec[rec.in_f24 | rec.in_harvard | dup]
    flagged[["gbif_id", "institutionCode", "n_images", "in_f24", "f24_by", "in_harvard", "harvard_by",
             "f25_dup_size", "f25_keep"]].to_csv(out, index=False)
    print(f"wrote {out}: {len(flagged)} flagged records")

    # 4. Uniform-k species tables after removing the overlap.
    base = C.master_species_counts().groupby(level="key").sum()
    sp = rec[(rec.kingdom == "Plantae") & rec.species_binomial.notna()].copy()
    sp["key"] = sp.species_binomial.str.lower().str.strip()
    new = ~(sp.in_f24 | sp.in_harvard)
    herb = sp.basisOfRecord == "PRESERVED_SPECIMEN"

    def f25_counts(mask):
        return sp[mask].groupby("key").n_images.sum()

    C.uniform_table("master_index + F25 (no removal)", base.add(f25_counts(np.ones(len(sp), bool)), fill_value=0))
    C.uniform_table("master_index + F25 minus F24/Harvard overlap", base.add(f25_counts(new), fill_value=0))
    C.uniform_table("... and minus within-F25 repeats", base.add(f25_counts(new & sp.f25_keep), fill_value=0))
    C.uniform_table("... and F25 PRESERVED_SPECIMEN only",
                    base.add(f25_counts(new & sp.f25_keep & herb), fill_value=0))


if __name__ == "__main__":
    main()
