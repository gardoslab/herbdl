"""Build a species-labelled image index for the GBIF-F25 download (images in GBIF-F25h/).

Every image with status='success' in GBIF-F25h/download_status.db is joined, by
gbifID, to its occurrence record in GBIF-F25/occurrence.txt (taxonomy,
institution, record type, catalogue number). The download is still running, so
the output is a dated snapshot:

    /projectnb/herbdl/data/GBIF-F25/index_snapshot_<YYYYMMDD>.csv

Core columns match master_index.csv (image_id, filepath, scientificName, family,
genus, species = epithet, creator = institutionCode); extra columns keep what the
overlap job and later filtering need. No `scientificNameEncoded`: label ids are
assigned when datasets are merged, not per source.
"""
import os
import random
import sqlite3
import time

import numpy as np
import pandas as pd

import common as C

OCC_COLS = ["gbifID", "institutionCode", "basisOfRecord", "occurrenceID", "catalogNumber",
            "kingdom", "family", "genus", "taxonRank", "datasetKey", "species"]


def main():
    out = C.F25_DIR + f"index_snapshot_{time.strftime('%Y%m%d')}.csv"
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite existing {out}")

    # 1. Downloaded images. Read-only connection: the downloader may still be writing.
    t0 = time.time()
    db = sqlite3.connect(f"file:{C.F25_DB}?mode=ro", uri=True)
    status = dict(db.execute("select status, count(*) from images group by status").fetchall())
    print("download status (images):", status)
    img = pd.read_sql("select gbif_id, image_no, file_path from images where status = 'success'", db)
    db.close()
    print(f"successful images: {len(img)} from {img.gbif_id.nunique()} records ({time.time() - t0:.0f}s)")

    # 2. Occurrence records for those gbifIDs only (the full file is 62 GB / tens of millions of rows).
    t0 = time.time()
    tsv = os.path.join(C.scratch(), "f25_occ_cols.tsv")
    names = C.cut_columns(C.F25_OCC, OCC_COLS, tsv)
    want = np.unique(img.gbif_id.to_numpy())
    parts, n_rows = [], 0
    for ch in C.read_tsv_chunks(tsv, names):
        n_rows += len(ch)
        ids = pd.to_numeric(ch.gbifID, errors="coerce").astype("Int64")
        keep = ids.notna() & np.isin(ids.fillna(-1).to_numpy(np.int64), want)
        ch = ch[keep.to_numpy()].copy()
        ch["gbif_id"] = ids[keep].astype(np.int64).to_numpy()
        parts.append(ch.drop(columns="gbifID"))
    occ = pd.concat(parts, ignore_index=True)
    del parts
    dup = occ.gbif_id.duplicated().sum()
    occ = occ.drop_duplicates("gbif_id")
    print(f"occurrence rows scanned: {n_rows}; matched records: {len(occ)} (dropped {dup} duplicate gbifIDs) "
          f"({time.time() - t0:.0f}s)")

    # 3. Join and shape like master_index.csv.
    df = img.merge(occ, on="gbif_id", how="inner")
    print(f"images with an occurrence record: {len(df)} / {len(img)} "
          f"({len(img) - len(df)} downloaded images have no record in occurrence.txt)")
    del img, occ
    df["image_id"] = df.file_path.str.rsplit("/", n=1).str[1].str.replace(r"\.jpg$", "", regex=True)
    df["species_binomial"] = df.species
    df["species"] = C.epithet(df.genus, df.species_binomial)
    df["scientificName"] = (df.family + " " + df.genus + " " + df.species).where(df.species.notna())
    df = df.rename(columns={"file_path": "filepath", "institutionCode": "creator"})
    cols = ["image_id", "filepath", "scientificName", "family", "genus", "species", "creator",
            "gbif_id", "image_no", "species_binomial", "taxonRank", "basisOfRecord", "kingdom",
            "catalogNumber", "occurrenceID", "datasetKey"]
    df = df[cols]

    # 4. Sanity: do the indexed files exist? (A full stat of ~20M files is too slow; sample.)
    random.seed(0)
    sample = random.sample(range(len(df)), min(5000, len(df)))
    exists = np.mean([os.path.exists(p) for p in df.filepath.iloc[sample]])
    print(f"file-exists check on {len(sample)} random rows: {exists:.4f}")

    # 5. Summary evidence.
    print("\nkingdom:", df.kingdom.value_counts(dropna=False).head(5).to_dict())
    print("taxonRank (images):", df.taxonRank.value_counts(dropna=False).head(10).to_dict())
    print("basisOfRecord (images):", df.basisOfRecord.value_counts(dropna=False).to_dict())
    print("top institutions (images):", df.creator.value_counts().head(20).to_dict())
    print(f"images per record: mean {df.groupby('gbif_id').size().mean():.3f}")

    sp = df[(df.kingdom == "Plantae") & df.species.notna()]
    key = sp.species_binomial.str.lower().str.strip()
    print(f"\nspecies-labelled Plantae images: {len(sp)} ({len(sp) / len(df):.1%}), species: {key.nunique()}")
    C.uniform_table("F25 snapshot, all record types", key.value_counts())
    herb = sp.basisOfRecord == "PRESERVED_SPECIMEN"
    C.uniform_table("F25 snapshot, PRESERVED_SPECIMEN only", key[herb].value_counts())

    t0 = time.time()
    df.to_csv(out, index=False)
    print(f"\nwrote {out}: {len(df)} rows, {os.path.getsize(out) / 1e9:.2f} GB ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
