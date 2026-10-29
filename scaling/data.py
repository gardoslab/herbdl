"""Manifest loading and the class census for the Herbarium pool.

The manifest is whatever the project already ships (CSV or JSON/JSONL). We are
deliberately tolerant about column naming because the 2021/2022/merged
manifests disagree, and we fail loudly with the actual column list rather than
silently picking the wrong field.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

IMAGE_COLUMN_CANDIDATES = ["filename", "filepath", "image", "path", "file_name"]
LABEL_COLUMN_CANDIDATES = ["scientificNameEncoded", "scientific_name", "scientificName", "species", "label"]


def _pick(df, requested, candidates, what):
    if requested and requested in df.columns:
        return requested
    if requested:
        print(f"__LCS__: requested {what} column '{requested}' not present; falling back", file=sys.stderr)
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No usable {what} column. Tried {([requested] if requested else []) + candidates}. "
                   f"Manifest columns: {list(df.columns)}")


def load_manifest(path, image_column=None, label_column=None, image_root=""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".json", ".jsonl"):
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            with open(path) as fh:
                df = pd.DataFrame(json.load(fh))
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported manifest extension '{ext}' for {path}")

    img_col = _pick(df, image_column, IMAGE_COLUMN_CANDIDATES, "image")
    lab_col = _pick(df, label_column, LABEL_COLUMN_CANDIDATES, "label")

    paths = df[img_col].astype(str)
    if image_root:
        paths = paths.map(lambda p: p if os.path.isabs(p) else os.path.join(image_root, p))

    out = pd.DataFrame({"path": paths.values, "raw_label": df[lab_col].values})
    # Re-encode densely and deterministically: manifest encodings may be sparse
    # or carry ordering from a different split, and every downstream index
    # (centroids, per-class metrics) assumes labels are 0..K-1.
    uniq = np.sort(pd.unique(out["raw_label"]))
    remap = {v: i for i, v in enumerate(uniq)}
    out["label"] = out["raw_label"].map(remap).astype(np.int64)

    for extra in ("family", "genus", "species"):
        if extra in df.columns:
            out[extra] = df[extra].values

    return out, {"image_column": img_col, "label_column": lab_col,
                 "n_images": len(out), "n_classes": len(uniq),
                 "manifest_columns": list(df.columns)}


def census(df, info, sample_check=8):
    """Print what we actually loaded. Cheap, and it is the first thing the log
    should show: every later number depends on this being the pool we think."""
    counts = df["label"].value_counts().to_numpy()
    q = np.percentile(counts, [0, 5, 25, 50, 75, 95, 100]).astype(int)

    print("__LCS__ census ---------------------------------------------")
    print(f"  manifest columns : {info['manifest_columns']}")
    print(f"  image column     : {info['image_column']}")
    print(f"  label column     : {info['label_column']}")
    print(f"  images           : {info['n_images']:,}")
    print(f"  classes          : {info['n_classes']:,}")
    print(f"  images/class     : min={q[0]} p5={q[1]} p25={q[2]} med={q[3]} "
          f"p75={q[4]} p95={q[5]} max={q[6]} mean={counts.mean():.1f}")
    for thr in (10, 20, 30, 50, 100, 200):
        n = int((counts >= thr).sum())
        print(f"  classes with >={thr:4d} images : {n:6,d}  ({n / len(counts):6.1%})")
    # Long-tail shape, so the head/tail story has a documented starting point.
    srt = np.sort(counts)[::-1]
    frac = np.cumsum(srt) / srt.sum()
    for p in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(p * len(srt)))
        print(f"  top {p:4.0%} of classes hold {frac[k - 1]:6.1%} of images")

    missing = [p for p in df["path"].head(sample_check) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{sample_check} sampled image paths do not exist, e.g. {missing[0]}. "
            "Check `data.image_root` / the manifest's image column.")
    print(f"  path check       : first {sample_check} paths all exist "
          f"(e.g. {df['path'].iloc[0]})")
    print("__LCS__ census end -----------------------------------------")
    return counts


def load_pair(train_path, eval_path, image_column=None, label_column=None, image_root=""):
    """Load a train pool and a DISJOINT eval pool onto one shared label space.

    This becomes mandatory the moment the representation is itself fine-tuned on
    the training pool. Scoring such a backbone on images it was fine-tuned on
    measures memorisation, not recognition, and would flatten the cardinality
    curve for entirely uninteresting reasons.

    The two manifests are encoded independently by load_manifest, so their dense
    label ids do NOT line up. We rebuild both from the *train* pool's raw labels
    and drop eval rows whose class never appears in train.
    """
    tr, tinfo = load_manifest(train_path, image_column, label_column, image_root)
    ev, einfo = load_manifest(eval_path, image_column, label_column, image_root)

    uniq = np.sort(pd.unique(tr["raw_label"]))
    remap = {v: i for i, v in enumerate(uniq)}
    tr["label"] = tr["raw_label"].map(remap).astype(np.int64)

    known = ev["raw_label"].isin(remap)
    n_dropped = int((~known).sum())
    ev = ev[known].copy()
    ev["label"] = ev["raw_label"].map(remap).astype(np.int64)

    tr = tr.copy()
    tr["is_eval"] = False
    ev["is_eval"] = True
    both = pd.concat([tr, ev], ignore_index=True)

    info = dict(tinfo)
    info.update({
        "n_classes": int(len(uniq)),
        "n_images": int(len(both)),
        "n_train_images": int(len(tr)),
        "n_eval_images": int(len(ev)),
        "eval_rows_dropped_unknown_class": n_dropped,
        "eval_manifest_columns": einfo["manifest_columns"],
    })
    return both, info


def pool_census(df, info):
    """Second census for two-pool mode: per-pool sizes and per-class coverage."""
    tr = df[~df["is_eval"]]
    ev = df[df["is_eval"]]
    tc = tr["label"].value_counts()
    ec = ev["label"].value_counts()
    print("__LCS__ pool census ----------------------------------------")
    print(f"  train pool       : {len(tr):,} images over {tc.size:,} classes")
    print(f"  eval  pool       : {len(ev):,} images over {ec.size:,} classes")
    print(f"  eval rows dropped (class absent from train): "
          f"{info['eval_rows_dropped_unknown_class']:,}")
    if tc.size:
        print("  train imgs/class : min={} med={} max={}".format(
            int(tc.min()), int(tc.median()), int(tc.max())))
    if ec.size:
        print("  eval  imgs/class : min={} med={} max={}".format(
            int(ec.min()), int(ec.median()), int(ec.max())))
    overlap = set(tr["path"]) & set(ev["path"])
    print(f"  path overlap between pools : {len(overlap):,} "
          f"{'<-- LEAKAGE, eval is not held out' if overlap else '(clean)'}")
    print("__LCS__ pool census end ------------------------------------", flush=True)
    return len(overlap)
