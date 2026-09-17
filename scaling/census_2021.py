"""CPU-only census of the Kaggle Herbarium data on SCC.

No training, no GPU, no writes outside stdout. The point is to replace what we
*believe* about Kaggle 2021 ("~65k species, skewed") with what is actually on
disk, because the top rung of any N-ladder is bounded by the number of species
that clear the per-class image floor -- not by the number of species the Kaggle
description advertises.

Everything is printed under __K21__ markers so the run log is greppable.

Deliberately defensive about layout: the repo's constants.py disagrees with
itself across modules, and a 2.5M-image tree on a shared GPFS punishes naive
recursive walks. So we discover the tree shallowly, sample rather than stat
everything, and report what we could not read instead of dying on it.
"""

import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/projectnb/herbdl/data/kaggle-herbaria"

# Same tolerant column resolution as scaling/data.py, kept local so this script
# has no import-time dependency on the sweep code.
IMAGE_COLS = ["filename", "filepath", "image", "path", "file_name"]
LABEL_COLS = ["scientificName", "scientific_name", "scientificNameEncoded", "species", "label"]

MANIFEST_EXT = (".csv", ".json", ".jsonl", ".parquet", ".txt")


def hr(title):
    print(f"\n__K21__ {'=' * 6} {title} {'=' * (60 - len(title))}", flush=True)


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------
# 1. Filesystem discovery
# --------------------------------------------------------------------------

IMAGE_DIR_NAMES = {"images", "train_images", "test_images"}


def scan_tree(root, max_depth=3, line_budget=400):
    """Shallow listing that deliberately stops at the image shards.

    The first version of this guarded on ">2000 entries" and descended anyway:
    the 2021 shards hold exactly 1,000 files each, so it walked the whole
    2.5M-image tree and produced a 15MB log. The shard *names* are the reliable
    signal, not their size, so we stop at them by name and keep a hard line
    budget as a backstop.
    """
    hr("filesystem tree")
    if not os.path.isdir(root):
        print(f"__K21__ FATAL: {root} is not a directory", flush=True)
        return

    printed = [0]

    def emit(line):
        if printed[0] >= line_budget:
            return False
        printed[0] += 1
        print(line, flush=True)
        return True

    def walk(path, depth, prefix=""):
        try:
            entries = sorted(os.scandir(path), key=lambda e: e.name)
        except OSError as exc:
            emit(f"{prefix}  <unreadable: {exc}>")
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        # Summarise rather than list when a directory is mostly data files.
        if len(files) > 40:
            ext_counts = {}
            total = 0
            for e in files:
                ext_counts[os.path.splitext(e.name)[1].lower()] = \
                    ext_counts.get(os.path.splitext(e.name)[1].lower(), 0) + 1
                try:
                    total += e.stat().st_size
                except OSError:
                    pass
            emit(f"{prefix}  <{len(files):,} files, {human(total)}, by ext: "
                 f"{dict(sorted(ext_counts.items(), key=lambda t: -t[1]))}>")
        else:
            for e in files:
                try:
                    size = e.stat().st_size
                except OSError:
                    size = -1
                mark = "  <-- manifest?" if e.name.lower().endswith(MANIFEST_EXT) else ""
                emit(f"{prefix}  {e.name:<45s} {human(size):>10s}{mark}")

        for e in dirs:
            try:
                n = sum(1 for _ in os.scandir(e.path))
            except OSError as exc:
                emit(f"{prefix}  {e.name}/  <unreadable: {exc}>")
                continue
            if e.name in IMAGE_DIR_NAMES:
                emit(f"{prefix}  {e.name}/  ({n:,} shard dirs)  <-- image root, not descending")
                shard_stats(e.path, prefix + "      ")
                continue
            emit(f"{prefix}  {e.name}/  ({n:,} entries)")
            if depth < max_depth:
                walk(e.path, depth + 1, prefix + "    ")

    def shard_stats(image_root, prefix):
        """Estimate the image count without stat-ing 2.5M files: count every
        shard's entries for a sample of shards and extrapolate, then report the
        exact total only if the shard count is small enough to be cheap."""
        try:
            shards = [e for e in os.scandir(image_root) if e.is_dir()]
        except OSError as exc:
            print(f"{prefix}<unreadable: {exc}>", flush=True)
            return
        if not shards:
            return
        rng = random.Random(0)
        sample = rng.sample(shards, min(12, len(shards)))
        t0 = time.time()
        per = []
        for sh in sample:
            try:
                sub = [e for e in os.scandir(sh.path) if e.is_dir()]
            except OSError:
                continue
            if sub:  # two-level shards (2021: images/NNN/MM/*.jpg)
                inner = rng.sample(sub, min(6, len(sub)))
                counts = []
                for d in inner:
                    try:
                        counts.append(sum(1 for _ in os.scandir(d.path)))
                    except OSError:
                        pass
                if counts:
                    per.append(len(sub) * (sum(counts) / len(counts)))
            else:
                try:
                    per.append(sum(1 for _ in os.scandir(sh.path)))
                except OSError:
                    pass
        if per:
            est = len(shards) * (sum(per) / len(per))
            print(f"{prefix}~{est:,.0f} image files estimated "
                  f"({len(shards):,} shards x ~{sum(per) / len(per):,.0f} each, "
                  f"sampled in {time.time() - t0:.1f}s)", flush=True)

    print(root, flush=True)
    walk(root, 0)
    if printed[0] >= line_budget:
        print(f"  <line budget of {line_budget} reached; tree truncated>", flush=True)


def find_manifests(root):
    """Candidate manifests from the top two levels only.

    os.walk is pruned in place: without `dirnames[:] = ...` it still descends
    into every image shard even though we skip those levels, which is minutes
    of pointless metadata traffic on a shared filesystem.
    """
    found = []
    for cur, dirnames, files in os.walk(root, topdown=True):
        rel_depth = cur[len(root):].strip(os.sep).count(os.sep) if cur != root else -1
        if rel_depth >= 1:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if d not in IMAGE_DIR_NAMES]
        for f in files:
            if f.lower().endswith(MANIFEST_EXT):
                p = os.path.join(cur, f)
                try:
                    found.append((p, os.path.getsize(p)))
                except OSError:
                    pass
    return sorted(found, key=lambda t: -t[1])


def peek(path, size):
    """Show a manifest's shape without loading it. Large JSON gets byte-level
    inspection only -- Kaggle 2021's metadata.json is a multi-hundred-MB COCO
    blob and json.load on it would balloon memory for no benefit here."""
    print(f"\n__K21__ --- {path}  ({human(size)})", flush=True)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".csv", ".txt"):
            head = pd.read_csv(path, nrows=3, sep=None, engine="python")
            print(f"    columns: {list(head.columns)}", flush=True)
            print(head.to_string(max_colwidth=40), flush=True)
        elif ext in (".json", ".jsonl"):
            with open(path, "r", errors="replace") as fh:
                first = fh.read(1200)
            print(f"    first 1200 bytes: {first[:1200]!r}", flush=True)
            if first.lstrip().startswith("{") and size < 50 * 1024 ** 2:
                with open(path) as fh:
                    obj = json.load(fh)
                if isinstance(obj, dict):
                    print("    top-level keys: "
                          + ", ".join(f"{k}(n={len(v) if hasattr(v, '__len__') else '?'})"
                                      for k, v in obj.items()), flush=True)
        elif ext == ".parquet":
            print(f"    columns: {list(pd.read_parquet(path).columns)}", flush=True)
    except Exception as exc:
        print(f"    <could not peek: {type(exc).__name__}: {exc}>", flush=True)


# --------------------------------------------------------------------------
# 2. Species census -- the numbers that bound the ladder
# --------------------------------------------------------------------------

def resolve(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    print(f"    <no {what} column among {candidates}; have {list(df.columns)}>", flush=True)
    return None


def species_census(name, df, label_col=None, image_col=None):
    """The distribution that decides which N values are reachable."""
    hr(f"species census: {name}")
    label_col = label_col or resolve(df, LABEL_COLS, "label")
    image_col = image_col or resolve(df, IMAGE_COLS, "image")
    if label_col is None:
        return None

    print(f"    rows        : {len(df):,}", flush=True)
    print(f"    label column: {label_col}", flush=True)
    print(f"    image column: {image_col}", flush=True)

    labels = df[label_col]
    n_null = int(labels.isna().sum())
    if n_null:
        print(f"    NULL labels : {n_null:,}  (excluded below)", flush=True)
        labels = labels.dropna()

    counts = labels.value_counts()
    print(f"    species     : {counts.size:,}", flush=True)

    q = np.percentile(counts.to_numpy(), [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("    images/species: min={:.0f} p1={:.0f} p5={:.0f} p25={:.0f} med={:.0f} "
          "p75={:.0f} p95={:.0f} p99={:.0f} max={:.0f} mean={:.1f}".format(
              *q, counts.mean()), flush=True)

    # The head/tail shape -- this is the "skew" the ladder has to survive.
    srt = np.sort(counts.to_numpy())[::-1]
    frac = np.cumsum(srt) / srt.sum()
    for p in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(p * len(srt)))
        print(f"    top {p:4.0%} of species hold {frac[k - 1]:6.1%} of images", flush=True)

    # THE decision table: an N-ladder needs k_train + k_eval images per class,
    # so the reachable top rung is exactly the count in this column.
    print("\n    species surviving a per-species image floor:", flush=True)
    print("      floor   n_species   (as % of all)", flush=True)
    for thr in (2, 3, 5, 8, 10, 12, 15, 16, 20, 25, 30, 40, 50, 75, 100):
        n = int((counts >= thr).sum())
        print(f"      >={thr:4d}   {n:9,d}   {n / counts.size:6.1%}", flush=True)

    for k_tr, k_ev in ((10, 5), (8, 4), (5, 3), (4, 2), (3, 2), (2, 1)):
        n = int((counts >= k_tr + k_ev).sum())
        print(f"    k_train={k_tr:2d} k_eval={k_ev:2d}  ->  top rung N = {n:,}", flush=True)

    return {"counts": counts, "label_col": label_col, "image_col": image_col}


def taxonomic_levels(name, df):
    """Species/genus/family cardinality -- the depth node will want these."""
    have = [c for c in ("family", "genus", "species", "scientificName") if c in df.columns]
    if not have:
        return
    print(f"\n    taxonomic cardinality ({name}):", flush=True)
    for c in have:
        print(f"      unique {c:<16s}: {df[c].nunique():,}", flush=True)


def path_check(name, df, image_col, image_root, n=300, seed=0):
    """Do the manifest's images actually exist? Sampled -- stat'ing 2.5M files
    on a shared filesystem is antisocial and unnecessary for a yes/no."""
    if image_col is None:
        return
    hr(f"image path check: {name}")
    rng = random.Random(seed)
    rows = df[image_col].dropna().astype(str).tolist()
    if not rows:
        print("    <no image paths>", flush=True)
        return
    sample = rng.sample(rows, min(n, len(rows)))

    variants = {"as-given": lambda p: p}
    if image_root:
        variants[f"joined to {image_root}"] = (
            lambda p: p if os.path.isabs(p) else os.path.join(image_root, p))

    for vname, fn in variants.items():
        t0 = time.time()
        ok = sum(1 for p in sample if os.path.exists(fn(p)))
        print(f"    {vname:<55s}: {ok}/{len(sample)} exist "
              f"({time.time() - t0:.1f}s)", flush=True)
        if ok:
            ex = next(fn(p) for p in sample if os.path.exists(fn(p)))
            print(f"        e.g. {ex}", flush=True)
        else:
            print(f"        e.g. MISSING {fn(sample[0])}", flush=True)


WANTED_COLS = set(IMAGE_COLS) | set(LABEL_COLS) | {
    "family", "genus", "species", "image_id", "category_id", "caption", "institution"}


def load_any(path):
    """Load a manifest, keeping only columns we actually use.

    index.csv alone is 842MB on disk; read in full with object dtypes it is
    several GB in memory for no benefit, so we read the header first and
    restrict with usecols.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
        keep = [c for c in cols if c in WANTED_COLS]
        if not keep:
            keep = cols
        dropped = [c for c in cols if c not in keep]
        if dropped:
            print(f"    (ignoring columns: {dropped})", flush=True)
        return pd.read_csv(path, usecols=keep, low_memory=False)
    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with open(path) as fh:
                return pd.DataFrame(json.load(fh))
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported {ext}")


def report_label_mappings(root):
    """Someone has already built label mappings here -- read them rather than
    re-deriving, and check whether the 64k one matches the 2021 pool."""
    hr("existing label mappings")
    for fn in ("label_mapping_15k.json", "label_mapping_64k.json",
               "label_mapping_merged.json"):
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            print(f"    MISSING  {fn}", flush=True)
            continue
        try:
            with open(p) as fh:
                obj = json.load(fh)
        except Exception as exc:
            print(f"    {fn}: <unreadable: {type(exc).__name__}: {exc}>", flush=True)
            continue
        if isinstance(obj, dict):
            items = list(obj.items())
            print(f"    {fn}: dict with {len(obj):,} entries; "
                  f"first 3 = {items[:3]}", flush=True)
        elif isinstance(obj, list):
            print(f"    {fn}: list of {len(obj):,}; first 3 = {obj[:3]}", flush=True)



def coco_categories(path, max_array_bytes=64 * 1024 ** 2):
    """Pull just the "categories" array out of a COCO-style metadata.json.

    The 2021 train metadata is 398MB and the 2022 one 636MB, so json.load is
    out. The categories array is the authoritative taxon list -- the CSVs are
    derived from it and may have been filtered -- and it is only a few MB, so
    we stream to the key and bracket-match the array.

    Wrapped by the caller in try/except: this is a cross-check, and a parse
    failure must not cost us the census that already printed above.
    """
    key = b'"categories"'
    buf = b""
    found_at = None
    with open(path, "rb") as fh:
        # Locate the key, keeping a small overlap so it cannot straddle reads.
        while True:
            chunk = fh.read(4 * 1024 ** 2)
            if not chunk:
                break
            buf += chunk
            idx = buf.find(key)
            if idx != -1:
                found_at = idx
                break
            buf = buf[-len(key):]
        if found_at is None:
            return None

        # Advance to the opening bracket, then bracket-match to the close.
        rest = buf[found_at + len(key):]
        while b"[" not in rest:
            more = fh.read(4 * 1024 ** 2)
            if not more:
                return None
            rest += more
        rest = rest[rest.index(b"["):]

        depth, out, i = 0, bytearray(), 0
        while True:
            while i < len(rest):
                c = rest[i]
                out.append(c)
                if c == 0x5B:      # [
                    depth += 1
                elif c == 0x5D:    # ]
                    depth -= 1
                    if depth == 0:
                        return json.loads(bytes(out).decode("utf-8"))
                i += 1
                if len(out) > max_array_bytes:
                    raise ValueError(f"categories array exceeded {human(max_array_bytes)}")
            more = fh.read(4 * 1024 ** 2)
            if not more:
                return None
            rest, i = more, 0


def report_coco(name, path):
    hr(f"authoritative taxon list: {name}")
    if not os.path.exists(path):
        print(f"    MISSING {path}", flush=True)
        return
    t0 = time.time()
    try:
        cats = coco_categories(path)
    except Exception as exc:
        print(f"    <parse failed: {type(exc).__name__}: {exc}>", flush=True)
        return
    if cats is None:
        print("    <no \"categories\" array found>", flush=True)
        return
    print(f"    categories in metadata.json : {len(cats):,}  "
          f"(read in {time.time() - t0:.1f}s)", flush=True)
    print(f"    example entry               : {cats[0]}", flush=True)
    keys = sorted({k for c in cats[:5000] for k in c})
    print(f"    category fields             : {keys}", flush=True)
    for level in ("family", "genus", "species", "name", "scientificName"):
        if level in keys:
            vals = {str(c.get(level)) for c in cats}
            print(f"    unique {level:<16s}     : {len(vals):,}", flush=True)


def main():
    print(f"__K21__ host={os.uname().nodename}  python={sys.version.split()[0]}", flush=True)
    print(f"__K21__ pandas={pd.__version__} numpy={np.__version__}", flush=True)
    t0 = time.time()

    scan_tree(ROOT)

    report_label_mappings(ROOT)

    hr("manifest inventory")
    manifests = find_manifests(ROOT)
    for p, s in manifests:
        print(f"    {human(s):>10s}  {p}", flush=True)
    for p, s in manifests:
        peek(p, s)

    # Load whichever of the expected manifests are actually present. We do not
    # assume the constants.py paths are right -- we verify each one.
    expected = {
        "kaggle21_train": f"{ROOT}/train_2021.csv",
        "kaggle21_val": f"{ROOT}/val_2021.csv",
        "kaggle22_train": f"{ROOT}/train_2022.csv",
        "kaggle22_val": f"{ROOT}/val_2022.csv",
        "merged_index": f"{ROOT}/index.csv",
    }
    roots = {
        "kaggle21": f"{ROOT}/herbarium-2021/train/images/",
        "kaggle22": f"{ROOT}/herbarium-2022/train_images/",
    }
    hr("expected paths from repo constants.py")
    for k, v in list(expected.items()) + list(roots.items()):
        print(f"    {'OK ' if os.path.exists(v) else 'MISSING'}  {k:<16s} {v}", flush=True)

    loaded = {}
    for name, path in expected.items():
        if not os.path.exists(path):
            continue
        try:
            loaded[name] = load_any(path)
            print(f"__K21__ loaded {name}: {len(loaded[name]):,} rows, "
                  f"columns={list(loaded[name].columns)}", flush=True)
        except Exception as exc:
            print(f"__K21__ FAILED to load {name}: {type(exc).__name__}: {exc}", flush=True)

    # Kaggle 2021: train+val concatenated is the real pool, since our own split
    # is what we will redraw anyway.
    pools = {}
    if "kaggle21_train" in loaded:
        parts = [loaded["kaggle21_train"]]
        if "kaggle21_val" in loaded:
            parts.append(loaded["kaggle21_val"])
        pools["kaggle21_full"] = pd.concat(parts, ignore_index=True)
    if "kaggle22_train" in loaded:
        parts = [loaded["kaggle22_train"]]
        if "kaggle22_val" in loaded:
            parts.append(loaded["kaggle22_val"])
        pools["kaggle22_full"] = pd.concat(parts, ignore_index=True)
    if "merged_index" in loaded:
        pools["merged_index"] = loaded["merged_index"]

    info = {}
    for name, df in pools.items():
        info[name] = species_census(name, df)
        taxonomic_levels(name, df)

    for name, df in pools.items():
        if info.get(name) is None:
            continue
        root = roots["kaggle21"] if "21" in name else roots["kaggle22"]
        path_check(name, df, info[name]["image_col"], root)

    # Species overlap -- decides whether 21+22 can be pooled into one label space.
    hr("2021 / 2022 species overlap")
    if "kaggle21_full" in info and "kaggle22_full" in info and \
            info["kaggle21_full"] and info["kaggle22_full"]:
        def names(pool):
            col = info[pool]["label_col"]
            return set(pools[pool][col].dropna().astype(str).str.strip().str.lower())
        a, b = names("kaggle21_full"), names("kaggle22_full")
        print(f"    kaggle21 species        : {len(a):,}", flush=True)
        print(f"    kaggle22 species        : {len(b):,}", flush=True)
        print(f"    shared                  : {len(a & b):,}", flush=True)
        print(f"    2021-only               : {len(a - b):,}", flush=True)
        print(f"    2022-only               : {len(b - a):,}", flush=True)
        print(f"    union (pooled label set): {len(a | b):,}", flush=True)
    else:
        print("    <one or both pools unavailable>", flush=True)

    # Cross-check the CSVs against the raw Kaggle metadata, last: if this is
    # slow or fails, the decision-relevant census above has already printed.
    report_coco("kaggle21 train", f"{ROOT}/herbarium-2021/train/metadata.json")
    report_coco("kaggle22 train", f"{ROOT}/herbarium-2022/train_metadata.json")

    print(f"\n__K21__ done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
