"""Inventory the herbaria data pools so a >15k label ladder can be designed.

Why this exists as a job rather than a shell session: SCC login requires
interactive 2FA, so the agent driving these experiments cannot get a shell on
the cluster. Everything it needs to know about the data therefore has to be
printed by something that runs *on* the cluster.

The survey answers exactly the questions that determine whether a bigger ladder
is possible at all:

  1. What manifests exist, and what is in them (columns, rows, classes)?
  2. Do their image paths actually resolve?
  3. How many classes clear each (k_train, k_eval) budget? That number, not the
     raw class count, is the real ceiling of a balanced ladder.
  4. Do the 2021 and 2022 label spaces refer to the same taxa? A merged ladder
     is incoherent unless we know.

It is deliberately read-only and deliberately avoids walking image directories,
which hold millions of files.
"""

import json
import os

import numpy as np
import pandas as pd

MANIFEST_EXT = (".csv", ".json", ".jsonl", ".parquet")
# Anything under these is image data: millions of files, never worth walking.
SKIP_DIR_HINTS = ("images", "train_images", "test_images", "thumbnails")

IMAGE_COLS = ["filename", "filepath", "image", "path", "file_name"]
LABEL_COLS = ["scientificNameEncoded", "category_id", "species", "label",
              "scientific_name", "scientificName"]
# Full binomials only. A bare species EPITHET ("alba") is not a taxon key --
# it recurs across genera -- so `species` is a last resort, and genus+species
# is composed in preference to it when both are present.
NAME_COLS = ["scientificName", "scientific_name", "name", "species"]


def _h(size):
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.0f}P"


def _count_entries(path, cap=200):
    """Count directory entries, giving up at `cap` so image dirs stay cheap."""
    n = 0
    try:
        with os.scandir(path) as it:
            for _ in it:
                n += 1
                if n >= cap:
                    return n, True
    except OSError:
        return -1, False
    return n, False


def list_tree(root, max_depth=2):
    print(f"__SURVEY__ tree of {root} (depth<={max_depth}, image dirs not walked)")
    if not os.path.isdir(root):
        print(f"  MISSING: {root}")
        return
    root_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count("/") - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        # Never descend into image stores.
        dirnames[:] = [d for d in dirnames
                       if not any(h in d.lower() for h in SKIP_DIR_HINTS)]
        indent = "  " * (depth + 1)
        n, capped = _count_entries(dirpath)
        print(f"{indent}{dirpath}/  [{n}{'+' if capped else ''} entries]")
        for f in sorted(filenames)[:25]:
            fp = os.path.join(dirpath, f)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                sz = 0
            mark = "  <-- manifest" if f.lower().endswith(MANIFEST_EXT) else ""
            print(f"{indent}  {f:<44} {_h(sz):>8}{mark}")
        if len(filenames) > 25:
            print(f"{indent}  ... {len(filenames) - 25} more files")
    # Report the image stores we deliberately skipped, at top level only.
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            for sub in sorted(os.listdir(p)) if os.path.isdir(p) else []:
                if any(h in sub.lower() for h in SKIP_DIR_HINTS):
                    sp = os.path.join(p, sub)
                    n, capped = _count_entries(sp)
                    print(f"  [image store, not walked] {sp}  "
                          f"[{n}{'+' if capped else ''} top-level entries]")
    print("__SURVEY__ tree end", flush=True)


def find_manifests(root, max_depth=3):
    out = []
    if not os.path.isdir(root):
        return out
    root_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count("/") - root_depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames
                       if not any(h in d.lower() for h in SKIP_DIR_HINTS)]
        for f in filenames:
            if f.lower().endswith(MANIFEST_EXT):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _coco_to_frame(obj):
    """Flatten a COCO-style metadata dict into one row per image.

    Kaggle Herbarium ships `annotations` (image_id -> category_id), `images`
    (id -> file_name) and `categories` (id -> name/family/genus/species) as
    separate lists. Individually none of them is a usable manifest; joined,
    they are exactly one. Reading only `annotations` would silently lose every
    filename, and reading the dict as-is yields a useless 1-row frame.
    """
    ann = pd.DataFrame(obj.get("annotations") or [])
    imgs = pd.DataFrame(obj.get("images") or [])
    cats = pd.DataFrame(obj.get("categories") or [])
    notes = [f"annotations={len(ann):,}", f"images={len(imgs):,}",
             f"categories={len(cats):,}"]

    if ann.empty and not imgs.empty:
        df = imgs.copy()
    elif imgs.empty:
        df = ann.copy()
    else:
        left_key = "image_id" if "image_id" in ann.columns else None
        right_key = "id" if "id" in imgs.columns else None
        if left_key and right_key:
            df = ann.merge(imgs, left_on=left_key, right_on=right_key,
                           how="left", suffixes=("", "_img"))
            notes.append("joined annotations<-images on image_id=id")
        else:
            df = ann.copy()
            notes.append("COULD NOT JOIN images (missing image_id/id)")

    if not cats.empty and "category_id" in df.columns and "id" in cats.columns:
        cat_cols = [c for c in ("id", "name", "family", "genus", "species")
                    if c in cats.columns]
        df = df.merge(cats[cat_cols], left_on="category_id", right_on="id",
                      how="left", suffixes=("", "_cat"))
        notes.append("joined categories on category_id=id")

    df.attrs["coco_notes"] = "; ".join(notes)
    df.attrs["sibling_keys"] = [k for k in obj
                                if k not in ("annotations", "images", "categories")]
    return df


def _load_any(path, nrows=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, nrows=nrows)
    if ext == ".json":
        # Try whole-document JSON FIRST. read_json(lines=True) happily returns a
        # 1-row frame for a COCO document instead of raising, so trusting it
        # would hide the nested rows rather than fail loudly.
        try:
            with open(path) as fh:
                obj = json.load(fh)
        except (ValueError, MemoryError) as e:
            if isinstance(e, MemoryError):
                raise
            return pd.read_json(path, lines=True, nrows=nrows)  # real JSONL
        if isinstance(obj, dict):
            if any(k in obj for k in ("annotations", "images", "categories")):
                return _coco_to_frame(obj)
            for key in ("data", "rows"):
                if isinstance(obj.get(key), list):
                    df = pd.DataFrame(obj[key])
                    df.attrs["json_key"] = key
                    return df
            return pd.DataFrame([obj])
        return pd.DataFrame(obj)
    raise ValueError(f"unsupported extension {ext}")


def _pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def profile_manifest(path, probe_paths=12, image_roots=()):
    print(f"\n__SURVEY__ manifest {path}  ({_h(os.path.getsize(path))})")
    try:
        df = _load_any(path)
    except Exception as e:
        print(f"  UNREADABLE: {type(e).__name__}: {e}")
        return None

    if df.attrs.get("coco_notes"):
        print(f"  COCO document: {df.attrs['coco_notes']}")
        print(f"  other top-level keys: {df.attrs.get('sibling_keys')}")
    elif df.attrs.get("json_key"):
        print(f"  (json object; rows taken from key '{df.attrs['json_key']}')")
    print(f"  rows    : {len(df):,}")
    print(f"  columns : {list(df.columns)}")

    img_col = _pick(df, IMAGE_COLS)
    lab_col = _pick(df, LABEL_COLS)
    name_col = _pick(df, NAME_COLS)
    print(f"  image col={img_col!r}  label col={lab_col!r}  name col={name_col!r}")

    info = {"path": path, "rows": len(df), "image_col": img_col,
            "label_col": lab_col, "name_col": name_col, "n_classes": None,
            "counts": None, "names": None}

    if lab_col is None:
        print("  no usable label column -- not a classification manifest")
        return info

    counts = df[lab_col].value_counts()
    info["n_classes"] = int(counts.size)
    info["counts"] = counts.to_numpy()
    print(f"  classes : {counts.size:,}")
    q = np.percentile(counts.to_numpy(), [0, 5, 25, 50, 75, 95, 100]).astype(int)
    print(f"  imgs/class: min={q[0]} p5={q[1]} p25={q[2]} med={q[3]} "
          f"p75={q[4]} p95={q[5]} max={q[6]} mean={counts.mean():.1f}")

    if "genus" in df.columns and "species" in df.columns:
        names = (df["genus"].astype(str).str.strip() + " "
                 + df["species"].astype(str).str.strip())
        info["names"] = set(pd.unique(names))
        info["name_source"] = "genus + ' ' + species"
        print(f"  distinct binomials from genus+species: {len(info['names']):,}")
    elif name_col is not None:
        names = df[name_col].astype(str).str.strip()
        info["names"] = set(pd.unique(names))
        info["name_source"] = name_col
        print(f"  distinct names in {name_col!r}: {len(info['names']):,}")
        if name_col == "species":
            print("    WARNING: this looks like a bare species epithet, not a "
                  "binomial; overlap figures below may be inflated.")

    # Eligibility is what actually caps a balanced ladder.
    print("  eligible classes by (k_train + k_eval):")
    c = counts.to_numpy()
    for kt, ke in ((10, 5), (10, 3), (5, 3), (5, 2), (3, 2), (20, 5)):
        n_ok = int((c >= kt + ke).sum())
        print(f"    k_train={kt:>2} k_eval={ke}  (need {kt + ke:>2}) -> "
              f"{n_ok:>6,} classes  ({n_ok / counts.size:5.1%})")

    if img_col is not None:
        vals = df[img_col].astype(str).head(probe_paths).tolist()
        roots = list(image_roots) + [os.path.dirname(path), ""]
        print("  path resolution probe:")
        best = None
        for r in roots:
            hits = sum(1 for v in vals
                       if os.path.exists(v if os.path.isabs(v) else os.path.join(r, v)))
            tag = r or "(as-is)"
            print(f"    root={tag:<62} {hits}/{len(vals)} exist")
            if best is None or hits > best[1]:
                best = (r, hits)
        info["best_root"] = best
        print(f"    example value: {vals[0] if vals else '(none)'}")
    return info


def compare_label_spaces(profiles):
    named = [p for p in profiles if p and p.get("names")]
    if len(named) < 2:
        print("\n__SURVEY__ label-space overlap: fewer than two manifests carry "
              "a usable name column; cannot compare.")
        return
    print("\n__SURVEY__ label-space overlap (by scientific name)")
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            a, b = named[i], named[j]
            sa, sb = a["names"], b["names"]
            inter = sa & sb
            union = sa | sb
            print(f"  {os.path.basename(a['path'])} ({len(sa):,}, from "
                  f"{a.get('name_source')}) vs {os.path.basename(b['path'])} "
                  f"({len(sb):,}, from {b.get('name_source')})")
            print(f"    shared={len(inter):,}  only-A={len(sa - sb):,}  "
                  f"only-B={len(sb - sa):,}  union={len(union):,}")
            if sa:
                print(f"    -> merging would give {len(union):,} distinct names "
                      f"({len(union) / max(len(sa), len(sb)):.2f}x the larger pool)")
    print("  NOTE: name matching is exact string equality. The Herbarium 2021")
    print("        paper flags that taxa appear under multiple names, so treat")
    print("        'shared' as a LOWER bound on true taxonomic overlap.")
    print("__SURVEY__ overlap end", flush=True)


def run(cfg):
    roots = cfg.get("survey", {}).get("roots") or [
        "/projectnb/herbdl/data/kaggle-herbaria"]
    image_roots = cfg.get("survey", {}).get("image_roots") or []
    print("=" * 72)
    print("POOL SURVEY -- read-only inventory of the herbaria data")
    print("=" * 72, flush=True)

    profiles = []
    for root in roots:
        list_tree(root, cfg.get("survey", {}).get("tree_depth", 2))
        mans = find_manifests(root)
        print(f"\n__SURVEY__ {len(mans)} manifest file(s) under {root}")
        for m in mans:
            try:
                profiles.append(profile_manifest(m, image_roots=image_roots))
            except Exception as e:
                print(f"  profile failed for {m}: {type(e).__name__}: {e}")
    compare_label_spaces(profiles)

    print("\n" + "=" * 72)
    print("LADDER CEILINGS -- largest N available per manifest at each budget")
    print("=" * 72)
    hdr = "{:<46} {:>9} {:>10} {:>10} {:>10}".format(
        "manifest", "classes", "k=10+5", "k=5+3", "k=3+2")
    print(hdr)
    print("-" * len(hdr))
    for p in profiles:
        if not p or p.get("counts") is None:
            continue
        c = p["counts"]
        print("{:<46} {:>9,} {:>10,} {:>10,} {:>10,}".format(
            os.path.basename(p["path"])[:46], p["n_classes"],
            int((c >= 15).sum()), int((c >= 8).sum()), int((c >= 5).sum())))
    print("-" * len(hdr))
    print("Read the k=10+5 column against the current ladder's 13,548-class ceiling.")
    print("=" * 72, flush=True)
