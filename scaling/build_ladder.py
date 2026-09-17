"""Build nested per-rung train/val datasets for the label-cardinality ladder.

Source pool is /projectnb/herbdl/data/master_index.csv -- the unified index over
GBIF-F24, harvard-herbaria, kaggle-2021 and kaggle-2022 (7,966,043 images,
271,529 distinct scientificName values).

Design rules, which mirror sweep.py's build_task so the fine-tuned arm and the
frozen-probe arm are measuring the same thing:

  * BALANCED. Every species contributes exactly k_train train images and k_eval
    val images at every rung. "More labels" must not also mean "more or less
    evidence per label", or the two effects are inseparable.
  * NESTED. Rung N is the first N species of one seeded shuffle of the eligible
    pool, so every species in a small rung is also in every larger rung.
  * STABLE IMAGES. A species' chosen images are seeded by (seed, species) and
    never by N, so a species carries byte-identical images up the whole ladder.
  * DENSE RELABELLING. Each rung re-encodes its species to 0..N-1, because the
    classifier head is sized N.

Two passes over the CSV, streaming, so an 8M-row file never lands in memory.

Writes, per rung, into <out_dir>:
    ladder_N<N>_train.json    JSONL: {"image_id","filename","scientificNameEncoded","scientificName"}
    ladder_N<N>_val.json      same shape, disjoint images
    ladder_N<N>_classes.csv   rung-local id -> scientificName, plus source counts
and one ladder_manifest.json describing every rung.

JSONL with an absolute `filename` and an integer `scientificNameEncoded` is
exactly what finetuning/SWIN/SWIN_finetuning_advanced.py already consumes, so
the fine-tuning configs need no new data-loading code.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

# Deliberately no numpy import. SCC's system python3 ships a numpy too old for
# the modern Generator API, and this script must run on a login node without
# loading a conda env. stdlib `random` with an explicit integer seed is just as
# reproducible and has no version floor.

csv.field_size_limit(10 ** 9)


def log(msg):
    print(f"__LADDER__: {msg}", flush=True)


def pass1_counts(path, label_col):
    """Count images per species without holding the file in memory."""
    counts = defaultdict(int)
    rows = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            v = row.get(label_col)
            if v:
                counts[v] += 1
            if rows % 2_000_000 == 0:
                log(f"pass 1: {rows:,} rows")
    log(f"pass 1 done: {rows:,} rows, {len(counts):,} distinct {label_col}")
    return counts, rows


def pass2_collect(path, label_col, keep, need):
    """Collect up to `need` rows per eligible species.

    Bounded memory: at most len(keep) * need rows are retained, which for
    69k species at 15 images is ~1M rows, not 8M.
    """
    bucket = defaultdict(list)
    rows = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            sp = row.get(label_col)
            if sp in keep and len(bucket[sp]) < need:
                bucket[sp].append({
                    "image_id": row.get("image_id", ""),
                    "filename": row.get("filepath", ""),
                    "scientificName": sp,
                    "source": _source_of(row.get("filepath", "")),
                })
            if rows % 2_000_000 == 0:
                log(f"pass 2: {rows:,} rows")
    log(f"pass 2 done: retained {sum(len(v) for v in bucket.values()):,} rows "
        f"for {len(bucket):,} species")
    return bucket


def _source_of(fp):
    """Which upstream collection a path belongs to, for the per-rung report."""
    parts = fp.split("/")
    if len(parts) > 5 and parts[1] == "projectnb":
        tail = parts[5]
        return {"IMAGES": "gbif-f24", "images": "harvard",
                "herbarium-2021": "kaggle-2021",
                "herbarium-2022": "kaggle-2022"}.get(tail, tail)
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/projectnb/herbdl/data/master_index.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-col", default="scientificName")
    ap.add_argument("--k-train", type=int, default=10)
    ap.add_argument("--k-eval", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rungs", default="50,200,1000,5000,13548,29689,69462")
    args = ap.parse_args()

    need = args.k_train + args.k_eval
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"index={args.index}")
    log(f"k_train={args.k_train} k_eval={args.k_eval} (need {need}/species) seed={args.seed}")

    counts, total_rows = pass1_counts(args.index, args.label_col)
    eligible = sorted(sp for sp, c in counts.items() if c >= need)
    log(f"eligible species (>= {need} images): {len(eligible):,} "
        f"of {len(counts):,} ({len(eligible) / len(counts):.1%})")

    requested = [int(x) for x in args.rungs.split(",") if x.strip()]
    rungs = sorted({n for n in requested if n <= len(eligible)} | {len(eligible)})
    dropped = [n for n in requested if n > len(eligible)]
    if dropped:
        log(f"NOTE: requested rungs {dropped} exceed the {len(eligible):,} eligible "
            f"species; the ladder tops out at the full eligible pool instead.")
    log(f"rungs: {rungs}")

    bucket = pass2_collect(args.index, args.label_col, set(eligible), need)

    # One shuffle drives every rung, so rungs nest.
    shuffled = list(eligible)
    random.Random(args.seed).shuffle(shuffled)
    # Stable per-species seed. Python's hash() of a str is salted per process
    # (PYTHONHASHSEED), so using it here would silently produce a DIFFERENT
    # ladder on every run. Position in the sorted eligible list is stable.
    sp_index = {sp: i for i, sp in enumerate(eligible)}

    manifest = {"index": args.index, "label_col": args.label_col,
                "k_train": args.k_train, "k_eval": args.k_eval,
                "seed": args.seed, "total_rows_scanned": total_rows,
                "distinct_species": len(counts), "eligible_species": len(eligible),
                "rungs": []}

    for n in rungs:
        chosen = shuffled[:n]
        tr_path = os.path.join(args.out_dir, f"ladder_N{n}_train.json")
        ev_path = os.path.join(args.out_dir, f"ladder_N{n}_val.json")
        cls_path = os.path.join(args.out_dir, f"ladder_N{n}_classes.csv")
        src_mix = defaultdict(int)
        n_tr = n_ev = 0

        with open(tr_path, "w") as ftr, open(ev_path, "w") as fev, \
                open(cls_path, "w", newline="") as fcls:
            cw = csv.writer(fcls)
            cw.writerow(["scientificNameEncoded", "scientificName", "n_train", "n_val"])
            for new_id, sp in enumerate(chosen):
                imgs = bucket[sp]
                # Integer seed from (seed, species position) -- never from n,
                # so a species keeps the same images at every rung.
                sel = list(imgs)
                random.Random(args.seed * 1000003 + sp_index[sp]).shuffle(sel)
                tr, ev = sel[:args.k_train], sel[args.k_train:need]
                for rec, fh in ((tr, ftr), (ev, fev)):
                    for r in rec:
                        fh.write(json.dumps({
                            "image_id": r["image_id"],
                            "filename": r["filename"],
                            "scientificNameEncoded": new_id,
                            "scientificName": r["scientificName"],
                        }) + "\n")
                        src_mix[r["source"]] += 1
                cw.writerow([new_id, sp, len(tr), len(ev)])
                n_tr += len(tr)
                n_ev += len(ev)

        entry = {"n_classes": n, "train_file": tr_path, "validation_file": ev_path,
                 "classes_file": cls_path, "n_train_images": n_tr,
                 "n_val_images": n_ev, "source_mix": dict(src_mix)}
        manifest["rungs"].append(entry)
        log(f"rung N={n:>6,}: train={n_tr:>8,} val={n_ev:>7,}  "
            f"sources={dict(src_mix)}")

    man_path = os.path.join(args.out_dir, "ladder_manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"wrote {man_path}")

    print("\n" + "=" * 78)
    print("LADDER BUILT")
    print("=" * 78)
    print("{:>8} {:>12} {:>10} {:>10}".format("N", "train imgs", "val imgs", "files"))
    print("-" * 78)
    for e in manifest["rungs"]:
        print("{:>8,} {:>12,} {:>10,} {:>10}".format(
            e["n_classes"], e["n_train_images"], e["n_val_images"], "3"))
    print("-" * 78)
    print(f"out_dir: {args.out_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
