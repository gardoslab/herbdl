"""Size-matched control for a taxonomy-aware (sibling) margin.

sibling_ceiling.py showed that a species' genus siblings and its nearest class templates
overlap a lot, but it compared a sibling set of any size with a fixed nearest-5 set, which
is not fair: a bigger set catches more errors by construction, and every competitor a
margin is applied to has a cost. Here the two competitor sets have the same size.

For a true species y with s siblings (the other species of its genus) there are two sets of
s competitors:
  siblings         the s other species of y's genus
  nearest(s)       the s species whose templates have the largest cosine with y's template
For each validation error (the model predicted the wrong species w for an image of y) we ask
whether w is in each set. Reported per training-image band of y and per genus size:
  coverage         share of errors whose wrong species is in the set
  overlap          share of the s siblings that are also among the s nearest templates
  only siblings / only nearest / both / neither     the four ways an error can fall
  nearest(k)       coverage of a fixed-size nearest set, k = 1, 3, 5, 10, 20, 50
Intervals are 95% bootstrap intervals over species (each species brings all its errors).

Species without a sibling are left out: the sibling set is empty. No training and no images;
CPU only. Reads the same cached PHASE3_GEOM features and templates as sibling_ceiling.py, and
repeats its numbers (read from its summary.json) in the final RESULT_JSON line so one log
holds everything a report needs.

Usage (from finetuning/SWIN):
  python loss_ablation/sibling_matched_control.py
  python loss_ablation/sibling_matched_control.py --models normface dbm
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sibling_ceiling import BANDS, band_name, band_of, normalise, taxonomy_ids  # noqa: E402
from species_sphere import (EXTRA_MODELS, GEOM, TRAIN_JSON, load_model,  # noqa: E402
                            load_rows, taxonomy)

OUT_ROOT = f"/projectnb/herbdl/workspaces/{os.environ.get('USER', 'unknown')}/herbdl/finetuning/output/SWIN"
DEFAULT_OUT = f"{OUT_ROOT}/sibling_matched_control"
CEILING_SUMMARY = f"{OUT_ROOT}/sibling_ceiling/summary.json"
SIZE_GROUPS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]  # number of siblings
FIXED_K = (1, 3, 5, 10, 20, 50)
BOOT = 1000


# ---------------------------------------------------------------- scoring

def predict(feats, Wn, chunk=4096):
    """Argmax-cosine prediction for every image."""
    out = np.zeros(len(feats), dtype=np.int64)
    for s in range(0, len(feats), chunk):
        out[s:s + chunk] = (normalise(feats[s:s + chunk]) @ Wn.T).argmax(1)
    return out


def species_counts(Wn, labels, pred, gid, chunk=256):
    """Per-species counts of errors in each competitor set, and the set overlap.

    Returns dict of (C,) arrays: n_err, in_sib, in_near (nearest s), both, in_near_k for each
    fixed k, overlap (share of siblings among the s nearest), s (number of siblings).
    """
    C = len(Wn)
    s = np.bincount(gid, minlength=gid.max() + 1)[gid] - 1
    err = pred != labels
    order = np.argsort(labels[err], kind="stable")
    e_true, e_wrong = labels[err][order], pred[err][order]
    start = np.searchsorted(e_true, np.arange(C + 1))  # errors of species c: start[c]:start[c+1]
    out = {k: np.zeros(C, dtype=np.int64) for k in ("n_err", "in_sib", "in_near", "both")}
    out.update({f"near_{k}": np.zeros(C, dtype=np.int64) for k in FIXED_K})
    out["overlap"] = np.full(C, np.nan)
    for c0 in range(0, C, chunk):
        ys = np.arange(c0, min(c0 + chunk, C))
        T = Wn[ys] @ Wn.T
        T[np.arange(len(ys)), ys] = -np.inf  # a species is not its own competitor
        for i, y in enumerate(ys):
            if s[y] >= 1:
                near = np.argpartition(-T[i], s[y] - 1)[:s[y]]
                out["overlap"][y] = float((gid[near] == gid[y]).mean())
            lo, hi = start[y], start[y + 1]
            if hi == lo or s[y] < 1:
                continue
            w = e_wrong[lo:hi]
            rank = 1 + (T[i][None, :] > T[i, w][:, None]).sum(1)  # 1 = most similar other template
            sib = gid[w] == gid[y]
            near = rank <= s[y]
            out["n_err"][y] = hi - lo
            out["in_sib"][y] = sib.sum()
            out["in_near"][y] = near.sum()
            out["both"][y] = (sib & near).sum()
            for k in FIXED_K:
                out[f"near_{k}"][y] = (rank <= k).sum()
    out["s"] = s
    return out


# ---------------------------------------------------------------- aggregation

def boot_ci(idx, num_cols, den_col, rng):
    """95% intervals of sum(num)/sum(den) for each numerator over species resampled with replacement."""
    draw = rng.integers(0, len(idx), size=(BOOT, len(idx)))
    den = den_col[idx][draw].sum(1)
    res = []
    for num in num_cols:
        v = num[idx][draw].sum(1) / np.maximum(den, 1)
        res.append([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    return res


def summarise_group(cnt, mask, rng):
    idx = np.flatnonzero(mask)
    n = cnt["n_err"]
    tot = int(n[idx].sum())
    if len(idx) == 0 or tot == 0:
        return None
    a, b, both = cnt["in_sib"], cnt["in_near"], cnt["both"]
    r = {"species": int(len(idx)), "errors": tot,
         "mean_set_size_species": float(cnt["s"][idx].mean()),
         "mean_set_size_error_weighted": float((cnt["s"][idx] * n[idx]).sum() / tot),
         "coverage_siblings": float(a[idx].sum() / tot),
         "coverage_nearest_matched": float(b[idx].sum() / tot),
         "only_siblings": float((a[idx] - both[idx]).sum() / tot),
         "only_nearest": float((b[idx] - both[idx]).sum() / tot),
         "both": float(both[idx].sum() / tot),
         "neither": float((n[idx] - a[idx] - b[idx] + both[idx]).sum() / tot),
         "overlap_species_mean": float(np.nanmean(cnt["overlap"][idx]))}
    for k in FIXED_K:
        r[f"coverage_nearest_{k}"] = float(cnt[f"near_{k}"][idx].sum() / tot)
    ci = boot_ci(idx, [a, b, a - b], n, rng)
    r["ci_coverage_siblings"], r["ci_coverage_nearest_matched"], r["ci_diff_siblings_minus_nearest"] = ci
    r["diff_siblings_minus_nearest"] = r["coverage_siblings"] - r["coverage_nearest_matched"]
    return r


def summarise(cnt, n_train, present):
    rng = np.random.default_rng(0)
    base = present & (cnt["s"] >= 1)
    sp_band = band_of(n_train)
    groups = {"all": base}
    for b in range(len(BANDS)):
        groups[f"band {band_name(b)}"] = base & (sp_band == b)
    for lo, hi in SIZE_GROUPS:
        label = f"siblings {lo}+" if hi >= 10 ** 9 else f"siblings {lo}-{hi}"
        groups[label] = base & (cnt["s"] >= lo) & (cnt["s"] <= hi)
    return {g: summarise_group(cnt, m, rng) for g, m in groups.items()}


# ---------------------------------------------------------------- printing

def pct(x):
    return "  -  " if x != x else f"{100 * x:5.1f}"


def ci_pct(ci):
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def print_tables(model, res):
    print(f"\n=== {model}")
    print("| group | species | errors | siblings per species | siblings cover | nearest(s) covers | difference [95% CI] | only siblings | only nearest | both | neither | overlap of the two sets |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g, r in res.items():
        if r is None:
            continue
        print(f"| {g} | {r['species']} | {r['errors']} | {r['mean_set_size_species']:.1f} | {pct(r['coverage_siblings'])} | "
              f"{pct(r['coverage_nearest_matched'])} | {100 * r['diff_siblings_minus_nearest']:+.1f} {ci_pct(r['ci_diff_siblings_minus_nearest'])} | "
              f"{pct(r['only_siblings'])} | {pct(r['only_nearest'])} | {pct(r['both'])} | {pct(r['neither'])} | {pct(r['overlap_species_mean'])} |")
    print("\nCoverage of a fixed-size nearest set (% of errors)")
    print("| group | " + " | ".join(f"nearest {k}" for k in FIXED_K) + " |")
    print("|---|" + "---|" * len(FIXED_K))
    for g, r in res.items():
        if r is not None and (g == "all" or g.startswith("band")):
            print(f"| {g} | " + " | ".join(pct(r[f"coverage_nearest_{k}"]) for k in FIXED_K) + " |")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["normface", "dbm", "kappaface", "arcface"],
                    help="names from phase3_geometry.MODELS or species_sphere.EXTRA_MODELS")
    ap.add_argument("--geom", default=GEOM, help="folder with <model>/features.npy and predictions.npz")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ceiling-summary", default=CEILING_SUMMARY,
                    help="summary.json of sibling_ceiling.py, repeated in RESULT_JSON if it exists")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from phase3_geometry import MODELS, N
    registry = {name: (name, run_dir, kind) for name, (run_dir, kind) in MODELS.items()}
    registry.update(EXTRA_MODELS)
    for m in args.models:
        if m not in registry:
            sys.exit(f"unknown model {m!r}; known: {sorted(registry)}")
    _, genus, family, n_train = taxonomy(load_rows(TRAIN_JSON), N)
    gid, _ = taxonomy_ids(genus, family)
    os.makedirs(args.out, exist_ok=True)
    summary, ref_labels = {}, None
    for m in args.models:
        feats, labels, stored, W = load_model(args.geom, *registry[m])
        if W.shape[1] != 1:
            sys.exit(f"{m}: {W.shape[1]} centres per species; this test assumes one")
        if ref_labels is not None and not np.array_equal(labels, ref_labels):
            sys.exit(f"{m}: validation labels differ from {args.models[0]}'s")
        ref_labels = labels
        Wn = normalise(W[:, 0, :])
        pred = predict(feats, Wn)
        print(f"{m}: {len(labels)} validation images; prediction agrees with the stored one on "
              f"{(pred == stored).mean():.5f}; {int((pred != labels).sum())} errors")
        cnt = species_counts(Wn, labels, pred, gid)
        present = np.bincount(labels, minlength=N) > 0
        summary[m] = summarise(cnt, n_train, present)
        print_tables(m, summary[m])
        np.savez_compressed(f"{args.out}/{m}.npz", **cnt)
        sys.stdout.flush()
    ceiling = None
    if os.path.exists(args.ceiling_summary):
        with open(args.ceiling_summary) as f:
            ceiling = json.load(f)
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump({"matched": summary, "ceiling": ceiling}, f)
    print("RESULT_JSON " + json.dumps({"matched": summary, "ceiling": ceiling}))


if __name__ == "__main__":
    main()
