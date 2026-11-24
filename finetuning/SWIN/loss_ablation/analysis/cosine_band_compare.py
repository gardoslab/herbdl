"""Where does DBM beat the zero-margin cosine head? Per-band comparison, base data and ladder.

Settings: base 15.5k data and the ladder rungs n15k, n32k, n64k.

Question: on the 15.5k base data the cosine head (s=30, m=0) is within 0.15 macro-F1 points of
DBM, but on the 32k ladder rung it is 1.7 points behind. If DBM's only real difference from the
cosine head is its margin for the rarest species, its lead over the cosine head should sit in the
rare bands, and the larger aggregate gap at 32k should come from the larger share of rare species.

For every setting (base 15.5k data; each ladder rung with a scored cosine model) and every loss we
compute per-species F1 on the full validation set, average it within bands of training images per
species (species with at least one validation image), and report paired leads with 95% bootstrap
intervals over species. Overall macro-F1 is printed as a check against the known scores. The
"rare share" is the fraction of the overall DBM-minus-cosine gap (species-weighted mean of the
per-band leads) contributed by species with at most 10 training images.

Inputs (SCC):
  base:   PHASE3_GEOM/<model>/predictions.npz (labels, predictions); train_2022.json counts
  ladder: LADDER_EVAL/<rung>_<loss>/predictions.npz; data/ladder/<rung>_train.json counts
Output: <OUT>/MARGIN_DIAG/cosine_band_compare.json; lines prefixed __COSBAND__.
"""

import json
import os

import numpy as np

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
BASE_TRAIN = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
RESULT = f"{OUT}/MARGIN_DIAG/cosine_band_compare.json"
PREFIX = "__COSBAND__"
N_BOOT, SEED = 1000, 0

BASE_BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 73)]
LADDER_BANDS = [(1, 1), (2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 100), (101, 10 ** 6)]
LOSSES = ["cosine_m0", "dbm", "kappaface", "arcface"]
PAIRS = [("dbm", "cosine_m0"), ("kappaface", "cosine_m0"), ("cosine_m0", "arcface")]
SETTINGS = {  # name -> (prediction file per loss, training manifest, bands)
    "base15k": ({l: f"{OUT}/PHASE3_GEOM/{l}/predictions.npz" for l in LOSSES}, BASE_TRAIN, BASE_BANDS),
    "n15k": ({l: f"{OUT}/LADDER_EVAL/n15k_{l}/predictions.npz" for l in LOSSES}, f"{LAD}/n15k_train.json", LADDER_BANDS),
    "n32k": ({l: f"{OUT}/LADDER_EVAL/n32k_{l}/predictions.npz" for l in LOSSES}, f"{LAD}/n32k_train.json", LADDER_BANDS),
    "n64k": ({l: f"{OUT}/LADDER_EVAL/n64k_{l}/predictions.npz" for l in LOSSES}, f"{LAD}/n64k_train.json", LADDER_BANDS),
}


def band_name(lo, hi):
    return f"{lo}+" if hi >= 10 ** 6 else (str(lo) if lo == hi else f"{lo}-{hi}")


def per_species_f1(y, p, n):
    """F1 of every class; classes absent from both labels and predictions are NaN."""
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    n_true = np.bincount(y, minlength=n).astype(float)
    n_pred = np.bincount(p, minlength=n).astype(float)
    denom = n_true + n_pred
    f1 = np.full(n, np.nan)
    ok = denom > 0
    f1[ok] = 2 * tp[ok] / denom[ok]
    return f1, n_true


def boot_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    b = d[idx].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def run_setting(name, files, manifest, bands, rng):
    missing = [l for l, f in files.items() if not os.path.isfile(f)]
    if missing:
        print(f"{PREFIX} SKIP {name}: missing {','.join(missing)}", flush=True)
        return None
    with open(manifest) as f:
        enc = np.array([json.loads(l)[LABEL] for l in f if l.strip()])
    preds, y = {}, None
    for loss, path in files.items():
        z = np.load(path)
        yy = z["labels"].astype(np.int64)
        if y is None:
            y = yy
        assert np.array_equal(y, yy), f"{name}: labels differ for {loss}"
        preds[loss] = z["predictions"].astype(np.int64)
    n = int(max(enc.max(), y.max(), *(p.max() for p in preds.values()))) + 1
    count = np.bincount(enc, minlength=n)
    f1 = {}
    for loss, p in preds.items():
        f1[loss], n_true = per_species_f1(y, p, n)
    in_val = n_true > 0  # species with at least one validation image
    res = {"n_val_images": int(len(y)), "macro_f1": {}, "bands": {}, "overall_leads": {}}
    for loss in LOSSES:
        res["macro_f1"][loss] = float(np.nanmean(f1[loss]))  # over species in labels or predictions
    print(f"{PREFIX} {name}: {len(y):,} val images | macro-F1 " +
          " ".join(f"{l}={res['macro_f1'][l]:.4f}" for l in LOSSES), flush=True)
    n_in = int(in_val.sum())
    for a, b in PAIRS:
        d = (f1[a] - f1[b])[in_val]
        res["overall_leads"][f"{a}-{b}"] = {"lead": float(100 * d.mean()), "ci": [100 * v for v in boot_ci(d, rng)]}
    rare_part = {f"{a}-{b}": 0.0 for a, b in PAIRS}
    for lo, hi in bands:
        m = in_val & (count >= lo) & (count <= hi)
        if not m.any():
            continue
        bn = band_name(lo, hi)
        entry = {"n_species": int(m.sum()), "share_of_species": float(m.sum() / n_in),
                 "f1": {l: float(np.nanmean(f1[l][m])) for l in LOSSES}, "leads": {}}
        parts = []
        for a, b in PAIRS:
            d = (f1[a] - f1[b])[m]
            lead = float(100 * d.mean())
            ci = [100 * v for v in boot_ci(d, rng)]
            entry["leads"][f"{a}-{b}"] = {"lead": lead, "ci": ci}
            parts.append(f"{a}-{b} {lead:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
            if hi <= 10:
                rare_part[f"{a}-{b}"] += lead * m.sum() / n_in
        res["bands"][bn] = entry
        print(f"{PREFIX} {name} band {bn:>6} (n={m.sum():>6,}, {100*m.sum()/n_in:4.1f}% of species): "
              + " ".join(f"{l}={entry['f1'][l]:.4f}" for l in LOSSES) + " | " + " | ".join(parts), flush=True)
    for key, v in res["overall_leads"].items():
        share = rare_part[key] / v["lead"] if abs(v["lead"]) > 1e-9 else float("nan")
        v["from_species_with_at_most_10_images"] = float(rare_part[key])
        print(f"{PREFIX} {name} overall {key}: {v['lead']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] points "
              f"(species-weighted); from species with <=10 images: {rare_part[key]:+.2f} ({100*share:.0f}%)", flush=True)
    return res


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    for name, (files, manifest, bands) in SETTINGS.items():
        r = run_setting(name, files, manifest, bands, rng)
        if r is not None:
            out[name] = r
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{PREFIX} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
