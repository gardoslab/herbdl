"""Separation gap by number of training images per species.

The earlier geometry table averages the separation gap over all 15,334 species that have
validation images, which counts every species once. Species with many training images are
the majority (51-73 training images: about half of all species), so they may dominate that
average. Here the same quantity is computed per species and averaged within bands of
training images per species.

Definitions (identical to phase3_geometry.py / repgeom_geometry.py, on the VALIDATION features
the eval-only runs saved to PHASE3_GEOM/<model>/features.npy):

  within-species cos     mean cosine between each validation image of a species and that
                         species' centroid (the mean validation feature of the species)
  nearest-species cos    cosine between that species' centroid and the closest OTHER species'
                         centroid
  separation gap         within-species cos - nearest-species cos   (higher = better separated)

Small-sample caveat, checked here: a species with ONE validation image has a centroid equal to
that image, so its within-species cosine is exactly 1 whatever the model. Rare species have few
validation images, so this inflates their "tightness" for reasons unrelated to the model. The
leave-one-out version scores each validation image against the centroid of the species' OTHER
validation images (species with >= 2 validation images only), which removes that effect.

Outputs (PHASE3_GEOM/analysis/): geometry_by_rarity.json (band tables) and
geometry_by_rarity_species.npz (per-species arrays). Usage (1 GPU): python loss_ablation/geometry_by_rarity.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repgeom_geometry as geometry  # noqa: E402

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
GEOM = f"{OUT}/PHASE3_GEOM"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
N = 15501
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
MODELS = ["dbm", "kappaface", "subcenter_arcface", "arcface_tara", "arcface", "arcface_ecc", "hace", "x2softmax"]
SEED_PAIRS = [("dbm_seed43", "dbm"), ("kappaface_seed43", "kappaface"), ("arcface_seed43", "arcface")]
N_BOOT = 1000


def loo_within_cos(feat, y, device):
    """Per species: mean cosine of each validation image to the centroid of the species' OTHER
    validation images. NaN for species with fewer than 2 validation images."""
    C = feat.shape[1]
    sums = torch.zeros(N, C, dtype=torch.float64, device=device)
    cnt = torch.zeros(N, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.long)
    for lo in range(0, len(feat), 4096):
        x = torch.as_tensor(feat[lo:lo + 4096]).to(device).double()
        yy = yt[lo:lo + 4096].to(device)
        sums.index_add_(0, yy, x)
        cnt.index_add_(0, yy, torch.ones(len(yy), dtype=torch.float64, device=device))
    cos_sum = torch.zeros(N, dtype=torch.float64, device=device)
    cos_n = torch.zeros(N, dtype=torch.float64, device=device)
    for lo in range(0, len(feat), 4096):
        x = torch.as_tensor(feat[lo:lo + 4096]).to(device).double()
        yy = yt[lo:lo + 4096].to(device)
        n_c = cnt[yy]
        ok = n_c >= 2
        others = (sums[yy] - x) / (n_c - 1).clamp(min=1)[:, None]
        c = torch.nn.functional.cosine_similarity(x, others, dim=1)
        cos_sum.index_add_(0, yy[ok], c[ok])
        cos_n.index_add_(0, yy[ok], torch.ones(int(ok.sum()), dtype=torch.float64, device=device))
    out = np.full(N, np.nan)
    has = cos_n.cpu().numpy() > 0
    out[has] = (cos_sum.cpu().numpy()[has] / cos_n.cpu().numpy()[has])
    return out


def band_stats(values, train_count, rng, mask=None):
    """Mean of `values` (per species) within each training-image band, with a species-bootstrap 95% interval."""
    out = {}
    for lo, hi in BANDS:
        m = (train_count >= lo) & (train_count <= hi) & np.isfinite(values)
        if mask is not None:
            m &= mask
        v = values[m]
        if len(v) == 0:
            continue
        boots = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(N_BOOT)]
        out[f"{lo}-{hi}"] = {"n_species": int(len(v)), "mean": float(v.mean()),
                             "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    with open(TRAIN_JSON) as f:
        train_count = np.bincount([json.loads(l)[LABEL] for l in f if l.strip()], minlength=N)
    res = {"bands": [f"{lo}-{hi}" for lo, hi in BANDS], "models": {}}
    arrays = {}
    for name in MODELS + [s for s, _ in SEED_PAIRS]:
        fpath, ppath = f"{GEOM}/{name}/features.npy", f"{GEOM}/{name}/predictions.npz"
        if not (os.path.isfile(fpath) and os.path.isfile(ppath)):
            print(f"__GBR__ MISSING {name}")
            continue
        feat = np.load(fpath).astype(np.float32)
        y = np.load(ppath)["labels"].astype(np.int64)
        mu, n_val, mu_g, present = geometry.class_centroids(feat, y, N, device)
        within, _ = geometry.intra_class_spread(feat, y, mu, N, device)
        nearest, _ = geometry.nearest_neighbour(mu, present, device, k=2)
        within = np.where(present, within, np.nan)
        nearest = nearest[:, 0]
        gap = within - nearest
        within_loo = loo_within_cos(feat, y, device)
        gap_loo = within_loo - nearest
        r = {"species_weighted_gap": float(np.nanmean(gap[present])), "n_species": int(present.sum()),
             "gap": band_stats(gap, train_count, rng), "within": band_stats(within, train_count, rng),
             "nearest": band_stats(nearest, train_count, rng),
             "gap_loo_2plus_val": band_stats(gap_loo, train_count, rng),
             "gap_5plus_val": band_stats(gap, train_count, rng, mask=n_val >= 5)}
        bm = [r["gap"][b]["mean"] for b in res["bands"]]
        r["band_equal_weighted_gap"] = float(np.mean(bm))
        res["models"][name] = r
        arrays[name] = {"gap": gap, "within": within, "nearest": nearest, "within_loo": within_loo}
        print(f"__GBR__ {name}: species-weighted gap {r['species_weighted_gap']:.4f}; by band "
              + ", ".join(f"{b} {r['gap'][b]['mean']:.4f}" for b in res["bands"]), flush=True)
        del feat, mu
        if device == "cuda":
            torch.cuda.empty_cache()
        if name == MODELS[0]:  # val counts and train counts per band are model independent
            vc = n_val
            res["species_per_band"] = {f"{lo}-{hi}": int(((train_count >= lo) & (train_count <= hi) & present).sum())
                                       for lo, hi in BANDS}
            res["mean_val_images_per_species"] = {f"{lo}-{hi}": float(vc[(train_count >= lo) & (train_count <= hi) & present].mean())
                                                  for lo, hi in BANDS}
            res["share_single_val_image"] = {f"{lo}-{hi}": float((vc[(train_count >= lo) & (train_count <= hi) & present] == 1).mean())
                                             for lo, hi in BANDS}
            res["share_of_species_average"] = {k: v / sum(res["species_per_band"].values()) for k, v in res["species_per_band"].items()}
            print(f"__GBR__ species per band {res['species_per_band']}; mean val images per species "
                  f"{ {k: round(v, 2) for k, v in res['mean_val_images_per_species'].items()} }; share with a single val image "
                  f"{ {k: round(v, 2) for k, v in res['share_single_val_image'].items()} }", flush=True)

    os.makedirs(f"{GEOM}/analysis", exist_ok=True)
    json.dump(res, open(f"{GEOM}/analysis/geometry_by_rarity.json", "w"), indent=1)
    np.savez_compressed(f"{GEOM}/analysis/geometry_by_rarity_species.npz",
                        **{f"{n}_{k}": v for n, d in arrays.items() for k, v in d.items()}, train_count=train_count)
    print(f"__GBR__ wrote {GEOM}/analysis/geometry_by_rarity.json")


if __name__ == "__main__":
    main()
