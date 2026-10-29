"""Separation gap by training-image band at an equal sample size (2 validation images per species).

geometry_by_rarity.py showed the band comparison is confounded by how many validation images a
species has (rare species: about 2.6; common: about 14). Here every species with >= 2 validation
images contributes exactly 2 randomly chosen validation images, so centroids, tightness and
neighbour distances are estimated from the same number of images in every band. Repeated for
N_DRAWS random draws (each draw: new image pairs); per-species values are averaged over draws.

  centroid            mean of the 2 images' features
  within-species cos  mean cosine of the 2 images to that centroid
  between-image cos   cosine between the 2 images themselves (no self-inclusion; reported as a check)
  nearest-species cos cosine between this centroid and the closest OTHER species' centroid
  separation gap      within-species cos - nearest-species cos

Species with a single validation image (25% of the 2-10 band) cannot be included.
Usage (1 GPU): python loss_ablation/geometry_fixed_n.py
"""

import json
import os

import numpy as np
import torch

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
GEOM = f"{OUT}/PHASE3_GEOM"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
N = 15501
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
MODELS = ["dbm", "kappaface", "subcenter_arcface", "arcface_tara", "arcface", "arcface_ecc", "hace", "x2softmax",
          "dbm_seed43", "kappaface_seed43", "arcface_seed43"]
N_DRAWS, N_BOOT = 10, 1000


def nearest_other(M, device):
    """Max cosine of each row of M to any other row (chunked on the GPU)."""
    Mn = torch.nn.functional.normalize(M, dim=1)
    best = torch.empty(len(Mn), device=device)
    for lo in range(0, len(Mn), 2048):
        s = Mn[lo:lo + 2048] @ Mn.T
        s[torch.arange(s.shape[0], device=device), torch.arange(lo, lo + s.shape[0], device=device)] = -2.0
        best[lo:lo + 2048] = s.max(dim=1).values
    return best.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    with open(TRAIN_JSON) as f:
        train_count = np.bincount([json.loads(l)[LABEL] for l in f if l.strip()], minlength=N)
    res = {"bands": [f"{lo}-{hi}" for lo, hi in BANDS], "n_draws": N_DRAWS, "models": {}}
    for name in MODELS:
        fpath, ppath = f"{GEOM}/{name}/features.npy", f"{GEOM}/{name}/predictions.npz"
        if not (os.path.isfile(fpath) and os.path.isfile(ppath)):
            print(f"__GFN__ MISSING {name}")
            continue
        feat = torch.as_tensor(np.load(fpath).astype(np.float32), device=device)
        y = np.load(ppath)["labels"].astype(np.int64)
        order = np.argsort(y, kind="stable")
        n_val = np.bincount(y, minlength=N)
        starts = np.concatenate([[0], np.cumsum(n_val)])[:-1]
        elig = np.where(n_val >= 2)[0]
        gap, within, between, nearest = [np.zeros((N_DRAWS, len(elig))) for _ in range(4)]
        for d in range(N_DRAWS):
            n = n_val[elig]
            j1 = np.floor(rng.random(len(elig)) * n).astype(np.int64)
            j2 = np.floor(rng.random(len(elig)) * (n - 1)).astype(np.int64)
            j2 += (j2 >= j1)
            i1 = torch.as_tensor(order[starts[elig] + j1], device=device)
            i2 = torch.as_tensor(order[starts[elig] + j2], device=device)
            x1, x2 = feat[i1], feat[i2]
            m = (x1 + x2) / 2
            cs = torch.nn.functional.cosine_similarity
            w = ((cs(x1, m, dim=1) + cs(x2, m, dim=1)) / 2).cpu().numpy()
            within[d], between[d] = w, cs(x1, x2, dim=1).cpu().numpy()
            nearest[d] = nearest_other(m, device)
            gap[d] = within[d] - nearest[d]
        tc = train_count[elig]
        r = {"n_species_used": int(len(elig)), "species_weighted_gap": float(gap.mean())}
        for key, arr in (("gap", gap), ("within", within), ("between_image_cos", between), ("nearest", nearest)):
            per_species = arr.mean(axis=0)
            band = {}
            for lo, hi in BANDS:
                mk = (tc >= lo) & (tc <= hi)
                v = per_species[mk]
                boots = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(N_BOOT)]
                band[f"{lo}-{hi}"] = {"n_species": int(mk.sum()), "mean": float(v.mean()),
                                      "draw_sd": float(arr[:, mk].mean(axis=1).std()),
                                      "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            r[key] = band
        res["models"][name] = r
        print(f"__GFN__ {name}: species used {r['n_species_used']}; gap by band "
              + ", ".join(f"{b} {r['gap'][b]['mean']:.4f}" for b in res["bands"])
              + " | within " + ", ".join(f"{r['within'][b]['mean']:.3f}" for b in res["bands"])
              + " | nearest " + ", ".join(f"{r['nearest'][b]['mean']:.3f}" for b in res["bands"]), flush=True)
        del feat
        if device == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(f"{GEOM}/analysis", exist_ok=True)
    json.dump(res, open(f"{GEOM}/analysis/geometry_fixed_n.json", "w"), indent=1)
    print(f"__GFN__ wrote {GEOM}/analysis/geometry_fixed_n.json")


if __name__ == "__main__":
    main()
