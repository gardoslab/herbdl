"""The DBM paper's own separability metrics, computed on the 384px models.

Our geometry analysis used within-species cosine to the feature centroid, the
centroid-to-centroid separation gap, NC1 and a trace ratio. The DBM paper
(2412.15477, "Intra-class Compactness and Inter-class Separability") used two
different quantities, and the comparison should use theirs:

Intra-class compactness
    the distribution of angular distances between each sample feature and its
    positive class center. In the paper the class center is the classifier's
    weight vector for the correct class (Eq. 2: theta_i is the angle between
    W_i and f(x)), not the mean feature. We report both:
      intra_classifier  mean/median/std of arccos(cos(f, W_y)), degrees
      intra_centroid    the same angle to the mean feature of the class

Inter-class separability
    Fisher's criterion for each class pair (Eq. 12):
        J(W_ij) = (mu_i - mu_j)^2 / (sigma_i^2 + sigma_j^2)
    along the optimal LDA projection W_ij, then averaged per class (Eq. 13):
        S_i = mean_{j != i} J(W_ij)
    With a pooled within-class covariance Sw, the optimal two-class projection
    gives J(W_ij) = (mu_i - mu_j)^T Sw^-1 (mu_i - mu_j), the squared
    Mahalanobis distance between the two class means. We use the global pooled
    Sw (the per-class covariance is singular at 2-73 images in 1024 dims) and
    report S_i averaged within rarity bands.

Two feature variants are computed: L2-normalised features (the space the cosine
classifier works in) and raw pooled features. The normalised variant is the
primary one; the raw variant checks that the ordering does not depend on it.

Inputs (SCC): PHASE3_GEOM/<model>/features.npy and predictions.npz, plus
PHASE3_GEOM/<model>/boundary.npz for the per-image cos_true written by
boundary_margins.py.
Output: PHASE3_GEOM/analysis/paper_separability.json

Usage: python loss_ablation/paper_separability.py [--models dbm arcface]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
GEOM = f"{OUT}/PHASE3_GEOM"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
N = 15501
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]

# Same model set as boundary_margins.py: no HACE (tree decision), no X2 (collapsed).
MODELS = ["dbm", "dbm_seed43", "kappaface", "kappaface_seed43", "arcface",
          "arcface_seed43", "subcenter_arcface", "arcface_tara", "arcface_ecc"]


def angle_stats(cos, mask, n):
    """Mean/median/std of arccos(cos), in degrees, over one slice."""
    th = np.degrees(np.arccos(np.clip(np.asarray(cos, dtype=np.float64), -1.0, 1.0)))
    sl = th[mask]
    return {"n_images": int(mask.sum()),
            "mean_deg": float(sl.mean()),
            "median_deg": float(np.median(sl)),
            "std_deg": float(sl.std())}


def fisher_scores(mu, Sw, device):
    """Pairwise Fisher criterion J_ij and per-class means S_i (Eq. 12-13).

    mu  (N, C) tensor of class means
    Sw  (C, C) pooled within-class covariance
    """
    A = torch.linalg.inv(Sw)
    B = mu @ A                       # (N, C), rows are A mu_i
    diag = (mu * B).sum(1)           # mu_i^T A mu_i
    # J_ij = diag_i + diag_j - 2 mu_i^T A mu_j. The cross term is B @ mu.T,
    # NOT B @ B.T: the latter would be mu_i^T A^2 mu_j, a different quadratic form.
    D = diag[:, None] + diag[None, :] - 2.0 * (B @ mu.T)
    S = (D.sum(1) - D.diagonal()) / (len(mu) - 1)
    return S.detach().cpu().numpy()


def band_means(values, train_count):
    out = {}
    for lo, hi in BANDS:
        m = (train_count >= lo) & (train_count <= hi)
        if m.any():
            out[f"{lo}-{hi}"] = {"n_species": int(m.sum()),
                                 "mean": float(values[m].mean()),
                                 "median": float(np.median(values[m]))}
    return out


def run_model(name, train_count, device):
    fpath = f"{GEOM}/{name}/features.npy"
    ppath = f"{GEOM}/{name}/predictions.npz"
    bpath = f"{GEOM}/{name}/boundary.npz"
    for p in (fpath, ppath, bpath):
        if not os.path.isfile(p):
            print(f"__PAPERSEP__ MISSING {p}", flush=True)
            return None

    z = np.load(ppath)
    y = z["labels"].astype(np.int64)
    n = len(y)
    feat = np.load(fpath, mmap_mode="r")
    assert len(feat) == n

    x_raw = torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device)
    x = F.normalize(x_raw, dim=1)
    y_t = torch.as_tensor(y, device=device)
    counts = torch.bincount(y_t, minlength=N).float()

    def class_means(vectors):
        mu = torch.zeros(N, vectors.size(1), device=device)
        mu.index_add_(0, y_t, vectors)
        return mu / counts.clamp(min=1)[:, None]

    mu_norm = class_means(x)
    mu_raw = class_means(x_raw)

    def within_cov(vectors, mu):
        centered = vectors - mu[y_t]
        Sw = centered.T @ centered / n
        # Shrinkage-style ridge: the spectrum is wide and the inverse is noisy at
        # 2-73 images per class in 1024 dims. 1e-3 of the mean variance is small
        # enough to leave well-sampled directions alone and large enough to stop
        # the inverse from amplifying the tail.
        ridge = 1e-3 * torch.trace(Sw) / Sw.size(0)
        return Sw + ridge * torch.eye(Sw.size(0), device=device)

    result = {"n_images": n}

    # Intra: angle to the classifier's own class row (the paper's class center).
    b = np.load(bpath)
    cos_true = b["cos_true"].astype(np.float64)
    result["intra_classifier"] = {
        "overall": angle_stats(cos_true, np.ones(n, bool), n),
        "bands": {f"{lo}-{hi}": angle_stats(cos_true,
                                            (train_count[y] >= lo) & (train_count[y] <= hi), n)
                  for lo, hi in BANDS},
    }

    # Intra: angle to the mean feature of the class, for comparison with our old metric.
    cos_centroid = (x * F.normalize(mu_norm, dim=1)[y_t]).sum(1)
    cos_centroid = cos_centroid.detach().cpu().numpy()
    result["intra_centroid"] = {
        "overall": angle_stats(cos_centroid, np.ones(n, bool), n),
        "bands": {f"{lo}-{hi}": angle_stats(cos_centroid,
                                            (train_count[y] >= lo) & (train_count[y] <= hi), n)
                  for lo, hi in BANDS},
    }

    # Inter: Fisher criterion, normalised features (primary).
    for variant, mu in (("norm", mu_norm), ("raw", mu_raw)):
        vectors = x if variant == "norm" else x_raw
        Sw = within_cov(vectors, mu)
        S = fisher_scores(mu, Sw, device)
        # Global Fisher ratio tr(Sb)/tr(Sw), with the count-weighted global mean.
        mu_bar = (mu * counts[:, None]).sum(0) / counts.sum()
        centered_mu = mu - mu_bar
        Sb = centered_mu.T @ (centered_mu * counts[:, None])
        global_ratio = float(torch.trace(Sb).item() / torch.trace(Sw).item()) \
            if variant == "norm" else None
        entry = {
            "overall": {"mean": float(S.mean()), "median": float(np.median(S)),
                        "min": float(S.min()), "max": float(S.max())},
            "bands": band_means(S, train_count),
            "global_tr_Sb_over_tr_Sw": global_ratio,
        }
        if variant == "norm":
            ev = torch.linalg.eigvalsh(Sw)
            entry["sw_condition_number"] = float((ev[-1] / ev[0]).item())
        result[f"inter_fisher_{variant}"] = entry

    del x_raw, x, mu_norm, mu_raw, feat
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"__PAPERSEP__ device={device}", flush=True)

    train_count = np.bincount(
        [json.loads(line)[LABEL] for line in open(TRAIN_JSON) if line.strip()],
        minlength=N)
    out = {}
    for name in args.models:
        r = run_model(name, train_count, device)
        if r is None:
            continue
        out[name] = r
        ic = r["intra_classifier"]["overall"]
        f = r["inter_fisher_norm"]["overall"]
        print(f"__PAPERSEP__ {name}: intra_classifier_mean={ic['mean_deg']:.2f}deg "
              f"intra_centroid_mean={r['intra_centroid']['overall']['mean_deg']:.2f}deg "
              f"fisher_mean={f['mean']:.3f} fisher_median={f['median']:.3f} "
              f"min={f['min']:.3f} cond={r['inter_fisher_norm']['sw_condition_number']:.1e} "
              f"| bands: " + " ".join(
                  f"{b}:{r['intra_classifier']['bands'][b]['mean_deg']:.1f}deg/"
                  f"F{r['inter_fisher_norm']['bands'][b]['mean']:.2f}"
                  for b in r["intra_classifier"]["bands"]), flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(f"{GEOM}/analysis", exist_ok=True)
    path = f"{GEOM}/analysis/paper_separability.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"__PAPERSEP__ wrote {path} ({len(out)} models)", flush=True)


if __name__ == "__main__":
    main()
