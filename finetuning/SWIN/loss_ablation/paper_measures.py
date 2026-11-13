"""The DBM paper's compactness and separability measures for every model the paper reports.

The paper's representation analysis uses one metric set, the one the DBM paper (Son et al.)
uses for its own claims:
  compactness   mean angle, in degrees, between an image's L2-normalised feature and the
                classifier weight of its true species (the paper's class centre);
  separability  Fisher's criterion for each pair of species along the optimal two-class
                projection, J_ij = (mu_i - mu_j)^T Sw^-1 (mu_i - mu_j), with a within-species
                covariance Sw pooled over all species, averaged per species
                (S_i = mean_{j != i} J_ij) and then over species.
Both use paper_separability.py's helpers (same ridge, same Fisher code), so numbers are
comparable with PHASE3_GEOM/analysis/paper_separability.json. Unlike that script, the angle is
computed here from the model's own classifier weights instead of a boundary.npz, so it also
runs for the two margin controls and for training-image features.

Jobs (split, model):
  val    cosine_m0, arcface_m01      the two margin controls (not in paper_separability.json)
  val    dbm, arcface                check: must reproduce 59.43 / 47.08 deg, 1840 / 2698
  train  dbm, arcface                671,817 training images, PHASE3_GEOM_TRAIN features

Output: lines prefixed __PAPERMEAS__; MARGIN_DIAG/paper_measures.json.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_separability import BANDS, LABEL, N, TRAIN_JSON, angle_stats, band_means, fisher_scores  # noqa: E402
from phase3_geometry import GEOM, MODELS as P3_MODELS, OUT, class_directions  # noqa: E402

TRAIN = f"{OUT}/PHASE3_GEOM_TRAIN"
RESULT = f"{OUT}/MARGIN_DIAG/paper_measures.json"
PREFIX = "__PAPERMEAS__"
RUNS = dict(P3_MODELS)
RUNS["cosine_m0"] = ("SWIN_BASE_COSINE_M0_384", "registry")
RUNS["arcface_m01"] = ("SWIN_BASE_ARCFACE_M01_384", "registry")
JOBS = [("val", "cosine_m0"), ("val", "arcface_m01"), ("val", "dbm"), ("val", "arcface"),
        ("train", "dbm"), ("train", "arcface")]


def measure(split, name, train_count, device):
    root = GEOM if split == "val" else TRAIN
    fpath, ppath = f"{root}/{name}/features.npy", f"{root}/{name}/predictions.npz"
    for p in (fpath, ppath):
        if not os.path.isfile(p):
            print(f"{PREFIX} MISSING {p}", flush=True)
            return None
    y = np.load(ppath)["labels"].astype(np.int64)
    feat = np.load(fpath, mmap_mode="r")
    assert len(feat) == len(y), (split, name, feat.shape, y.shape)
    n = len(y)

    x = F.normalize(torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device), dim=1)
    y_t = torch.as_tensor(y, device=device)

    # Compactness: angle to the true species' classifier weight. With sub-centres (k > 1) the
    # closest sub-centre counts; every model measured here has k = 1.
    w = F.normalize(class_directions(*RUNS[name]).to(device), dim=-1)  # (N, k, C)
    cos_true = torch.empty(n, device=device)
    for s in range(0, n, 65536):
        xs, ys = x[s:s + 65536], y_t[s:s + 65536]
        cos_true[s:s + 65536] = torch.einsum("bc,bkc->bk", xs, w[ys]).max(1).values
    cos_true = cos_true.cpu().numpy()
    ones = np.ones(n, bool)
    res = {"n_images": n,
           "angle_to_weight": {"overall": angle_stats(cos_true, ones, n),
                               "bands": {f"{lo}-{hi}": angle_stats(cos_true, (train_count[y] >= lo) & (train_count[y] <= hi), n)
                                         for lo, hi in BANDS}}}

    # Separability: pairwise Fisher criterion with a pooled within-species covariance
    # (same ridge as paper_separability.py: 1e-3 of the mean variance).
    counts = torch.bincount(y_t, minlength=N).float()
    mu = torch.zeros(N, x.size(1), device=device).index_add_(0, y_t, x) / counts.clamp(min=1)[:, None]
    Sw = torch.zeros(x.size(1), x.size(1), device=device)
    for s in range(0, n, 65536):
        c = x[s:s + 65536] - mu[y_t[s:s + 65536]]
        Sw += c.T @ c
    Sw /= n
    Sw += 1e-3 * torch.trace(Sw) / Sw.size(0) * torch.eye(Sw.size(0), device=device)
    # All N species, as in paper_separability.py, so the DBM/ArcFace check reproduces it exactly.
    S = fisher_scores(mu, Sw, device)
    res["fisher"] = {"overall": {"mean": float(S.mean()), "median": float(np.median(S))},
                     "bands": band_means(S, train_count)}
    del x, mu, Sw
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_count = np.bincount([json.loads(l)[LABEL] for l in open(TRAIN_JSON) if l.strip()], minlength=N)
    print(f"{PREFIX} device={device}", flush=True)
    out = {}
    for split, name in JOBS:
        r = measure(split, name, train_count, device)
        if r is None:
            continue
        out.setdefault(split, {})[name] = r
        a, f = r["angle_to_weight"], r["fisher"]
        print(f"{PREFIX} {split} {name}: n={r['n_images']:,} angle_mean={a['overall']['mean_deg']:.2f}deg "
              f"fisher_mean={f['overall']['mean']:.1f} fisher_median={f['overall']['median']:.1f} | bands: "
              + " ".join(f"{b}:{a['bands'][b]['mean_deg']:.1f}deg/F{f['bands'][b]['mean']:.0f}" for b in a["bands"]),
              flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(out, open(RESULT, "w"), indent=1)
    print(f"{PREFIX} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
