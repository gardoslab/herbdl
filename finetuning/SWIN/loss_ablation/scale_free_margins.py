"""Scale-free margins for the 15.5k base models: are the losses really that different?

boundary_margins.py reports the raw cosine margin cos_true - max_wrong cos.  That number
depends on how spread out the cosines are (a model whose cosines are all squeezed together
has small raw margins whatever it does).  This script reports the same margin in three
scale-free forms and splits the errors by how far down the true species is ranked:

  raw margin        m = cos_y - max_{k != y} cos_k
  standardised      z = m / std_k(cos_k), std over all species for that image (population std)
  softmax margin    p_y - max_{k != y} p_k, p = softmax(30 * cos)  (30 = the training scale s)
  error by rank     close call = true species ranked 2nd, mid = rank 3-5, far = rank > 5,
                    each as a % of ALL validation images (so close+mid+far = error rate).
                    The old raw split (m in [-0.02, 0) vs m < -0.02) is kept for reference.
  separation gap    per species with >= 2 val images: within-species cosine to the feature
                    centroid minus the cosine of its centroid to the nearest other species'
                    centroid; reported raw and divided by the std of all off-diagonal
                    centroid-centroid cosines (estimated from 2M random pairs, fixed seed).
  trained margin    the angular margin each loss actually trained with, per species,
                    summarised (species-weighted mean / median, degrees) per training-count
                    band: DBM m = K (n_min / n_y)^tau, K = 0.1, tau = 1, n_min from the train
                    counts (cross-checked against the head.m_class buffer in the checkpoint; this
                    is the class margin m_C -- DBM adds up to 1.5x of it on samples whose nearest
                    species is wrong, which is not included);
                    KappaFace = head.per_class_margin from the checkpoint (value after the last
                    epoch's recompute); ArcFace family = constant 0.3 rad (the warm-up to 0.3
                    over 5 epochs is over by the end); cosine_m0 = 0; arcface_m01 = 0.1.

Inputs (SCC): PHASE3_GEOM/<model>/{features.npy, predictions.npz}; classifier rows from the
model's own model.safetensors, located exactly as phase3_geometry.class_directions does
(head.weight for registry heads, arcface.weight for the native ArcFace and sub-center runs).
cosine_m0 and arcface_m01 are registry-arcface runs: SWIN_BASE_COSINE_M0_384 and
SWIN_BASE_ARCFACE_M01_384.  HACE is skipped: its decision is tree-aggregated, there is no flat
cosine classifier (boundary_margins.py skips it for the same reason).  A model with any missing
file prints a MISSING line and is skipped.

Output: MARGIN_DIAG/scale_free_margins.json.  Prefix __SCALEFREE__.

Usage: python loss_ablation/scale_free_margins.py [--models dbm kappaface ...]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repgeom_geometry as geometry  # noqa: E402
from phase3_geometry import GEOM, MODELS as P3_MODELS, N, OUT, class_directions  # noqa: E402

TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
RESULT = f"{OUT}/MARGIN_DIAG/scale_free_margins.json"
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
CHUNK = 4096
SCALE = 30.0  # training logit scale s
RAW_SPLIT = 0.02
N_PAIRS = 2_000_000
PREFIX = "__SCALEFREE__"

DEFAULT_MODELS = ["dbm", "kappaface", "arcface", "arcface_ecc", "subcenter_arcface",
                  "arcface_tara", "hace", "cosine_m0", "arcface_m01"]

# name -> (checkpoint dir, head kind as in phase3_geometry.class_directions)
RUNS = {k: v for k, v in P3_MODELS.items()}
RUNS["cosine_m0"] = ("SWIN_BASE_COSINE_M0_384", "registry")
RUNS["arcface_m01"] = ("SWIN_BASE_ARCFACE_M01_384", "registry")
# name -> how the trained margin is obtained (radians)
MARGIN_SPEC = {
    "dbm": ("dbm",), "dbm_seed43": ("dbm",),
    "kappaface": ("kappaface",), "kappaface_seed43": ("kappaface",),
    "arcface": ("const", 0.3), "arcface_seed43": ("const", 0.3), "arcface_ecc": ("const", 0.3),
    "arcface_tara": ("const", 0.3), "subcenter_arcface": ("const", 0.3),
    "cosine_m0": ("const", 0.0), "arcface_m01": ("const", 0.1),
}
SKIP = {"hace": "tree-aggregated decision, no flat cosine classifier (boundary_margins.py skips it too)",
        "x2softmax": "collapsed run, not a trained-well comparison"}


@torch.no_grad()
def margin_scores(feat, labels, dirs, device, chunk=CHUNK, scale=SCALE):
    """Per-image scores against every species' direction, in chunks.

    feat (n, C) array, labels (n,) ints, dirs (N, k, C) classifier rows (k sub-centers;
    best sub-center per species, as the head does at inference).  Memory is bounded by
    chunk x (N*k) floats.
    """
    W = F.normalize(dirs.to(device).float(), dim=2)
    n_classes, k, dim = W.shape
    Wf = W.reshape(n_classes * k, dim)
    keys = ("cos_true", "cos_other", "margin", "z", "p_margin", "rank", "predicted")
    out = {key: np.empty(len(feat), dtype=np.float32) for key in keys}
    y_all = torch.as_tensor(labels, dtype=torch.long)
    for lo in range(0, len(feat), chunk):
        x = torch.as_tensor(np.ascontiguousarray(feat[lo:lo + chunk]), device=device).float()
        xn = F.normalize(x, dim=1)
        cos = (xn @ Wf.T).view(-1, n_classes, k).max(dim=2).values
        y = y_all[lo:lo + chunk].to(device)
        cos_true = cos.gather(1, y[:, None]).squeeze(1)
        std = cos.std(dim=1, unbiased=False)
        pred = cos.argmax(1)
        rank = (cos > cos_true[:, None]).sum(1) + 1
        lse = torch.logsumexp(scale * cos, dim=1)
        cos.scatter_(1, y[:, None], -2.0)  # mask the true species before the max
        other = cos.max(1).values
        m = cos_true - other
        p_true = torch.exp(scale * cos_true - lse)
        p_other = torch.exp(scale * other - lse)
        sl = slice(lo, lo + len(y))
        for key, val in (("cos_true", cos_true), ("cos_other", other), ("margin", m),
                         ("z", m / std.clamp(min=1e-8)), ("p_margin", p_true - p_other),
                         ("rank", rank), ("predicted", pred)):
            out[key][sl] = val.float().cpu().numpy()
    return out


def summarise(s, y, mask=None):
    """Error / rank split and margin statistics over one slice of images."""
    if mask is None:
        mask = np.ones(len(y), bool)
    n = int(mask.sum())
    if n == 0:
        return {"n_images": 0}
    rank, m = s["rank"][mask], s["margin"][mask]
    pct = lambda b: float(100.0 * np.mean(b))  # noqa: E731
    return {
        "n_images": n,
        "err_pct": pct(s["predicted"][mask] != y[mask]),
        "close_rank2_pct": pct(rank == 2),
        "mid_rank3_5_pct": pct((rank >= 3) & (rank <= 5)),
        "far_rank_gt5_pct": pct(rank > 5),
        "raw_close_pct": pct((m < 0) & (m >= -RAW_SPLIT)),
        "raw_far_pct": pct(m < -RAW_SPLIT),
        "raw_margin_mean": float(m.mean()), "raw_margin_median": float(np.median(m)),
        "z_margin_mean": float(s["z"][mask].mean()), "z_margin_median": float(np.median(s["z"][mask])),
        "p_margin_mean": float(s["p_margin"][mask].mean()),
        "p_margin_median": float(np.median(s["p_margin"][mask])),
    }


@torch.no_grad()
def centroid_geometry(feat, y, dirs, device, n_pairs=N_PAIRS, seed=0):
    """Separation gap (raw and standardised) and classifier-row/centroid alignment.

    Uses the repgeom definitions: centroid = mean of the raw features of a species,
    within-species cos = mean cosine of its images to that centroid direction, nearest-species
    cos = cosine from its centroid to the closest other species' centroid (over all species
    with >= 1 image).  The gap is averaged over species with >= 2 images.  Alignment is the
    cosine between a species' classifier row (best sub-center) and its centroid, both for the
    globally-centred centroid (repgeom's "self-duality") and the uncentred one.
    """
    n_classes = dirs.shape[0]
    mu, counts, mu_g, present = geometry.class_centroids(feat, y, n_classes, device)
    within, _ = geometry.intra_class_spread(feat, y, mu, n_classes, device)
    sims, _ = geometry.nearest_neighbour(mu, present, device, k=1)
    nearest = sims[:, 0]
    use = present & (counts >= 2)
    gap_raw = float(np.mean(within[use] - nearest[use]))

    # std of the off-diagonal centroid-centroid cosines, from a fixed-seed sample of pairs
    mu_n = F.normalize(mu, dim=1)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(present)
    i, j = rng.choice(idx, n_pairs), rng.choice(idx, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    cs = []
    for lo in range(0, len(i), 100_000):
        a = torch.as_tensor(i[lo:lo + 100_000], device=device)
        b = torch.as_tensor(j[lo:lo + 100_000], device=device)
        cs.append((mu_n[a] * mu_n[b]).sum(1).cpu().numpy())
    cs = np.concatenate(cs)
    off_std = float(cs.std())

    W = F.normalize(dirs.to(device).float(), dim=2)
    p = torch.as_tensor(present, device=device)
    cen_c = F.normalize(mu[p] - mu_g[None, :], dim=1)
    cen_u = mu_n[p]
    align_c = (W[p] * cen_c[:, None, :]).sum(2).max(1).values
    align_u = (W[p] * cen_u[:, None, :]).sum(2).max(1).values
    return {
        "n_species_present": int(present.sum()), "n_species_gap": int(use.sum()),
        "within_cos": float(np.mean(within[use])), "nearest_cos": float(np.mean(nearest[use])),
        "gap_raw": gap_raw, "offdiag_centroid_cos_mean": float(cs.mean()),
        "offdiag_centroid_cos_std": off_std, "gap_std": gap_raw / off_std,
        "align_centred_mean": float(align_c.mean()), "align_uncentred_mean": float(align_u.mean()),
    }


def analyse(feat, y, dirs, device):
    """Margin summary + centroid geometry for one split.  Returns (summary dict, per-image scores)."""
    s = margin_scores(feat, y, dirs, device)
    r = summarise(s, y)
    r["argmax_matches_labels"] = float(np.mean(s["predicted"].astype(np.int64) == y))
    r.update(centroid_geometry(feat, y, dirs, device))
    return r, s


def read_buffer(run_dir, key):
    path = f"{OUT}/{run_dir}/model.safetensors"
    with safe_open(path, "pt") as f:
        if key not in f.keys():
            return None
        return f.get_tensor(key).float().numpy()


def trained_margin_rad(name, run_dir, train_count):
    """(N,) angular margin per species that this model trained with, in radians (or None)."""
    spec = MARGIN_SPEC.get(name)
    if spec is None:
        return None, {}
    note = {}
    if spec[0] == "const":
        return np.full(len(train_count), spec[1], dtype=np.float64), note
    if spec[0] == "dbm":
        K, tau = 0.1, 1.0
        n_min = train_count[train_count > 0].min()
        m = K * (n_min / np.maximum(train_count, 1)) ** tau
        buf = read_buffer(run_dir, "head.m_class")
        if buf is not None:
            note["max_abs_diff_vs_head.m_class"] = float(np.abs(buf - m).max())
        note["n_min"] = int(n_min)
        return m, note
    if spec[0] == "kappaface":
        buf = read_buffer(run_dir, "head.per_class_margin")
        return (None if buf is None else buf.astype(np.float64)), note
    return None, note


def stats_deg(rad, mask):
    deg = np.degrees(rad[mask])
    return {"n_species": int(mask.sum()), "mean_deg": float(deg.mean()), "median_deg": float(np.median(deg))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{PREFIX} device={device} models={','.join(args.models)}", flush=True)

    with open(TRAIN_JSON) as f:
        train_count = np.bincount([json.loads(l)[LABEL] for l in f if l.strip()], minlength=N)
    results = {}
    for name in args.models:
        if name in SKIP:
            print(f"{PREFIX} SKIP {name}: {SKIP[name]}", flush=True)
            continue
        if name not in RUNS:
            print(f"{PREFIX} MISSING {name}: unknown model name", flush=True)
            continue
        run_dir, kind = RUNS[name]
        fpath, ppath = f"{GEOM}/{name}/features.npy", f"{GEOM}/{name}/predictions.npz"
        ckpt = f"{OUT}/{run_dir}/model.safetensors"
        if not all(os.path.isfile(p) for p in (fpath, ppath, ckpt)):
            print(f"{PREFIX} MISSING {name}", flush=True)
            continue
        feat = np.load(fpath)
        z = np.load(ppath)
        y = z["labels"].astype(np.int64)
        saved = z["predictions"].astype(np.int64)
        assert len(feat) == len(y), (name, feat.shape, y.shape)
        dirs = class_directions(run_dir, kind)

        r, s = analyse(feat, y, dirs, device)
        match = float(np.mean(s["predicted"].astype(np.int64) == saved))
        if match < 0.999:
            print(f"{PREFIX} WARN {name}: recomputed argmax matches the saved predictions for only "
                  f"{match:.4%} of images -- check the head key", flush=True)
        r["argmax_matches_saved"] = match

        tm, note = trained_margin_rad(name, run_dir, train_count)
        has_species = train_count > 0
        if tm is not None:
            r["trained_margin"] = {"overall": stats_deg(tm, has_species), "bands": {}, **note}
        tc = train_count[y]
        r["bands"] = {}
        for lo, hi in BANDS:
            b = summarise(s, y, (tc >= lo) & (tc <= hi))
            if tm is not None:
                b["trained_margin"] = stats_deg(tm, (train_count >= lo) & (train_count <= hi))
                r["trained_margin"]["bands"][f"{lo}-{hi}"] = b["trained_margin"]
            r["bands"][f"{lo}-{hi}"] = b
        results[name] = r

        tmtxt = (f"trained_margin_deg mean={r['trained_margin']['overall']['mean_deg']:.2f} "
                 f"median={r['trained_margin']['overall']['median_deg']:.2f}"
                 if tm is not None else "trained_margin_deg n/a")
        print(f"{PREFIX} {name}: err={r['err_pct']:.2f}% close(rank2)={r['close_rank2_pct']:.2f}% "
              f"mid(3-5)={r['mid_rank3_5_pct']:.2f}% far(rank>5)={r['far_rank_gt5_pct']:.2f}% "
              f"[raw split: [-{RAW_SPLIT},0)={r['raw_close_pct']:.2f}% <-{RAW_SPLIT}={r['raw_far_pct']:.2f}%] | "
              f"raw_margin={r['raw_margin_mean']:+.4f}/{r['raw_margin_median']:+.4f} "
              f"z_margin={r['z_margin_mean']:+.3f}/{r['z_margin_median']:+.3f} "
              f"p_margin={r['p_margin_mean']:+.3f}/{r['p_margin_median']:+.3f} (mean/median) | "
              f"within={r['within_cos']:.4f} nearest={r['nearest_cos']:.4f} gap_raw={r['gap_raw']:.4f} "
              f"gap_std={r['gap_std']:.2f} (offdiag std {r['offdiag_centroid_cos_std']:.4f}) "
              f"align={r['align_centred_mean']:.4f} | {tmtxt}", flush=True)
        for bname, b in r["bands"].items():
            if b["n_images"] == 0:
                continue
            btm = (f" trained_margin_deg mean={b['trained_margin']['mean_deg']:.2f} "
                   f"median={b['trained_margin']['median_deg']:.2f}" if "trained_margin" in b else "")
            print(f"{PREFIX} {name} band {bname:>5} (n_img={b['n_images']:6,}): "
                  f"z_margin={b['z_margin_mean']:+.3f} raw_margin={b['raw_margin_mean']:+.4f} "
                  f"err={b['err_pct']:.2f}% close={b['close_rank2_pct']:.2f}% "
                  f"mid={b['mid_rank3_5_pct']:.2f}% far={b['far_rank_gt5_pct']:.2f}% |{btm}", flush=True)
        del feat, dirs, s
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(results, open(RESULT, "w"), indent=1)
    print(f"{PREFIX} wrote {RESULT} ({len(results)} models)", flush=True)


if __name__ == "__main__":
    main()
