"""Margins each loss trains with, and scale-free decision margins, for the 15.5k base models.

The raw cosine margin cos_true - max_wrong cos depends on how spread out the cosines are (a model whose
cosines are all squeezed together has small raw margins whatever it does). This script reports the same
margin in three scale-free forms and splits the errors by how far down the true species is ranked:

  raw margin        m = cos_y - max_{k != y} cos_k
  standardised      z = m / std_k(cos_k), std over all species for that image (population std)
  softmax margin    p_y - max_{k != y} p_k, p = softmax(30 * cos)  (30 = the training scale s)
  error by rank     close call = true species ranked 2nd, mid = rank 3-5, far = rank > 5,
                    each as a % of ALL validation images (so close+mid+far = error rate).
                    The old raw split (m in [-0.02, 0) vs m < -0.02) is kept for reference.
  trained margin    the angular margin each loss actually trained with, per species,
                    summarised (species-weighted mean / median, degrees) per training-count
                    band: DBM m = K (n_min / n_y)^tau, K = 0.1, tau = 1, n_min from the train
                    counts (cross-checked against the head.m_class buffer in the checkpoint; this
                    is the class margin m_C -- DBM adds up to 1.5x of it on samples whose nearest
                    species is wrong, which is not included);
                    KappaFace = head.per_class_margin from the checkpoint (value after the last
                    epoch's recompute); ArcFace family = constant 0.3 rad (the warm-up to 0.3
                    over 5 epochs is over by the end); cosine_m0 = 0; arcface_m01 = 0.1.

Also holds boundary_scores, the per-sample cosine margin to the nearest wrong species (cos_true,
cos_best_other, margin, rank, predicted) that head_swap.py uses to score a set of classifier rows on
cached features.

Inputs (SCC): PHASE3_GEOM/<model>/{features.npy, predictions.npz}; classifier rows from the
model's own model.safetensors, located as common.class_directions does (head.weight for registry heads,
arcface.weight for the native ArcFace and sub-center runs). cosine_m0 and arcface_m01 are registry-arcface
runs: SWIN_BASE_COSINE_M0_384 and SWIN_BASE_ARCFACE_M01_384. HACE is skipped: its decision is
tree-aggregated, there is no flat cosine classifier. A model with any missing file prints a MISSING
line and is skipped.

Output: MARGIN_DIAG/scale_free_margins.json. Prefix __SCALEFREE__.

Usage (from finetuning/SWIN): python loss_ablation/analysis/margins.py [--models dbm kappaface ...]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BANDS, DIAG, LABEL, N, OUT, PHASE3_GEOM, RUNS, TRAIN_JSON, class_directions  # noqa: E402

RESULT = f"{DIAG}/scale_free_margins.json"
CHUNK = 4096
BOUNDARY_CHUNK = 1024
SCALE = 30.0  # training logit scale s
RAW_SPLIT = 0.02
PREFIX = "__SCALEFREE__"

DEFAULT_MODELS = ["dbm", "kappaface", "arcface", "arcface_ecc", "subcenter_arcface",
                  "arcface_tara", "hace", "cosine_m0", "arcface_m01"]

# name -> how the trained margin is obtained (radians)
MARGIN_SPEC = {
    "dbm": ("dbm",), "dbm_seed43": ("dbm",),
    "kappaface": ("kappaface",), "kappaface_seed43": ("kappaface",),
    "arcface": ("const", 0.3), "arcface_seed43": ("const", 0.3), "arcface_ecc": ("const", 0.3),
    "arcface_tara": ("const", 0.3), "subcenter_arcface": ("const", 0.3),
    "cosine_m0": ("const", 0.0), "arcface_m01": ("const", 0.1),
}
SKIP = {"hace": "tree-aggregated decision, no flat cosine classifier",
        "x2softmax": "collapsed run, not a trained-well comparison"}


@torch.no_grad()
def boundary_scores(feat, labels, dirs, device, chunk=BOUNDARY_CHUNK):
    """Per-sample cosine margin to the nearest wrong species.

    feat    (n, C) float array          pooled features the head reads
    labels  (n,) int array              true species, 0..N-1
    dirs    (N, k, C) float tensor      classifier directions (k sub-centers per species)

    Returns a dict of length-n arrays.  The head at inference is pure cosine with no
    margin, taking the best sub-center per species, so this reproduces its argmax.
    """
    W = F.normalize(dirs.to(device).float(), dim=2)
    n_classes, k, dim = W.shape
    Wf = W.reshape(n_classes * k, dim)
    out = {key: np.empty(len(feat), dtype=np.float32)
           for key in ("cos_true", "cos_best_other", "margin", "rank", "predicted")}
    y_all = torch.as_tensor(labels, dtype=torch.long)

    for lo in range(0, len(feat), chunk):
        x = torch.as_tensor(feat[lo:lo + chunk], device=device).float()
        xn = F.normalize(x, dim=1)
        # (b, N*k) -> (b, N, k) -> max over sub-centers = the species logits / s
        cos = (xn @ Wf.T).view(-1, n_classes, k).max(dim=2).values
        y = y_all[lo:lo + chunk].to(device)
        cos_true = cos.gather(1, y[:, None]).squeeze(1)
        pred = cos.argmax(1)
        rank = (cos > cos_true[:, None]).sum(1) + 1
        cos.scatter_(1, y[:, None], -2.0)  # mask the true class before the max
        cos_best_other = cos.max(1).values
        sl = slice(lo, lo + len(y))
        out["cos_true"][sl] = cos_true.cpu().numpy()
        out["cos_best_other"][sl] = cos_best_other.cpu().numpy()
        out["margin"][sl] = (cos_true - cos_best_other).cpu().numpy()
        out["rank"][sl] = rank.cpu().numpy()
        out["predicted"][sl] = pred.cpu().numpy()
    return out


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


def analyse(feat, y, dirs, device):
    """Margin summary for one split.  Returns (summary dict, per-image scores)."""
    s = margin_scores(feat, y, dirs, device)
    r = summarise(s, y)
    r["argmax_matches_labels"] = float(np.mean(s["predicted"].astype(np.int64) == y))
    return r, s


def read_buffer(run_dir, key):
    from safetensors import safe_open
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
        fpath, ppath = f"{PHASE3_GEOM}/{name}/features.npy", f"{PHASE3_GEOM}/{name}/predictions.npz"
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
              f"p_margin={r['p_margin_mean']:+.3f}/{r['p_margin_median']:+.3f} (mean/median) | {tmtxt}", flush=True)
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
