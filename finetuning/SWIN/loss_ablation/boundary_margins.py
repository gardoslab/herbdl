"""Per-sample decision-boundary margins for the 384px loss models.

The geometry study measures the *data clouds* (within-species cosine, nearest-species
cosine, separation gap).  Those are mean-level properties of each species' features.
Accuracy is not: it is the per-sample ranking that the classifier head produces.  This
script measures that ranking directly, from the same saved features and classifier
rows, and asks whether the loss that wins macro-F1 also puts its samples further from
the decision boundary.

For every validation image i with true species y_i it computes, using the head exactly
as it runs at inference (cosine, no margin, max over sub-centers):

    cos_true(i)        cosine to the correct species' direction
    cos_best_other(i)  cosine to the closest wrong species' direction
    margin(i)          cos_true - cos_best_other   (negative = wrong answer)
    rank(i)            1 + number of species scoring above the correct one
    predicted(i)       argmax over species (checked against the saved predictions)

A margin near zero means the image sits near the boundary; the sign says which side.
The mean of that distribution can improve while the class-cloud statistics barely move,
because the error rate is driven by the mass near zero, not by the average tightness.

Outputs (SCC paths below):
  PHASE3_GEOM/<model>/boundary.npz          per-image margin, components, labels, preds
  PHASE3_GEOM/analysis/boundary_summary.json  overall / rarity-band / error-side stats,
                                              joined to geometry_summary.json, plus the
                                              difference from ArcFace per band.

Usage: python loss_ablation/boundary_margins.py [--selftest] [--models dbm arcface]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
CHUNK = 1024
NEAR = 0.02  # "near the boundary": |margin| < 0.02 in cosine units

# Same model set as phase3_geometry.py, minus HACE (its species rows are only part of a
# tree-aggregated decision, so a cosine margin over those rows is not the model's answer)
# and X2-Softmax (collapsed, not a trained-well comparison).
MODELS = {
    "arcface": ("SWIN_BASE_ARCFACE_384", "arcface"),
    "arcface_seed43": ("SWIN_BASE_ARCFACE_384_SEED43", "arcface"),
    "subcenter_arcface": ("SWIN_BASE_SUBCENTER_ARCFACE_384_PILOT", "arcface"),
    "dbm": ("SWIN_BASE_DBM_384_PILOT", "registry"),
    "dbm_seed43": ("SWIN_BASE_DBM_384_PILOT_SEED43", "registry"),
    "kappaface": ("SWIN_BASE_KAPPAFACE_384_PILOT", "registry"),
    "kappaface_seed43": ("SWIN_BASE_KAPPAFACE_384_PILOT_SEED43", "registry"),
    "arcface_tara": ("SWIN_BASE_ARCFACE_TARA_384_PILOT", "registry"),
    "arcface_ecc": ("SWIN_BASE_ARCFACE_ECC_384_PILOT", "registry"),
}


@torch.no_grad()
def boundary_scores(feat, labels, dirs, device, chunk=CHUNK):
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


def band_stats(mask, labels, pred, cos_true, cos_best, margin):
    """Summary of the margin distribution for one slice of the validation set."""
    n = int(mask.sum())
    if n == 0:
        return {"n_images": 0}
    m, ct, cb = margin[mask], cos_true[mask], cos_best[mask]
    return {
        "n_images": n,
        "n_species": int(len(np.unique(labels[mask]))),
        "accuracy": float(np.mean(pred[mask] == labels[mask])),
        "mean_cos_true": float(ct.mean()),
        "mean_cos_best_other": float(cb.mean()),
        "mean_margin": float(m.mean()),
        "median_margin": float(np.median(m)),
        "frac_error": float(np.mean(m < 0)),
        f"frac_near_correct_{NEAR}": float(np.mean((m >= 0) & (m < NEAR))),
        f"frac_near_wrong_{NEAR}": float(np.mean((m < 0) & (m >= -NEAR))),
        f"frac_margin_below_{NEAR}": float(np.mean(m < NEAR)),
    }


def selftest():
    """Tiny synthetic case: brute-force reference, plus scale invariance."""
    torch.manual_seed(0)
    n_classes, k, dim, n = 7, 3, 16, 50
    W = torch.randn(n_classes * k, dim)
    feat = (torch.randn(n, dim) * 3.0).numpy()
    y = np.random.default_rng(0).integers(0, n_classes, n)
    got = boundary_scores(feat, y, W.view(n_classes, k, dim), "cpu", chunk=8)

    Wn = F.normalize(W.view(n_classes, k, dim), dim=2)
    xn = F.normalize(torch.as_tensor(feat), dim=1)
    ref_margin, ref_pred = [], []
    for i in range(n):
        c = torch.einsum("ckd,d->ck", Wn, xn[i]).max(1).values
        others = torch.cat([c[:y[i]], c[y[i] + 1:]])
        ref_margin.append(float(c[y[i]] - others.max()))
        ref_pred.append(int(c.argmax()))
    assert np.allclose(got["margin"], ref_margin, atol=1e-6), (
        np.abs(got["margin"] - np.array(ref_margin)).max())
    assert np.array_equal(got["predicted"], np.array(ref_pred))

    # Cosine margins must not depend on feature magnitude.
    scaled = boundary_scores(feat * 100.0, y, W.view(n_classes, k, dim), "cpu", chunk=8)
    assert np.allclose(got["margin"], scaled["margin"], atol=1e-6)
    print("__BOUNDARY__ SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the synthetic check and exit")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--no-per-image", action="store_true", help="do not write boundary.npz")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    # Imported here so --selftest needs only torch, not the checkpoint reader.
    from phase3_geometry import GEOM, N, class_directions  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(TRAIN_JSON) as f:
        train_count = np.bincount([json.loads(line)[LABEL] for line in f if line.strip()],
                                  minlength=N)
    names = args.models or list(MODELS)
    results = {}

    for name in names:
        run_dir, kind = MODELS[name]
        fpath, ppath = f"{GEOM}/{name}/features.npy", f"{GEOM}/{name}/predictions.npz"
        if not (os.path.isfile(fpath) and os.path.isfile(ppath)):
            print(f"__BOUNDARY__ MISSING {name}")
            continue
        feat = np.load(fpath).astype(np.float32)
        z = np.load(ppath)
        y = z["labels"].astype(np.int64)
        saved_pred = z["predictions"].astype(np.int64)
        assert len(feat) == len(y), (name, feat.shape, y.shape)

        dirs = class_directions(run_dir, kind)  # (N, k, C)
        s = boundary_scores(feat, y, dirs, device)

        match = float(np.mean(s["predicted"] == saved_pred))
        if match < 0.999:
            print(f"__BOUNDARY__ WARN {name}: recomputed argmax matches saved predictions "
                  f"for only {match:.4%} of images -- check the head key")

        tc = train_count[y]
        r = {
            "overall": band_stats(np.ones(len(y), bool), y, s["predicted"], s["cos_true"],
                                  s["cos_best_other"], s["margin"]),
            "by_correctness": {
                "correct": band_stats(s["predicted"] == y, y, s["predicted"], s["cos_true"],
                                      s["cos_best_other"], s["margin"]),
                "wrong": band_stats(s["predicted"] != y, y, s["predicted"], s["cos_true"],
                                    s["cos_best_other"], s["margin"]),
            },
            "bands": {},
            "argmax_matches_saved": match,
        }
        for lo, hi in BANDS:
            r["bands"][f"{lo}-{hi}"] = band_stats(
                (tc >= lo) & (tc <= hi), y, s["predicted"], s["cos_true"],
                s["cos_best_other"], s["margin"])

        gpath = f"{GEOM}/analysis/geometry_summary.json"
        if os.path.isfile(gpath):
            geo = json.load(open(gpath)).get(name)
            if geo:
                r["geometry"] = {key: geo[key] for key in
                                 ("separation_gap", "within_species_cos",
                                  "nearest_species_cos", "alignment_mean")}

        if not args.no_per_image:
            np.savez_compressed(f"{GEOM}/{name}/boundary.npz",
                                labels=y, predictions=s["predicted"],
                                cos_true=s["cos_true"], cos_best_other=s["cos_best_other"],
                                margin=s["margin"], rank=s["rank"])
        results[name] = r

        o = r["overall"]
        band_txt = " ".join(
            f"{b}:m={r['bands'][b]['mean_margin']:+.3f},err={r['bands'][b]['frac_error']:.3f}"
            for b in r["bands"])
        print(f"__BOUNDARY__ {name} acc={o['accuracy']:.4f} match={match:.4f} "
              f"mean_m={o['mean_margin']:+.4f} med={o['median_margin']:+.4f} "
              f"err={o['frac_error']:.4f} near={o['frac_near_correct_0.02']:.4f} "
              f"near_wrong={o['frac_near_wrong_0.02']:.4f} | {band_txt}", flush=True)
        del feat, dirs
        if device == "cuda":
            torch.cuda.empty_cache()

    # Difference from ArcFace, band by band: the comparison the plot will show.
    if "arcface" in results:
        base = results["arcface"]
        for name, r in results.items():
            if name == "arcface":
                continue
            r["vs_arcface"] = {
                "bands": {
                    b: {
                        "delta_mean_margin": r["bands"][b]["mean_margin"] - base["bands"][b]["mean_margin"],
                        "delta_frac_error": r["bands"][b]["frac_error"] - base["bands"][b]["frac_error"],
                        "delta_frac_near": r["bands"][b]["frac_near_correct_0.02"]
                                           - base["bands"][b]["frac_near_correct_0.02"],
                    } for b in r["bands"] if b in base["bands"]
                },
                "overall": {
                    "delta_mean_margin": r["overall"]["mean_margin"] - base["overall"]["mean_margin"],
                    "delta_frac_error": r["overall"]["frac_error"] - base["overall"]["frac_error"],
                },
            }
            d = r["vs_arcface"]["bands"]
            print(f"__BOUNDARY__ vs arcface {name}: " + " ".join(
                f"{b}:dm={d[b]['delta_mean_margin']:+.3f},derr={d[b]['delta_frac_error']:+.3f}"
                for b in d), flush=True)

    os.makedirs(f"{GEOM}/analysis", exist_ok=True)
    out = f"{GEOM}/analysis/boundary_summary.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"__BOUNDARY__ wrote {out} ({len(results)} models)")


if __name__ == "__main__":
    main()
