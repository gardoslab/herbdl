"""Phase 3 representation geometry of the 384px loss models.

For each model, reads the full-validation-set pooled features and predictions written by
the eval-only runs (PHASE3_GEOM/<name>/features.npy, predictions.npz) and the classifier
weights from that model's own model.safetensors, then computes the same geometry as the
earlier ArcFace-vs-CE study, with the same code (repgeom_geometry.py is a verbatim copy):

  within-species cos   mean cosine of an image to its own species centroid (tightness)
  nearest-species cos  mean cosine between a species centroid and its nearest other one
  separation gap       within-species cos - nearest-species cos (higher = better separated)
  NC1                  tr(Sw pinv(Sb)) / K, within- vs between-species scatter
  trace ratio          tr(Sw) / tr(Sb)
  alignment            cosine between each species' classifier direction and its centred
                       centroid (repgeom's "self-duality"; the earlier study's "detector
                       alignment")
  alignment~recall     Spearman correlation, across species within one model, between that
                       species' alignment and its recall (0.60 in the earlier 224px study)

Classifier directions: native ArcFace/sub-center (arcface.weight, rows c*k+j), registry
heads (head.weight), HACE (first 15,501 rows of hace.linear.weight). For sub-center k=3,
alignment uses sub-center 0 like repgeom; the best of the 3 sub-centers is also reported.

Usage: python loss_ablation/phase3_geometry.py   (GPU if available; SCC paths below)
"""

import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repgeom_geometry as geometry  # noqa: E402

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
GEOM = f"{OUT}/PHASE3_GEOM"
N = 15501
# name -> (weights dir, head kind)
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
    "hace": ("SWIN_BASE_HACE_384_PILOT", "hace"),
    "x2softmax": ("SWIN_BASE_X2SOFTMAX_384_PILOT", "registry"),
}


def class_directions(run_dir, kind):
    """(N, k, C) classifier directions read by tensor name from the model file."""
    with safe_open(f"{OUT}/{run_dir}/model.safetensors", "pt") as f:
        if kind == "arcface":
            w = f.get_tensor("arcface.weight").float()
            return w.view(N, w.shape[0] // N, w.shape[1])  # row c*k + j = species c, center j
        if kind == "hace":
            return f.get_tensor("hace.linear.weight").float()[:N].unsqueeze(1)
        return f.get_tensor("head.weight").float().unsqueeze(1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    for name, (run_dir, kind) in MODELS.items():
        fpath, ppath = f"{GEOM}/{name}/features.npy", f"{GEOM}/{name}/predictions.npz"
        if not (os.path.isfile(fpath) and os.path.isfile(ppath)):
            print(f"__GEOM__ MISSING {name}")
            continue
        feat = np.load(fpath).astype(np.float32)
        z = np.load(ppath)
        y, pred = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
        assert len(feat) == len(y), (name, feat.shape, y.shape)

        mu, counts, mu_g, present = geometry.class_centroids(feat, y, N, device)
        intra_cos, _ = geometry.intra_class_spread(feat, y, mu, N, device)
        cen_sim, _ = geometry.nearest_neighbour(mu, present, device, k=2)
        Wk = class_directions(run_dir, kind).to(device)
        coll, dual = geometry.collapse_metrics(feat, y, mu, mu_g, Wk[:, 0], present, device)

        # per-species recall, and alignment-vs-recall rank correlation within this model
        n_val = np.bincount(y, minlength=N)
        hit = np.bincount(y[pred == y], minlength=N)
        recall = hit[present] / n_val[present]
        r = {
            "accuracy": float(np.mean(pred == y)),
            "within_species_cos": float(np.nanmean(intra_cos[present])),
            "nearest_species_cos": float(np.nanmean(cen_sim[present, 0])),
            "nc1": coll["nc1_tr_sw_pinv_sb_over_k"],
            "trace_ratio_sw_over_sb": coll["trace_ratio_sw_over_sb"],
            "alignment_mean": coll["self_duality_mean_cos"],
            "alignment_median": coll["self_duality_median_cos"],
            "spearman_alignment_vs_recall": spearman(dual, recall),
            "n_species": int(present.sum()), "sub_centers": int(Wk.shape[1]),
        }
        r["separation_gap"] = r["within_species_cos"] - r["nearest_species_cos"]
        if Wk.shape[1] > 1:  # best-matching sub-center per species
            cen = torch.nn.functional.normalize(mu[torch.as_tensor(present, device=device)] - mu_g, dim=1)
            wn = torch.nn.functional.normalize(Wk[torch.as_tensor(present, device=device)], dim=2)
            r["alignment_best_subcenter_mean"] = float((wn * cen[:, None, :]).sum(2).max(1).values.mean().item())
        rows[name] = r
        print(f"__GEOM__ {name}: acc={r['accuracy']:.4f} within={r['within_species_cos']:.4f} "
              f"nearest={r['nearest_species_cos']:.4f} gap={r['separation_gap']:.4f} "
              f"nc1={r['nc1']:.4f} trace_ratio={r['trace_ratio_sw_over_sb']:.4f} "
              f"align={r['alignment_mean']:.4f} (median {r['alignment_median']:.4f}"
              + (f", best-subcenter {r['alignment_best_subcenter_mean']:.4f}" if "alignment_best_subcenter_mean" in r else "")
              + f") spearman(align,recall)={r['spearman_alignment_vs_recall']:.3f}", flush=True)
        del feat, mu, Wk
        torch.cuda.empty_cache() if device == "cuda" else None

    os.makedirs(f"{GEOM}/analysis", exist_ok=True)
    json.dump(rows, open(f"{GEOM}/analysis/geometry_summary.json", "w"), indent=1)
    print(f"__GEOM__ wrote {GEOM}/analysis/geometry_summary.json ({len(rows)} models)")


if __name__ == "__main__":
    main()
