"""Robustness of the paper's two geometry measures to sample size (reviewer feedback).

1. Separability. The pairwise Fisher score J_ij = (mu_i - mu_j)^T A (mu_i - mu_j), with
   A = (Sw + ridge)^-1, is computed from *estimated* species means. Each estimated mean carries
   sampling noise with covariance Sigma_i / n_i, so
       E[J_ij_hat] = J_ij + tr(A Sigma_i) / n_i + tr(A Sigma_j) / n_j,
   a positive bias that is larger on the validation split (about a quarter of the training
   images per species) and in the bands with fewest images. Reported variants:
     original    the paper's numbers: all 15,501 species (species absent from a split enter
                 with a zero mean), Sw divided by n. Must reproduce the stored values.
     fixed       only species present in the split; Sw divided by n - C (unbiased pooled
                 covariance). Same ridge (1e-3 of the mean variance).
     corr_pooled fixed, minus the bias with tr(A Sigma_i) ~ d_eff = tr(A Sw) for every species
                 (one shared within-species covariance).
     corr_class  fixed, minus the bias with tr(A Sigma_i) estimated per species from its own
                 residuals, sum (x - mu_i)^T A (x - mu_i) / (n_i - 1); species with one image
                 fall back to d_eff.
     matched_k   k images drawn per species in both splits (species with >= k images in both),
                 means from the draw, covariance from the whole split; averaged over several
                 draws. Equal n on both splits, so the bias is the same on both (and is also
                 removed: _corr).
   The inverse of an estimated covariance is itself inflated by about dof / (dof - d - 1); with
   dof = n - C that is 1.007 on validation and 1.002 on training images, so it is only reported.
   S_i = mean_{j != i} J_ij; the corrected S_i subtracts b_i + mean_{j != i} b_j.
   Summaries average S_i over species with at least two images in the split.

2. Compactness. The paper's angle uses the classifier weight w_y. For unit features f of species
   i with population mean m_i, the mean cosine to the weight factorises exactly:
       E[cos(f, w_i)] = m_i . w_i = |m_i| * cos(m_i, w_i) = R_i * alignment_i.
   R_i (the mean resultant length) is the dispersion of the species' images around their own
   centroid; alignment_i is how well the weight points at that centroid. Estimators (n_i >= 2):
     cos_w       mean cos(f, w_i)                      (unbiased: w_i is fixed)
     R2          (|sum f|^2 - n_i) / (n_i (n_i - 1))   = mean pairwise within-species cosine,
                                                         unbiased for |m_i|^2
     R           sqrt(max(R2, 0))
     align       mean cos_w / sqrt(mean R2)            (population alignment, from the averages)
     align_naive cos(mean f, w_i)                      (biased low at small n_i)
     loo_cos     mean cos(f, centroid of the species' other images)  (biased LOW at small n_i:
                 the other images' centroid is noisy, so its direction is off)
     in_cos      mean cos(f, centroid including f)     (biased HIGH at small n_i: f pulls the
                 centroid towards itself). R2 is the unbiased one; these two bracket it.
   Species-weighted means over species with n_i >= 2, overall and by training-image band.
   With sub-centres (k > 1) the closest sub-centre counts for cos_w, as in the paper.

Inputs: PHASE3_GEOM/<model> (validation) and PHASE3_GEOM_TRAIN/<model> (training) features and
predictions. Output: lines prefixed __GEOROB__; MARGIN_DIAG/geometry_robustness.json.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_measures import RUNS, TRAIN  # noqa: E402
from paper_separability import BANDS, LABEL, N, TRAIN_JSON, fisher_scores  # noqa: E402
from phase3_geometry import GEOM, OUT, class_directions  # noqa: E402

MODELS = ["cosine_m0", "dbm", "arcface_m01", "kappaface", "subcenter_arcface",
          "arcface_tara", "arcface", "arcface_ecc"]
RESULT = f"{OUT}/MARGIN_DIAG/geometry_robustness.json"
PREFIX = "__GEOROB__"
RIDGE = 1e-3
MATCHED_K = (2, 5)
MATCHED_DRAWS = 5
CHUNK = 65536
# Stored paper values (run d87d734a / train_measures): the "original" variant must reproduce them.
EXPECT = {("val", "dbm"): 1840.2, ("val", "arcface"): 2698.2,
          ("train", "dbm"): 1731.8, ("train", "arcface"): 3977.4}


def load(split, name, device):
    root = GEOM if split == "val" else TRAIN
    y = np.load(f"{root}/{name}/predictions.npz")["labels"].astype(np.int64)
    feat = np.load(f"{root}/{name}/features.npy", mmap_mode="r")
    assert len(feat) == len(y), (split, name)
    x = F.normalize(torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device), dim=1)
    return x, torch.as_tensor(y, device=device)


def pooled(x, y, n_species, divide_by_dof=True):
    """Species sums/means/counts and the pooled within-species covariance (no ridge)."""
    d = x.size(1)
    counts = torch.bincount(y, minlength=n_species).double()
    sums = torch.zeros(n_species, d, device=x.device, dtype=torch.float64)
    for s in range(0, len(y), CHUNK):  # chunked: a 2M x 1024 double copy would not fit next to x
        sums.index_add_(0, y[s:s + CHUNK], x[s:s + CHUNK].double())
    mu = (sums / counts.clamp(min=1)[:, None]).float()
    Sw = torch.zeros(d, d, device=x.device, dtype=torch.float64)
    for s in range(0, len(y), CHUNK):
        c = (x[s:s + CHUNK] - mu[y[s:s + CHUNK]]).double()
        Sw += c.T @ c
    present = int((counts > 0).sum())
    Sw /= (len(y) - present) if divide_by_dof else len(y)
    return sums, mu, counts, Sw.float()


def with_ridge(Sw):
    return Sw + RIDGE * torch.trace(Sw) / Sw.size(0) * torch.eye(Sw.size(0), device=Sw.device)


def per_class_trace(x, y, mu, A, counts):
    """tr(A Sigma_i) per species from its residuals: sum r^T A r / (n_i - 1); NaN if n_i < 2."""
    q = torch.zeros(len(counts), device=x.device, dtype=torch.float64)
    for s in range(0, len(y), CHUNK):
        r = x[s:s + CHUNK] - mu[y[s:s + CHUNK]]
        q.index_add_(0, y[s:s + CHUNK], ((r @ A) * r).sum(1).double())
    q = q / (counts - 1).clamp(min=1)
    q[counts < 2] = float("nan")
    return q


def fisher_variants(x, y, device):
    """Original, fixed and bias-corrected per-species S_i (arrays over all N; NaN if absent)."""
    out = {}
    # original: exactly the paper's computation (all N species, Sw / n)
    _, mu, counts, Sw_n = pooled(x, y, N, divide_by_dof=False)
    out["original"] = fisher_scores(mu, with_ridge(Sw_n), device)

    # fixed: present species only, Sw / (n - C)
    _, mu, counts, Sw = pooled(x, y, N, divide_by_dof=True)
    pres = counts > 0
    idx = torch.nonzero(pres).squeeze(1)
    Sr = with_ridge(Sw)
    A = torch.linalg.inv(Sr.double()).float()
    S_fix = fisher_scores(mu[idx], Sr, device)
    C = len(idx)
    n_i = counts[idx].double().cpu().numpy()
    d_eff = float(torch.trace(A.double() @ Sw.double()))

    def corrected(b):
        return S_fix - b - (b.sum() - b) / (C - 1)

    b_pool = d_eff / n_i
    q = per_class_trace(x, y, mu, A, counts)[idx].cpu().numpy()
    b_class = np.where(np.isnan(q), d_eff, q) / n_i

    def scatter(v):
        full = np.full(N, np.nan)
        full[idx.cpu().numpy()] = v
        return full

    out["fixed"] = scatter(S_fix)
    out["corr_pooled"] = scatter(corrected(b_pool))
    out["corr_class"] = scatter(corrected(b_class))
    out["bias_pooled_mean"] = scatter(b_pool + (b_pool.sum() - b_pool) / (C - 1))
    dof = len(y) - C
    meta = {"d_eff": d_eff, "n_present": C, "n_images": int(len(y)),
            "mean_trace_class": float(np.nanmean(q)),
            # inverse-Wishart inflation of A, ~dof / (dof - d - 1); about 1.007 (val), 1.002 (train)
            "inv_wishart_factor": dof / (dof - Sw.size(0) - 1)}
    return out, counts.cpu().numpy(), meta, Sr


def matched(x, y, keep_species, k, seed, Sr, d_eff, device):
    """S_i on k random images per species in keep_species (bool over N); raw and corrected.

    Means come from the draw; the covariance Sr (with ridge) and d_eff come from the whole split,
    because a covariance estimated from k images per species has few degrees of freedom and its
    inverse is inflated by about dof / (dof - d) (inverse-Wishart bias).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(y), generator=g).to(y.device)
    yp = y[perm]
    order = torch.argsort(yp, stable=True)
    idx_all = perm[order]
    ys = y[idx_all]
    counts = torch.bincount(ys, minlength=N)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(ys), device=y.device) - start[ys]
    keep = (rank < k) & torch.as_tensor(keep_species, device=y.device)[ys]
    sel = idx_all[keep]
    xs, ys = x[sel], y[sel]
    cnt = torch.bincount(ys, minlength=N).float()
    mu = torch.zeros(N, x.size(1), device=x.device).index_add_(0, ys, xs) / cnt.clamp(min=1)[:, None]
    idx = torch.nonzero(cnt > 0).squeeze(1)
    S = fisher_scores(mu[idx], Sr, device)
    full, full_c = np.full(N, np.nan), np.full(N, np.nan)
    full[idx.cpu().numpy()] = S
    full_c[idx.cpu().numpy()] = S - 2.0 * d_eff / k  # b_i = b_j = d_eff / k
    return full, full_c


def compactness(x, y, w):
    """Per-species cos_w, R2, align_naive, loo_cos, in_cos (NaN where n_i < 2)."""
    sums, mu, counts, _ = pooled(x, y, N, divide_by_dof=False)
    n = counts
    # cos to the weight, closest sub-centre
    cos_w_img = torch.empty(len(y), device=x.device)
    for s in range(0, len(y), CHUNK):
        cos_w_img[s:s + CHUNK] = torch.einsum("bc,bkc->bk", x[s:s + CHUNK], w[y[s:s + CHUNK]]).max(1).values
    cos_w = torch.zeros(N, device=x.device, dtype=torch.float64).index_add_(0, y, cos_w_img.double()) / n.clamp(min=1)
    R2 = ((sums * sums).sum(1) - n) / (n * (n - 1)).clamp(min=1)
    mu_hat = F.normalize(sums.float(), dim=1)
    align_naive = torch.einsum("nc,nkc->nk", mu_hat, w).max(1).values.double()
    # leave-one-out and in-sample centroid cosines per image
    loo = torch.zeros(N, device=x.device, dtype=torch.float64)
    ins = torch.zeros(N, device=x.device, dtype=torch.float64)
    for s in range(0, len(y), CHUNK):
        xs, ys = x[s:s + CHUNK].double(), y[s:s + CHUNK]
        S = sums[ys]
        dot = (xs * S).sum(1)
        other = S - xs
        loo.index_add_(0, ys, (dot - 1.0) / other.norm(dim=1).clamp(min=1e-12))
        ins.index_add_(0, ys, dot / S.norm(dim=1).clamp(min=1e-12))
    loo, ins = loo / n.clamp(min=1), ins / n.clamp(min=1)
    ok = n >= 2
    R = R2.clamp(min=0).sqrt()
    out = {"cos_w": cos_w, "R2": R2, "R": R, "align_naive": align_naive, "loo_cos": loo, "in_cos": ins}
    return {k: torch.where(ok, v, torch.full_like(v, float("nan"))).cpu().numpy() for k, v in out.items()}


def summarise(v, train_count, ok):
    res = {"all": float(np.nanmean(v[ok]))}
    for lo, hi in BANDS:
        m = ok & (train_count >= lo) & (train_count <= hi)
        res[f"{lo}-{hi}"] = float(np.nanmean(v[m])) if m.any() else None
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_count = np.bincount([json.loads(l)[LABEL] for l in open(TRAIN_JSON) if l.strip()], minlength=N)
    print(f"{PREFIX} device={device}", flush=True)
    out = {}
    for name in MODELS:
        w = F.normalize(class_directions(*RUNS[name]).to(device), dim=-1)
        data, per_split = {}, {}
        for split in ("val", "train"):
            x, y = load(split, name, device)
            fv, counts, meta, Sr = fisher_variants(x, y, device)
            comp = compactness(x, y, w)
            ok = counts >= 2
            r = {"meta": meta,
                 "fisher": {k: summarise(v, train_count, ok) for k, v in fv.items()},
                 "compactness": {k: summarise(v, train_count, ok) for k, v in comp.items()},
                 "n_species_ge2": int(ok.sum())}
            # Alignment as mean cos_w / sqrt(mean R2): a per-species cos_w / R explodes when R is
            # estimated from two images, and sqrt of a per-species R2 is biased at small n_i.
            cmp_ = r["compactness"]
            cmp_["align"] = {b: (cmp_["cos_w"][b] / np.sqrt(cmp_["R2"][b])
                                 if cmp_["R2"][b] and cmp_["R2"][b] > 0 else None) for b in cmp_["R2"]}
            r["fisher"]["original_paper_mean"] = float(np.mean(fv["original"]))
            exp = EXPECT.get((split, name))
            if exp is not None:
                got = r["fisher"]["original_paper_mean"]
                print(f"{PREFIX} CHECK {split} {name}: original {got:.1f} vs stored {exp:.1f}", flush=True)
            data[split], per_split[split] = (x, y, Sr, meta["d_eff"]), counts
            out.setdefault(name, {})[split] = r
            m = r["compactness"]
            print(f"{PREFIX} {name} {split}: d_eff={meta['d_eff']:.0f} trAS_i={meta['mean_trace_class']:.0f} "
                  f"Fisher orig={r['fisher']['original_paper_mean']:.0f} fixed={r['fisher']['fixed']['all']:.0f} "
                  f"corr_pooled={r['fisher']['corr_pooled']['all']:.0f} corr_class={r['fisher']['corr_class']['all']:.0f} "
                  f"bias={r['fisher']['bias_pooled_mean']['all']:.0f} | cos_w={m['cos_w']['all']:.3f} "
                  f"R2={m['R2']['all']:.3f} align={m['align']['all']:.3f} align_naive={m['align_naive']['all']:.3f} "
                  f"loo={m['loo_cos']['all']:.3f} in={m['in_cos']['all']:.3f}", flush=True)

        # size-matched Fisher: k images per species in both splits
        for k in MATCHED_K:
            keep = (per_split["val"] >= k) & (per_split["train"] >= k)
            res = {"n_species": int(keep.sum())}
            for split in ("val", "train"):
                x, y, Sr, d_eff = data[split]
                raws, cors = [], []
                for seed in range(MATCHED_DRAWS):
                    S, Sc = matched(x, y, keep, k, seed, Sr, d_eff, device)
                    raws.append(summarise(S, train_count, keep))
                    cors.append(summarise(Sc, train_count, keep))
                avg = lambda rs: {b: (float(np.mean([r[b] for r in rs])) if rs[0][b] is not None else None)
                                  for b in rs[0]}
                res[split] = {"raw": avg(raws), "corr": avg(cors),
                              "raw_draw_sd": float(np.std([r["all"] for r in raws]))}
            out[name][f"matched_k{k}"] = res
            print(f"{PREFIX} {name} matched k={k} ({res['n_species']} species): "
                  f"val raw={res['val']['raw']['all']:.0f} corr={res['val']['corr']['all']:.0f} | "
                  f"train raw={res['train']['raw']['all']:.0f} corr={res['train']['corr']['all']:.0f} | "
                  f"val/train-1 raw={100 * (res['val']['raw']['all'] / res['train']['raw']['all'] - 1):+.1f}% "
                  f"corr={100 * (res['val']['corr']['all'] / res['train']['corr']['all'] - 1):+.1f}%", flush=True)

        # train -> val change of every Fisher variant, overall and by band
        ch = {}
        for var in ("original", "fixed", "corr_pooled", "corr_class"):
            fv_, ft_ = out[name]["val"]["fisher"][var], out[name]["train"]["fisher"][var]
            ch[var] = {b: (100 * (fv_[b] / ft_[b] - 1) if fv_[b] and ft_[b] else None) for b in fv_}
        out[name]["train_to_val_change_pct"] = ch
        print(f"{PREFIX} {name} train->val Fisher change: " +
              " ".join(f"{v}={ch[v]['all']:+.1f}%" for v in ch), flush=True)
        del data
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(out, open(RESULT, "w"), indent=1)
    print(f"{PREFIX} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
