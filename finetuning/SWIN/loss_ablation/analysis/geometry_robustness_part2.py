"""Bias-corrected Fisher for the scale control, the cone pair table and the ladder.

Part 2 of geometry_robustness.py (see its docstring for the estimator). The pairwise Fisher score
J_ij = (mu_i - mu_j)^T A (mu_i - mu_j) uses estimated species means, so E[J_ij_hat] = J_ij +
tr(A Sigma_i)/n_i + tr(A Sigma_j)/n_j. With one pooled covariance, tr(A Sigma) is estimated by
d_eff = tr(A Sw), and every J_ij is corrected by d_eff * (1/n_i + 1/n_j). Three further choices:
only species present in the split enter (a species with no image has no mean), the pooled
covariance divides by n - C (images minus species), and the same ridge (1e-3 of the mean variance)
is kept. Variants reported per model:
    original   the paper's computation: all species, covariance / n (must reproduce stored values)
    fixed      present species, covariance / (n - C)
    corrected  fixed, minus the bias above (S_i: b_i + mean_{j != i} b_j, b_i = d_eff / n_i)

Parts (--parts):
  scale   s=15 / s=30 NormFace and ArcFace, validation images (PHASE3_GEOM): overall and by band.
  cone    pairwise Fisher by taxonomic relation of the pair (same genus / same family other genus /
          other family) and the nearest same-genus species, original and corrected, for the eight
          base-data losses (the table in Supp. "cone test").
  ladder  the ladder's validation features, per (rung, loss) in --losses, overall and by band.
          Reads LADDER_GEOM/<rung>_<loss>/features.npy; the class count comes from the head.
          Output file is tagged with the losses so a later pass (NormFace) does not overwrite.

Output: lines prefixed __GEOROB2__; MARGIN_DIAG/geometry_robustness2_{scale,cone}.json and
LADDER_GEOM/analysis/ladder_fisher_corrected_<losses>.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_measures as pm  # noqa: E402
import ladder_diagnostics as ld  # noqa: E402
import cone_test as ct  # noqa: E402

P = "__GEOROB2__"
RIDGE = 1e-3
CHUNK = 4096
STEP = 65536
BANDS_BASE = [(2, 10), (11, 25), (26, 50), (51, 73)]


def ridge(Sw):
    return Sw + RIDGE * torch.trace(Sw) / Sw.size(0) * torch.eye(Sw.size(0), device=Sw.device)


def split_stats(x, y, n_species):
    """Species means (float32), counts, and the summed within-species scatter (double)."""
    counts = torch.bincount(y, minlength=n_species).double()
    sums = torch.zeros(n_species, x.size(1), device=x.device, dtype=torch.float64)
    for s in range(0, len(y), STEP):  # chunked: a 2M x 1024 double copy would not fit next to x
        sums.index_add_(0, y[s:s + STEP], x[s:s + STEP].double())
    mu = (sums / counts.clamp(min=1)[:, None]).float()
    scatter = torch.zeros(x.size(1), x.size(1), device=x.device, dtype=torch.float64)
    for s in range(0, len(y), STEP):
        c = (x[s:s + STEP] - mu[y[s:s + STEP]]).double()
        scatter += c.T @ c
    return mu, counts, scatter


def chunked_S(mu, A):
    """S_i = mean_{j != i} J_ij for all rows of mu, without an N x N matrix."""
    B = mu @ A
    diag = (mu * B).sum(1)
    n = len(mu)
    out = torch.empty(n, device=mu.device)
    for lo in range(0, n, CHUNK):
        d = diag[lo:lo + CHUNK, None] + diag[None, :] - 2.0 * (B[lo:lo + CHUNK] @ mu.T)
        rows = torch.arange(len(d), device=mu.device)
        out[lo:lo + CHUNK] = (d.sum(1) - d[rows, rows + lo]) / (n - 1)
    return out.cpu().numpy()


def fisher_variants(x, y, n_species, with_original=True):
    """S_i arrays (NaN where a species is absent) for original / fixed / corrected, plus meta."""
    mu, counts, scatter = split_stats(x, y, n_species)
    pres = counts > 0
    idx = torch.nonzero(pres).squeeze(1)
    C = len(idx)
    n_img = len(y)
    out = {}

    def scatter_back(v):
        full = np.full(n_species, np.nan)
        full[idx.cpu().numpy()] = v
        return full

    if with_original:
        Sn = ridge((scatter / n_img).float())
        out["original"] = chunked_S(mu, torch.linalg.inv(Sn.double()).float())  # all species, /n
    Sw = (scatter / (n_img - C)).float()
    Sr = ridge(Sw)
    A = torch.linalg.inv(Sr.double()).float()
    S_fix = chunked_S(mu[idx], A)
    d_eff = float(torch.trace(A.double() @ Sw.double()))
    n_i = counts[idx].cpu().numpy()
    b = d_eff / n_i
    S_cor = S_fix - b - (b.sum() - b) / (C - 1)
    out["fixed"] = scatter_back(S_fix)
    out["corrected"] = scatter_back(S_cor)
    meta = {"n_images": int(n_img), "n_species_present": C, "d_eff": d_eff,
            "mean_bias": float(np.mean(b + (b.sum() - b) / (C - 1)))}
    return out, counts.cpu().numpy(), meta, (mu, idx, A, d_eff)


def summarise(v, band_count, bands, ok):
    res = {"all": float(np.nanmean(v[ok]))}
    for lo, hi in bands:
        m = ok & (band_count >= lo) & (band_count <= hi)
        res[f"{lo}-{hi}"] = float(np.nanmean(v[m])) if m.any() else None
    return res


def load(path_feat, path_pred, device):
    y = np.load(path_pred)["labels"].astype(np.int64)
    feat = np.load(path_feat, mmap_mode="r")
    assert len(feat) == len(y), path_feat
    x = F.normalize(torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device), dim=1)
    return x, torch.as_tensor(y, device=device)


def report(tag, fv, counts, meta, band_count, bands, expect=None):
    ok = counts >= 2
    res = {"meta": meta, "n_species_ge2": int(ok.sum()),
           "fisher": {k: summarise(v, band_count, bands, ok) for k, v in fv.items() if k != "original"}}
    if "original" in fv:
        res["fisher"]["original_all_species_mean"] = float(np.mean(fv["original"]))
        res["fisher"]["original"] = summarise(fv["original"], band_count, bands, np.ones(len(fv["original"]), bool))
        if expect is not None:
            print(f"{P} CHECK {tag}: original {res['fisher']['original_all_species_mean']:.1f} vs stored {expect:.1f}",
                  flush=True)
    f = res["fisher"]
    print(f"{P} {tag}: d_eff={meta['d_eff']:.0f} bias={meta['mean_bias']:.0f} "
          f"orig={f.get('original_all_species_mean', float('nan')):.0f} fixed={f['fixed']['all']:.0f} "
          f"corrected={f['corrected']['all']:.0f} | corrected by band: "
          + " ".join(f"{b}={v:.0f}" for b, v in f["corrected"].items() if b != "all" and v is not None),
          flush=True)
    return res


# ---------------------------------------------------------------- scale control
SCALE_MODELS = [("cosine_m0", "NormFace s=30", 1837.0), ("cosine_m0_s15", "NormFace s=15", 5096.0),
                ("arcface", "ArcFace s=30", 2698.2), ("arcface_s15", "ArcFace s=15", 7213.0)]


def part_scale(train_count, device):
    out = {}
    for name, label, expect in SCALE_MODELS:
        fp, pp = f"{pm.GEOM}/{name}/features.npy", f"{pm.GEOM}/{name}/predictions.npz"
        if not (os.path.isfile(fp) and os.path.isfile(pp)):
            print(f"{P} MISSING {name}", flush=True)
            continue
        x, y = load(fp, pp, device)
        fv, counts, meta, _ = fisher_variants(x, y, pm.N)
        out[name] = report(f"scale {label}", fv, counts, meta, train_count, BANDS_BASE, expect)
        del x, y
        torch.cuda.empty_cache()
    os.makedirs(f"{pm.OUT}/MARGIN_DIAG", exist_ok=True)
    json.dump(out, open(f"{pm.OUT}/MARGIN_DIAG/geometry_robustness2_scale.json", "w"), indent=1)
    return out


# ---------------------------------------------------------------- cone pairs
def pair_groups(D, g, f):
    same_g, same_f = g[:, None] == g[None, :], f[:, None] == f[None, :]
    offd = ~torch.eye(len(g), dtype=torch.bool, device=D.device)
    groups = {"same_genus": same_g & offd, "same_family_other_genus": same_f & ~same_g, "other_family": ~same_f}
    res = {k: {"n_pairs": int(m.sum().item() // 2), "mean_J": float(D[m].mean().item())} for k, m in groups.items()}
    has = (same_g & offd).any(1)
    Dg = D.masked_fill(~(same_g & offd), float("inf"))
    res["nearest_sister_mean_J"] = float(Dg.min(1).values[has].mean().item())
    Dn = D.masked_fill(~offd, float("inf"))
    nn_val, nn_idx = Dn.min(1)
    res["nearest_neighbour"] = {"mean_J": float(nn_val.mean().item()),
                                "same_genus_pct": float(100 * (g[nn_idx] == g).float().mean().item())}
    return res


def part_cone(device):
    counts, cls_fam, cls_gen, _, _ = ct.load_meta()
    out = {}
    for m in ct.MODELS:
        x, y = load(f"{pm.GEOM}/{m}/features.npy", f"{pm.GEOM}/{m}/predictions.npz", device)
        mu, cnt, scatter = split_stats(x, y, pm.N)
        idx = torch.nonzero(cnt > 0).squeeze(1)
        C, n = len(idx), len(y)
        mu_p, n_i = mu[idx], cnt[idx].float()
        gi = torch.as_tensor(cls_gen[idx.cpu().numpy()], device=device)
        fi = torch.as_tensor(cls_fam[idx.cpu().numpy()], device=device)

        def J(Sw):
            A = torch.linalg.inv(ridge(Sw.float()).double()).float()
            B = mu_p @ A
            dg = (mu_p * B).sum(1)
            return dg[:, None] + dg[None, :] - 2.0 * (B @ mu_p.T), A

        D_orig, _ = J(scatter / n)  # paper: covariance / n (present species only, as in cone_test)
        Sw = scatter / (n - C)
        D_fix, A = J(Sw)
        d_eff = float(torch.trace(A.double() @ Sw))
        bias = d_eff * (1.0 / n_i[:, None] + 1.0 / n_i[None, :])
        res = {"n_species": C, "d_eff": d_eff,
               "original": pair_groups(D_orig, gi, fi),
               "corrected": pair_groups(D_fix - bias, gi, fi)}
        out[m] = res
        o, c = res["original"], res["corrected"]
        print(f"{P} cone {ct.NAMES[m]}: same genus {o['same_genus']['mean_J']:.0f}->{c['same_genus']['mean_J']:.0f} | "
              f"same family {o['same_family_other_genus']['mean_J']:.0f}->{c['same_family_other_genus']['mean_J']:.0f} | "
              f"other family {o['other_family']['mean_J']:.0f}->{c['other_family']['mean_J']:.0f} | "
              f"nearest sister {o['nearest_sister_mean_J']:.0f}->{c['nearest_sister_mean_J']:.0f}", flush=True)
        del x, y, D_orig, D_fix, bias
        torch.cuda.empty_cache()
    for kind in ("original", "corrected"):
        base = out["dbm"][kind]
        for m in ("arcface", "arcface_ecc", "cosine_m0"):
            r = out[m][kind]
            ratios = {k: r[k]["mean_J"] / base[k]["mean_J"]
                      for k in ("same_genus", "same_family_other_genus", "other_family")}
            ratios["nearest_sister"] = r["nearest_sister_mean_J"] / base["nearest_sister_mean_J"]
            out[m][f"over_dbm_{kind}"] = ratios
            print(f"{P} cone {ct.NAMES[m]} / DBM ({kind}): same genus x{ratios['same_genus']:.2f}, same family "
                  f"x{ratios['same_family_other_genus']:.2f}, other family x{ratios['other_family']:.2f}, "
                  f"nearest sister x{ratios['nearest_sister']:.2f}", flush=True)
    json.dump(out, open(f"{pm.OUT}/MARGIN_DIAG/geometry_robustness2_cone.json", "w"), indent=1)
    return out


# ---------------------------------------------------------------- ladder
def part_ladder(rungs, losses, device):
    prior = json.load(open(f"{ld.FEAT}/analysis/ladder_diagnostics.json"))
    out = {}
    for rung in rungs:
        out[rung] = {}
        for loss in losses:
            name = f"{rung}_{loss}"
            fp, pp = f"{ld.FEAT}/{name}/features.npy", f"{ld.PRED}/{name}/predictions.npz"
            if not (os.path.isfile(fp) and os.path.isfile(pp)):
                print(f"{P} MISSING {name}", flush=True)
                continue
            n_classes = ld.read_directions(rung, loss).shape[0]
            train_count = np.bincount([json.loads(l)[ld.LABEL] for l in open(f"{ld.DATA}/{rung}_train.json")
                                       if l.strip()], minlength=n_classes)
            x, y = load(fp, pp, device)
            fv, counts, meta, _ = fisher_variants(x, y, n_classes)
            expect = prior.get(rung, {}).get(loss, {}).get("fisher", {}).get("mean")
            out[rung][loss] = report(f"ladder {name}", fv, counts, meta, train_count, ld.BANDS, expect)
            del x, y
            torch.cuda.empty_cache()
    tag = "_".join(losses)
    path = f"{ld.FEAT}/analysis/ladder_fisher_corrected_{tag}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"{P} wrote {path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["scale", "cone", "ladder"])
    ap.add_argument("--losses", nargs="*", default=["dbm", "kappaface", "arcface"])
    ap.add_argument("--rungs", nargs="*", default=["n15k", "n32k", "n64k"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    pm.RUNS["arcface_s15"] = ("SWIN_BASE_ARCFACE_S15_384", "arcface")
    pm.RUNS["cosine_m0_s15"] = ("SWIN_BASE_COSINE_M0_S15_384", "registry")
    if "scale" in args.parts:
        tc = np.bincount([json.loads(l)[pm.LABEL] for l in open(pm.TRAIN_JSON) if l.strip()], minlength=pm.N)
        part_scale(tc, device)
    if "cone" in args.parts:
        part_cone(device)
    if "ladder" in args.parts:
        part_ladder(args.rungs, args.losses, device)


if __name__ == "__main__":
    main()
