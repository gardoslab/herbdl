"""Representation geometry of the loss models: compactness angle, bias-corrected Fisher, compactness split.

Measures (all on L2-normalised pooled features, per species, averaged over species with >= 2 images in
the split, overall and by band of training images per species):
  angle        mean angle, in degrees, between an image's feature and the classifier weight of its true
               species (the DBM paper's compactness; with sub-centres the closest one counts);
  Fisher       pairwise Fisher score J_ij = (mu_i - mu_j)^T A (mu_i - mu_j), A = (Sw + ridge)^-1 with a
               pooled within-species covariance, S_i = mean_{j != i} J_ij. The species means are
               estimated, so E[J_hat] = J + tr(A Sigma_i)/n_i + tr(A Sigma_j)/n_j, a positive bias that is
               larger on validation (fewer images per species) and in the rare bands. Reported:
                 fixed      only species present in the split, covariance divided by n - C, ridge 1e-3 of
                            the mean variance;
                 corrected  fixed, minus the bias with tr(A Sigma_i) ~ d_eff = tr(A Sw): S_i - b_i -
                            mean_{j != i} b_j, b_i = d_eff / n_i. This is the number the paper reports.
  compactness  E[cos(f, w_i)] = |m_i| * cos(m_i, w_i) = R_i * alignment_i: cos_w (mean cosine to the weight),
               R2 (unbiased squared mean resultant length = within-species spread, from the mean pairwise
               within-species cosine) and align = mean cos_w / sqrt(mean R2) (weight-centre alignment).
plus macro-F1 and error from the saved predictions, and the change from training to validation images.
The uncorrected ("original", covariance / n, all species) Fisher is kept only in `cone`, whose table
reports original and corrected side by side.

Subcommands:
  base       the eight base-data SWIN losses on validation (PHASE3_GEOM) and training (PHASE3_GEOM_TRAIN)
             images; classifier rows from each model.safetensors. Train vs validation by band is read off
             this output (angle_deg and fisher per split, train_to_val_change_pct).
             Output MARGIN_DIAG/geometry_base.json; prefix __GEOROB__.
  scale      logit-scale control, s=15 vs s=30 for NormFace and ArcFace, validation images, Fisher only.
             Output MARGIN_DIAG/geometry_robustness2_scale.json; prefix __GEOROB2__.
  cone       pairwise Fisher by taxonomic relation of the pair (same genus / same family other genus / other
             family), nearest same-genus species and nearest neighbour, original and corrected, for the
             eight base-data losses, validation images. Output MARGIN_DIAG/geometry_robustness2_cone.json.
  backbones  ConvNeXt / DINOv2 base models (--models <name> ...), PHASE4_GEOM{,_TRAIN}/<name> and
             BACKBONES/<name>/model.safetensors; CE and CE+LA use the linear head's row as the weight (bias
             ignored). Output MARGIN_DIAG/backbone_geometry.json (merged with earlier results); prefix
             __BACKGEOM__.
  ladder     every loss and backbone on the ladder rungs, validation (LADDER_GEOM + LADDER_EVAL) and training
             (LADDER_GEOM_TRAIN) images; --rungs --losses --splits. Losses are named <loss> for SWIN-B and
             <convnext|dinov2>_<loss> for the other backbones. Output LADDER_GEOM/analysis/
             ladder_geometry_full.json (merged, so cells can be added later); prefix __LADDERFULL__.
  table      collect the supplement table (macro-F1, accuracy, angle, corrected Fisher on train and val) from
             the outputs above and the saved base predictions; CPU only. Output MARGIN_DIAG/supplement_table.json;
             prefix __SUPPTABLE__.

Usage (from finetuning/SWIN):
  python loss_ablation/analysis/geometry.py base [--models cosine_m0 dbm ...] [--splits val train]
  python loss_ablation/analysis/geometry.py scale
  python loss_ablation/analysis/geometry.py cone
  python loss_ablation/analysis/geometry.py backbones --models convnext_dbm dinov2_arcface
  python loss_ablation/analysis/geometry.py ladder --rungs n15k --losses dbm arcface --splits val train
  python loss_ablation/analysis/geometry.py table
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BANDS, BASE_MODELS, DIAG, LABEL, LADDER_BANDS, LADDER_DATA, LADDER_EVAL,  # noqa: E402
                    LADDER_GEOM, LADDER_GEOM_TRAIN, N, NAMES, OUT, PHASE3_GEOM, PHASE3_GEOM_TRAIN,
                    PHASE4_GEOM, PHASE4_GEOM_TRAIN, RUNS, class_directions, classifier, ladder_classifier,
                    load, load_taxonomy, load_train_counts, macro_f1_labels, split_backbone)

RIDGE = 1e-3
CHUNK = 4096
STEP = 65536
POOL_CHUNK = 65536  # chunk of pooled() and compactness()


def angle_stats(cos, mask, n):
    """Mean/median/std of arccos(cos), in degrees, over one slice."""
    th = np.degrees(np.arccos(np.clip(np.asarray(cos, dtype=np.float64), -1.0, 1.0)))
    sl = th[mask]
    return {"n_images": int(mask.sum()),
            "mean_deg": float(sl.mean()),
            "median_deg": float(np.median(sl)),
            "std_deg": float(sl.std())}


# ---------------------------------------------------------------- bias-corrected Fisher
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


def fisher_variants(x, y, n_species):
    """S_i arrays (NaN where a species is absent) for fixed / corrected, plus meta."""
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


# ---------------------------------------------------------------- compactness split
def pooled(x, y, n_species, divide_by_dof=True):
    """Species sums/means/counts and the pooled within-species covariance (no ridge)."""
    d = x.size(1)
    counts = torch.bincount(y, minlength=n_species).double()
    sums = torch.zeros(n_species, d, device=x.device, dtype=torch.float64)
    for s in range(0, len(y), POOL_CHUNK):  # chunked: a 2M x 1024 double copy would not fit next to x
        sums.index_add_(0, y[s:s + POOL_CHUNK], x[s:s + POOL_CHUNK].double())
    mu = (sums / counts.clamp(min=1)[:, None]).float()
    Sw = torch.zeros(d, d, device=x.device, dtype=torch.float64)
    for s in range(0, len(y), POOL_CHUNK):
        c = (x[s:s + POOL_CHUNK] - mu[y[s:s + POOL_CHUNK]]).double()
        Sw += c.T @ c
    present = int((counts > 0).sum())
    Sw /= (len(y) - present) if divide_by_dof else len(y)
    return sums, mu, counts, Sw.float()


def compactness(x, y, w, n_species=N):
    """Per-species cos_w, R2, align_naive, loo_cos, in_cos (NaN where n_i < 2).

    The species count is an argument so the same code serves the 15.5k base data and the ladder rungs.
    """
    sums, mu, counts, _ = pooled(x, y, n_species, divide_by_dof=False)
    n = counts
    # cos to the weight, closest sub-centre
    cos_w_img = torch.empty(len(y), device=x.device)
    for s in range(0, len(y), POOL_CHUNK):
        cos_w_img[s:s + POOL_CHUNK] = torch.einsum("bc,bkc->bk", x[s:s + POOL_CHUNK], w[y[s:s + POOL_CHUNK]]).max(1).values
    cos_w = torch.zeros(n_species, device=x.device, dtype=torch.float64).index_add_(0, y, cos_w_img.double()) / n.clamp(min=1)
    R2 = ((sums * sums).sum(1) - n) / (n * (n - 1)).clamp(min=1)
    mu_hat = F.normalize(sums.float(), dim=1)
    align_naive = torch.einsum("nc,nkc->nk", mu_hat, w).max(1).values.double()
    # leave-one-out and in-sample centroid cosines per image
    loo = torch.zeros(n_species, device=x.device, dtype=torch.float64)
    ins = torch.zeros(n_species, device=x.device, dtype=torch.float64)
    for s in range(0, len(y), POOL_CHUNK):
        xs, ys = x[s:s + POOL_CHUNK].double(), y[s:s + POOL_CHUNK]
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


# ---------------------------------------------------------------- one (model, split) cell
def measure(x, y, w, y_np, p_np, train_count, bands, n_classes, tag, prefix):
    """Angle to the weight, corrected Fisher, compactness split, error and macro-F1 of one cell.

    x, y  normalised features and labels (device tensors); w (n_classes, k, C) unit classifier rows;
    y_np, p_np  labels and saved predictions; train_count  training images per species (length n_classes).
    """
    n = len(y_np)
    cos_true = torch.empty(n, device=x.device)
    for s in range(0, n, 65536):
        cos_true[s:s + 65536] = torch.einsum("bc,bkc->bk", x[s:s + 65536], w[y[s:s + 65536]]).max(1).values
    cos_true = cos_true.cpu().numpy()
    tc = train_count[y_np]
    angle = {"overall": angle_stats(cos_true, np.ones(n, bool), n)["mean_deg"]}
    for lo, hi in bands:
        m = (tc >= lo) & (tc <= hi)
        angle[f"{lo}-{hi}"] = angle_stats(cos_true, m, n)["mean_deg"] if m.any() else None
    fv, counts, meta, _ = fisher_variants(x, y, n_classes)
    ok = counts >= 2
    fisher = {k: summarise(v, train_count, bands, ok) for k, v in fv.items()}
    comp = compactness(x, y, w, n_classes)
    cs = {k: summarise(v, train_count, bands, ok) for k, v in comp.items() if k in ("cos_w", "R2")}
    cs["align"] = {b: (cs["cos_w"][b] / np.sqrt(cs["R2"][b]) if cs["R2"][b] and cs["R2"][b] > 0 else None)
                   for b in cs["R2"]}
    res = {"n_images": n, "n_classes": n_classes, "feature_dim": int(x.size(1)),
           "error_pct": float(100 * np.mean(p_np != y_np)),
           "macro_f1": macro_f1_labels(y_np, p_np), "angle_deg": angle, "fisher": fisher, "fisher_meta": meta,
           "compactness": cs}
    print(f"{prefix} {tag}: n={n:,} dim={res['feature_dim']} err={res['error_pct']:.2f}% "
          f"macroF1={res['macro_f1']:.4f} angle={angle['overall']:.1f}deg "
          f"Fisher fixed={fisher['fixed']['all']:.0f} corrected={fisher['corrected']['all']:.0f} "
          f"(d_eff={meta['d_eff']:.0f}, bias {meta['mean_bias']:.0f}) R2={cs['R2']['all']:.3f} "
          f"align={cs['align']['all']:.3f}", flush=True)
    return res


def train_to_val_change(val, train):
    """Percent change of the corrected Fisher from training to validation images, overall and by band."""
    v, t = val["fisher"]["corrected"], train["fisher"]["corrected"]
    return {b: (100 * (v[b] / t[b] - 1) if v[b] and t[b] else None) for b in v}


# ---------------------------------------------------------------- base
BASE_RESULT = f"{DIAG}/geometry_base.json"
P_BASE = "__GEOROB__"


def main_base(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_count = load_train_counts()
    print(f"{P_BASE} device={device}", flush=True)
    roots = {"val": PHASE3_GEOM, "train": PHASE3_GEOM_TRAIN}
    out = {}
    for name in args.models:
        w = F.normalize(class_directions(*RUNS[name]).to(device), dim=-1)
        out[name] = {}
        for split in args.splits:
            fp, pp = f"{roots[split]}/{name}/features.npy", f"{roots[split]}/{name}/predictions.npz"
            if not (os.path.isfile(fp) and os.path.isfile(pp)):
                print(f"{P_BASE} MISSING {name} {split}", flush=True)
                continue
            z = np.load(pp)
            y_np, p_np = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
            x, y = load(fp, pp, device)
            out[name][split] = measure(x, y, w, y_np, p_np, train_count, BANDS, N, f"{name} {split}", P_BASE)
            del x, y
            if device == "cuda":
                torch.cuda.empty_cache()
        if "val" in out[name] and "train" in out[name]:
            out[name]["train_to_val_fisher_change_pct"] = train_to_val_change(out[name]["val"], out[name]["train"])
            print(f"{P_BASE} {name} train->val corrected Fisher "
                  f"{out[name]['train_to_val_fisher_change_pct']['all']:+.1f}% | angle "
                  f"{out[name]['train']['angle_deg']['overall']:.1f} -> {out[name]['val']['angle_deg']['overall']:.1f} deg",
                  flush=True)
    os.makedirs(os.path.dirname(BASE_RESULT), exist_ok=True)
    json.dump(out, open(BASE_RESULT, "w"), indent=1)
    print(f"{P_BASE} wrote {BASE_RESULT}", flush=True)


# ---------------------------------------------------------------- scale control
P = "__GEOROB2__"
SCALE_MODELS = [("cosine_m0", "NormFace s=30"), ("cosine_m0_s15", "NormFace s=15"),
                ("arcface", "ArcFace s=30"), ("arcface_s15", "ArcFace s=15")]


def report(tag, fv, counts, meta, band_count, bands):
    ok = counts >= 2
    res = {"meta": meta, "n_species_ge2": int(ok.sum()),
           "fisher": {k: summarise(v, band_count, bands, ok) for k, v in fv.items()}}
    f = res["fisher"]
    print(f"{P} {tag}: d_eff={meta['d_eff']:.0f} bias={meta['mean_bias']:.0f} "
          f"fixed={f['fixed']['all']:.0f} corrected={f['corrected']['all']:.0f} | corrected by band: "
          + " ".join(f"{b}={v:.0f}" for b, v in f["corrected"].items() if b != "all" and v is not None),
          flush=True)
    return res


def part_scale(train_count, device):
    out = {}
    for name, label in SCALE_MODELS:
        fp, pp = f"{PHASE3_GEOM}/{name}/features.npy", f"{PHASE3_GEOM}/{name}/predictions.npz"
        if not (os.path.isfile(fp) and os.path.isfile(pp)):
            print(f"{P} MISSING {name}", flush=True)
            continue
        x, y = load(fp, pp, device)
        fv, counts, meta, _ = fisher_variants(x, y, N)
        out[name] = report(f"scale {label}", fv, counts, meta, train_count, BANDS)
        del x, y
        torch.cuda.empty_cache()
    os.makedirs(DIAG, exist_ok=True)
    json.dump(out, open(f"{DIAG}/geometry_robustness2_scale.json", "w"), indent=1)
    return out


def main_scale(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    part_scale(load_train_counts(), device)


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
    counts, cls_fam, cls_gen = load_taxonomy()
    out = {}
    for m in BASE_MODELS:
        x, y = load(f"{PHASE3_GEOM}/{m}/features.npy", f"{PHASE3_GEOM}/{m}/predictions.npz", device)
        mu, cnt, scatter = split_stats(x, y, N)
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

        D_orig, _ = J(scatter / n)  # paper: covariance / n (present species only)
        Sw = scatter / (n - C)
        D_fix, A = J(Sw)
        d_eff = float(torch.trace(A.double() @ Sw))
        bias = d_eff * (1.0 / n_i[:, None] + 1.0 / n_i[None, :])
        res = {"n_species": C, "d_eff": d_eff,
               "original": pair_groups(D_orig, gi, fi),
               "corrected": pair_groups(D_fix - bias, gi, fi)}
        out[m] = res
        o, c = res["original"], res["corrected"]
        print(f"{P} cone {NAMES[m]}: same genus {o['same_genus']['mean_J']:.0f}->{c['same_genus']['mean_J']:.0f} | "
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
            print(f"{P} cone {NAMES[m]} / DBM ({kind}): same genus x{ratios['same_genus']:.2f}, same family "
                  f"x{ratios['same_family_other_genus']:.2f}, other family x{ratios['other_family']:.2f}, "
                  f"nearest sister x{ratios['nearest_sister']:.2f}", flush=True)
    json.dump(out, open(f"{DIAG}/geometry_robustness2_cone.json", "w"), indent=1)
    return out


def main_cone(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    os.makedirs(DIAG, exist_ok=True)
    part_cone(device)


# ---------------------------------------------------------------- backbones
BACKBONE_RESULT = f"{DIAG}/backbone_geometry.json"
P_BACK = "__BACKGEOM__"
BACKBONE_GEOM = {"val": PHASE4_GEOM, "train": PHASE4_GEOM_TRAIN}


def backbone_cell(name, split, train_count, device):
    fp, pp = f"{BACKBONE_GEOM[split]}/{name}/features.npy", f"{BACKBONE_GEOM[split]}/{name}/predictions.npz"
    if not (os.path.isfile(fp) and os.path.isfile(pp)):
        print(f"{P_BACK} MISSING {name} {split}", flush=True)
        return None
    z = np.load(pp)
    y_np, p_np = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    x, y = load(fp, pp, device)
    w, key = classifier(name, device)
    res = measure(x, y, w, y_np, p_np, train_count, BANDS, N, f"{name} {split}", P_BACK)
    res["classifier_key"] = key
    del x, y, w
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def main_backbones(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P_BACK} device={device}", flush=True)
    train_count = load_train_counts()
    out = json.load(open(BACKBONE_RESULT)) if os.path.isfile(BACKBONE_RESULT) else {}
    for name in args.models:
        r = {s: backbone_cell(name, s, train_count, device) for s in ("val", "train")}
        if r["val"] and r["train"]:
            r["train_to_val_fisher_change_pct"] = train_to_val_change(r["val"], r["train"])
            print(f"{P_BACK} {name} train->val corrected Fisher {r['train_to_val_fisher_change_pct']['all']:+.1f}% | angle "
                  f"{r['train']['angle_deg']['overall']:.1f} -> {r['val']['angle_deg']['overall']:.1f} deg", flush=True)
        out[name] = r
        os.makedirs(os.path.dirname(BACKBONE_RESULT), exist_ok=True)
        json.dump(out, open(BACKBONE_RESULT, "w"), indent=1)
    print(f"{P_BACK} wrote {BACKBONE_RESULT}", flush=True)


# ---------------------------------------------------------------- ladder
LADDER_RESULT = f"{LADDER_GEOM}/analysis/ladder_geometry_full.json"
P_LAD = "__LADDERFULL__"
LADDER_PATHS = {"val": (LADDER_GEOM, LADDER_EVAL), "train": (LADDER_GEOM_TRAIN, LADDER_GEOM_TRAIN)}


def cell_name(rung, loss):
    """Feature folder name: <rung>_<loss> for SWIN, <backbone>_<rung>_<loss> for losses given as <backbone>_<loss>."""
    bb, l = split_backbone(loss)
    return f"{bb}_{rung}_{l}" if bb else f"{rung}_{loss}"


def ladder_cell(rung, loss, split, train_count, device):
    fdir, pdir = LADDER_PATHS[split]
    name = cell_name(rung, loss)
    if split_backbone(loss)[0]:
        pdir = fdir  # the ConvNeXt feature passes save predictions next to the features
    fp, pp = f"{fdir}/{name}/features.npy", f"{pdir}/{name}/predictions.npz"
    if not (os.path.isfile(fp) and os.path.isfile(pp)):
        print(f"{P_LAD} MISSING {name} {split}", flush=True)
        return None
    w = ladder_classifier(rung, loss, device)
    n_classes = w.size(0)
    train_count = np.pad(train_count[:n_classes], (0, max(0, n_classes - len(train_count))))
    z = np.load(pp)
    y_np, p_np = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    x, y = load(fp, pp, device)
    res = measure(x, y, w, y_np, p_np, train_count, LADDER_BANDS, n_classes, f"{rung}_{loss} {split}", P_LAD)
    del x, y, w
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def main_ladder(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P_LAD} device={device}", flush=True)
    out = json.load(open(LADDER_RESULT)) if os.path.isfile(LADDER_RESULT) else {}
    for rung in args.rungs:
        train_count = None
        for loss in args.losses:
            if train_count is None:
                train_count = load_train_counts(f"{LADDER_DATA}/{rung}_train.json", minlength=0)
            entry = out.setdefault(rung, {}).setdefault(loss, {})
            for split in args.splits:
                r = ladder_cell(rung, loss, split, train_count, device)
                if r is not None:
                    entry[split] = r
            if "val" in entry and "train" in entry:
                entry["train_to_val_fisher_change_pct"] = train_to_val_change(entry["val"], entry["train"])
                print(f"{P_LAD} {rung}_{loss} train->val corrected Fisher "
                      f"{entry['train_to_val_fisher_change_pct']['all']:+.1f}% | angle "
                      f"{entry['train']['angle_deg']['overall']:.1f} -> {entry['val']['angle_deg']['overall']:.1f} deg",
                      flush=True)
            os.makedirs(os.path.dirname(LADDER_RESULT), exist_ok=True)
            json.dump(out, open(LADDER_RESULT, "w"), indent=1)
    print(f"{P_LAD} wrote {LADDER_RESULT}", flush=True)


# ---------------------------------------------------------------- supplement table
SWIN_BASE = ["cosine_m0", "dbm", "arcface_m01", "kappaface", "subcenter_arcface", "arcface", "arcface_ecc"]
LADDER = ["cosine_m0", "dbm", "kappaface", "arcface", "ce"]
BACKBONE = ["ce", "ce_la", "cosine_m0", "dbm", "arcface"]


def load_json(path):
    return json.load(open(path)) if os.path.isfile(path) else {}


def f1_acc(path):
    if not os.path.isfile(path):
        return None, None
    z = np.load(path)
    y, p = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    return macro_f1_labels(y, p), float(np.mean(y == p))


def cell(f1=None, acc=None, angle=None, fisher=None):
    return {"macro_f1": f1, "accuracy": acc, "angle_deg": angle, "fisher": fisher}


def main_table(args):
    rows = []
    gb = load_json(BASE_RESULT)
    for m in SWIN_BASE:
        e = gb.get(m, {})
        r = {"setting": "SWIN-B, 15.5k (base)", "loss": m}
        for s, root in (("train", PHASE3_GEOM_TRAIN), ("val", PHASE3_GEOM)):
            c = e.get(s)
            f1, acc = f1_acc(f"{root}/{m}/predictions.npz")
            r[s] = cell(f1, acc, c["angle_deg"]["overall"], c["fisher"]["corrected"]["all"]) if c else cell(f1, acc)
        rows.append(r)

    lg = load_json(LADDER_RESULT)
    for rung, label in (("n15k", "15.5k"), ("n32k", "32k"), ("n64k", "64k")):
        for m in LADDER:
            e = lg.get(rung, {}).get(m, {})
            r = {"setting": f"SWIN-B, ladder {label}", "loss": m}
            for s in ("train", "val"):
                c = e.get(s)
                r[s] = cell(c["macro_f1"], 1 - c["error_pct"] / 100, c["angle_deg"]["overall"],
                            c["fisher"]["corrected"]["all"]) if c else cell()
            rows.append(r)

    bg = load_json(BACKBONE_RESULT)
    for bb, label in (("convnext", "ConvNeXt-B, 15.5k (base)"), ("dinov2", "DINOv2-B, 15.5k (base)")):
        for m in BACKBONE:
            e = bg.get(f"{bb}_{m}", {})
            r = {"setting": label, "loss": m}
            for s in ("train", "val"):
                c = e.get(s)
                r[s] = cell(c["macro_f1"], 1 - c["error_pct"] / 100, c["angle_deg"]["overall"],
                            c["fisher"]["corrected"]["all"]) if c else cell()
            rows.append(r)

    for bb, label in (("convnext", "ConvNeXt-B"), ("dinov2", "DINOv2-B")):
        for rung, rl in (("n32k", "32k"), ("n64k", "64k")):
            for m in ("cosine_m0", "dbm", "arcface"):  # filled as each model's geometry exists
                e = lg.get(rung, {}).get(f"{bb}_{m}", {})
                r = {"setting": f"{label}, ladder {rl}", "loss": m}
                for s in ("train", "val"):
                    c = e.get(s)
                    r[s] = cell(c["macro_f1"], 1 - c["error_pct"] / 100, c["angle_deg"]["overall"],
                                c["fisher"]["corrected"]["all"]) if c else cell()
                rows.append(r)

    json.dump(rows, open(f"{DIAG}/supplement_table.json", "w"), indent=1)
    filled = sum(1 for r in rows for s in ("train", "val") if r[s]["fisher"] is not None)
    print(f"__SUPPTABLE__ {len(rows)} rows, {filled} of {2 * len(rows)} split cells filled -> {DIAG}/supplement_table.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("base")
    p.add_argument("--models", nargs="+", default=BASE_MODELS)
    p.add_argument("--splits", nargs="+", default=["val", "train"])
    p.set_defaults(fn=main_base)
    sub.add_parser("scale").set_defaults(fn=main_scale)
    sub.add_parser("cone").set_defaults(fn=main_cone)
    p = sub.add_parser("backbones")
    p.add_argument("--models", nargs="+", required=True)
    p.set_defaults(fn=main_backbones)
    p = sub.add_parser("ladder")
    p.add_argument("--rungs", nargs="+", default=["n15k", "n32k", "n64k"])
    p.add_argument("--losses", nargs="+", default=["cosine_m0", "dbm", "kappaface", "arcface", "ce"])
    p.add_argument("--splits", nargs="+", default=["val", "train"])
    p.set_defaults(fn=main_ladder)
    sub.add_parser("table").set_defaults(fn=main_table)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
