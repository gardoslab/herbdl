"""Results of the sibling-margin model on the 32k ladder rung, against NormFace and DBM.

Three questions, all on the 32k validation set (417,176 images), CPU only:

1. Per band of training images (1, 2-5, ..., 101+): mean per-species F1 of the sibling-margin
   model, NormFace (cosine head, m=0, saved as cosine_m0) and DBM, and the paired leads
   sibling-NormFace and sibling-DBM in F1 points with 95% bootstrap intervals over species.
   Same per-species F1 definition as cosine_band_compare.py, so the NormFace and DBM numbers
   must reproduce that script's 32k rows.
2. Where errors land, per band and model: share of wrong predictions in the true genus,
   in the true family but another genus, or in another family; and genus accuracy (the
   predicted species is in the true genus, right or wrong species).
3. Margin distribution.
   a. Designed margin m_y = m_max * n_min / n_y per band (the rule in sibling_margin.py).
   b. Realised gap on validation images: cos(f, w_y) - max over genus siblings j of
      cos(f, w_j), from the saved pooled features and the head rows of each model. Positive
      means the true species beats every sibling. Quantiles per band and model, and the share
      of images with gap >= m_y (the designed margin is met).
   c. Applied margins during training: the __SIBLING__ lines of the four 12-hour training
      segments, read from the full run logs on SCC (orx keeps only a short copy).

Inputs (SCC): LADDER_EVAL/n32k_<model>/predictions.npz, LADDER_GEOM/n32k_<model>/features.npy,
the model.safetensors of each training run, data/ladder/n32k_train.json.
Output: MARGIN_DIAG/sibling_results.json; printed lines start with __SIBRES__.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sibling_margin import species_margin, taxonomy_arrays  # noqa: E402

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
TRAIN = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder/n32k_train.json"
RUNS = "/projectnb/proactiveai/workspaces/herb/faridkar/.orx/runs"  # runDir in the orx submission handles
RESULT = f"{OUT}/MARGIN_DIAG/sibling_results.json"
PREFIX = "__SIBRES__"
LABEL = "scientificNameEncoded"
M_MAX = 0.15
N_BOOT, SEED, CHUNK = 1000, 0, 4096

# model -> (LADDER_EVAL / LADDER_GEOM folder, training run folder)
MODELS = {
    "normface": ("n32k_cosine_m0", "SWIN_BASE_LADDER_N32K_COSINE_M0"),
    "dbm": ("n32k_dbm", "SWIN_BASE_LADDER_N32K_DBM"),
    "sibling": ("n32k_sibling", "SWIN_BASE_LADDER_N32K_SIBLING_R3"),
}
PAIRS = [("sibling", "normface"), ("sibling", "dbm"), ("dbm", "normface")]
BANDS = [(1, 1), (2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 100), (101, 10 ** 6)]
# The four training segments, in order, with the optimizer step each resumed from.
SEGMENTS = [("0f48263b-fae6-4458-8355-387be310af52", 0),
            ("4a61187f-56c3-409f-b284-c15a2f541e18", 32580),
            ("260ebf80-b0ae-490d-a0d0-743c5843758f", 71676),
            ("bd6b6321-f7d5-45d2-9f7e-7aaaf60c2044", 104256)]
GRAD_ACCUM = 2  # head forward calls per optimizer step on rank 0
SAME_GENUS, SAME_FAMILY, OTHER_FAMILY = 0, 1, 2


def band_name(lo, hi):
    return f"{lo}+" if hi >= 10 ** 6 else (str(lo) if lo == hi else f"{lo}-{hi}")


def per_species_f1(y, p, n):
    """F1 of every class; classes absent from both labels and predictions are NaN."""
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    n_true = np.bincount(y, minlength=n).astype(float)
    n_pred = np.bincount(p, minlength=n).astype(float)
    denom = n_true + n_pred
    f1 = np.full(n, np.nan)
    ok = denom > 0
    f1[ok] = 2 * tp[ok] / denom[ok]
    return f1, n_true


def boot_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    b = d[idx].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def load_taxonomy():
    with open(TRAIN) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    sp = np.array([r[LABEL] for r in rows])
    n = int(sp.max()) + 1
    gid, fid = taxonomy_arrays(sp, [r.get("genus") or "" for r in rows], [r["family"] for r in rows], n)
    count = np.bincount(sp, minlength=n)
    return gid, fid, count


def relation(y, p, gid, fid):
    """Relation of predicted to true species for wrong predictions."""
    r = np.full(len(y), OTHER_FAMILY)
    r[fid[p] == fid[y]] = SAME_FAMILY
    r[gid[p] == gid[y]] = SAME_GENUS
    return r


def head_rows(run_dir):
    with safe_open(f"{OUT}/{run_dir}/model.safetensors", "pt") as f:
        keys = set(f.keys())
        name = "head.weight" if "head.weight" in keys else "arcface.weight"
        return F.normalize(f.get_tensor(name).float(), dim=1)


def sibling_gaps(feat_path, run_dir, y, gid):
    """Per image: cos to own row minus the best genus-sibling cos (NaN if no sibling),
    and cos to own row minus the best non-sibling cos."""
    feats = np.load(feat_path, mmap_mode="r")
    if len(feats) != len(y):
        raise ValueError(f"{feat_path}: {len(feats)} features for {len(y)} labels")
    W = head_rows(run_dir)
    g = torch.as_tensor(gid)
    gap_sib = np.full(len(y), np.nan, dtype=np.float32)
    gap_far = np.full(len(y), np.nan, dtype=np.float32)
    for s in range(0, len(y), CHUNK):
        f = F.normalize(torch.as_tensor(np.asarray(feats[s:s + CHUNK], dtype=np.float32)), dim=1)
        yy = torch.as_tensor(y[s:s + CHUNK])
        cos = f @ W.T  # (B, N)
        own = cos.gather(1, yy[:, None]).squeeze(1)
        cos.scatter_(1, yy[:, None], -2.0)  # never its own competitor
        sib = g[None, :] == g[yy][:, None]
        best_sib = cos.masked_fill(~sib, -2.0).max(1).values
        best_far = cos.masked_fill(sib, -2.0).max(1).values
        has = best_sib > -1.5
        gs = (own - best_sib).numpy()
        gs[~has.numpy()] = np.nan
        gap_sib[s:s + CHUNK] = gs
        gap_far[s:s + CHUNK] = (own - best_far).numpy()
    return gap_sib, gap_far


def parse_sibling_logs():
    """All __SIBLING__ lines of the training segments, with an approximate optimizer step."""
    out = []
    for run_id, start in SEGMENTS:
        path = f"{RUNS}/{run_id}/log"
        if not os.path.isfile(path):
            hits = glob.glob(f"{RUNS}/{run_id}*/log")
            path = hits[0] if hits else None
        if path is None:
            print(f"{PREFIX} MISSING log of run {run_id}", flush=True)
            continue
        with open(path, errors="replace") as f:
            text = f.read().replace("\r", "\n")
        n_seg = 0
        for line in re.findall(r"__SIBLING__[^\n]*", text):
            kv = dict(re.findall(r"(\w+(?:_ge_[\d.]+)?)=([-\d.eE+naif]+)", line))
            try:
                rec = {k: float(v) for k, v in kv.items()}
            except ValueError:
                continue
            rec["segment"] = run_id[:8]
            rec["step"] = start + (rec.get("call", 1) - 1) / GRAD_ACCUM
            out.append(rec)
            n_seg += 1
        print(f"{PREFIX} log {run_id[:8]}: {n_seg} __SIBLING__ lines", flush=True)
    return out


def q(a, qs=(10, 25, 50, 75, 90)):
    a = a[~np.isnan(a)]
    return {f"p{k}": float(np.percentile(a, k)) for k in qs} if len(a) else {}


def main():
    rng = np.random.default_rng(SEED)
    gid, fid, count = load_taxonomy()
    n = len(count)
    m_y = species_margin(count, M_MAX)
    n_sib = np.bincount(gid, minlength=gid.max() + 1)[gid] - 1
    print(f"{PREFIX} taxonomy: {n} species, {gid.max() + 1} genera, {fid.max() + 1} families, "
          f"{(n_sib > 0).mean():.3f} of species have a sibling", flush=True)

    preds, y = {}, None
    for m, (folder, _) in MODELS.items():
        z = np.load(f"{OUT}/LADDER_EVAL/{folder}/predictions.npz")
        yy = z["labels"].astype(np.int64)
        if y is None:
            y = yy
        assert np.array_equal(y, yy), f"labels differ for {m}"
        preds[m] = z["predictions"].astype(np.int64)
    res = {"n_val_images": int(len(y)), "macro_f1": {}, "accuracy": {}, "bands": {}, "overall": {}}

    # ---- 1. per-band F1 and leads
    f1 = {}
    for m, p in preds.items():
        f1[m], n_true = per_species_f1(y, p, n)
        res["macro_f1"][m] = float(np.nanmean(f1[m]))
        res["accuracy"][m] = float((p == y).mean())
    in_val = n_true > 0
    print(f"{PREFIX} macro-F1 " + " ".join(f"{m}={v:.4f}" for m, v in res["macro_f1"].items())
          + " | accuracy " + " ".join(f"{m}={v:.4f}" for m, v in res["accuracy"].items()), flush=True)
    for a, b in PAIRS:
        d = (f1[a] - f1[b])[in_val]
        ci = [100 * v for v in boot_ci(d, rng)]
        res["overall"][f"{a}-{b}"] = {"lead": float(100 * d.mean()), "ci": ci}
        print(f"{PREFIX} overall {a}-{b}: {100 * d.mean():+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] F1 points", flush=True)
    # species with and without a genus sibling
    for lab, mask in (("with_sibling", in_val & (n_sib > 0)), ("no_sibling", in_val & (n_sib == 0))):
        e = {"n_species": int(mask.sum())}
        for a, b in PAIRS:
            d = (f1[a] - f1[b])[mask]
            e[f"{a}-{b}"] = {"lead": float(100 * d.mean()), "ci": [100 * v for v in boot_ci(d, rng)]}
        res["overall"][lab] = e
        print(f"{PREFIX} {lab} (n={mask.sum():,}): " + " | ".join(
            f"{k} {v['lead']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]" for k, v in e.items() if k != "n_species"),
            flush=True)

    # ---- 2. where errors land, per band (by the true species' training count)
    cnt_img = count[y]
    rel = {m: relation(y, p, gid, fid) for m, p in preds.items()}

    # ---- 3b. realised sibling gaps
    gaps = {}
    for m, (folder, run_dir) in MODELS.items():
        fp = f"{OUT}/LADDER_GEOM/{folder}/features.npy"
        if not (os.path.isfile(fp) and os.path.isfile(f"{OUT}/{run_dir}/model.safetensors")):
            print(f"{PREFIX} MISSING features or weights for {m}", flush=True)
            continue
        gaps[m] = sibling_gaps(fp, run_dir, y, gid)
        # check: the argmax of the recomputed cosines must give the stored prediction for most images
        gs, gf = gaps[m]
        right = (np.nan_to_num(gs, nan=1.0) > 0) & (gf > 0)
        print(f"{PREFIX} {m}: recomputed top-1 accuracy {right.mean():.4f} (stored {res['accuracy'][m]:.4f})", flush=True)
        res.setdefault("gap_check", {})[m] = {"recomputed_acc": float(right.mean())}

    for lo, hi in BANDS:
        bn = band_name(lo, hi)
        ms = in_val & (count >= lo) & (count <= hi)
        mi = (cnt_img >= lo) & (cnt_img <= hi)
        if not ms.any():
            continue
        e = {"n_species": int(ms.sum()), "n_images": int(mi.sum()),
             "designed_margin": q(m_y[ms]), "f1": {m: float(np.nanmean(f1[m][ms])) for m in preds},
             "leads": {}, "errors": {}, "gaps": {}}
        for a, b in PAIRS:
            d = (f1[a] - f1[b])[ms]
            e["leads"][f"{a}-{b}"] = {"lead": float(100 * d.mean()), "ci": [100 * v for v in boot_ci(d, rng)]}
        for m, p in preds.items():
            wrong = mi & (p != y)
            r = rel[m][wrong]
            e["errors"][m] = {"error_rate": float(wrong.sum() / mi.sum()),
                              "same_genus": float((r == SAME_GENUS).mean()) if wrong.any() else float("nan"),
                              "same_family": float((r == SAME_FAMILY).mean()) if wrong.any() else float("nan"),
                              "other_family": float((r == OTHER_FAMILY).mean()) if wrong.any() else float("nan"),
                              "genus_accuracy": float((gid[p[mi]] == gid[y[mi]]).mean())}
        for m, (gs, gf) in gaps.items():
            v = gs[mi]
            ok = ~np.isnan(v)
            e["gaps"][m] = {"n_with_sibling": int(ok.sum()), **q(v),
                            "share_positive": float((v[ok] > 0).mean()) if ok.any() else float("nan"),
                            "share_meets_designed_margin": float((v[ok] >= m_y[y[mi]][ok]).mean()) if ok.any() else float("nan")}
        res["bands"][bn] = e
        print(f"{PREFIX} band {bn:>6} (species={ms.sum():>6,}, images={mi.sum():>7,}, m_y median {np.median(m_y[ms]):.4f}): "
              + " ".join(f"{m}={e['f1'][m]:.4f}" for m in preds) + " | "
              + " | ".join(f"{k} {v['lead']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]" for k, v in e["leads"].items()),
              flush=True)
        print(f"{PREFIX}   errors {bn:>6}: " + " | ".join(
            f"{m} err={v['error_rate']:.4f} genus={v['same_genus']:.3f} fam={v['same_family']:.3f} "
            f"far={v['other_family']:.3f} genus_acc={v['genus_accuracy']:.4f}" for m, v in e["errors"].items()), flush=True)
        if gaps:
            print(f"{PREFIX}   gaps   {bn:>6}: " + " | ".join(
                f"{m} p10={v.get('p10', float('nan')):+.3f} p50={v.get('p50', float('nan')):+.3f} "
                f"p90={v.get('p90', float('nan')):+.3f} pos={v['share_positive']:.3f} met={v['share_meets_designed_margin']:.3f}"
                for m, v in e["gaps"].items()), flush=True)

    # gap histograms for plotting (all images with a sibling, per coarse band)
    edges = np.linspace(-0.6, 0.8, 141)
    res["gap_hist"] = {"edges": edges.tolist(), "hist": {}}
    for m, (gs, _) in gaps.items():
        for lab, lo, hi in (("2-5", 2, 5), ("6-25", 6, 25), ("26+", 26, 10 ** 6)):
            mi = (cnt_img >= lo) & (cnt_img <= hi) & ~np.isnan(gs)
            res["gap_hist"]["hist"][f"{m}|{lab}"] = np.histogram(np.clip(gs[mi], -0.6, 0.8), edges)[0].tolist()

    # ---- 3c. applied margins during training
    logs = parse_sibling_logs()
    res["training_log"] = logs
    if logs:
        late = [r for r in logs if r["step"] >= 5000]
        keys = ["top_is_sibling", "top_margin_mean", "m_true_p50", "m_true_p99", "images_margin_ge_0.03",
                "competitors_with_margin_mean", "cos_true_mean"]
        summ = {k: float(np.mean([r[k] for r in late if k in r])) for k in keys if any(k in r for r in late)}
        res["training_log_summary"] = {"n_lines": len(logs), "n_after_step_5000": len(late), "means": summ}
        print(f"{PREFIX} training log: {len(logs)} lines, {len(late)} after step 5,000; means " +
              " ".join(f"{k}={v:.4f}" for k, v in summ.items()), flush=True)
        for lo, hi in ((0, 20000), (20000, 60000), (60000, 100000), (100000, 132000)):
            sel = [r for r in logs if lo <= r["step"] < hi]
            if sel:
                print(f"{PREFIX}   steps {lo:>6}-{hi:<6} n={len(sel):>3}: " + " ".join(
                    f"{k}={np.mean([r[k] for r in sel if k in r]):.4f}" for k in ("top_is_sibling", "top_margin_mean", "cos_true_mean")),
                    flush=True)

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{PREFIX} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
