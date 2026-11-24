"""Everything prediction-based: per-species F1 by rarity band, bootstrap intervals, band leads, error taxonomy.

Subcommands (all read saved per-image predictions; the numbers are macro-F1 = mean per-class F1 over
classes in the labels OR the predictions, as in sklearn 'macro'):

  species   the 15.5k base models (PHASE3_EVAL/<model>/predictions.npz, val_2022 order): accuracy,
            macro-F1 and top-5 recomputed from the predictions and checked against each run's own
            all_results.json; mean per-species F1 within bands of training images per species;
            how often an error is still in the right genus / family; for the key pairs the
            per-species wins/losses and a paired bootstrap over validation images (1000 resamples)
            of the accuracy and macro-F1 gaps. That bootstrap is evaluation-set noise only, not
            seed-to-seed training noise.
            Output: PHASE3_EVAL/analysis/per_species_summary.json and per_species_f1.csv; prefix __P3__.
  ci        paired image-bootstrap intervals for the main table's margin controls
            (DBM - ArcFace, NormFace - ArcFace, ArcFace m=0.1 - ArcFace), same procedure and resamples
            (resample i drawn from rng([SEED, i])) as `species`. Input PHASE3_GEOM/<model>/predictions.npz.
            Output: MARGIN_DIAG/main_table_ci.json; prefix __CI__.
  compare   per-band comparison of DBM / KappaFace / ArcFace against the zero-margin cosine head (NormFace),
            on the base data and the ladder rungs n15k, n32k, n64k: per-species F1 averaged in bands of
            training images per species, paired leads with 95% bootstrap intervals over species, and the
            share of the overall DBM-minus-cosine gap from species with at most 10 training images.
            Inputs: PHASE3_GEOM/<model>/predictions.npz (base), LADDER_EVAL/<rung>_<loss>/predictions.npz
            and data/ladder/<rung>_train.json (ladder). Output: MARGIN_DIAG/cosine_band_compare.json;
            prefix __COSBAND__.
  taxonomy  "far misses by taxonomy" (15.5k base data, eight losses): for every validation image the rank of
            the true species among the model's scores (cosine to each class weight, max over sub-centres);
            for errors with the true species at rank 2, 3-5 and >5, the share whose predicted species is in
            the same genus / same family other genus / other family; and the far misses ArcFace has and DBM
            does not. Inputs: PHASE3_GEOM features and predictions, classifier weights from each model's
            model.safetensors. Output: MARGIN_DIAG/cone_test_far_misses.json; prefix __CONE__.

Usage (from finetuning/SWIN): python loss_ablation/analysis/per_band.py species|ci|compare|taxonomy
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BANDS, BASE_MODELS, DIAG, LABEL, LADDER_DATA, LADDER_EVAL, NAMES, OUT, PHASE3_EVAL, PHASE3_GEOM,  # noqa: E402
                    RUNS, TRAIN_JSON, VAL_JSON, class_directions, load_rows, load_taxonomy, macro_f1,
                    per_class_counts, per_species_f1, per_species_f1_labels)

N_BOOT, SEED = 1000, 0

# ---------------------------------------------------------------- species
# model name -> the training run's own output dir (for its all_results.json)
SPECIES_MODELS = {
    "arcface": "SWIN_BASE_ARCFACE_384",
    "subcenter_arcface": "SWIN_BASE_SUBCENTER_ARCFACE_384_PILOT",
    "hace": "SWIN_BASE_HACE_384_PILOT",
    "dbm": "SWIN_BASE_DBM_384_PILOT",
    "x2softmax": "SWIN_BASE_X2SOFTMAX_384_PILOT",
    "kappaface": "SWIN_BASE_KAPPAFACE_384_PILOT",
    "arcface_ecc": "SWIN_BASE_ARCFACE_ECC_384_PILOT",
    "arcface_tara": "SWIN_BASE_ARCFACE_TARA_384_PILOT",
}
SPECIES_PAIRS = [("kappaface", "dbm"), ("kappaface", "arcface"), ("dbm", "arcface"),
                 ("subcenter_arcface", "arcface"), ("arcface_tara", "arcface"),
                 ("arcface_ecc", "arcface"), ("hace", "arcface")]


def main_species():
    train = load_rows(TRAIN_JSON)
    n_classes = max(r[LABEL] for r in train) + 1
    train_count = np.bincount([r[LABEL] for r in train], minlength=n_classes)
    genus_of, family_of = {}, {}
    for r in train:
        genus_of[r[LABEL]] = (r["family"], r["genus"])
        family_of[r[LABEL]] = r["family"]
    genus_ids = {g: i for i, g in enumerate(sorted(set(genus_of.values())))}
    family_ids = {f: i for i, f in enumerate(sorted(set(family_of.values())))}
    cls_genus = np.full(n_classes, -1)
    cls_family = np.full(n_classes, -1)
    for c, g in genus_of.items():
        cls_genus[c] = genus_ids[g]
        cls_family[c] = family_ids[family_of[c]]
    val_labels = np.array([r[LABEL] for r in load_rows(VAL_JSON)])
    print(f"__P3__ classes={n_classes} train_images={len(train)} val_images={len(val_labels)} "
          f"genera={len(genus_ids)} families={len(family_ids)}")

    preds, f1s, results = {}, {}, {}
    for name, run_dir in SPECIES_MODELS.items():
        path = os.path.join(PHASE3_EVAL, name, "predictions.npz")
        if not os.path.isfile(path):
            print(f"__P3__ MISSING {name}: {path}")
            continue
        z = np.load(path)
        p, y, top5 = z["predictions"].astype(np.int64), z["labels"].astype(np.int64), z["top5"]
        if not np.array_equal(y, val_labels):
            sys.exit(f"{name}: saved labels are not val_2022.json in order -- refusing to analyze")
        tp, nt, npd = per_class_counts(y, p, n_classes)
        acc, mf1 = float(np.mean(p == y)), macro_f1(tp, nt, npd)
        ref = json.load(open(os.path.join(OUT, run_dir, "all_results.json")))
        wrong = p != y
        r = {
            "accuracy": acc, "macro_f1": mf1,
            "top5_accuracy": float(np.mean((top5 == y[:, None]).any(axis=1))),
            "training_run_accuracy": ref["eval_accuracy"], "training_run_macro_f1": ref["eval_f1"],
            "matches_training_run": bool(abs(acc - ref["eval_accuracy"]) < 1e-6
                                         and abs(mf1 - ref["eval_f1"]) < 1e-6),
            "errors_in_right_genus": float(np.mean(cls_genus[p[wrong]] == cls_genus[y[wrong]])),
            "errors_in_right_family": float(np.mean(cls_family[p[wrong]] == cls_family[y[wrong]])),
        }
        f1 = per_species_f1(tp, nt, npd)
        r["bands"] = {}
        for lo, hi in BANDS:
            m = (train_count >= lo) & (train_count <= hi) & ~np.isnan(f1)
            r["bands"][f"{lo}-{hi}"] = {"n_species": int(m.sum()), "mean_species_f1": float(np.mean(f1[m]))}
        preds[name], f1s[name], results[name] = p, f1, r
        print(f"__P3__ {name}: acc={acc:.4f} macroF1={mf1:.4f} top5={r['top5_accuracy']:.4f} "
              f"matches_training_run={r['matches_training_run']} "
              f"errors_same_genus={r['errors_in_right_genus']:.3f} errors_same_family={r['errors_in_right_family']:.3f} | "
              + " ".join(f"band{k}={v['mean_species_f1']:.4f}(n={v['n_species']})" for k, v in r["bands"].items()))

    n = len(val_labels)

    def boot_weights(i):
        # Resample i is regenerated from (SEED, i), so every pair sees the same resamples
        # without holding all N_BOOT weight vectors in memory at once.
        idx = np.random.default_rng([SEED, i]).integers(0, n, size=n)
        return np.bincount(idx, minlength=n).astype(np.float64)

    pairs = {}
    for a, b in SPECIES_PAIRS:
        if a not in preds or b not in preds:
            continue
        d_acc, d_f1 = [], []
        for i in range(N_BOOT):  # same resampled images for both models (paired)
            w = boot_weights(i)
            fa = macro_f1(*per_class_counts(val_labels, preds[a], n_classes, w))
            fb = macro_f1(*per_class_counts(val_labels, preds[b], n_classes, w))
            d_f1.append(fa - fb)
            d_acc.append((np.sum(w * (preds[a] == val_labels)) - np.sum(w * (preds[b] == val_labels))) / n)
        both = ~np.isnan(f1s[a]) & ~np.isnan(f1s[b])
        diff = f1s[a][both] - f1s[b][both]
        band_delta = {k: results[a]["bands"][k]["mean_species_f1"] - results[b]["bands"][k]["mean_species_f1"]
                      for k in results[a]["bands"]}
        pr = {
            "accuracy_gap": results[a]["accuracy"] - results[b]["accuracy"],
            "accuracy_gap_95ci": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
            "macro_f1_gap": results[a]["macro_f1"] - results[b]["macro_f1"],
            "macro_f1_gap_95ci": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
            "species_better": int(np.sum(diff > 0)), "species_worse": int(np.sum(diff < 0)),
            "species_tied": int(np.sum(diff == 0)),
            "band_mean_species_f1_gap": band_delta,
        }
        pairs[f"{a}_vs_{b}"] = pr
        print(f"__P3__ PAIR {a} vs {b}: acc_gap={pr['accuracy_gap']:+.4f} "
              f"[{pr['accuracy_gap_95ci'][0]:+.4f},{pr['accuracy_gap_95ci'][1]:+.4f}] "
              f"macroF1_gap={pr['macro_f1_gap']:+.4f} "
              f"[{pr['macro_f1_gap_95ci'][0]:+.4f},{pr['macro_f1_gap_95ci'][1]:+.4f}] "
              f"species better/worse/tied={pr['species_better']}/{pr['species_worse']}/{pr['species_tied']} | "
              + " ".join(f"band{k}={v:+.4f}" for k, v in band_delta.items()))

    out = os.path.join(PHASE3_EVAL, "analysis")
    os.makedirs(out, exist_ok=True)
    json.dump({"models": results, "pairs": pairs, "bands": BANDS, "n_bootstrap": N_BOOT,
               "bootstrap_note": "paired image-level bootstrap of the validation set; "
                                 "evaluation-set noise only, not seed noise"},
              open(os.path.join(out, "per_species_summary.json"), "w"), indent=1)
    # One row per species: training count, val count, F1 under every model.
    names = list(f1s)
    with open(os.path.join(out, "per_species_f1.csv"), "w") as f:
        f.write("species_id,family,genus,train_images,val_images," + ",".join(f"f1_{m}" for m in names) + "\n")
        n_val = np.bincount(val_labels, minlength=n_classes)
        for c in range(n_classes):
            fam, gen = genus_of.get(c, ("", ""))
            vals = ",".join("" if np.isnan(f1s[m][c]) else f"{f1s[m][c]:.4f}" for m in names)
            f.write(f"{c},{fam},{gen},{train_count[c]},{n_val[c]},{vals}\n")
    print(f"__P3__ wrote {out}/per_species_summary.json and per_species_f1.csv")


# ---------------------------------------------------------------- ci
CI_RESULT = f"{DIAG}/main_table_ci.json"
CI_PAIRS = [("dbm", "arcface"), ("cosine_m0", "arcface"), ("arcface_m01", "arcface")]


def main_ci():
    preds, y = {}, None
    for name in sorted({m for p in CI_PAIRS for m in p}):
        z = np.load(f"{PHASE3_GEOM}/{name}/predictions.npz")
        yy = z["labels"].astype(np.int64)
        y = yy if y is None else y
        assert np.array_equal(y, yy), f"labels differ for {name}"
        preds[name] = z["predictions"].astype(np.int64)
    n_img = len(y)
    n = int(max(y.max(), *(p.max() for p in preds.values()))) + 1
    ones = np.ones(n_img)
    point = {k: macro_f1(*per_class_counts(y, p, n, ones)) for k, p in preds.items()}
    for k, v in point.items():
        print(f"__CI__ {k}: accuracy={np.mean(preds[k] == y):.4f} macroF1={v:.4f}", flush=True)
    diffs = {f"{a}-{b}": [] for a, b in CI_PAIRS}
    for i in range(N_BOOT):
        idx = np.random.default_rng([SEED, i]).integers(0, n_img, size=n_img)
        w = np.bincount(idx, minlength=n_img).astype(np.float64)
        f = {k: macro_f1(*per_class_counts(y, p, n, w)) for k, p in preds.items()}
        for a, b in CI_PAIRS:
            diffs[f"{a}-{b}"].append(f[a] - f[b])
    res = {"n_val_images": n_img, "macro_f1": point, "pairs": {}}
    for a, b in CI_PAIRS:
        d = np.array(diffs[f"{a}-{b}"]) * 100
        gap = 100 * (point[a] - point[b])
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        res["pairs"][f"{a}-{b}"] = {"gap_points": gap, "ci95_points": ci}
        print(f"__CI__ {a} - {b}: {gap:+.2f} points [{ci[0]:+.2f}, {ci[1]:+.2f}]", flush=True)
    os.makedirs(os.path.dirname(CI_RESULT), exist_ok=True)
    with open(CI_RESULT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"__CI__ wrote {CI_RESULT}", flush=True)


# ---------------------------------------------------------------- compare
COMPARE_RESULT = f"{DIAG}/cosine_band_compare.json"
PREFIX = "__COSBAND__"

COMPARE_BASE_BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 73)]
COMPARE_LADDER_BANDS = [(1, 1), (2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 100), (101, 10 ** 6)]
LOSSES = ["cosine_m0", "dbm", "kappaface", "arcface"]
COMPARE_PAIRS = [("dbm", "cosine_m0"), ("kappaface", "cosine_m0"), ("cosine_m0", "arcface")]
SETTINGS = {  # name -> (prediction file per loss, training manifest, bands)
    "base15k": ({l: f"{PHASE3_GEOM}/{l}/predictions.npz" for l in LOSSES}, TRAIN_JSON, COMPARE_BASE_BANDS),
    "n15k": ({l: f"{LADDER_EVAL}/n15k_{l}/predictions.npz" for l in LOSSES}, f"{LADDER_DATA}/n15k_train.json", COMPARE_LADDER_BANDS),
    "n32k": ({l: f"{LADDER_EVAL}/n32k_{l}/predictions.npz" for l in LOSSES}, f"{LADDER_DATA}/n32k_train.json", COMPARE_LADDER_BANDS),
    "n64k": ({l: f"{LADDER_EVAL}/n64k_{l}/predictions.npz" for l in LOSSES}, f"{LADDER_DATA}/n64k_train.json", COMPARE_LADDER_BANDS),
}


def band_name(lo, hi):
    return f"{lo}+" if hi >= 10 ** 6 else (str(lo) if lo == hi else f"{lo}-{hi}")


def boot_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    b = d[idx].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def run_setting(name, files, manifest, bands, rng):
    missing = [l for l, f in files.items() if not os.path.isfile(f)]
    if missing:
        print(f"{PREFIX} SKIP {name}: missing {','.join(missing)}", flush=True)
        return None
    with open(manifest) as f:
        enc = np.array([json.loads(l)[LABEL] for l in f if l.strip()])
    preds, y = {}, None
    for loss, path in files.items():
        z = np.load(path)
        yy = z["labels"].astype(np.int64)
        if y is None:
            y = yy
        assert np.array_equal(y, yy), f"{name}: labels differ for {loss}"
        preds[loss] = z["predictions"].astype(np.int64)
    n = int(max(enc.max(), y.max(), *(p.max() for p in preds.values()))) + 1
    count = np.bincount(enc, minlength=n)
    f1 = {}
    for loss, p in preds.items():
        f1[loss], n_true = per_species_f1_labels(y, p, n)
    in_val = n_true > 0  # species with at least one validation image
    res = {"n_val_images": int(len(y)), "macro_f1": {}, "bands": {}, "overall_leads": {}}
    for loss in LOSSES:
        res["macro_f1"][loss] = float(np.nanmean(f1[loss]))  # over species in labels or predictions
    print(f"{PREFIX} {name}: {len(y):,} val images | macro-F1 " +
          " ".join(f"{l}={res['macro_f1'][l]:.4f}" for l in LOSSES), flush=True)
    n_in = int(in_val.sum())
    for a, b in COMPARE_PAIRS:
        d = (f1[a] - f1[b])[in_val]
        res["overall_leads"][f"{a}-{b}"] = {"lead": float(100 * d.mean()), "ci": [100 * v for v in boot_ci(d, rng)]}
    rare_part = {f"{a}-{b}": 0.0 for a, b in COMPARE_PAIRS}
    for lo, hi in bands:
        m = in_val & (count >= lo) & (count <= hi)
        if not m.any():
            continue
        bn = band_name(lo, hi)
        entry = {"n_species": int(m.sum()), "share_of_species": float(m.sum() / n_in),
                 "f1": {l: float(np.nanmean(f1[l][m])) for l in LOSSES}, "leads": {}}
        parts = []
        for a, b in COMPARE_PAIRS:
            d = (f1[a] - f1[b])[m]
            lead = float(100 * d.mean())
            ci = [100 * v for v in boot_ci(d, rng)]
            entry["leads"][f"{a}-{b}"] = {"lead": lead, "ci": ci}
            parts.append(f"{a}-{b} {lead:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]")
            if hi <= 10:
                rare_part[f"{a}-{b}"] += lead * m.sum() / n_in
        res["bands"][bn] = entry
        print(f"{PREFIX} {name} band {bn:>6} (n={m.sum():>6,}, {100*m.sum()/n_in:4.1f}% of species): "
              + " ".join(f"{l}={entry['f1'][l]:.4f}" for l in LOSSES) + " | " + " | ".join(parts), flush=True)
    for key, v in res["overall_leads"].items():
        share = rare_part[key] / v["lead"] if abs(v["lead"]) > 1e-9 else float("nan")
        v["from_species_with_at_most_10_images"] = float(rare_part[key])
        print(f"{PREFIX} {name} overall {key}: {v['lead']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] points "
              f"(species-weighted); from species with <=10 images: {rare_part[key]:+.2f} ({100*share:.0f}%)", flush=True)
    return res


def main_compare():
    rng = np.random.default_rng(SEED)
    out = {}
    for name, (files, manifest, bands) in SETTINGS.items():
        r = run_setting(name, files, manifest, bands, rng)
        if r is not None:
            out[name] = r
    os.makedirs(os.path.dirname(COMPARE_RESULT), exist_ok=True)
    with open(COMPARE_RESULT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{PREFIX} wrote {COMPARE_RESULT}", flush=True)


# ---------------------------------------------------------------- taxonomy
TAXONOMY_RESULT = f"{DIAG}/cone_test_far_misses.json"
P = "__CONE__"
RANK_CHUNK = 8192


def ranks_for(m, device):
    import torch
    import torch.nn.functional as F
    z = np.load(f"{PHASE3_GEOM}/{m}/predictions.npz")
    y = torch.as_tensor(z["labels"].astype(np.int64), device=device)
    saved_pred = z["predictions"].astype(np.int64)
    feat = np.load(f"{PHASE3_GEOM}/{m}/features.npy", mmap_mode="r")
    w = F.normalize(class_directions(*RUNS[m]).float(), dim=-1).to(device)  # (N, k, C)
    nk, k, c = w.shape
    w2 = w.reshape(nk * k, c)
    rank = torch.empty(len(y), dtype=torch.long, device=device)
    pred = torch.empty(len(y), dtype=torch.long, device=device)
    for s in range(0, len(y), RANK_CHUNK):
        f = F.normalize(torch.as_tensor(np.asarray(feat[s:s + RANK_CHUNK], dtype=np.float32), device=device), dim=1)
        sc = (f @ w2.T).view(len(f), nk, k).max(-1).values
        yy = y[s:s + RANK_CHUNK]
        t = sc.gather(1, yy[:, None])
        rank[s:s + RANK_CHUNK] = (sc > t).sum(1) + 1
        pred[s:s + RANK_CHUNK] = sc.argmax(1)
    pred = pred.cpu().numpy()
    agree = float(np.mean(pred == saved_pred))
    return y.cpu().numpy(), rank.cpu().numpy(), pred, agree


def far_misses(cls_fam, cls_gen, device):
    out, keep = {}, {}
    for m in BASE_MODELS:
        y, rank, pred, agree = ranks_for(m, device)
        wrong = pred != y
        res = {"n": int(len(y)), "pred_matches_saved": agree}
        for lab, sel in (("rank2", rank == 2), ("rank3_5", (rank >= 3) & (rank <= 5)), ("rank_gt5", rank > 5)):
            sel = sel & wrong
            same_gen = cls_gen[pred[sel]] == cls_gen[y[sel]]
            same_fam = cls_fam[pred[sel]] == cls_fam[y[sel]]
            res[lab] = {"pct_of_images": float(100 * sel.mean()),
                        "pred_same_genus_pct": float(100 * same_gen.mean()) if sel.any() else None,
                        "pred_same_family_other_genus_pct": float(100 * (same_fam & ~same_gen).mean()) if sel.any() else None,
                        "pred_other_family_pct": float(100 * (~same_fam).mean()) if sel.any() else None}
        out[m] = res
        keep[m] = (y, rank, pred)
        r2, r5 = res["rank2"], res["rank_gt5"]
        print(f"{P} [2] {NAMES[m]}: pred==saved {agree:.4f} | near misses (rank 2) {r2['pct_of_images']:.2f}% of images: "
              f"same genus {r2['pred_same_genus_pct']:.0f}%, same family other genus {r2['pred_same_family_other_genus_pct']:.0f}%, "
              f"other family {r2['pred_other_family_pct']:.0f}% | far misses (rank >5) {r5['pct_of_images']:.2f}%: "
              f"same genus {r5['pred_same_genus_pct']:.0f}%, same family other genus {r5['pred_same_family_other_genus_pct']:.0f}%, "
              f"other family {r5['pred_other_family_pct']:.0f}%", flush=True)
    # Far misses ArcFace has and DBM does not.
    ya, ra, pa = keep["arcface"]
    yd, rd, pd = keep["dbm"]
    arc_far, dbm_ok = ra > 5, pd == yd
    only = np.flatnonzero(arc_far & dbm_ok)
    both_far = int(np.sum(arc_far & (rd > 5)))
    same_gen = cls_gen[pa[only]] == cls_gen[ya[only]]
    same_fam = cls_fam[pa[only]] == cls_fam[ya[only]]
    out["arcface_far_dbm_correct"] = {
        "n": int(len(only)), "arcface_far_total": int(arc_far.sum()), "both_far": both_far,
        "arcface_pred_same_genus_pct": float(100 * same_gen.mean()),
        "arcface_pred_same_family_other_genus_pct": float(100 * (same_fam & ~same_gen).mean()),
        "arcface_pred_other_family_pct": float(100 * (~same_fam).mean()),
        "dbm_rank_of_true_in_arcface_far_misses_median": float(np.median(rd[arc_far]))}
    o = out["arcface_far_dbm_correct"]
    print(f"{P} [2] ArcFace far misses: {o['arcface_far_total']}; of these DBM gets {o['n']} right and also ranks below "
          f"fifth {both_far}. ArcFace's predictions on the {o['n']}: same genus {o['arcface_pred_same_genus_pct']:.0f}%, "
          f"same family other genus {o['arcface_pred_same_family_other_genus_pct']:.0f}%, other family "
          f"{o['arcface_pred_other_family_pct']:.0f}%", flush=True)
    return out


def main_taxonomy():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    _, cls_fam, cls_gen = load_taxonomy()
    result = {"part2_far_misses": far_misses(cls_fam, cls_gen, device)}
    os.makedirs(DIAG, exist_ok=True)
    json.dump(result, open(TAXONOMY_RESULT, "w"), indent=1)
    print(f"{P} wrote {TAXONOMY_RESULT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["species", "ci", "compare", "taxonomy"])
    args = ap.parse_args()
    {"species": main_species, "ci": main_ci, "compare": main_compare, "taxonomy": main_taxonomy}[args.command]()


if __name__ == "__main__":
    main()
