"""Per-band F1 leads at each ladder rung, and how much of the overall lead is composition.

Question: DBM's macro-F1 lead over ArcFace changes with the number of species (15.5k, 32k,
64k).  Is that because DBM's per-band advantage changes, or only because the mix of rare
and common species changes (the ladder adds mostly rare species)?  Same question for
KappaFace vs ArcFace.

Inputs (SCC): per-image predictions of the finished ladder models,
LADDER_EVAL/<rung>_<loss>/predictions.npz (keys labels, predictions) and label_mapping.json
(integer label id -> "Family Genus species"), plus training-image counts per species from
data/ladder/<rung>_train.json.  Rungs n15k / n32k / n64k; losses dbm, kappaface, arcface,
and ce if LADDER_EVAL/<rung>_ce exists (a missing cell is skipped, not an error).

Part 1 -- per band of training images (2-5 ... 101+; a "1" band is added because the 32k and
64k rungs contain single-image species, and without it the bands would not sum to the
macro-F1): number of species, mean per-species F1 per loss, and the leads DBM-ArcFace,
KappaFace-ArcFace, DBM-KappaFace (CE-ArcFace if CE exists), in F1 points, with a 95% paired
bootstrap CI over species (1000 resamples, fixed seed).

Part 2 -- composition.  Macro-F1 lead = mean over species of (F1_a - F1_b) = sum_b w_rb *
lead_rb, with w_rb the fraction of species in band b at rung r and lead_rb the mean lead in
that band.  Printed per rung: the actual lead, the lead with the 15.5k band weights and
rung r's per-band leads ("same per-band skill, 15.5k mix"), and with rung r's weights and
the 15.5k per-band leads ("rung r's mix, 15.5k per-band skill").

Part 3 -- macro-F1 on the ORIGINAL 15.5k species only, at each rung.  Species are matched
by NAME through label_mapping.json (integer ids may differ across rungs).  Only validation
images whose true species is one of the 15.5k are kept.  Each model's full-label-space
prediction is used unchanged, so predicting an added (non-15.5k) species for such an image
is a miss for recall.  Precision for a 15.5k species counts only the kept images predicted
as it: a non-15.5k val image that is predicted as a 15.5k species is NOT counted as a false
positive (it is outside the restricted set).  This isolates "does the lead shrink on the old
species because the added species steal predictions" from "the lead shrinks in the mix".

Output: MARGIN_DIAG/band_leads_by_rung.json.  Prefix __BANDLEADS__.

Usage: python loss_ablation/band_leads_by_rung.py   (CPU, numpy only)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_routing import BANDS as BASE_BANDS, band_name, macro_f1, per_species_f1  # noqa: E402

LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
EV = f"{OUT}/LADDER_EVAL"
RESULT = f"{OUT}/MARGIN_DIAG/band_leads_by_rung.json"
RUNGS = ("n15k", "n32k", "n64k")
LOSSES = ("dbm", "kappaface", "arcface", "ce")
BANDS = [(1, 1)] + BASE_BANDS
PAIRS = [("dbm", "arcface"), ("kappaface", "arcface"), ("dbm", "kappaface"), ("ce", "arcface")]
DECOMP_PAIRS = [("dbm", "arcface"), ("kappaface", "arcface")]
N_BOOT = 1000
SEED = 0
PREFIX = "__BANDLEADS__"


def pair_label(a, b):
    short = {"dbm": "DBM", "kappaface": "Kappa", "arcface": "Arc", "ce": "CE"}
    return f"{short[a]}-{short[b]}"


def boot_ci(d, rng, idx=None):
    """95% CI of mean(d) over species; idx is a shared (N_BOOT, len(d)) resample index."""
    if idx is None:
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def load_rung(rung):
    """Labels, predictions per loss, name<->id maps per loss, and training counts."""
    with open(f"{LAD}/{rung}_train.json") as f:
        enc = [json.loads(l)["scientificNameEncoded"] for l in f if l.strip()]
    d = {"preds": {}, "id2name": {}, "y": None}
    n = max(enc) + 1
    for loss in LOSSES:
        p = f"{EV}/{rung}_{loss}/predictions.npz"
        m = f"{EV}/{rung}_{loss}/label_mapping.json"
        if not (os.path.isfile(p) and os.path.isfile(m)):
            print(f"{PREFIX} MISSING {rung}_{loss}", flush=True)
            continue
        z = np.load(p)
        y = z["labels"].astype(np.int64)
        if d["y"] is None:
            d["y"] = y
        elif not np.array_equal(d["y"], y):
            print(f"{PREFIX} MISSING {rung}_{loss} (val labels differ from the other losses; skipped)")
            continue
        d["preds"][loss] = z["predictions"].astype(np.int64)
        with open(m) as f:
            d["id2name"][loss] = {int(k): v for k, v in json.load(f).items()}
        n = max(n, len(d["id2name"][loss]), int(d["preds"][loss].max()) + 1, int(y.max()) + 1)
    d["n"] = n
    d["count"] = np.bincount(enc, minlength=n)
    return d


def band_leads(rung, d, rng):
    """Per-band F1 table and leads for one rung.  Returns (json dict, per-pair per-band lead arrays)."""
    n, y, count = d["n"], d["y"], d["count"]
    f1 = {loss: per_species_f1(y, p, n) for loss, p in d["preds"].items()}
    has = ~np.isnan(next(iter(f1.values())))
    res = {}
    leads = {pair: {} for pair in PAIRS}  # pair -> band -> (n_species, mean lead)
    for lo, hi in BANDS:
        m = has & (count >= lo) & (count <= hi)
        if m.sum() == 0:
            continue
        idx = rng.integers(0, int(m.sum()), size=(N_BOOT, int(m.sum())))  # shared by all pairs
        r = {"n_species": int(m.sum()), "mean_f1": {loss: float(f1[loss][m].mean()) for loss in f1},
             "leads": {}}
        txt = []
        for a, b in PAIRS:
            if a not in f1 or b not in f1:
                continue
            dif = f1[a][m] - f1[b][m]
            ci = boot_ci(dif, rng, idx)
            r["leads"][f"{a}-{b}"] = {"mean": float(dif.mean()), "ci95": ci}
            leads[(a, b)][band_name(lo, hi)] = (int(m.sum()), float(dif.mean()))
            txt.append(f"{pair_label(a, b)} {100*dif.mean():+.2f} [{100*ci[0]:+.2f},{100*ci[1]:+.2f}]")
        res[band_name(lo, hi)] = r
        f1txt = " ".join(f"{loss}={r['mean_f1'][loss]:.4f}" for loss in f1)
        print(f"{PREFIX} {rung} band {band_name(lo, hi):>7} (n={int(m.sum()):6,}): F1 {f1txt} | "
              + "  ".join(txt), flush=True)
    return res, leads, f1, has


def decomposition(all_leads, all_nsp):
    """Composition decomposition of the macro-F1 lead, one line per (rung, pair)."""
    out = {}
    base = "n15k"
    for pair in DECOMP_PAIRS:
        a, b = pair
        if base not in all_leads or not all_leads[base][pair]:
            continue
        for rung in RUNGS:
            if rung not in all_leads or not all_leads[rung][pair]:
                continue
            bands = [band_name(lo, hi) for lo, hi in BANDS]
            w_r = {k: all_nsp[rung].get(k, 0) for k in bands}
            tot_r = sum(w_r.values())
            w_r = {k: v / tot_r for k, v in w_r.items()}
            w_b = {k: all_nsp[base].get(k, 0) for k in bands}
            tot_b = sum(w_b.values())
            w_b = {k: v / tot_b for k, v in w_b.items()}
            L_r = {k: all_leads[rung][pair][k][1] for k in bands if k in all_leads[rung][pair]}
            L_b = {k: all_leads[base][pair][k][1] for k in bands if k in all_leads[base][pair]}

            def mix(w, lead):
                ks = [k for k in lead if w.get(k, 0) > 0]
                s = sum(w[k] for k in ks)
                return float(sum(w[k] * lead[k] for k in ks) / s) if s > 0 else float("nan")

            actual = mix(w_r, L_r)
            cf_base_w = mix(w_b, L_r)   # 15.5k mix, rung r per-band leads
            cf_base_l = mix(w_r, L_b)   # rung r mix, 15.5k per-band leads
            out[f"{rung}/{a}-{b}"] = {
                "actual_lead": actual, "w15k_with_rung_leads": cf_base_w,
                "rung_w_with_15k_leads": cf_base_l,
                "weights_rung": w_r, "weights_15k": w_b}
            print(f"{PREFIX} decomp {rung} {pair_label(a, b)}: actual={100*actual:+.2f} | "
                  f"15.5k band weights x {rung} per-band leads={100*cf_base_w:+.2f} | "
                  f"{rung} band weights x 15.5k per-band leads={100*cf_base_l:+.2f} | "
                  f"mix effect (actual - 15.5k weights)={100*(actual - cf_base_w):+.2f} F1 pts", flush=True)
    return out


def restricted_to_base(rung, d, base_names, rng):
    """Macro-F1 over the original 15.5k species at this rung (see module docstring, Part 3)."""
    n, y = d["n"], d["y"]
    res, per_species = {}, {}
    for loss, p in d["preds"].items():
        name2id = {v: k for k, v in d["id2name"][loss].items()}
        ids = np.array(sorted(name2id[nm] for nm in base_names if nm in name2id), dtype=np.int64)
        in_set = np.zeros(n, bool)
        in_set[ids] = True
        keep = in_set[y]
        f = per_species_f1(y[keep], p[keep], n)  # recall over kept images; precision over kept images
        has = np.zeros(n, bool)
        has[ids] = True
        has &= ~np.isnan(f)
        per_species[loss] = (f, has)
        res[loss] = {"macro_f1_orig15k": float(f[has].mean()), "n_species": int(has.sum()),
                     "n_images": int(keep.sum()), "macro_f1_full": macro_f1(y, p, n),
                     "accuracy_orig15k": float(np.mean(p[keep] == y[keep]))}
    txt = " ".join(f"{loss}={r['macro_f1_orig15k']:.4f}(full {r['macro_f1_full']:.4f})"
                   for loss, r in res.items())
    first = next(iter(res.values()))
    print(f"{PREFIX} orig15k {rung}: {first['n_species']:,} species, {first['n_images']:,} val images | "
          f"macro-F1 on orig 15.5k species {txt}", flush=True)
    leads = {}
    for a, b in PAIRS:
        if a in per_species and b in per_species:
            fa, ha = per_species[a]
            fb, hb = per_species[b]
            m = ha & hb
            dif = fa[m] - fb[m]
            ci = boot_ci(dif, rng)
            leads[f"{a}-{b}"] = {"mean": float(dif.mean()), "ci95": ci}
            print(f"{PREFIX} orig15k {rung} lead {pair_label(a, b)}: {100*dif.mean():+.2f} "
                  f"[{100*ci[0]:+.2f},{100*ci[1]:+.2f}] F1 pts", flush=True)
    return {"models": res, "leads": leads}


def main():
    rng = np.random.default_rng(SEED)
    result = {"bands": {}, "decomposition": {}, "orig15k": {}, "rung_summary": {}}
    all_leads, all_nsp, data = {}, {}, {}
    for rung in RUNGS:
        d = load_rung(rung)
        if d["y"] is None or not {"dbm", "arcface"} <= set(d["preds"]):
            print(f"{PREFIX} MISSING {rung}: need at least dbm and arcface predictions; skipped", flush=True)
            continue
        data[rung] = d
        macro = {loss: macro_f1(d["y"], p, d["n"]) for loss, p in d["preds"].items()}
        print(f"{PREFIX} {rung}: {d['n']:,} species, {len(d['y']):,} val images | macro-F1 "
              + " ".join(f"{k}={v:.4f}" for k, v in macro.items()), flush=True)
        res, leads, f1, has = band_leads(rung, d, rng)
        result["bands"][rung] = res
        result["rung_summary"][rung] = {"n_species": int(d["n"]), "val_images": int(len(d["y"])),
                                        "macro_f1": macro}
        all_leads[rung] = leads
        all_nsp[rung] = {k: v["n_species"] for k, v in res.items()}

    result["decomposition"] = decomposition(all_leads, all_nsp)

    if "n15k" in data:
        base_names = set(data["n15k"]["id2name"]["dbm"].values())
        for rung, d in data.items():
            result["orig15k"][rung] = restricted_to_base(rung, d, base_names, rng)
    else:
        print(f"{PREFIX} MISSING n15k: no original-15.5k restriction", flush=True)

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    json.dump(result, open(RESULT, "w"), indent=1)
    print(f"{PREFIX} wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
