"""Driver: one forward pass over the val split, then the four analyses.

Run as `python analyze.py --config configs/<name>.yml`. Everything it concludes
is printed to stdout under `__RG__` markers so the run log is self-contained
evidence; the per-class table is additionally written as CSV for later figures.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

import depth
import extract
import capacity
import geometry
import spatial
import taxonomy


# ---------------------------------------------------------------- statistics

def rankdata(a):
    """Average ranks, ties included -- counts like n_train are heavily tied."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def standardized_ols(X, y, names):
    """z-scored multiple regression: betas are directly comparable in size."""
    Xz = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    yz = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    A = np.hstack([Xz, np.ones((len(yz), 1))])
    beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
    resid = yz - A @ beta
    r2 = 1.0 - (resid @ resid) / (yz @ yz)
    return dict(zip(names, beta[:-1])), float(r2)


# ---------------------------------------------------------------- data

def load_val(cfg):
    df = pd.read_json(cfg["data"]["val_manifest"], lines=True)
    lab = cfg["data"]["label_column"]
    img = cfg["data"]["image_column"]
    df = df.rename(columns={lab: "label", img: "path"})
    # The training script fed scientificNameEncoded straight to the loss without
    # re-encoding, so these integers ARE the model's class indices. Re-encoding
    # them here (as a generic loader would) would silently shift every label.
    df["label"] = df["label"].astype(np.int64)
    return df


def train_counts(path, label_key, n_classes):
    """Per-class training frequency, streamed -- the train manifest is 285 MB."""
    counts = np.zeros(n_classes, dtype=np.int64)
    bad = 0
    with open(path) as fh:
        for line in fh:
            try:
                v = json.loads(line)[label_key]
            except Exception:
                bad += 1
                continue
            if 0 <= v < n_classes:
                counts[v] += 1
    if bad:
        print(f"__RG__: {bad} unparseable train manifest lines skipped")
    return counts


def run_depth(cfg, df, n_classes, device):
    """Depth x taxonomic-level resolution table."""
    y = df["label"].to_numpy()
    fam, gen, spe = taxonomy.build_taxonomy(df, cfg["model"]["label_mapping"], n_classes)

    # Genus / family ids per image, derived from the species label.
    gen_ids = {g: i for i, g in enumerate(sorted(set(gen)))}
    fam_ids = {f: i for i, f in enumerate(sorted(set(fam)))}
    gen_of_class = np.array([gen_ids[g] for g in gen])
    fam_of_class = np.array([fam_ids[f] for f in fam])
    y_gen, y_fam = gen_of_class[y], fam_of_class[y]
    print(f"__RG__: levels -> species {n_classes:,}  genus {len(gen_ids):,}  family {len(fam_ids):,}")

    section("DEPTH EXTRACT")
    raw, names = depth.build_depth_cache(df["path"].tolist(), cfg, device)

    # Three descriptors per stage. mean is what the parent node used; std is the
    # spatial variation it discarded; meanstd is both, to check they are not
    # merely redundant.
    feats = {}
    for nm in names:
        feats[(nm, "mean")] = raw[(nm, "mean")]
        feats[(nm, "std")] = raw[(nm, "std")]
        feats[(nm, "mean+std")] = np.hstack([raw[(nm, "mean")], raw[(nm, "std")]])

    fit = depth.alternating_split(y, n_classes)
    print(f"__RG__: split  fit={fit.sum():,}  eval={(~fit).sum():,} "
          f"(per-class alternation)")

    section("REPRESENTATION DIAGNOSTIC")
    # A nearest-centroid score is only readable if the vectors at that depth
    # actually point in different directions. Check that before believing any of
    # the accuracies below.
    print(f"__RG__: {'stage':10s} {'descr':9s} {'dim':>6s} {'pairwise cos (raw)':>26s} "
          f"{'(centred)':>22s}")
    for (nm, d), X in feats.items():
        m0, s0 = depth.cosine_concentration(X, device)
        m1, s1 = depth.cosine_concentration(depth.prep(X, True), device)
        print(f"__RG__: {nm:10s} {d:9s} {X.shape[1]:6d} {m0:14.4f} +/-{s0:8.4f} "
              f"{m1:12.4f} +/-{s1:8.4f}")

    rows = []
    # Centring is settled by the parent node as necessary; only that path is run.
    for centre in (True,):
        tag = "centred"
        section(f"DEPTH x DESCRIPTOR x TAXONOMIC LEVEL [{tag}] (nearest-centroid)")
        print(f"__RG__: {'stage':10s} {'descr':9s} {'dim':>6s} {'family':>9s} {'genus':>9s} "
              f"{'species':>9s} {'nn-is-congener':>16s}")
        for (nm, d), X0 in feats.items():
            X = depth.prep(X0, centre)
            a_f, _ = depth.nearest_centroid_accuracy(X[fit], y_fam[fit], X[~fit], y_fam[~fit],
                                                     len(fam_ids), device)
            a_g, _ = depth.nearest_centroid_accuracy(X[fit], y_gen[fit], X[~fit], y_gen[~fit],
                                                     len(gen_ids), device)
            a_s, live = depth.nearest_centroid_accuracy(X[fit], y[fit], X[~fit], y[~fit],
                                                        n_classes, device)
            cong, tot = depth.congener_neighbour_rate(X, y, n_classes, gen, device)
            print(f"__RG__: {nm:10s} {d:9s} {X.shape[1]:6d} {a_f:9.4f} {a_g:9.4f} {a_s:9.4f} "
                  f"{cong:16.4f}", flush=True)
            rows.append({"mode": tag, "stage": nm, "descriptor": d, "dim": int(X.shape[1]),
                         "family": a_f, "genus": a_g, "species": a_s, "nn_is_congener": cong,
                         "species_centroids_live": live, "classes_with_centroid": tot})

    # Chance baselines, so "low" can be read against something.
    section("CHANCE BASELINES (majority-class on the held-out half)")
    for lvl, yy, k in (("family", y_fam, len(fam_ids)), ("genus", y_gen, len(gen_ids)),
                       ("species", y, n_classes)):
        maj = np.bincount(yy[~fit], minlength=k).max() / (~fit).sum()
        print(f"__RG__: {lvl:8s} groups={k:6,d}  majority-class acc={maj:.4f}  "
              f"uniform={1.0 / k:.6f}")

    section("DOES VARIANCE RECOVER EARLY STRUCTURE? (species accuracy)")
    print(f"__RG__: {'stage':10s} {'mean':>10s} {'std':>10s} {'mean+std':>10s} "
          f"{'std/mean':>10s}")
    by = {(r["stage"], r["descriptor"]): r for r in rows}
    for nm in names:
        m, sd, ms = (by[(nm, k)]["species"] for k in ("mean", "std", "mean+std"))
        print(f"__RG__: {nm:10s} {m:10.4f} {sd:10.4f} {ms:10.4f} "
              f"{(sd / m if m > 0 else float('nan')):10.2f}")

    section("FRACTION OF FINAL-STAGE ACCURACY REACHED, per descriptor")
    for d in ("mean", "std", "mean+std"):
        last = by[(names[-1], d)]
        print(f"__RG__: [{d}]  " + "  ".join(
            f"{nm}:" + "/".join(f"{by[(nm, d)][k] / last[k]:.3f}"
                                for k in ("family", "genus", "species")) for nm in names))
    print("__RG__: (each entry is family/genus/species as a fraction of that "
          "descriptor's own final stage)")

    section("ARTIFACTS")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "depth_taxonomy.csv"), index=False)
    print(f"__RG__: wrote {os.path.join(outdir, 'depth_taxonomy.csv')}")
    section("DONE")


def run_f1(cfg, df, n_classes, device):
    """Does detector misalignment cost F1, or only recall?

    The geometry node found detector alignment is the strongest single
    predictor of per-class *recall* (Spearman +0.595). Recall alone cannot tell
    apart two very different failures: a detector that under-fires and misses
    its own species, versus one that over-fires and grabs other species'
    images. The second costs precision, which recall is blind to. F1 sees both,
    so if alignment predicts F1 as strongly as it predicts recall, the effect
    is about detector quality generally; if it predicts only recall, the
    detector is specifically failing to claim its own.

    No forward pass: the predictions this needs are already in the geometry
    node's cache.
    """
    section("F1: LOAD CACHE")
    cache = extract.build_cache(df["path"].tolist(), df["label"].to_numpy(), cfg, device)
    ok = cache["valid"]
    y = df["label"].to_numpy()[ok]
    pred = cache["pred"][ok]
    feat = cache["feat"][ok]

    section("F1: PER-CLASS PRECISION / RECALL / F1")
    n_val = np.bincount(y, minlength=n_classes).astype(np.int64)          # true support
    n_pred = np.bincount(pred, minlength=n_classes).astype(np.int64)      # predicted support
    tp = np.bincount(y[y == pred], minlength=n_classes).astype(np.int64)
    present = n_val > 0

    precision = np.full(n_classes, np.nan)
    np.divide(tp, n_pred, out=precision, where=n_pred > 0)
    # A class that is never predicted has no precision to speak of; scoring it 0
    # is the sklearn convention and is what makes macro-F1 comparable to the
    # figure the checkpoint reports.
    precision[(n_pred == 0) & present] = 0.0

    recall = np.full(n_classes, np.nan)
    np.divide(tp, n_val, out=recall, where=n_val > 0)

    denom = precision + recall
    f1 = np.full(n_classes, np.nan)
    np.divide(2 * precision * recall, denom, out=f1, where=np.isfinite(denom) & (denom > 0))
    f1[present & np.isfinite(denom) & (denom == 0)] = 0.0

    never_predicted = int(((n_pred == 0) & present).sum())
    print(f"__RG__: species with val images            {present.sum():,}")
    print(f"__RG__: species NEVER predicted, any image {never_predicted:,} "
          f"({never_predicted / present.sum():.2%} of them)")
    print(f"__RG__: macro precision {np.nanmean(precision[present]):.4f}   "
          f"macro recall {np.nanmean(recall[present]):.4f}   "
          f"macro F1 {np.nanmean(f1[present]):.4f}")
    # Cross-check against the checkpoint's own reported eval_f1 (0.7455).
    print(f"__RG__: macro F1 over all {n_classes:,} classes "
          f"(absent ones scored 0) = {np.nan_to_num(f1).sum() / n_classes:.4f}")

    section("F1: DETECTOR ALIGNMENT")
    mu, _, mu_g, _p = geometry.class_centroids(feat, y, n_classes, device)
    from transformers import SwinForImageClassification
    model = SwinForImageClassification.from_pretrained(cfg["model"]["checkpoint"])
    W = model.classifier.weight.detach().cpu().numpy()
    del model
    mu_np, mu_g_np = mu.cpu().numpy(), mu_g.cpu().numpy()
    Mc = mu_np - mu_g_np[None, :]
    denom_a = np.linalg.norm(W, axis=1) * np.linalg.norm(Mc, axis=1)
    align = np.full(n_classes, np.nan)
    num = (W * Mc).sum(axis=1)
    np.divide(num, denom_a, out=align, where=denom_a > 0)
    align[~present] = np.nan
    print(f"__RG__: alignment mean={np.nanmean(align):.4f} median={np.nanmedian(align):.4f}")

    section("F1: DOES ALIGNMENT PREDICT F1, OR ONLY RECALL?")
    # Reported at several minimum-support thresholds on purpose. The subspace
    # node produced a headline correlation on this same data that reversed sign
    # once small-sample classes were excluded, so a single unfiltered number is
    # not trustworthy evidence here.
    print(f"__RG__: {'min n_val':>10s} {'species':>8s} {'vs precision':>13s} "
          f"{'vs recall':>10s} {'vs F1':>8s}")
    rows = []
    for thr in (1, 5, 10, 20):
        m = present & (n_val >= thr) & np.isfinite(align)
        if m.sum() < 50:
            continue
        rp = spearman(align[m], precision[m])
        rr = spearman(align[m], recall[m])
        rf = spearman(align[m], f1[m])
        print(f"__RG__: {thr:>10d} {m.sum():>8,d} {rp:>13.4f} {rr:>10.4f} {rf:>8.4f}")
        rows.append({"min_n_val": thr, "n_species": int(m.sum()),
                     "spearman_align_precision": rp, "spearman_align_recall": rr,
                     "spearman_align_f1": rf})

    sel = present & (n_val >= int(cfg["analysis"]["min_val_per_class"])) & np.isfinite(align)
    print(f"__RG__: quintiles of detector alignment (n_val >= "
          f"{cfg['analysis']['min_val_per_class']}, {sel.sum():,} species):")
    q = np.quantile(align[sel], np.linspace(0, 1, 6))
    for i in range(5):
        m = sel & (align >= q[i]) & (align <= q[i + 1])
        if m.sum():
            print(f"__RG__:     Q{i + 1}  align {q[i]:.3f}-{q[i + 1]:.3f}  "
                  f"species {m.sum():5d}  precision {np.nanmean(precision[m]):.4f}  "
                  f"recall {np.nanmean(recall[m]):.4f}  F1 {np.nanmean(f1[m]):.4f}")

    section("ARTIFACTS")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    table = pd.DataFrame({
        "class_id": np.arange(n_classes), "n_val": n_val, "n_pred": n_pred, "tp": tp,
        "precision": precision, "recall": recall, "f1": f1, "detector_alignment": align,
    })
    p = os.path.join(outdir, "per_class_f1.csv")
    table.to_csv(p, index=False)
    print(f"__RG__: wrote {p}  ({len(table):,} rows)")
    with open(os.path.join(outdir, "f1_summary.json"), "w") as fh:
        json.dump({"macro_precision": float(np.nanmean(precision[present])),
                   "macro_recall": float(np.nanmean(recall[present])),
                   "macro_f1": float(np.nanmean(f1[present])),
                   "never_predicted": never_predicted,
                   "n_present": int(present.sum()),
                   "correlations": rows}, fh, indent=2)
    print(f"__RG__: wrote {os.path.join(outdir, 'f1_summary.json')}")
    section("DONE")


def run_capacity(cfg, df, n_classes, device):
    """How many of the classifier's 1024 dimensions carry independent weight."""
    section("CAPACITY: LOAD CLASSIFIER")
    from transformers import SwinForImageClassification
    model = SwinForImageClassification.from_pretrained(cfg["model"]["checkpoint"])
    W = model.classifier.weight.detach().cpu().numpy()   # (K, C)
    del model
    print(f"__RG__: classifier weight matrix W: {W.shape}")

    section("CAPACITY: CLASS CENTROIDS (reuses the geometry node's cached val "
            "features; no forward pass if that cache is already complete)")
    cache = extract.build_cache(df["path"].tolist(), df["label"].to_numpy(), cfg, device)
    ok = cache["valid"]
    y = df["label"].to_numpy()[ok]
    feat = cache["feat"][ok]
    mu, _, _mu_g, present = geometry.class_centroids(feat, y, n_classes, device)
    mu_present = mu.cpu().numpy()[present]
    print(f"__RG__: class-centroid matrix (classes with >=1 val image): {mu_present.shape}")

    section("CAPACITY: SINGULAR VALUE SPECTRA")
    spectra = {
        "classifier_W": capacity.singular_values(W),
        "class_centroids": capacity.singular_values(mu_present),
        # The honest null: 15,501 rows into 1024 columns cannot be fully
        # decorrelated even with zero structure, so "1024" itself is not the
        # right ceiling to compare the trained matrix against -- this is.
        "random_baseline": capacity.random_baseline(W.shape, seed=0),
    }
    shapes = {"classifier_W": W.shape, "class_centroids": mu_present.shape,
              "random_baseline": W.shape}

    results = {}
    print(f"__RG__: {'matrix':18s} {'shape':>16s} {'entropy-erank':>14s} "
          f"{'particip.ratio':>15s} {'dims@90%':>9s} {'dims@99%':>9s} {'cond#':>10s}")
    for name, s in spectra.items():
        st = capacity.effective_rank(s)
        results[name] = {**st, "shape": list(shapes[name])}
        print(f"__RG__: {name:18s} {str(shapes[name]):>16s} {st['erank_entropy']:14.1f} "
              f"{st['participation_ratio']:15.1f} {st['dims_for_energy'][0.90]:9d} "
              f"{st['dims_for_energy'][0.99]:9d} {st['condition_number']:10.1f}")

    section("CAPACITY: VERDICT")
    d = W.shape[1]
    ceiling = results["random_baseline"]["erank_entropy"]
    actual = results["classifier_W"]["erank_entropy"]
    print(f"__RG__: classifier effective rank  = {actual:.1f} of {d} raw dimensions ({actual / d:.1%})")
    print(f"__RG__: random-matrix ceiling      = {ceiling:.1f} of {d} raw dimensions ({ceiling / d:.1%})")
    print(f"__RG__: classifier / ceiling       = {actual / ceiling:.1%}")

    section("ARTIFACTS")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    for name, s in spectra.items():
        np.save(os.path.join(outdir, f"singular_values_{name}.npy"), s)
    with open(os.path.join(outdir, "capacity_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"__RG__: wrote {os.path.join(outdir, 'capacity_summary.json')}")
    section("DONE")


def run_subspace(cfg, df, n_classes, device):
    """Does a species' reliance on the near-degenerate directions predict how
    poorly its detector is aligned with it?

    The capacity node found detectors spread across ~879 of 1024 dimensions
    while species positions carry real information in only ~467 of them, and
    unevenly -- the loudest direction there carries ~925x the quietest. The
    reading offered was that a detector's reach into those near-empty
    directions dilutes its alignment with its own species. This tests that
    directly rather than leaving it as a plausible story: split each species'
    centred position into its share of energy in the informative directions
    versus the near-degenerate tail, and correlate that split against the
    detector-alignment measure from the geometry node.

    No forward pass: reuses the geometry node's cached val features and the
    checkpoint's classifier weights, identically to the capacity node.
    """
    section("SUBSPACE: LOAD CLASSIFIER + CENTROIDS")
    from transformers import SwinForImageClassification
    model = SwinForImageClassification.from_pretrained(cfg["model"]["checkpoint"])
    W = model.classifier.weight.detach().cpu().numpy()   # (K, C)
    del model

    cache = extract.build_cache(df["path"].tolist(), df["label"].to_numpy(), cfg, device)
    ok = cache["valid"]
    y = df["label"].to_numpy()[ok]
    pred = cache["pred"][ok]
    feat = cache["feat"][ok]
    mu, _, mu_g, present = geometry.class_centroids(feat, y, n_classes, device)
    mu_np = mu.cpu().numpy()
    mu_g_np = mu_g.cpu().numpy()
    # Centred exactly as the geometry node's self-duality measure centres it --
    # an off-centre matrix here would let the discarded global-mean direction
    # masquerade as "informative" for every species alike (see capacity.py).
    Mc = mu_np[present] - mu_g_np[None, :]
    print(f"__RG__: centred centroid matrix: {Mc.shape}")

    section("SUBSPACE: DECOMPOSE THE CENTRED CENTROID MATRIX")
    s, vt = capacity.svd_full(Mc)
    st = capacity.effective_rank(s)
    r = int(round(st["erank_entropy"]))
    print(f"__RG__: centred effective rank = {st['erank_entropy']:.1f}  "
          f"-> top {r} directions treated as informative, "
          f"remaining {Mc.shape[1] - r} as the near-degenerate tail")

    tail_present = capacity.tail_energy_fraction(Mc, vt, r)
    tail_frac = np.full(n_classes, np.nan)
    tail_frac[present] = tail_present
    print(f"__RG__: tail-energy fraction across species: "
          f"mean={np.nanmean(tail_frac):.4f} median={np.nanmedian(tail_frac):.4f} "
          f"p10={np.nanpercentile(tail_frac,10):.4f} p90={np.nanpercentile(tail_frac,90):.4f}")

    section("SUBSPACE: DOES TAIL RELIANCE PREDICT MISALIGNMENT?")
    # Detector alignment, computed identically to geometry.collapse_metrics'
    # self_duality_cos: cosine between the classifier row and the centred
    # centroid, so this compares directly against the parent line's Section 5.
    denom = np.linalg.norm(W, axis=1) * np.linalg.norm(mu_np - mu_g_np[None, :], axis=1)
    num = (W * (mu_np - mu_g_np[None, :])).sum(axis=1)
    dual = np.full(n_classes, np.nan)
    dual[present] = num[present] / np.where(denom[present] > 0, denom[present], np.nan)

    correct = (y == pred)
    rec = spatial.per_class_mean(correct.astype(np.float32), y, n_classes)

    sel = present & np.isfinite(tail_frac) & np.isfinite(dual)
    print(f"__RG__: {sel.sum():,} species enter the comparison")
    r_align = spearman(tail_frac[sel], dual[sel])
    r_recall = spearman(tail_frac[sel], rec[sel])
    print(f"__RG__: Spearman(tail_energy_fraction, detector_alignment) = {r_align:+.4f}")
    print(f"__RG__: Spearman(tail_energy_fraction, recall)             = {r_recall:+.4f}")

    print("__RG__: quintiles of tail-energy fraction "
          "(Q1 = lives in the informative directions, Q5 = lives in the tail):")
    q = np.quantile(tail_frac[sel], np.linspace(0, 1, 6))
    for i in range(5):
        m = sel & (tail_frac >= q[i]) & (tail_frac <= q[i + 1])
        if m.sum():
            print(f"__RG__:     Q{i + 1}  tail_frac {q[i]:.3f}-{q[i + 1]:.3f}  "
                  f"species {m.sum():5d}  alignment {np.nanmean(dual[m]):.4f}  "
                  f"recall {np.nanmean(rec[m]):.4f}")

    section("ARTIFACTS")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    table = pd.DataFrame({
        "class_id": np.arange(n_classes),
        "tail_energy_fraction": tail_frac,
        "detector_alignment": dual,
        "recall": rec,
    })
    p = os.path.join(outdir, "subspace_alignment.csv")
    table.to_csv(p, index=False)
    print(f"__RG__: wrote {p}  ({len(table):,} rows)")
    with open(os.path.join(outdir, "subspace_summary.json"), "w") as fh:
        json.dump({"effective_rank_centred": st["erank_entropy"], "n_loud_dims": r,
                   "n_tail_dims": Mc.shape[1] - r, "n_species": int(sel.sum()),
                   "spearman_tail_vs_alignment": r_align,
                   "spearman_tail_vs_recall": r_recall}, fh, indent=2)
    print(f"__RG__: wrote {os.path.join(outdir, 'subspace_summary.json')}")
    section("DONE")


def section(title):
    print(f"\n__RG__ ===== {title} " + "=" * max(0, 58 - len(title)), flush=True)


def run_loss_ablation_geometry(cfg, device):
    """mode: loss_ablation_geometry. Reads checkpoints exported by
    finetuning/SWIN/loss_ablation/checkpoint_io.py (a different branch, see
    loss_ablation_geometry.py's module docstring for why this is a file
    contract rather than a Python import) and compares, across every loss
    found under `export_root`:

        geometry           within-class tightness, nearest-class distance,
                            detector alignment -- needs only pooled features
                            and class_directions.pt, head-agnostic
        recall + spatial   top-1 accuracy, macro recall, and where on the
                            sheet the evidence comes from -- needs real
                            logits, computed from head_inference_weight.pt
                            via the two-formula scheme (see
                            loss_ablation_geometry.py's module docstring).
                            Only present for exports that carry that
                            metadata; older geometry-only exports are
                            reported as geometry-only, not silently skipped.

    Deliberately NOT the same shape as the rest of this file's `main()`: no
    val_manifest, no `config.json`, no full-corpus forward pass -- those are
    specific to analyzing one 15.5k-class HF checkpoint over the full val
    split, which does not apply here at all.
    """
    import loss_ablation_geometry as lag
    export_root = cfg["loss_ablation"]["export_root"]
    section("LOSS-ABLATION GEOMETRY EXPORTS")
    print(f"__RG__: export_root = {export_root}")
    results = lag.compare_configs(export_root, device)

    section("COMPARISON TABLE: GEOMETRY")
    if not results:
        print("__RG__: no exports found -- nothing to compare")
        return

    def print_table(cols):
        print("__RG__: " + "  ".join(f"{c:>22s}" for c in cols))
        for r in results:
            cells = []
            for c in cols:
                v = r.get(c, "")
                cells.append(f"{v:>22.4f}" if isinstance(v, float) else f"{v!s:>22}")
            print("__RG__: " + "  ".join(cells))

    print_table(["config", "n_images", "intra_class_cos", "nearest_class_cos",
                "self_duality_cos", "nc1_tr_sw_pinv_sb_over_k", "equinorm_cv_classifier"])

    if any("top1_accuracy" in r for r in results):
        section("COMPARISON TABLE: RECALL + SPATIAL EVIDENCE")
        print_table(["config", "top1_accuracy", "macro_recall", "mean_true_rank",
                    "evidence_diffuseness", "border_ring_mass", "bottom_right_2x2_mass"])
    else:
        section("COMPARISON TABLE: RECALL + SPATIAL EVIDENCE")
        print("__RG__: no export had head_inference_weight.pt metadata -- "
              "geometry-only exports, nothing to show here")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    df_out = pd.DataFrame(results)
    p = os.path.join(outdir, "loss_ablation_geometry.csv")
    df_out.to_csv(p, index=False)
    print(f"__RG__: wrote {p}  ({len(df_out)} configs)")
    section("DONE")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if cfg.get("analysis", {}).get("mode") == "loss_ablation_geometry":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        section("CONFIG")
        print(json.dumps(cfg, indent=2))
        print(f"__RG__: device={device} torch={torch.__version__}")
        run_loss_ablation_geometry(cfg, device)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    section("CONFIG")
    print(json.dumps(cfg, indent=2))
    print(f"__RG__: device={device} torch={torch.__version__}")

    # --- pool ------------------------------------------------------------
    df = load_val(cfg)
    ck_cfg = json.load(open(os.path.join(cfg["model"]["checkpoint"], "config.json")))
    n_classes = len(ck_cfg["id2label"])
    section("CENSUS")
    print(f"__RG__: checkpoint       {cfg['model']['checkpoint']}")
    print(f"__RG__: classes          {n_classes:,}")
    print(f"__RG__: val images       {len(df):,}")
    print(f"__RG__: val label range  [{df['label'].min()}, {df['label'].max()}]  "
          f"distinct={df['label'].nunique():,}")
    if df["label"].max() >= n_classes or df["label"].min() < 0:
        sys.exit("FATAL: validation labels fall outside the model's class range")

    n_tr = train_counts(cfg["data"]["train_manifest"], cfg["data"]["label_column"], n_classes)
    print(f"__RG__: train images     {n_tr.sum():,}  "
          f"(classes with 0 train images: {(n_tr == 0).sum():,})")

    if cfg["analysis"].get("mode", "geometry") == "depth":
        run_depth(cfg, df, n_classes, device)
        return
    if cfg["analysis"].get("mode", "geometry") == "f1":
        run_f1(cfg, df, n_classes, device)
        return
    if cfg["analysis"].get("mode", "geometry") == "capacity":
        run_capacity(cfg, df, n_classes, device)
        return
    if cfg["analysis"].get("mode", "geometry") == "subspace":
        run_subspace(cfg, df, n_classes, device)
        return

    # --- preprocessing fidelity ------------------------------------------
    # The reported accuracy of a checkpoint is only reproducible under the
    # preprocessing it was scored with, and this analysis is only about the
    # reported model if it matches. Probe, print, then proceed.
    n_probe = int(cfg["analysis"].get("preprocess_probe_n", 0))
    if n_probe > 0:
        section("PREPROCESSING PROBE")
        res = extract.probe_preprocessing(df["path"].tolist(), df["label"].to_numpy(),
                                          cfg, device, n_probe)
        best = max(res, key=res.get)
        print(f"__RG__: best on probe = {best} ({res[best]:.4f}); "
              f"configured = {cfg['extract']['preprocess']}")

    # --- forward pass ----------------------------------------------------
    section("EXTRACT")
    cache = extract.build_cache(df["path"].tolist(), df["label"].to_numpy(), cfg, device)

    ok = cache["valid"]
    y = df["label"].to_numpy()[ok]
    pred = cache["pred"][ok]
    feat = cache["feat"][ok]
    rank = cache["true_rank"][ok]
    ent = cache["entropy"][ok]
    correct = (y == pred)

    section("HEADLINE")
    print(f"__RG__: images scored    {len(y):,}")
    print(f"__RG__: top-1 accuracy   {correct.mean():.4f}")
    print(f"__RG__: top-5 accuracy   {(rank < 5).mean():.4f}")
    print(f"__RG__: mean true rank   {rank.mean():.1f}   median {np.median(rank):.0f}")
    print(f"__RG__: mean entropy     {ent.mean():.3f} nats  (ln K = {np.log(n_classes):.2f})")

    # --- per-class performance -------------------------------------------
    n_val = np.bincount(y, minlength=n_classes).astype(np.int64)
    rec = spatial.per_class_mean(correct.astype(np.float32), y, n_classes)
    rec5 = spatial.per_class_mean((rank < 5).astype(np.float32), y, n_classes)
    mrank = spatial.per_class_mean(rank.astype(np.float32), y, n_classes)
    ment = spatial.per_class_mean(ent, y, n_classes)
    present = n_val > 0
    print(f"__RG__: macro recall     {np.nanmean(rec[present]):.4f} "
          f"over {present.sum():,} classes with val images")

    # --- geometry ---------------------------------------------------------
    section("GEOMETRY")
    mu, _, mu_g, present_t = geometry.class_centroids(feat, y, n_classes, device)
    intra_cos, intra_l2 = geometry.intra_class_spread(feat, y, mu, n_classes, device)
    cen_sim, cen_idx = geometry.nearest_neighbour(mu, present, device, k=2)
    nn_cos, nn_class = cen_sim[:, 0], cen_idx[:, 0]

    # Read the head by name, never via from_pretrained -- an ArcFace checkpoint
    # has no classifier.* and would silently yield a random matrix here.
    W_cpu, b_cpu, head = extract.load_head(cfg["model"]["checkpoint"], cfg)
    W = W_cpu.to(device)
    b = b_cpu.numpy() if b_cpu is not None else np.zeros(n_classes, dtype=np.float32)
    w_norm = W.norm(dim=1).cpu().numpy()
    if head == "arcface":
        # ArcFace L2-normalises its rows at inference, so row length plays no
        # part in the decision. Equinorm/row-norm statistics are reported for
        # completeness but are not comparable to the linear-head baseline.
        print("__RG__: NOTE head=arcface -- classifier row norms are not used at "
              "inference; row-norm and equinorm figures are not comparable to the "
              "linear baseline. Direction-based measures (alignment, nearest-row "
              "cosine) are comparable.")
    w_sim, _ = geometry.nearest_neighbour(W, np.ones(n_classes, dtype=bool), device, k=2)

    coll, dual_present = geometry.collapse_metrics(feat, y, mu, mu_g, W, present, device)
    dual = np.full(n_classes, np.nan)
    dual[present] = dual_present
    for k, v in coll.items():
        print(f"__RG__: {k:34s} {v}")
    print(f"__RG__: {'intra-class cos to own centroid':34s} mean={np.nanmean(intra_cos[present]):.4f}")
    print(f"__RG__: {'nearest-class centroid cos':34s} mean={np.nanmean(nn_cos[present]):.4f}")
    print(f"__RG__: {'classifier row norm':34s} mean={w_norm.mean():.3f} cv={w_norm.std()/w_norm.mean():.4f}")

    # --- taxonomy ---------------------------------------------------------
    section("TAXONOMY")
    fam, gen, spe = taxonomy.build_taxonomy(df, cfg["model"]["label_mapping"], n_classes)
    gsize = taxonomy.group_sizes(gen)
    fsize = taxonomy.group_sizes(fam)
    print(f"__RG__: families {len(set(fam)):,}   genera {len(set(gen)):,}")
    print(f"__RG__: species per genus  mean={gsize.mean():.1f} median={np.median(gsize):.0f} "
          f"max={gsize.max()}")
    es = taxonomy.error_structure(y, pred, fam, gen)
    for k, v in es.items():
        print(f"__RG__: {k:34s} {v}")
    names = np.array([f"{g} {s}" for g, s in zip(gen, spe)], dtype=object)
    print("__RG__: most frequent confusions (true -> predicted, count):")
    for a, bb, c in taxonomy.top_confusions(y, pred, names, k=15):
        print(f"__RG__:     {a:38s} -> {bb:38s} {c}")

    # --- spatial ----------------------------------------------------------
    section("SPATIAL EVIDENCE")
    share = spatial.evidence_share(cache["spat_true"][ok])
    diff = spatial.concentration(share)
    print(spatial.format_map(spatial.mean_map(share), "mean evidence map, all images (true class)"))
    print(spatial.format_map(spatial.mean_map(share[correct]), "correct predictions only"))
    print(spatial.format_map(spatial.mean_map(share[~correct]), "errors only"))
    for nm, rm in (("all", spatial.region_mass(share)),
                   ("correct", spatial.region_mass(share[correct])),
                   ("errors", spatial.region_mass(share[~correct]))):
        print(f"__RG__: region mass [{nm}]: " +
              "  ".join(f"{k}={v:.4f}" for k, v in rm.items()))
    print(f"__RG__: evidence diffuseness (norm. entropy, 1=uniform): "
          f"all={diff.mean():.4f} correct={diff[correct].mean():.4f} errors={diff[~correct].mean():.4f}")
    cls_diff = spatial.per_class_mean(diff, y, n_classes)

    # --- what predicts per-class recall ----------------------------------
    section("WHAT PREDICTS PER-CLASS RECALL")
    mn = int(cfg["analysis"]["min_val_per_class"])
    sel = present & (n_val >= mn) & np.isfinite(nn_cos) & (gsize > 0)
    print(f"__RG__: {sel.sum():,} classes with >= {mn} val images enter the analysis")

    covars = {
        "log_n_train": np.log1p(n_tr).astype(np.float64),
        "intra_class_cos": intra_cos,
        "nearest_class_cos": nn_cos,
        "classifier_row_norm": w_norm.astype(np.float64),
        "self_duality_cos": dual,
        "log_genus_size": np.log(gsize.astype(np.float64)),
        "evidence_diffuseness": cls_diff,
        "mean_entropy": ment,
    }
    print(f"__RG__: {'covariate':26s} {'spearman vs recall':>20s}")
    for k, v in covars.items():
        print(f"__RG__: {k:26s} {spearman(v[sel], rec[sel]):>20.4f}")

    reg_names = ["log_n_train", "intra_class_cos", "nearest_class_cos",
                 "classifier_row_norm", "log_genus_size", "evidence_diffuseness"]
    X = np.column_stack([covars[k][sel] for k in reg_names])
    betas, r2 = standardized_ols(X, rec[sel], reg_names)
    print(f"__RG__: standardized OLS on per-class recall, R^2={r2:.4f}")
    for k, v in sorted(betas.items(), key=lambda kv: -abs(kv[1])):
        print(f"__RG__:     beta[{k:24s}] = {v:+.4f}")

    # Head/tail: the single most-cited driver, shown without a model.
    print("__RG__: recall by training-frequency decile (selected classes):")
    q = np.quantile(n_tr[sel], np.linspace(0, 1, 11))
    for i in range(10):
        m = sel & (n_tr >= q[i]) & (n_tr <= q[i + 1])
        if m.sum():
            print(f"__RG__:     decile {i + 1:2d}  n_train {q[i]:6.0f}-{q[i + 1]:6.0f}  "
                  f"classes {m.sum():5d}  recall {np.nanmean(rec[m]):.4f}  "
                  f"intra_cos {np.nanmean(intra_cos[m]):.4f}  nn_cos {np.nanmean(nn_cos[m]):.4f}")

    # --- artifacts --------------------------------------------------------
    section("ARTIFACTS")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    table = pd.DataFrame({
        "class_id": np.arange(n_classes), "family": fam, "genus": gen, "species": spe,
        "n_train": n_tr, "n_val": n_val, "recall": rec, "recall_top5": rec5,
        "mean_true_rank": mrank, "mean_entropy": ment,
        "intra_class_cos": intra_cos, "intra_class_l2": intra_l2,
        "nearest_class_cos": nn_cos, "nearest_class": [names[i] if i >= 0 else "" for i in nn_class],
        "nearest_class_same_genus": [bool(i >= 0 and gen[i] == g) for i, g in zip(nn_class, gen)],
        "classifier_row_norm": w_norm, "classifier_bias": b,
        "nearest_classifier_cos": w_sim[:, 0], "self_duality_cos": dual,
        "genus_size": gsize, "family_size": fsize, "evidence_diffuseness": cls_diff,
    })
    p = os.path.join(outdir, "per_class.csv")
    table.to_csv(p, index=False)
    print(f"__RG__: wrote {p}  ({len(table):,} rows)")
    np.save(os.path.join(outdir, "mean_evidence_map_correct.npy"), spatial.mean_map(share[correct]))
    np.save(os.path.join(outdir, "mean_evidence_map_errors.npy"), spatial.mean_map(share[~correct]))
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump({"headline": {"top1": float(correct.mean()), "top5": float((rank < 5).mean()),
                                "macro_recall": float(np.nanmean(rec[present]))},
                   "collapse": coll, "errors": es, "ols_r2": r2, "ols_betas": betas,
                   "spearman": {k: spearman(v[sel], rec[sel]) for k, v in covars.items()}}, fh, indent=2)
    print(f"__RG__: wrote {os.path.join(outdir, 'summary.json')}")
    section("DONE")


if __name__ == "__main__":
    main()
