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
    feats = depth.build_depth_cache(df["path"].tolist(), cfg, device)

    fit = depth.alternating_split(y, n_classes)
    print(f"__RG__: split  fit={fit.sum():,}  eval={(~fit).sum():,} "
          f"(per-class alternation)")

    section("DEPTH x TAXONOMIC LEVEL (nearest-centroid accuracy, held-out half)")
    print(f"__RG__: {'stage':10s} {'dim':>6s} {'family':>9s} {'genus':>9s} {'species':>9s} "
          f"{'nn-is-congener':>16s}")
    rows = []
    for nm, X in feats.items():
        a_f, _ = depth.nearest_centroid_accuracy(X[fit], y_fam[fit], X[~fit], y_fam[~fit],
                                                 len(fam_ids), device)
        a_g, _ = depth.nearest_centroid_accuracy(X[fit], y_gen[fit], X[~fit], y_gen[~fit],
                                                 len(gen_ids), device)
        a_s, live = depth.nearest_centroid_accuracy(X[fit], y[fit], X[~fit], y[~fit],
                                                    n_classes, device)
        cong, tot = depth.congener_neighbour_rate(X, y, n_classes, gen, device)
        print(f"__RG__: {nm:10s} {X.shape[1]:6d} {a_f:9.4f} {a_g:9.4f} {a_s:9.4f} "
              f"{cong:16.4f}", flush=True)
        rows.append({"stage": nm, "dim": int(X.shape[1]), "family": a_f, "genus": a_g,
                     "species": a_s, "species_centroids_live": live,
                     "nn_is_congener": cong, "classes_with_centroid": tot})

    # How much of each level's final accuracy is already present at each depth?
    section("FRACTION OF FINAL-STAGE ACCURACY REACHED")
    last = rows[-1]
    print(f"__RG__: {'stage':10s} {'family':>9s} {'genus':>9s} {'species':>9s}")
    for r in rows:
        print(f"__RG__: {r['stage']:10s} " + " ".join(
            f"{r[k] / last[k]:9.3f}" for k in ("family", "genus", "species")))

    section("ARTIFACTS")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "depth_taxonomy.csv"), index=False)
    print(f"__RG__: wrote {os.path.join(outdir, 'depth_taxonomy.csv')}")
    section("DONE")


def section(title):
    print(f"\n__RG__ ===== {title} " + "=" * max(0, 58 - len(title)), flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

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

    from transformers import SwinForImageClassification
    model = SwinForImageClassification.from_pretrained(cfg["model"]["checkpoint"])
    W = model.classifier.weight.detach().to(device)
    b = model.classifier.bias.detach().cpu().numpy()
    del model
    w_norm = W.norm(dim=1).cpu().numpy()
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
