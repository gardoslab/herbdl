"""Ceiling test for a taxonomy-aware (sibling) margin and a coarse-first curriculum.

No training. For each model it reads the cached PHASE3_GEOM validation features and the
class templates (the species_sphere.py inputs) and asks, per training-image band:

1. Oracle ceilings. Accuracy and macro-F1 when the prediction is restricted to the true
   family, or to the true genus. A coarse-first scheme that separated families (genera)
   perfectly could not do better than this.
2. Error taxonomy. Of the wrong predictions, the share that land in the same genus
   (siblings), in the same family but another genus, or in another family.
3. Control. How often the wrong species is the nearest class template of the true species
   (the model-defined hard competitor), against a taxonomic sibling, and how much the
   two sets overlap. A sibling margin must beat a margin on the nearest templates to
   justify using the taxonomy.
4. Near misses. Share of errors where the top wrong species beats the true one by less
   than delta in cosine, split by whether that competitor is a sibling. A rough proxy
   for what a margin of size delta could recover; it is not a simulation of training.

Scores are cosines between L2-normalised features and templates, the head of every model
used here (k = 1 centre per species). The recomputed predictions are compared with the
stored ones as a check. CPU only; no images are loaded.

Writes to --out:
  summary.json       every number printed below, by model and band
  <model>.npz        per-image predictions and competitor types, for redrawing

Usage (from finetuning/SWIN):
  python loss_ablation/sibling_ceiling.py
  python loss_ablation/sibling_ceiling.py --models normface dbm
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species_sphere import (EXTRA_MODELS, GEOM, TRAIN_JSON, load_model,  # noqa: E402
                            load_rows, taxonomy)

DEFAULT_OUT = (f"/projectnb/herbdl/workspaces/{os.environ.get('USER', 'unknown')}"
               "/herbdl/finetuning/output/SWIN/sibling_ceiling")
BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 10 ** 9)]
DELTAS = (0.02, 0.05, 0.10)
SIBLING, FAMILY, FAR = 0, 1, 2  # relation of a wrong species to the true one
RELATIONS = {SIBLING: "same genus", FAMILY: "same family, other genus", FAR: "other family"}
NEG = np.float32(-np.inf)


# ---------------------------------------------------------------- taxonomy groups

def group_ids(keys):
    """Integer id per class for equal keys; an empty key gets an id of its own."""
    ids, seen = np.zeros(len(keys), dtype=np.int64), {}
    for c, key in enumerate(keys):
        if key in ("", ("", "")):
            ids[c] = len(seen) + 10 ** 7 + c  # unique, so a class with no label has no siblings
        else:
            ids[c] = seen.setdefault(key, len(seen))
    return ids


def taxonomy_ids(genus, family):
    """Genus ids keyed by (family, genus), so a genus name reused across families stays apart."""
    gid = group_ids([(f, g) if g else "" for f, g in zip(family, genus)])
    fid = group_ids(list(family))
    # compact ids, so bincount over them stays small
    return np.unique(gid, return_inverse=True)[1], np.unique(fid, return_inverse=True)[1]


def band_of(counts):
    """Band index per count, -1 outside every band."""
    out = np.full(len(counts), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(BANDS):
        out[(counts >= lo) & (counts <= hi)] = i
    return out


def band_name(i):
    lo, hi = BANDS[i]
    return f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}"


# ---------------------------------------------------------------- scoring

def normalise(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def nearest_templates(Wn, topk=5, chunk=2048):
    """(C, topk) indices of the most similar other templates, most similar first."""
    C = len(Wn)
    out = np.zeros((C, topk), dtype=np.int64)
    for s in range(0, C, chunk):
        sim = Wn[s:s + chunk] @ Wn.T
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = NEG
        top = np.argpartition(-sim, topk, axis=1)[:, :topk]
        order = np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1)
        out[s:s + chunk] = np.take_along_axis(top, order, axis=1)
    return out


def score_images(feats, labels, Wn, gid, fid, chunk=2048):
    """Per-image predictions and competitor structure, in chunks.

    Returns arrays over images: pred, pred_family (argmax among the true family),
    pred_genus (argmax among the true genus), top-wrong species and its relation to the true
    one, and the cosine of the true species and of the best competitor of each relation.
    """
    n = len(labels)
    out = {k: np.zeros(n, dtype=np.int64) for k in ("pred", "pred_family", "pred_genus", "wrong")}
    out["wrong_rel"] = np.zeros(n, dtype=np.int8)
    for k in ("true", "wrong_score", "best_sibling", "best_family", "best_far"):
        out[k] = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        y = labels[s:e]
        rows = np.arange(e - s)
        S = normalise(feats[s:e]) @ Wn.T
        same_genus = gid[None, :] == gid[y][:, None]
        same_family = fid[None, :] == fid[y][:, None]
        out["pred"][s:e] = S.argmax(1)
        out["pred_family"][s:e] = np.where(same_family, S, NEG).argmax(1)
        out["pred_genus"][s:e] = np.where(same_genus, S, NEG).argmax(1)
        true = S[rows, y].copy()
        S[rows, y] = NEG  # leaves the wrong species only
        sib = same_genus  # the true species itself is already masked out
        fam = same_family & ~same_genus
        far = ~same_family
        out["wrong"][s:e] = S.argmax(1)
        out["wrong_score"][s:e] = S.max(1)
        out["best_sibling"][s:e] = np.where(sib, S, NEG).max(1)
        out["best_family"][s:e] = np.where(fam, S, NEG).max(1)
        out["best_far"][s:e] = np.where(far, S, NEG).max(1)
        out["true"][s:e] = true
    w, y = out["wrong"], labels
    out["wrong_rel"] = np.where(gid[w] == gid[y], SIBLING, np.where(fid[w] == fid[y], FAMILY, FAR))
    return out


# ---------------------------------------------------------------- metrics

def species_f1(labels, preds, n_classes):
    """Per-species F1 and the species present in the labels."""
    tp = np.bincount(labels[preds == labels], minlength=n_classes)
    n_true = np.bincount(labels, minlength=n_classes)
    n_pred = np.bincount(preds, minlength=n_classes)
    f1 = np.where(n_true + n_pred > 0, 2 * tp / np.maximum(n_true + n_pred, 1), 0.0)
    return f1, n_true > 0


def share(num, den):
    return float(num) / den if den else float("nan")


def summarise(sc, labels, stored, n_train, n_classes, gid, fid, nn):
    """Every table entry for one model, by band (key 'all' for everything)."""
    pred = sc["pred"]
    f1 = {k: species_f1(labels, sc[k], n_classes)[0] for k in ("pred", "pred_family", "pred_genus")}
    present = species_f1(labels, pred, n_classes)[1]
    img_band, sp_band = band_of(n_train[labels]), band_of(n_train)
    has_sib = np.bincount(gid, minlength=gid.max() + 1)[gid] > 1
    nn1_sib = gid[nn[:, 0]] == gid
    nn5_sibs = (gid[nn] == gid[:, None]).sum(1)
    res = {"check": {"agree_with_stored": float((pred == stored).mean()),
                     "macro_f1_recomputed": float(f1["pred"][present].mean()),
                     "macro_f1_stored": float(species_f1(labels, stored, n_classes)[0][present].mean())}}
    groups = [("all", np.ones(len(labels), bool), present)]
    groups += [(band_name(b), img_band == b, present & (sp_band == b)) for b in range(len(BANDS))]
    for name, im, sp in groups:
        err = im & (pred != labels)
        n_img, n_err = int(im.sum()), int(err.sum())
        if not n_img:
            continue
        r = {"images": n_img, "species": int(sp.sum()), "errors": n_err,
             "species_with_sibling": share((sp & has_sib).sum(), sp.sum())}
        for tag, key in (("base", "pred"), ("oracle_family", "pred_family"), ("oracle_genus", "pred_genus")):
            r[f"acc_{tag}"] = float((sc[key][im] == labels[im]).mean())
            r[f"f1_{tag}"] = float(f1[key][sp].mean())
        r["errors_removed_by_family_oracle"] = share(n_err - (im & (sc["pred_family"] != labels)).sum(), n_err)
        r["errors_removed_by_genus_oracle"] = share(n_err - (im & (sc["pred_genus"] != labels)).sum(), n_err)
        for rel, label in RELATIONS.items():
            r[f"error_share_{label}"] = share((err & (sc["wrong_rel"] == rel)).sum(), n_err)
        # Control: the wrong species against the nearest templates of the true species.
        hit1 = nn[labels, 0] == pred
        hit5 = (nn[labels] == pred[:, None]).any(1)
        sib_err = err & (sc["wrong_rel"] == SIBLING)
        r["error_is_nearest_template"] = share((err & hit1).sum(), n_err)
        r["error_in_nearest5_templates"] = share((err & hit5).sum(), n_err)
        r["sibling_errors_in_nearest5"] = share((sib_err & hit5).sum(), sib_err.sum())
        r["nearest5_errors_that_are_siblings"] = share((err & hit5 & (sc["wrong_rel"] == SIBLING)).sum(),
                                                       (err & hit5).sum())
        r["species_nearest_template_is_sibling"] = share((sp & has_sib & nn1_sib).sum(), (sp & has_sib).sum())
        r["siblings_in_nearest5_per_species"] = float(nn5_sibs[sp & has_sib].mean()) if (sp & has_sib).any() else float("nan")
        # Engagement and near misses.
        r["top_wrong_is_sibling_all_images"] = share((im & (sc["wrong_rel"] == SIBLING)).sum(), n_img)
        deficit = sc["wrong_score"] - sc["true"]  # > 0 for errors
        for d in DELTAS:
            near = err & (deficit < d)
            r[f"near_miss_{d}"] = share(near.sum(), n_err)
            r[f"near_miss_{d}_sibling"] = share((near & (sc["wrong_rel"] == SIBLING)).sum(), n_err)
        with_sib = im & has_sib[labels]
        r["median_gap_to_best_sibling"] = float(np.median(sc["true"][with_sib] - sc["best_sibling"][with_sib])) if with_sib.any() else float("nan")
        other = np.maximum(sc["best_family"], sc["best_far"])
        r["median_gap_to_best_nonsibling"] = float(np.median(sc["true"][im] - other[im]))
        res[name] = r
    return res


# ---------------------------------------------------------------- printing

def pct(x):
    return "  -  " if x != x else f"{100 * x:5.1f}"


def print_tables(model, res):
    print(f"\n=== {model}")
    c = res["check"]
    print(f"check: stored-prediction agreement {c['agree_with_stored']:.5f}; macro-F1 recomputed "
          f"{c['macro_f1_recomputed']:.4f} vs stored {c['macro_f1_stored']:.4f}")
    bands = [k for k in res if k != "check"]
    print("\nCeilings (points; macro-F1 over species, accuracy over images)")
    print("| band | species | species w/ sibling | F1 base | F1 family oracle | F1 genus oracle | acc base | acc genus oracle | errors removed by genus oracle |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in bands:
        r = res[b]
        print(f"| {b} | {r['species']} | {pct(r['species_with_sibling'])} | {pct(r['f1_base'])} | {pct(r['f1_oracle_family'])} | "
              f"{pct(r['f1_oracle_genus'])} | {pct(r['acc_base'])} | {pct(r['acc_oracle_genus'])} | {pct(r['errors_removed_by_genus_oracle'])} |")
    print("\nWhere the errors land (% of errors) and the nearest-template control")
    print("| band | errors | same genus | same family | other family | = nearest template | in nearest 5 | sibling errors in nearest 5 | nearest-5 errors that are siblings |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in bands:
        r = res[b]
        print(f"| {b} | {r['errors']} | {pct(r['error_share_same genus'])} | {pct(r['error_share_same family, other genus'])} | "
              f"{pct(r['error_share_other family'])} | {pct(r['error_is_nearest_template'])} | {pct(r['error_in_nearest5_templates'])} | "
              f"{pct(r['sibling_errors_in_nearest5'])} | {pct(r['nearest5_errors_that_are_siblings'])} |")
    print("\nEngagement and near misses (% of errors with the top wrong species within delta cosine of the true one)")
    print("| band | top wrong is sibling (all images) | species: nearest template is sibling | siblings in nearest 5 | "
          + " | ".join(f"near miss {d}" for d in DELTAS) + " | " + " | ".join(f"...sibling {d}" for d in DELTAS)
          + " | median gap to best sibling | median gap to best non-sibling |")
    print("|" + "---|" * (6 + 2 * len(DELTAS)))
    for b in bands:
        r = res[b]
        print(f"| {b} | {pct(r['top_wrong_is_sibling_all_images'])} | {pct(r['species_nearest_template_is_sibling'])} | "
              f"{r['siblings_in_nearest5_per_species']:.2f} | "
              + " | ".join(pct(r[f"near_miss_{d}"]) for d in DELTAS) + " | "
              + " | ".join(pct(r[f"near_miss_{d}_sibling"]) for d in DELTAS)
              + f" | {r['median_gap_to_best_sibling']:.3f} | {r['median_gap_to_best_nonsibling']:.3f} |")


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["normface", "dbm", "kappaface", "arcface"],
                    help="names from phase3_geometry.MODELS or species_sphere.EXTRA_MODELS")
    ap.add_argument("--geom", default=GEOM, help="folder with <model>/features.npy and predictions.npz")
    ap.add_argument("--out", default=DEFAULT_OUT)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from phase3_geometry import MODELS, N
    registry = {name: (name, run_dir, kind) for name, (run_dir, kind) in MODELS.items()}
    registry.update(EXTRA_MODELS)
    for m in args.models:
        if m not in registry:
            sys.exit(f"unknown model {m!r}; known: {sorted(registry)}")

    _, genus, family, n_train = taxonomy(load_rows(TRAIN_JSON), N)
    gid, fid = taxonomy_ids(genus, family)
    print(f"{N} species; {int((np.bincount(gid)[gid] > 1).sum())} have at least one same-genus sibling; "
          f"{len(set(gid))} genera, {len(set(fid))} families")
    os.makedirs(args.out, exist_ok=True)
    summary, ref_labels = {}, None
    for m in args.models:
        feats, labels, stored, W = load_model(args.geom, *registry[m])
        if W.shape[1] != 1:
            sys.exit(f"{m}: {W.shape[1]} centres per species; this test assumes one")
        if ref_labels is not None and not np.array_equal(labels, ref_labels):
            sys.exit(f"{m}: validation labels differ from {args.models[0]}'s")
        ref_labels = labels
        Wn = normalise(W[:, 0, :])
        nn = nearest_templates(Wn)
        sc = score_images(feats, labels, Wn, gid, fid)
        summary[m] = summarise(sc, labels, stored, n_train, N, gid, fid, nn)
        print_tables(m, summary[m])
        np.savez_compressed(f"{args.out}/{m}.npz", labels=labels, **sc)
        sys.stdout.flush()
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
