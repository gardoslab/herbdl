"""Species-sphere figure: validation features of three species drawn on a sphere.

For one triple of dissimilar species (three families) and one triple of similar species
(one genus, the triple the reference model confuses most), each model's validation
features are projected onto the 3-D span of that triple's three class templates and
drawn on the unit sphere, viewed from the templates' mean direction.

The projection keeps every cosine between a feature and those three templates exactly,
because the templates lie in the span. What it drops is the part of the feature
orthogonal to all three, which does not affect how the model ranks the three species.
Renormalising the projected point onto the sphere rescales its three cosines together,
so which template wins is unchanged.

Reads the cached eval-only outputs that phase3_geometry.py uses
(PHASE3_GEOM/<model>/features.npy and predictions.npz), the class templates from each
model's model.safetensors, and the training manifest for genus, family and image
counts. CPU only; no images are loaded.

Writes to --out:
  species_sphere.png    one row per model, columns = dissimilar | similar
  species_sphere.npz    projected points and templates, for redrawing elsewhere
  triples.json          the chosen species, their taxonomy and per-model statistics

Usage (from finetuning/SWIN):
  python loss_ablation/species_sphere.py
  python loss_ablation/species_sphere.py --models arcface dbm kappaface \
      --add normface=<RUN_DIR>:registry
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

GEOM = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_GEOM"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL = "scientificNameEncoded"
DEFAULT_OUT = (f"/projectnb/herbdl/workspaces/{os.environ.get('USER', 'unknown')}"
               "/herbdl/finetuning/output/SWIN/species_sphere")

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
KINDS = ("dissimilar", "similar")


# ---------------------------------------------------------------- taxonomy and data

def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def taxonomy(train_rows, n_classes):
    """Per-class name, genus, family and training-image count from the training manifest."""
    name = np.array([f"class {c}" for c in range(n_classes)], dtype=object)
    genus = np.array([""] * n_classes, dtype=object)
    family = np.array([""] * n_classes, dtype=object)
    n_train = np.zeros(n_classes, dtype=np.int64)
    for r in train_rows:
        c = int(r[LABEL])
        genus[c], family[c] = r["genus"], r["family"]
        name[c] = r.get("scientificName") or r.get("scientific_name") or f"{r['genus']} {r.get('species', '')}".strip()
        n_train[c] += 1
    return name, genus, family, n_train


def load_model(geom, name, run_dir, kind):
    """Validation features (memory-mapped), labels, predictions and (N, k, d) templates."""
    from phase3_geometry import class_directions
    feats = np.load(f"{geom}/{name}/features.npy", mmap_mode="r")
    z = np.load(f"{geom}/{name}/predictions.npz")
    labels, preds = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    if len(feats) != len(labels):
        sys.exit(f"{name}: {len(feats)} features but {len(labels)} labels")
    return feats, labels, preds, np.asarray(class_directions(run_dir, kind))


# ---------------------------------------------------------------- choosing the triples

def pick_similar(labels, preds, genus, ok):
    """The three species of one genus that the model confuses most with each other.

    Start from the most-confused pair (both directions counted) within a genus that has
    at least three eligible species, then add the species of that genus with the most
    confusions with the two combined.
    """
    wrong = (labels != preds) & ok[labels] & ok[preds] & (genus[labels] == genus[preds])
    conf = Counter((min(a, b), max(a, b)) for a, b in zip(labels[wrong], preds[wrong]))
    eligible_per_genus = Counter(genus[np.flatnonzero(ok)])
    pairs = [pair for pair in conf if eligible_per_genus[genus[pair[0]]] >= 3]
    if not pairs:
        sys.exit("no confused pair inside a genus with 3+ eligible species; lower --min-val")
    a, b = min(pairs, key=lambda pair: (-conf[pair], pair))

    def confusion_with_pair(c):
        return conf[(min(a, c), max(a, c))] + conf[(min(b, c), max(b, c))]

    others = [c for c in np.flatnonzero(ok & (genus == genus[a])) if c not in (a, b)]
    c = min(others, key=lambda c: (-confusion_with_pair(c), c))
    return [int(a), int(b), int(c)]


def pick_dissimilar(family, ok, rng):
    """One eligible species from each of three randomly chosen families."""
    families = sorted(set(family[np.flatnonzero(ok)]))
    if len(families) < 3:
        sys.exit("fewer than 3 families have an eligible species; lower --min-val")
    chosen = rng.choice(families, size=3, replace=False)
    return [int(rng.choice(np.flatnonzero(ok & (family == f)))) for f in chosen]


def species_ids(names, wanted):
    ids = []
    for w in wanted:
        hits = np.flatnonzero(names == w)
        if len(hits) != 1:
            sys.exit(f"species {w!r} matched {len(hits)} classes")
        ids.append(int(hits[0]))
    return ids


# ---------------------------------------------------------------- projection

def unit(v, axis=-1):
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def triple_templates(W, triple, feats_by_class):
    """Unit template per species. For sub-center heads (k > 1), the sub-center closest to
    that species' mean validation feature."""
    out = []
    for c in triple:
        w = unit(W[c].astype(np.float64))
        out.append(w[np.argmax(w @ unit(feats_by_class[c].mean(0)))])
    return np.stack(out)


def sphere_basis(T):
    """(d, 3) orthonormal basis of span(T), rotated so that the templates' mean direction
    is +z (the viewing direction) and the first template points up (+y)."""
    q, _ = np.linalg.qr(T.T)
    t = T @ q
    z = unit(t.sum(0))
    y = unit(t[0] - (t[0] @ z) * z)
    x = np.cross(y, z)
    return q @ np.stack([x, y, z]).T


def project(feats, labels, preds, W, triple):
    idx = np.flatnonzero(np.isin(labels, triple))
    X = unit(np.asarray(feats[idx], dtype=np.float64))
    y = labels[idx]
    T = triple_templates(W, triple, {c: X[y == c] for c in triple})
    B = sphere_basis(T)
    u = X @ B
    t = T @ B
    slot = {c: k for k, c in enumerate(triple)}
    return {
        "coords": u,
        "templates": t,
        "position": np.array([slot[c] for c in y]),
        "full_correct": preds[idx] == y,
        "cos_exact": X @ T.T,
        "cos_projected": u @ t.T,
    }


def panel_stats(p):
    t, u, pos = p["templates"], p["coords"], p["position"]
    ang = np.degrees(np.arccos(np.clip(t @ t.T, -1, 1)))
    own = np.degrees(np.arccos(np.clip(p["cos_projected"][np.arange(len(pos)), pos], -1, 1)))
    return {
        "template_angles_deg": [round(float(ang[i, j]), 1) for i, j in ((0, 1), (0, 2), (1, 2))],
        "mean_angle_to_own_template_deg": [round(float(own[pos == k].mean()), 1) for k in range(3)],
        "three_way_accuracy": round(float(np.mean(p["cos_projected"].argmax(1) == pos)), 3),
        "full_model_accuracy": round(float(np.mean(p["full_correct"])), 3),
        "mean_fraction_in_span": round(float(np.linalg.norm(u, axis=1).mean()), 3),
        "hidden_behind_sphere": int(np.sum(u[:, 2] < 0)),
        "n_images": int(len(pos)),
    }


# ---------------------------------------------------------------- drawing

def boundary_arcs(t):
    """For each pair of templates, the visible arc of the great circle where their cosines
    tie and both beat the third one."""
    arcs = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        c = 3 - a - b
        n = unit(t[a] - t[b])
        e1 = unit(np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.99 else np.cross(n, [1.0, 0.0, 0.0]))
        e2 = np.cross(n, e1)
        s = np.linspace(0, 2 * np.pi, 1441)
        pts = np.outer(np.cos(s), e1) + np.outer(np.sin(s), e2)
        keep = (pts @ t[a] >= pts @ t[c]) & (pts[:, 2] >= 0)
        arcs.append(np.where(keep[:, None], pts, np.nan))
    return arcs


def draw_panel(ax, p, names, title, stats):
    from matplotlib.patches import Circle
    u = unit(p["coords"])
    t = p["templates"]
    ax.add_patch(Circle((0, 0), 1, fill=False, color=GRID, lw=1.5))
    for arc in boundary_arcs(t):
        ax.plot(arc[:, 0], arc[:, 1], color=INK2, lw=1, alpha=0.6)
    visible = u[:, 2] >= 0
    for k in range(3):
        m = visible & (p["position"] == k)
        right, wrong = m & p["full_correct"], m & ~p["full_correct"]
        ax.scatter(u[right, 0], u[right, 1], s=10, color=SLOTS[k], alpha=0.55, linewidths=0, zorder=2)
        ax.scatter(u[wrong, 0], u[wrong, 1], s=22, facecolor="none", edgecolor=SLOTS[k],
                   linewidths=1.2, zorder=3)
    for k in range(3):
        ax.scatter(t[k, 0], t[k, 1], s=150, marker="D", color=SLOTS[k], edgecolor="white",
                   linewidth=2, zorder=4, label=names[k])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ang = stats["template_angles_deg"]
    ax.text(-1.05, -1.12, f"templates {min(ang):.0f}–{max(ang):.0f}° apart · "
            f"3-way accuracy {stats['three_way_accuracy']:.2f}",
            fontsize=8.5, color=INK2, va="top")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False,
                    fontsize=8.5, handletextpad=0.4, borderaxespad=0)
    for txt in leg.get_texts():
        txt.set_fontstyle("italic")
        txt.set_color(INK)


def draw_figure(results, triples, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(results)
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 6.2 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, kind in enumerate(KINDS):
            labels = [names[c] for c in triples[kind]]
            draw_panel(axes[i, j], results[m][kind], labels, f"{m} · {kind} species",
                       results[m][kind]["stats"])
    fig.subplots_adjust(hspace=0.42, wspace=0.12)
    fig.text(0.125, 0.905 + 0.02 / len(models),
             "Validation features projected onto the span of three class templates (diamonds), "
             "viewed from their mean direction.\nLines: where two templates tie. "
             "Hollow rings: images the full model gets wrong.",
             fontsize=9, color=INK2, va="bottom")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["arcface", "dbm", "kappaface"],
                    help="names from phase3_geometry.MODELS (or added with --add)")
    ap.add_argument("--add", action="append", default=[], metavar="NAME=RUN_DIR:KIND",
                    help="a model not in phase3_geometry.MODELS; KIND is arcface, registry or hace")
    ap.add_argument("--reference", default="dbm", help="model whose confusions pick the similar triple")
    ap.add_argument("--min-val", type=int, default=10, help="minimum validation images per species")
    ap.add_argument("--seed", type=int, default=0, help="seed for the dissimilar triple")
    ap.add_argument("--similar", nargs=3, metavar="SPECIES", help="override: three species names")
    ap.add_argument("--dissimilar", nargs=3, metavar="SPECIES", help="override: three species names")
    ap.add_argument("--geom", default=GEOM, help="folder with <model>/features.npy and predictions.npz")
    ap.add_argument("--out", default=DEFAULT_OUT)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase3_geometry import MODELS, N
    registry = dict(MODELS)
    for spec in args.add:
        name, rest = spec.split("=", 1)
        run_dir, kind = rest.rsplit(":", 1)
        registry[name] = (run_dir, kind)
    for m in [args.reference, *args.models]:
        if m not in registry:
            sys.exit(f"unknown model {m!r}; known: {sorted(registry)} (use --add for others)")

    names, genus, family, n_train = taxonomy(load_rows(TRAIN_JSON), N)
    _, labels, preds, _ = load_model(args.geom, args.reference, *registry[args.reference])
    ok = np.bincount(labels, minlength=N) >= args.min_val
    print(f"species with >= {args.min_val} validation images: {int(ok.sum())}")

    triples = {
        "similar": species_ids(names, args.similar) if args.similar else pick_similar(labels, preds, genus, ok),
        "dissimilar": (species_ids(names, args.dissimilar) if args.dissimilar
                       else pick_dissimilar(family, ok, np.random.default_rng(args.seed))),
    }
    n_val = np.bincount(labels, minlength=N)
    summary = {kind: [{"class": c, "name": names[c], "genus": genus[c], "family": family[c],
                       "n_train": int(n_train[c]), "n_val": int(n_val[c])} for c in triple]
               for kind, triple in triples.items()}
    for kind in KINDS:
        print(f"{kind}: " + "; ".join(f"{s['name']} ({s['family']}, {s['n_val']} val)" for s in summary[kind]))

    os.makedirs(args.out, exist_ok=True)
    results, arrays = {}, {}
    for m in args.models:
        feats, lab, prd, W = load_model(args.geom, m, *registry[m])
        if not np.array_equal(lab, labels):
            sys.exit(f"{m}: validation labels differ from {args.reference}'s")
        results[m] = {}
        for kind in KINDS:
            p = project(feats, lab, prd, W, triples[kind])
            assert np.allclose(p["cos_exact"], p["cos_projected"], atol=1e-6), "projection lost a cosine"
            p["stats"] = panel_stats(p)
            results[m][kind] = p
            for key in ("coords", "templates", "position", "full_correct"):
                arrays[f"{m}__{kind}__{key}"] = p[key]
            print(f"{m:>18} {kind:>10}: {p['stats']}")

    draw_figure(results, triples, names, f"{args.out}/species_sphere.png")
    np.savez_compressed(f"{args.out}/species_sphere.npz", **arrays,
                        **{f"triple__{k}": np.array(v) for k, v in triples.items()})
    summary["stats"] = {m: {k: results[m][k]["stats"] for k in KINDS} for m in results}
    summary["args"] = vars(args)
    with open(f"{args.out}/triples.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {args.out}/species_sphere.png, species_sphere.npz, triples.json")


if __name__ == "__main__":
    main()
