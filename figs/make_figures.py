"""Figures for paper.tex (loss-function ablation + scaling ladder).

Every number is read from figs/data/, which holds outputs of the project's runs:
  per_species_f1.csv, per_species_summary.json  run 2964fe72 (full-val per-species F1, seed 42)
  rare_crossover.json                           run ab4f160f (KappaFace - DBM by band, 2 seeds)
  geometry_summary.json, macro_f1_full_val.json run bf50bd59 (+ re-scores 61e3b0a7, cf97fa8d, f4e0099f)
  pair_gaps.json                                run 2964fe72 (paired image bootstrap vs ArcFace)
  ladder.json                                   ladder run logs (see its _source field)

Regenerate from the repository root:
  uv run --no-project --with matplotlib --with numpy python figs/make_figures.py
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from orx_figstyle import BASELINE, COLUMN, PALETTE, TEXT, family, figure_grid, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


# One color per loss, shared by every figure. ArcFace is the grey reference.
LOSS = {
    "arcface": ("ArcFace", BASELINE),
    "dbm": ("DBM", PALETTE["orange"]),
    "kappaface": ("KappaFace", PALETTE["blue"]),
    "subcenter_arcface": ("Sub-center ArcFace", PALETTE["green"]),
    "arcface_tara": ("ArcFace + TARA", PALETTE["cyan"]),
    "arcface_ecc": ("ArcFace + ECC", PALETTE["red"]),
    "hace": ("HACE", PALETTE["purple"]),
}

# Training-image bands used by the crossover analysis (run ab4f160f).
BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 73)]
BAND_LABELS = [f"{lo}–{hi}" for lo, hi in BANDS]


# --------------------------------------------------------------- figure 1 ----
def fig_gaps():
    """Accuracy and macro-F1 gap of each loss over ArcFace, with bootstrap CIs."""
    gaps = load("pair_gaps.json")["vs_arcface"]
    order = sorted(gaps, key=lambda k: gaps[k]["f1"][0])  # smallest at the bottom
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.36, sharey=True)
    for ax, metric, xlabel in [(axes[0], "f1", "Δ macro-F1 vs ArcFace (points)"),
                               (axes[1], "acc", "Δ top-1 accuracy vs ArcFace (points)")]:
        ax.axvline(0, color=BASELINE, lw=0.8, zorder=0)
        for i, key in enumerate(order):
            mid, lo, hi = (100 * v for v in gaps[key][metric])
            color = LOSS[key][1]
            ax.plot([lo, hi], [i, i], color=color, lw=1.6, solid_capstyle="butt")
            ax.plot(mid, i, "o", color=color, markersize=4.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(-1.2, 2.4)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LOSS[k][0] for k in order])
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    axes[0].set_ylabel("Loss")
    panel_labels(axes)
    save(fig, os.path.join(HERE, "fig_loss_gaps"))


# --------------------------------------------------------------- figure 2 ----
def per_species():
    """Rows of the per-species CSV that have at least one validation image."""
    with open(os.path.join(DATA, "per_species_f1.csv")) as f:
        rows = [r for r in csv.DictReader(f) if int(r["val_images"]) > 0]
    n_train = np.array([int(r["train_images"]) for r in rows])
    f1 = {k: np.array([float(r[f"f1_{k}"]) for r in rows]) for k in LOSS}
    return n_train, f1


def bootstrap_mean(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fig_f1_by_count():
    n_train, f1 = per_species()
    masks = [(n_train >= lo) & (n_train <= hi) for lo, hi in BANDS]
    x = np.arange(len(BANDS))
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.42)

    # (a) absolute mean per-species F1 in each band.
    ax = axes[0]
    for key in ["arcface", "hace", "arcface_ecc", "subcenter_arcface", "kappaface", "dbm"]:
        label, color = LOSS[key]
        y = [100 * f1[key][m].mean() for m in masks]
        lw = 1.6 if key in ("dbm", "kappaface", "arcface") else 1.0
        ax.plot(x, y, "-o", color=color, lw=lw, markersize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("Training images per species")
    ax.set_ylabel("Mean per-species F1 (%)")
    ax.legend(loc="lower right", fontsize=6, handlelength=1.5)

    # (b) paired per-species gain over ArcFace, species-bootstrap 95% CI.
    ax = axes[1]
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
    offsets = {"dbm": -0.2, "kappaface": 0.0, "subcenter_arcface": 0.2}
    for key, dx in offsets.items():
        label, color = LOSS[key]
        stats = [bootstrap_mean(100 * (f1[key][m] - f1["arcface"][m])) for m in masks]
        mid = [s[0] for s in stats]
        err = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
        ax.errorbar(x + dx, mid, yerr=err, fmt="o", color=color, markersize=3.5,
                    elinewidth=1.0, capsize=0, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.set_xlabel("Training images per species")
    ax.set_ylabel("Per-species F1 gain vs ArcFace (points)")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.0)
    panel_labels(axes)
    save(fig, os.path.join(HERE, "fig_f1_by_count"))

    # Band sizes, printed so the caption can quote them from this run's data.
    print("band sizes:", {b: int(m.sum()) for b, m in zip(BAND_LABELS, masks)})


# --------------------------------------------------------------- figure 3 ----
def fig_crossover():
    rc = load("rare_crossover.json")["bands"]
    keys = [f"{lo}-{hi}" for lo, hi in BANDS]  # JSON keys use an ASCII hyphen
    x = np.arange(len(BAND_LABELS))
    fig, axes = figure_grid(1, 1, width=COLUMN, ratio=0.66)
    ax = axes[0] if hasattr(axes, "__len__") else axes
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
    for seed, dx, face in [("seed42", -0.15, PALETTE["blue"]), ("seed43", 0.15, "white")]:
        y = [100 * rc[b][seed]["kappa_minus_dbm"] for b in keys]
        lo = [100 * rc[b][seed]["ci95"][0] for b in keys]
        hi = [100 * rc[b][seed]["ci95"][1] for b in keys]
        ax.errorbar(x + dx, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o",
                    color=PALETTE["blue"], markerfacecolor=face, markersize=4,
                    elinewidth=1.0, capsize=0, label=f"seed {seed[-2:]} pair")
    pooled = [100 * rc[b]["both_seeds"]["kappa_minus_dbm"] for b in keys]
    ax.plot(x, pooled, "-", color=PALETTE["blue"], lw=1.0, alpha=0.6, label="mean of both seeds")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={rc[k]['n_species']:,})" for lab, k in zip(BAND_LABELS, keys)], fontsize=6)
    ax.set_xlabel("Training images per species (species in band)")
    ax.set_ylabel("KappaFace − DBM\nper-species F1 (points)")
    ax.legend(loc="upper right", fontsize=6)
    save(fig, os.path.join(HERE, "fig_crossover"))


# --------------------------------------------------------------- figure 4 ----
def fig_geometry():
    G = load("geometry_summary.json")
    F = load("macro_f1_full_val.json")
    keys = ["dbm", "kappaface", "subcenter_arcface", "arcface_tara", "arcface", "arcface_ecc", "hace"]
    fig, axes = figure_grid(1, 3, width=TEXT, ratio=0.46)

    # (a) where the clusters sit: own-species vs nearest-other-species cosine.
    ax = axes[0]
    ax.plot([0.5, 0.82], [0.5, 0.82], color=MUTED_LINE, lw=0.8, ls="--", zorder=0)
    for key in keys:
        label, color = LOSS[key]
        ax.plot(G[key]["within_species_cos"], G[key]["nearest_species_cos"], "o", color=color, markersize=4.5)
    ax.set_xlabel("Within-species cosine")
    ax.set_ylabel("Nearest-species cosine")
    ax.set_xlim(0.66, 0.80)
    ax.set_ylim(0.50, 0.69)

    # (b), (c) macro-F1 against two "tidiness" summaries; open markers = seed 43.
    for ax, metric, xlabel in [(axes[1], "separation_gap", "Separation gap"),
                               (axes[2], "alignment_mean", "Classifier alignment")]:
        for key in keys:
            label, color = LOSS[key]
            ax.plot(G[key][metric], 100 * F[key], "o", color=color, markersize=4.5, label=label)
            if f"{key}_seed43" in G:
                ax.plot(G[f"{key}_seed43"][metric], 100 * F[f"{key}_seed43"], "o", markersize=4.5,
                        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Macro-F1 (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6, frameon=False,
               handletextpad=0.2, columnspacing=1.0)
    panel_labels(axes)
    save(fig, os.path.join(HERE, "fig_geometry"))


MUTED_LINE = "#BBBBBB"


# --------------------------------------------------------------- figure 5 ----
def fig_ladder():
    """Scaling ladder: final scores, leads over ArcFace, and the lead during training."""
    L = load("ladder.json")
    rungs = ["15.5k", "32k", "64k"]
    n = np.array([L["rungs"][r]["n_species"] for r in rungs])
    fig, axes = figure_grid(1, 3, width=TEXT, ratio=0.5)
    ladder_losses = ["dbm", "kappaface", "arcface"]

    # (a) final macro-F1 (solid) and accuracy (dashed) vs number of species.
    ax = axes[0]
    for key in ladder_losses:
        label, color = LOSS[key]
        ax.plot(n, [100 * L["rungs"][r][key]["f1"][-1] for r in rungs], "-o", color=color, markersize=3.5)
        ax.plot(n, [100 * L["rungs"][r][key]["acc"][-1] for r in rungs], "--s", color=color, markersize=3)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels(rungs)
    ax.minorticks_off()
    ax.set_xlabel("Species in label set")
    ax.set_ylabel("Final score (%)")
    ax.text(0.03, 0.04, "solid: macro-F1\ndashed: accuracy", transform=ax.transAxes, fontsize=5.5, va="bottom")

    # (b) lead over ArcFace at the end of training, for DBM and KappaFace.
    ax = axes[1]
    for key in ["dbm", "kappaface"]:
        label, color = LOSS[key]
        for metric, ls, marker in [("f1", "-", "o"), ("acc", "--", "s")]:
            lead = [100 * (L["rungs"][r][key][metric][-1] - L["rungs"][r]["arcface"][metric][-1]) for r in rungs]
            ax.plot(n, lead, ls, marker=marker, color=color, markersize=3.5 if metric == "f1" else 3,
                    label=f"{label} {'macro-F1' if metric == 'f1' else 'accuracy'}")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels(rungs)
    ax.minorticks_off()
    ax.set_ylim(0, 4.5)
    ax.set_xlabel("Species in label set")
    ax.set_ylabel("Lead over ArcFace (points)")
    ax.legend(loc="upper left", fontsize=5.2, handlelength=1.8, labelspacing=0.25)

    # (c) macro-F1 lead over ArcFace at each evaluation during training.
    ax = axes[2]
    steps = np.array(L["steps"]) / 1000
    for r, marker in zip(rungs, ["o", "s", "^"]):
        base = np.array(L["rungs"][r]["arcface"]["f1"])
        for key in ["dbm", "kappaface"]:
            lead = 100 * (np.array(L["rungs"][r][key]["f1"]) - base)
            ax.plot(steps, lead, "-", marker=marker, color=LOSS[key][1], markersize=3, lw=1.0)
        ax.annotate(r, (steps[0], 100 * (L["rungs"][r]["dbm"]["f1"][0] - base[0])), xytext=(-3, 0),
                    textcoords="offset points", fontsize=5.5, va="center", ha="right")
    ax.set_xlabel("Optimizer steps (thousands)")
    ax.set_ylabel("Macro-F1 lead over ArcFace (points)")
    ax.set_xlim(-5, 140)
    ax.set_ylim(0, 6.5)
    handles = [plt.Line2D([], [], color=LOSS[k][1], lw=1.4, label=LOSS[k][0]) for k in ladder_losses]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=6, frameon=False)
    panel_labels(axes)
    save(fig, os.path.join(HERE, "fig_ladder"))


# --------------------------------------------------------------- figure 6 ----
def fig_ladder_bands():
    """KappaFace minus DBM per-species F1, by training-image band, on each ladder rung."""
    D = load("ladder_routing.json")
    rungs = [("n15k", "15.5k"), ("n32k", "32k"), ("n64k", "64k")]
    bands = ["2-5", "6-10", "11-15", "16-25", "26-50", "51-100", "101+"]
    labels = [b.replace("-", "–") for b in bands]
    shades = family("blue", 3)
    fig, axes = figure_grid(1, 1, width=TEXT, ratio=0.4)
    ax = axes[0] if hasattr(axes, "__len__") else axes
    x = np.arange(len(bands))
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
    ymin = -8.0
    off_scale = []
    for (key, name), color, marker, dx in zip(rungs, shades, ["o", "s", "^"], [-0.22, 0, 0.22]):
        mid, lo, hi = [], [], []
        for b in bands:
            r = D[key]["bands"][b]
            mid.append(100 * r["kappa_minus_dbm"]); lo.append(100 * r["ci95"][0]); hi.append(100 * r["ci95"][1])
        mid, lo, hi = map(np.array, (mid, lo, hi))
        on = mid >= ymin
        ax.errorbar((x + dx)[on], mid[on], yerr=[(mid - lo)[on], (hi - mid)[on]], fmt=marker, color=color,
                    markersize=3.8, elinewidth=1.0, capsize=0, label=f"{name} species")
        for i in np.where(~on)[0]:
            # A point far below the axis: draw an arrow at the floor and print its value instead.
            ax.plot(x[i] + dx, ymin + 0.35, "v", color=color, markersize=4.5, clip_on=False)
            ax.annotate(f"{mid[i]:.1f} (n={D[key]['bands'][bands[i]]['n_species']})", (x[i] + dx, ymin + 0.35),
                        xytext=(6, 0), textcoords="offset points", fontsize=5.5, va="center", color=color)
            off_scale.append((name, bands[i], round(float(mid[i]), 2), D[key]["bands"][bands[i]]["n_species"]))
    ax.set_ylim(ymin, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training images per species")
    ax.set_ylabel("KappaFace − DBM\nper-species F1 (points)")
    ax.legend(loc="lower right", fontsize=6)
    save(fig, os.path.join(HERE, "fig_ladder_bands"))
    print("off-scale points:", off_scale)


# --------------------------------------------------------------- figure 7 ----
def fig_routing():
    """Routing on the ladder: gain per rung (with the ArcFace control), and what changes."""
    R = load("ladder_routing.json")
    C = load("ladder_routing_control.json")
    counts = load("routing_counts.json")
    keys = [("n15k", "15.5k"), ("n32k", "32k"), ("n64k", "64k")]
    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.42)

    ax = axes[0]
    width = 0.26
    xi = np.arange(len(keys))
    series = [
        ("KappaFace, first split", [100 * (R[k]["routing"]["routed"]["half_B_macro_f1"] - R[k]["routing"]["dbm"]["half_B_macro_f1"]) for k, _ in keys],
         LOSS["kappaface"][1], None),
        ("KappaFace, second split", [100 * C[k]["kappaface"]["gain"]["macro_f1"] for k, _ in keys], LOSS["kappaface"][1], "//"),
        ("ArcFace, second split", [100 * C[k]["arcface"]["gain"]["macro_f1"] for k, _ in keys], LOSS["arcface"][1], None),
    ]
    for j, (label, vals, color, hatch) in enumerate(series):
        bars = ax.bar(xi + (j - 1) * width, vals, width, color=color if hatch is None else "white", edgecolor=color,
                      lw=1.0, hatch=hatch, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(xi)
    ax.set_xticklabels([n for _, n in keys])
    ax.set_xlabel("Species in label set")
    ax.set_ylabel("Macro-F1 gain from routing\nover DBM alone (points)")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", fontsize=5.5)

    ax = axes[1]
    xs = np.arange(2)
    for j, (label, key, color) in enumerate([("fixed", "fixed", PALETTE["green"]), ("broken", "broken", PALETTE["red"])]):
        vals = [counts[r][key] for r in ("n32k", "n64k")]
        bars = ax.bar(xs + (j - 0.5) * 0.34, vals, 0.34, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:,}", ha="center", va="bottom", fontsize=5.5)
    for i, r in enumerate(("n32k", "n64k")):
        ax.text(xs[i], 1480, f"net +{counts[r]['net']:,}", ha="center", fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"32k (T={counts['n32k']['T']})", f"64k (T={counts['n64k']['T']})"])
    ax.set_xlabel("Species in label set (cutoff T)")
    ax.set_ylabel("Validation images whose answer\nchanges from DBM's")
    ax.set_ylim(0, 1650)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=6)
    panel_labels(axes)
    save(fig, os.path.join(HERE, "fig_routing"))


if __name__ == "__main__":
    use_style()
    fig_gaps()
    fig_f1_by_count()
    fig_crossover()
    fig_geometry()
    fig_ladder()
    fig_ladder_bands()
    fig_routing()
