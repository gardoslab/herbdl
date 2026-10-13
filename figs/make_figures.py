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

import numpy as np
from orx_figstyle import BASELINE, COLUMN, PALETTE, TEXT, figure_grid, panel_labels, save, use_style

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
    L = load("ladder.json")
    rungs = ["15.5k", "32k", "64k"]
    n = np.array([L["rungs"][r]["n_species"] for r in rungs])
    fig, axes = figure_grid(1, 3, width=TEXT, ratio=0.36)

    # (a) final macro-F1 (solid) and accuracy (dashed) vs number of species.
    ax = axes[0]
    for key in ["dbm", "arcface"]:
        label, color = LOSS[key]
        f1 = [100 * L["rungs"][r][key]["f1"][-1] for r in rungs]
        acc = [100 * L["rungs"][r][key]["acc"][-1] for r in rungs]
        ax.plot(n, f1, "-o", color=color, markersize=3.5, label=f"{label} macro-F1")
        ax.plot(n, acc, "--s", color=color, markersize=3, label=f"{label} accuracy")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels(["15.5k", "32k", "64k"])
    ax.minorticks_off()
    ax.set_xlabel("Species in label set")
    ax.set_ylabel("Final score (%)")
    ax.legend(loc="lower left", fontsize=5.5, handlelength=1.8)

    # (b) DBM's lead over ArcFace at the end of training.
    ax = axes[1]
    width = 0.36
    xi = np.arange(len(rungs))
    for metric, dx, label, face in [("f1", -width / 2, "macro-F1", PALETTE["orange"]),
                                    ("acc", width / 2, "accuracy", "white")]:
        lead = [100 * (L["rungs"][r]["dbm"][metric][-1] - L["rungs"][r]["arcface"][metric][-1]) for r in rungs]
        bars = ax.bar(xi + dx, lead, width, color=face, edgecolor=PALETTE["orange"], lw=1.0, label=label)
        for b, v in zip(bars, lead):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"+{v:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(xi)
    ax.set_xticklabels(rungs)
    ax.set_ylim(0, 4.8)
    ax.set_xlabel("Species in label set")
    ax.set_ylabel("DBM − ArcFace (points)")
    ax.legend(loc="upper left", fontsize=6)

    # (c) macro-F1 over training at each rung.
    ax = axes[2]
    steps = np.array(L["steps"]) / 1000
    for r, marker in zip(rungs, ["o", "s", "^"]):
        for key in ["dbm", "arcface"]:
            color = LOSS[key][1]
            ax.plot(steps, 100 * np.array(L["rungs"][r][key]["f1"]), "-", marker=marker, color=color,
                    markersize=3, lw=1.0)
        ax.annotate(r, (steps[-1], 100 * L["rungs"][r]["dbm"]["f1"][-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=6, va="center")
    ax.set_xlabel("Optimizer steps (thousands)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlim(15, 150)
    panel_labels(axes)
    save(fig, os.path.join(HERE, "fig_ladder"))


if __name__ == "__main__":
    use_style()
    fig_gaps()
    fig_f1_by_count()
    fig_crossover()
    fig_geometry()
    fig_ladder()
