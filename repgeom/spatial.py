"""Where on the sheet the evidence comes from.

Swin-base at 224px ends on a 7x7 grid of 1024-d tokens which the pooler averages,
so the class logit is *exactly* the mean of 49 per-position terms w_c . h_p (see
extract.py). This module turns those terms into a spatial evidence distribution.

The grid size is NOT assumed to always be 7x7: it is read off each `share`
array's own width (`P = share.shape[1]`, `grid = sqrt(P)`), because the
loss-ablation pipeline (loss_ablation_geometry.py) runs a *different* SWIN
config -- `microsoft/swin-base-patch4-window12-384-in22k` at 384px ends on a
12x12=144 grid, not 49 -- through this same module. A hardcoded GRID=7
constant here silently produced an out-of-range boolean index the first time
that path was exercised (`share` had 144 columns, the region masks were sized
for 49), so region masks are computed per-call and cached by grid size
instead.

Negative contributions are real -- a position can argue against a class -- so a
position's "share" is taken after shifting the map to be non-negative. That
keeps the result a probability distribution over the grid without discarding the
relative ordering, which is what the concentration and region statistics read.

The herbarium-specific reason to care: a specimen sheet is a plant plus a
printed determination label, a barcode and a colour bar, and the mounting board
in between. If the evidence mass sits on the border ring rather than on the
pressed plant, the model may be reading the institution's stationery.
"""

import functools
import math

import numpy as np


def _grid_size(n_positions):
    grid = round(math.sqrt(n_positions))
    if grid * grid != n_positions:
        raise ValueError(
            f"{n_positions} positions is not a perfect square -- spatial.py "
            "assumes a square pooled token grid (e.g. 49 = 7x7, 144 = 12x12)")
    return grid


@functools.lru_cache(maxsize=None)
def _regions(grid):
    row = np.arange(grid * grid) // grid
    col = np.arange(grid * grid) % grid
    border = (row == 0) | (row == grid - 1) | (col == 0) | (col == grid - 1)
    interior = ~border
    corners = {
        "top_left": (row <= 1) & (col <= 1),
        "top_right": (row <= 1) & (col >= grid - 2),
        "bottom_left": (row >= grid - 2) & (col <= 1),
        "bottom_right": (row >= grid - 2) & (col >= grid - 2),
    }
    return border, interior, corners


def evidence_share(spat):
    """(N, P) raw contributions -> (N, P) non-negative shares summing to 1."""
    n_positions = spat.shape[1]
    s = spat.astype(np.float32)
    s = s - s.min(axis=1, keepdims=True)
    tot = s.sum(axis=1, keepdims=True)
    # A perfectly flat map (every position identical) has nothing to normalise;
    # call it uniform rather than dividing by zero.
    flat = tot[:, 0] <= 0
    s[flat] = 1.0
    tot[flat] = n_positions
    return s / tot


def concentration(share):
    """Normalised entropy of the evidence map: 0 = one position, 1 = uniform.

    Reported as-is (higher = more diffuse) so it can go straight into the
    per-class regression without a sign convention to remember.
    """
    p = np.clip(share, 1e-12, None)
    return (-(p * np.log(p)).sum(axis=1) / np.log(share.shape[1])).astype(np.float32)


def region_mass(share):
    grid = _grid_size(share.shape[1])
    border, interior, corners = _regions(grid)
    out = {
        "border_ring": float(share[:, border].sum(axis=1).mean()),
        "interior_5x5": float(share[:, interior].sum(axis=1).mean()),
    }
    for name, mask in corners.items():
        out[name + "_2x2"] = float(share[:, mask].sum(axis=1).mean())
    # What a model with no spatial preference would score, for reference.
    out["uniform_border_ring"] = float(border.sum() / (grid * grid))
    out["uniform_corner_2x2"] = float(4 / (grid * grid))
    return out


def mean_map(share):
    grid = _grid_size(share.shape[1])
    return share.mean(axis=0).reshape(grid, grid)


def format_map(m, title):
    grid = m.shape[0]
    lines = [f"  {title}"]
    for r in range(grid):
        lines.append("    " + " ".join(f"{v * grid * grid:5.2f}" for v in m[r]))
    lines.append("    (values are x uniform: 1.00 = exactly the uniform share)")
    return "\n".join(lines)


def per_class_mean(values, labels, n_classes):
    """Mean of a per-image quantity within each class."""
    s = np.bincount(labels, weights=values.astype(np.float64), minlength=n_classes)
    c = np.bincount(labels, minlength=n_classes).astype(np.float64)
    out = np.full(n_classes, np.nan)
    nz = c > 0
    out[nz] = s[nz] / c[nz]
    return out
