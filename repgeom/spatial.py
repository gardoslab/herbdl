"""Where on the sheet the evidence comes from.

Swin-base at 224px ends on a 7x7 grid of 1024-d tokens which the pooler averages,
so the class logit is *exactly* the mean of 49 per-position terms w_c . h_p (see
extract.py). This module turns those terms into a spatial evidence distribution.

Negative contributions are real -- a position can argue against a class -- so a
position's "share" is taken after shifting the map to be non-negative. That
keeps the result a probability distribution over the grid without discarding the
relative ordering, which is what the concentration and region statistics read.

The herbarium-specific reason to care: a specimen sheet is a plant plus a
printed determination label, a barcode and a colour bar, and the mounting board
in between. If the evidence mass sits on the border ring rather than on the
pressed plant, the model may be reading the institution's stationery.
"""

import numpy as np

GRID = 7
ROW = np.arange(GRID * GRID) // GRID
COL = np.arange(GRID * GRID) % GRID
BORDER = (ROW == 0) | (ROW == GRID - 1) | (COL == 0) | (COL == GRID - 1)
INTERIOR = ~BORDER
CORNERS = {
    "top_left": (ROW <= 1) & (COL <= 1),
    "top_right": (ROW <= 1) & (COL >= GRID - 2),
    "bottom_left": (ROW >= GRID - 2) & (COL <= 1),
    "bottom_right": (ROW >= GRID - 2) & (COL >= GRID - 2),
}


def evidence_share(spat):
    """(N, 49) raw contributions -> (N, 49) non-negative shares summing to 1."""
    s = spat.astype(np.float32)
    s = s - s.min(axis=1, keepdims=True)
    tot = s.sum(axis=1, keepdims=True)
    # A perfectly flat map (every position identical) has nothing to normalise;
    # call it uniform rather than dividing by zero.
    flat = tot[:, 0] <= 0
    s[flat] = 1.0
    tot[flat] = GRID * GRID
    return s / tot


def concentration(share):
    """Normalised entropy of the evidence map: 0 = one position, 1 = uniform.

    Reported as-is (higher = more diffuse) so it can go straight into the
    per-class regression without a sign convention to remember.
    """
    p = np.clip(share, 1e-12, None)
    return (-(p * np.log(p)).sum(axis=1) / np.log(GRID * GRID)).astype(np.float32)


def region_mass(share):
    out = {
        "border_ring": float(share[:, BORDER].sum(axis=1).mean()),
        "interior_5x5": float(share[:, INTERIOR].sum(axis=1).mean()),
    }
    for name, mask in CORNERS.items():
        out[name + "_2x2"] = float(share[:, mask].sum(axis=1).mean())
    # What a model with no spatial preference would score, for reference.
    out["uniform_border_ring"] = float(BORDER.sum() / (GRID * GRID))
    out["uniform_corner_2x2"] = float(4 / (GRID * GRID))
    return out


def mean_map(share):
    return share.mean(axis=0).reshape(GRID, GRID)


def format_map(m, title):
    lines = [f"  {title}"]
    for r in range(GRID):
        lines.append("    " + " ".join(f"{v * GRID * GRID:5.2f}" for v in m[r]))
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
