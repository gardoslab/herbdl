"""The margin rule of the sibling-margin head, in plain NumPy.

The head (losses.SiblingMarginHead), the distribution report (sibling_margin_distribution.py)
and the tests all read the rule from here, so there is one definition of the margin.

A cosine head scores an image against every species template by cosine. The sibling margin
adds a margin to the score of a *competing* species j when the image belongs to species y,
so y must beat j by at least that much. It depends on the pair, through how closely related
j is to y, and on how rare y is:

    scale_y  = (n_min / n_y) ** tau            1 for the rarest species, near 0 for common ones
    m_y      = m_max * scale_y                 margin against a genus sibling, in cosine units
    m(y, j)  = genus_weight  * m_y             j in the genus of y            (j != y)
             = family_weight * m_y             j in the family of y, other genus
             = 0                               j in another family, and j = y

Rarity uses DBM's profile, (n_y / n_min) ** -tau. A margin of 0.15 cosine units is about 10
degrees at the typical angle of 60 degrees between an image and its own template, which sits
between KappaFace's 5-13 degrees and DBM's 5.7 degrees for the rarest species.
"""

import numpy as np

BANDS = [(2, 5), (6, 10), (11, 15), (16, 25), (26, 50), (51, 10 ** 9)]  # training images of the species


def taxonomy_arrays(species_ids, genus, family, num_classes):
    """Integer genus and family id per species, from the rows of a training manifest.

    A genus is keyed by (family, genus), so a genus name reused in two families stays apart;
    a species with no genus gets an id of its own, so it has no siblings. Raises if a species
    is listed under two genera or families, because the margin assumes a strict hierarchy.
    """
    sp_genus, sp_family = [None] * num_classes, [None] * num_classes
    for s, g, f in zip(species_ids, genus, family):
        s = int(s)
        key = (f, g) if g else None
        if sp_genus[s] is not None and (sp_genus[s], sp_family[s]) != (key, f):
            raise ValueError(f"species id {s} appears under more than one genus or family")
        sp_genus[s], sp_family[s] = key, f
    missing = [i for i in range(num_classes) if sp_family[i] is None]
    if missing:
        raise ValueError(f"{len(missing)} species ids never appear in the manifest (e.g. {missing[:5]})")
    gid, fid, g_seen, f_seen = np.zeros(num_classes, np.int64), np.zeros(num_classes, np.int64), {}, {}
    for s in range(num_classes):
        key = sp_genus[s]
        gid[s] = g_seen.setdefault(key, len(g_seen)) if key is not None else len(g_seen) + 10 ** 7 + s
        fid[s] = f_seen.setdefault(sp_family[s], len(f_seen))
    return np.unique(gid, return_inverse=True)[1], fid  # compact genus ids


def species_margin(counts, m_max, tau=1.0):
    """m_y for every species: the margin against a genus sibling, in cosine units."""
    counts = np.asarray(counts, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError("every species needs at least one training image")
    return m_max * (counts.min() / counts) ** tau


def pair_margin(y, m_species, gid, fid, genus_weight=1.0, family_weight=0.25):
    """(len(y), C) matrix of m(y_i, j) for every competitor j; zero at j = y_i."""
    y = np.asarray(y)
    same_genus = gid[None, :] == gid[y][:, None]
    same_family = fid[None, :] == fid[y][:, None]
    w = genus_weight * same_genus + family_weight * (same_family & ~same_genus)
    w[np.arange(len(y)), y] = 0.0
    return w * m_species[y][:, None]


def competitor_counts(gid, fid):
    """Per species: the number of genus siblings and of same-family species in other genera."""
    n_genus = np.bincount(gid)[gid]
    n_family = np.bincount(fid)[fid]
    return n_genus - 1, n_family - n_genus


def band_index(counts):
    out = np.full(len(counts), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(BANDS):
        out[(counts >= lo) & (counts <= hi)] = i
    return out


def band_name(i):
    lo, hi = BANDS[i]
    return f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}"


def distribution(counts, gid, fid, m_max, tau=1.0, family_weight=0.25, thresholds=(0.01, 0.03, 0.05, 0.10)):
    """What the margins look like on one training set, as plain numbers.

    Everything is given per species and per training image (each image carries the margin of
    its species, so common species weigh more per image). 'Competitors with a margin' counts,
    for one image, the species that receive a non-zero margin against it.
    """
    counts = np.asarray(counts)
    m = species_margin(counts, m_max, tau)
    n_sib, n_fam = competitor_counts(gid, fid)
    n_imgs = counts.sum()
    out = {"species": int(len(counts)), "images": int(n_imgs), "n_min": int(counts.min()), "n_max": int(counts.max()),
           "m_max": m_max, "tau": tau, "family_weight": family_weight,
           "species_with_sibling": float((n_sib > 0).mean()),
           "mean_siblings_per_species": float(n_sib.mean()),
           "mean_siblings_per_image": float((n_sib * counts).sum() / n_imgs),
           "mean_other_genus_family_per_species": float(n_fam.mean())}
    for q in (0, 10, 25, 50, 75, 90, 100):
        out[f"species_margin_p{q}"] = float(np.percentile(m, q))
        order = np.argsort(m)
        cum = np.cumsum(counts[order]) / n_imgs  # image-weighted percentile
        out[f"image_margin_p{q}"] = float(m[order][min(np.searchsorted(cum, q / 100.0), len(m) - 1)])
    for t in thresholds:
        out[f"species_share_margin_ge_{t}"] = float((m >= t).mean())
        out[f"image_share_margin_ge_{t}"] = float(counts[m >= t].sum() / n_imgs)
    bands = band_index(counts)
    rows = []
    for b in range(len(BANDS)):
        sel = bands == b
        if not sel.any():
            continue
        rows.append({"band": band_name(b), "species": int(sel.sum()), "species_share": float(sel.mean()),
                     "images": int(counts[sel].sum()), "image_share": float(counts[sel].sum() / n_imgs),
                     "margin_min": float(m[sel].min()), "margin_median": float(np.median(m[sel])),
                     "margin_max": float(m[sel].max()),
                     "siblings_per_species": float(n_sib[sel].mean()),
                     "other_genus_family_per_species": float(n_fam[sel].mean()),
                     "competitors_with_margin_per_image": float(((n_sib + n_fam) * counts)[sel].sum() / counts[sel].sum())})
    out["bands"] = rows
    return out
