"""How many of the classifier's 1024 dimensions does it actually use?

The geometry node found each class's detector points, on average, only 0.449
toward its own class's cloud (about 63 degrees off), not the ~1.0 a fully
"collapsed" classifier would reach -- and that shortfall is the strongest
single predictor of which species are recognised well.

One explanation is a hard packing limit rather than a training deficiency:
15,501 classes cannot all have mutually well-separated directions in a
1024-dimensional space, so most detectors are forced to sit at narrow angles to
several others regardless of how training goes. That explanation makes a
checkable prediction -- the 15,501 x 1024 classifier weight matrix should not
be using all 1024 dimensions independently. Its *effective* rank should sit
well below 1024 if classes are being compressed into a smaller functional
subspace, or close to 1024 if the space is being used efficiently and the
alignment shortfall has some other cause.

Everything here is one matrix decomposition. No forward pass is needed: the
val-split feature cache the geometry node already built is reused as-is for the
class-centroid comparison, and a fresh extraction only happens if that cache is
genuinely absent.
"""

import numpy as np
import torch


def singular_values(M):
    # svdvals is the cheap path when only the spectrum is needed, not the
    # singular vectors -- and float64 because a 15,501 x 1024 matrix easily
    # produces singular values spanning several orders of magnitude, where
    # float32 would blur the tail that the effective-rank measures depend on.
    s = torch.linalg.svdvals(torch.as_tensor(np.asarray(M), dtype=torch.float64))
    return s.numpy()


def effective_rank(s):
    """Three complementary readings of "how many dimensions are really used".

    None requires eyeballing a knee in the spectrum by hand -- that judgement
    call is exactly what this analysis exists to avoid making unaccountably.
    """
    s = np.asarray(s, dtype=np.float64)
    energy = s ** 2
    total = energy.sum()
    p = energy / total

    # Roy & Vetterli's entropy-based effective rank: exp(Shannon entropy of the
    # normalised energy spectrum). A perfectly flat spectrum over d dimensions
    # gives exactly d; all the energy on one direction gives 1.
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum()
    erank_entropy = float(np.exp(ent))

    # Participation ratio: same idea via a different moment of the spectrum,
    # less sensitive than the entropy form to a long thin tail of near-zero
    # singular values, which a 15,501 x 1024 matrix will always have some of.
    participation_ratio = float((energy.sum() ** 2) / (energy ** 2).sum())

    cum = np.cumsum(p)
    dims_for_energy = {frac: int(np.searchsorted(cum, frac) + 1)
                       for frac in (0.50, 0.80, 0.90, 0.95, 0.99)}

    return {
        "erank_entropy": erank_entropy,
        "participation_ratio": participation_ratio,
        "dims_for_energy": dims_for_energy,
        "max_sv": float(s.max()),
        "min_sv": float(s.min()),
        "condition_number": float(s.max() / max(s.min(), 1e-12)),
    }


def random_baseline(shape, seed=0):
    """The honest null: what a matrix with no preferred dimension looks like.

    15,501 purely random directions in a 1024-dimensional space are still
    correlated on average -- that is geometry, nothing to do with training.
    Comparing the trained classifier's effective rank against "1024" itself
    would silently credit training with a shortfall every matrix this shape
    has; comparing against this baseline does not.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal(size=shape)
    return singular_values(M)
