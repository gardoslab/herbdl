"""Head/tail-decomposed classification metrics for large label spaces.

Aggregate macro-F1 hides the thing we actually care about at high label
cardinality: how much of the score is carried by a small, easy subset of the
label space. Everything here is computed from per-class tp / predicted-support /
true-support counts, so it stays O(n_samples + n_classes) and never
materialises an N x N confusion matrix (15k^2 float64 is 1.8 GB).
"""

import numpy as np


def class_counts(y_true, y_pred, n_classes):
    """tp, predicted support and true support per class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes).astype(np.float64)
    pred_support = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    true_support = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    return tp, pred_support, true_support


def per_class_f1(tp, pred_support, true_support):
    denom = pred_support + true_support
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def gini(x):
    """Gini coefficient of a non-negative vector. 0 = perfectly even, ->1 = concentrated."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total <= 0:
        return 0.0
    # (2*sum(i*x_i) / (n*sum(x))) - (n+1)/n  with i 1-indexed
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)


def effective_num_classes(f1):
    """exp(H(p)) where p is the per-class F1 vector normalised to a distribution.

    Reads as "this N-way model behaves like an N_eff-way model". If every class
    is recognised equally well it equals N regardless of the absolute F1 level;
    if the score is carried by m classes it collapses towards m. Pairs with
    f1_mass (= sum of per-class F1 = macro_f1 * N), which says how much total
    recognition there is, while this says how evenly it is spread.
    """
    f1 = np.asarray(f1, dtype=np.float64)
    total = f1.sum()
    if total <= 0:
        return 0.0
    p = f1 / total
    nz = p[p > 0]
    return float(np.exp(-np.sum(nz * np.log(nz))))


def decile_profile(values, weights_desc=None, n_bins=10):
    """Mean of `values` within each bin after sorting descending.

    Sorted by `weights_desc` when given (e.g. class frequency or separability),
    otherwise by `values` themselves. Returns a list of n_bins means; bin 0 is
    the top decile.
    """
    values = np.asarray(values, dtype=np.float64)
    key = values if weights_desc is None else np.asarray(weights_desc, dtype=np.float64)
    order = np.argsort(-key, kind="stable")
    return [float(np.mean(chunk)) if chunk.size else 0.0
            for chunk in np.array_split(values[order], n_bins)]


def shot_bins(true_support_train, bounds=(20, 100)):
    """Long-tail convention: few (<20), medium (20-100], many (>100) train images."""
    few, many = bounds
    counts = np.asarray(true_support_train, dtype=np.float64)
    return {
        "few_shot": counts < few,
        "medium_shot": (counts >= few) & (counts <= many),
        "many_shot": counts > many,
    }


def decompose(y_true, y_pred, n_classes, train_counts=None, separability=None):
    """Full metric bundle for one evaluation.

    train_counts   per-class number of training images (enables the frequency
                   head/tail split; meaningless in the balanced regime where it
                   is constant, so it is skipped there).
    separability   per-class distance to the nearest other class centroid in
                   feature space. This is the balanced-regime stand-in for
                   "head vs tail": when every class has equal evidence, what
                   separates the winners from the losers is geometry, not
                   frequency (cf. dataset-granularity results).
    """
    tp, pred_support, true_support = class_counts(y_true, y_pred, n_classes)
    f1 = per_class_f1(tp, pred_support, true_support)
    recall = np.divide(tp, true_support, out=np.zeros_like(tp), where=true_support > 0)

    n_eval = int(len(y_true))
    accuracy = float(tp.sum() / n_eval) if n_eval else 0.0
    profile = decile_profile(f1)

    out = {
        "n_classes": int(n_classes),
        "n_eval": n_eval,
        # --- aggregate (what gets reported today) ---
        "accuracy": accuracy,
        "micro_f1": accuracy,  # identical for single-label multiclass
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        # --- concentration: is the aggregate carried by a few classes? ---
        "f1_gini": gini(f1),
        "f1_mass": float(f1.sum()),           # == macro_f1 * n_classes
        "n_eff": effective_num_classes(f1),
        "n_eff_frac": effective_num_classes(f1) / n_classes if n_classes else 0.0,
        # --- coverage: how much of the label space was abandoned? ---
        "dead_class_rate": float(np.mean(f1 == 0.0)),
        "never_predicted_rate": float(np.mean(pred_support == 0.0)),
        # --- the head/tail split itself ---
        "f1_decile_profile": profile,
        "f1_top_decile": profile[0],
        "f1_bottom_decile": profile[-1],
        "f1_decile_ratio": (profile[0] / profile[-1]) if profile[-1] > 0 else None,
        "f1_median": float(np.median(f1)),
    }

    if train_counts is not None and len(np.unique(train_counts)) > 1:
        for name, mask in shot_bins(train_counts).items():
            out[f"macro_f1_{name}"] = float(f1[mask].mean()) if mask.any() else float("nan")
            out[f"n_classes_{name}"] = int(mask.sum())

    if separability is not None:
        sep_profile = decile_profile(f1, weights_desc=separability)
        out["f1_by_separability_decile"] = sep_profile
        out["f1_most_separable_decile"] = sep_profile[0]
        out["f1_least_separable_decile"] = sep_profile[-1]
        # Does geometry predict per-class outcome better than chance ordering?
        sep = np.asarray(separability, dtype=np.float64)
        if sep.std() > 0 and f1.std() > 0:
            out["corr_f1_separability"] = float(np.corrcoef(f1, sep)[0, 1])

    return out, f1
