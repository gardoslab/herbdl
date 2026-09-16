"""Taxonomic structure of the label space and of the model's mistakes.

At 15,501 species the label set is not flat: species sit inside genera inside
families, and a genus with 40 congeners is a harder neighbourhood than a
monotypic one. Two things follow that a plain accuracy number cannot show.

  * whether confusions respect the taxonomy (a within-genus error is the model
    being nearly right about a hard case; a cross-family error is it being
    confused about something else entirely), and
  * whether crowding -- how many congeneric species compete for the same
    prediction -- costs accuracy, which is the closest thing to a *structural*
    difficulty covariate this dataset offers.
"""

import json

import numpy as np


def build_taxonomy(val_df, label_mapping_path, n_classes):
    """class id -> (family, genus, species).

    The validation manifest carries explicit family/genus/species columns, so it
    is the authority. label_mapping.json ("Family Genus species") fills in the
    classes that have no validation image at all.
    """
    fam = np.array([""] * n_classes, dtype=object)
    gen = np.array([""] * n_classes, dtype=object)
    spe = np.array([""] * n_classes, dtype=object)

    if {"family", "genus", "species"}.issubset(val_df.columns):
        seen = val_df.drop_duplicates(subset=["label"])
        for lab, f, g, s in zip(seen["label"], seen["family"], seen["genus"], seen["species"]):
            fam[lab], gen[lab], spe[lab] = str(f), str(g), str(s)

    n_from_mapping = 0
    with open(label_mapping_path) as fh:
        mapping = json.load(fh)
    for k, name in mapping.items():
        i = int(k)
        if i >= n_classes or gen[i]:
            continue
        parts = str(name).split()
        if len(parts) >= 3:
            fam[i], gen[i], spe[i] = parts[0], parts[1], " ".join(parts[2:])
            n_from_mapping += 1

    missing = int(sum(1 for g in gen if not g))
    print(f"__RG__: taxonomy resolved for {n_classes - missing:,}/{n_classes:,} classes "
          f"({n_from_mapping:,} from label_mapping.json, {missing:,} unresolved)")
    return fam, gen, spe


def group_sizes(keys):
    """How many classes share each class's family / genus."""
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return np.array([counts[k] for k in keys], dtype=np.int64)


def error_structure(y_true, y_pred, fam, gen):
    """Decompose mistakes by how far up the taxonomy they land."""
    wrong = y_true != y_pred
    n_w = int(wrong.sum())
    if n_w == 0:
        return {"n_errors": 0}
    t, p = y_true[wrong], y_pred[wrong]
    same_gen = np.array([gen[a] == gen[b] and gen[a] != "" for a, b in zip(t, p)])
    same_fam = np.array([fam[a] == fam[b] and fam[a] != "" for a, b in zip(t, p)])
    return {
        "n_errors": n_w,
        "error_rate": float(wrong.mean()),
        "frac_errors_same_genus": float(same_gen.mean()),
        "frac_errors_same_family": float(same_fam.mean()),
        "frac_errors_cross_family": float((~same_fam).mean()),
        # The base rate a taxonomy-blind model would hit: probability that a
        # uniformly random wrong class happens to share the genus/family.
        "chance_same_genus": float(np.mean([(gen == gen[a]).sum() - 1 for a in t[:5000]]) / (len(gen) - 1)),
        "chance_same_family": float(np.mean([(fam == fam[a]).sum() - 1 for a in t[:5000]]) / (len(fam) - 1)),
    }


def top_confusions(y_true, y_pred, names, k=15):
    wrong = y_true != y_pred
    pairs = {}
    for a, b in zip(y_true[wrong], y_pred[wrong]):
        key = (int(a), int(b))
        pairs[key] = pairs.get(key, 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [(names[a], names[b], c) for (a, b), c in top]
