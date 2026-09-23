"""Standalone smoke test for the HACE taxonomy-tree-building logic that
SWIN_finetuning_advanced.py's `use_hace` branch now runs at startup, against
the REAL train_2022.json -- not the E2E harness's synthetic 40-species subset.

Deliberately outside SWIN_finetuning_advanced.py's own main(): this only needs
the dataset's `genus`/`family`/`scientificNameEncoded` columns, no image
loading, no model, no GPU -- cheap enough to run before committing to a real
50-epoch/2-GPU job, per this project's "test before launching" convention.

Run: cd finetuning/SWIN && python loss_ablation/hace_tree_smoke.py
"""
import sys

from datasets import load_dataset

sys.path.insert(0, ".")
from loss_ablation.hierarchy import TaxonomyTree

TRAIN_FILE = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL_COLUMN = "scientificNameEncoded"


def log(msg):
    print(f"__HACE_SMOKE__: {msg}", flush=True)


def build_tree(dataset_train, label_column):
    labels = dataset_train.unique(label_column)
    num_labels = len(labels)
    log(f"num_labels ({label_column}) = {num_labels}")

    sp_col = dataset_train[label_column]
    genus_col = dataset_train["genus"]
    family_col = dataset_train["family"]

    unique_genera = sorted(set(genus_col))
    genus2id = {g: i for i, g in enumerate(unique_genera)}
    unique_families = sorted(set(family_col))
    family2id = {f: i for i, f in enumerate(unique_families)}
    log(f"unique genera = {len(unique_genera)}, unique families = {len(unique_families)}")

    species_to_genus = [None] * num_labels
    genus_to_family = [None] * len(unique_genera)
    n_rows = 0
    for sp, g, f in zip(sp_col, genus_col, family_col):
        n_rows += 1
        gid, fid = genus2id[g], family2id[f]
        if species_to_genus[sp] is not None and species_to_genus[sp] != gid:
            raise ValueError(f"species id {sp} maps to more than one genus in the training data")
        if genus_to_family[gid] is not None and genus_to_family[gid] != fid:
            raise ValueError(f"genus {g!r} maps to more than one family in the training data")
        species_to_genus[sp] = gid
        genus_to_family[gid] = fid
    log(f"scanned {n_rows} training rows, no hierarchy inconsistency found")

    missing = [i for i, g in enumerate(species_to_genus) if g is None]
    if missing:
        raise ValueError(f"{len(missing)} species id(s) never appeared with a genus "
                          f"(e.g. {missing[:5]})")

    tree = TaxonomyTree(species_to_genus, genus_to_family)
    log(f"tree: {tree.n_species} species / {tree.n_genera} genera / "
        f"{tree.n_families} families / {tree.n_nodes} total nodes")
    return tree


def main():
    log(f"loading {TRAIN_FILE}")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})
    train = dataset["train"]
    log(f"columns: {train.column_names}")
    for col in (LABEL_COLUMN, "genus", "family"):
        if col not in train.column_names:
            raise ValueError(f"expected column {col!r} not found in {TRAIN_FILE}")

    tree = build_tree(train, LABEL_COLUMN)

    # Cross-check against this project's own previously-stated expectation
    # (Loss Function Ablations.md, Phase 1: "15,501 species -> 2,564 genera ->
    # 272 families -> root"), so a silent change in the dataset or an off-by-one
    # in this script doesn't go unnoticed.
    expected = (15501, 2564, 272)
    actual = (tree.n_species, tree.n_genera, tree.n_families)
    if actual != expected:
        log(f"WARNING: tree shape {actual} does not match the doc's previously stated "
            f"{expected} -- not necessarily wrong (the doc's figure may be stale), but "
            "worth a human looking at before trusting a 50-epoch run on it")
    else:
        log("tree shape matches the doc's previously stated (15501, 2564, 272) exactly")

    # Reachability-matrix + HACEHead sanity: build the head on CPU with a tiny
    # in_features, confirm forward()/labels round-trip without crashing, and
    # that R is correctly excluded from the checkpointed state (the fix this
    # smoke test is also implicitly covering, alongside the tree-building logic).
    import torch

    from loss_ablation.hierarchy import HACEHead
    head = HACEHead(in_features=8, tree=tree, dilution=0.6, label_smoothing=0.0)
    sd_keys = list(head.state_dict().keys())
    log(f"HACEHead.state_dict() keys: {sd_keys}")
    assert "R" not in sd_keys, "R leaked into the persistent state dict -- the persistent=False fix regressed"
    assert any(k.startswith("linear.") for k in sd_keys), "expected the linear layer's weight/bias in the state dict"

    B = 4
    x = torch.randn(B, 8)
    y = torch.randint(0, tree.n_species, (B,))
    loss, logits = head(x, y)
    assert torch.isfinite(loss), f"non-finite loss on random init: {loss}"
    assert logits.shape == (B, tree.n_species), f"unexpected species-logit shape {logits.shape}"
    log(f"HACEHead forward OK: loss={loss.item():.4f}, logits shape={tuple(logits.shape)}")

    inf_logits = head(x, labels=None)
    assert inf_logits.shape == (B, tree.n_species)
    log("HACEHead inference (labels=None) OK")

    log("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
