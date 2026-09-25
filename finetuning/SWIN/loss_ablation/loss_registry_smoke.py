"""Standalone smoke test for SWIN_finetuning_advanced.py's new `loss_registry`
branch, against REAL class_counts from train_2022.json (not the E2E harness's
synthetic 40-species subset).

Why this matters specifically for KappaFace/DBM: both adapt per-class margin
by training-sample count, and the real dataset is long-tailed (train counts
2-73 per this project's own prior note). A count of 2 is a genuine edge case
the harness's synthetic subset (12 images/species, uniform) never exercised.
This checks construction + a forward/backward pass with the REAL count
distribution doesn't crash or produce non-finite values, for all four
loss_registry configs (kappaface, x2softmax, dbm, arcface+ecc), before
committing to four real 50-epoch/2-GPU jobs.

No image loading, no SWIN backbone, no GPU -- pure tensor math on the actual
head/aux classes with a random embedding, cheap enough to run before every
one of those four jobs. Run: cd finetuning/SWIN && python
loss_ablation/loss_registry_smoke.py
"""
import sys
from collections import Counter

import torch
import torch.nn.functional as F

from datasets import load_dataset

sys.path.insert(0, ".")
from loss_ablation.losses import build_head
from loss_ablation.aux import AUX_REGISTRY

TRAIN_FILE = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
LABEL_COLUMN = "scientificNameEncoded"
IN_FEATURES = 1024  # swin-base pooled feature dim, matches the real backbone


def log(msg):
    print(f"__LR_SMOKE__: {msg}", flush=True)


def real_class_counts():
    log(f"loading {TRAIN_FILE}")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})
    train = dataset["train"]
    labels = train.unique(LABEL_COLUMN)
    num_labels = len(labels)
    cnt = Counter(train[LABEL_COLUMN])
    counts = [cnt.get(i, 0) for i in range(num_labels)]
    zero = [i for i, c in enumerate(counts) if c == 0]
    if zero:
        raise ValueError(f"{len(zero)} species with zero training examples: {zero[:5]}")
    log(f"num_labels={num_labels}, min_count={min(counts)}, max_count={max(counts)}, "
        f"p10={sorted(counts)[len(counts)//10]}, median={sorted(counts)[len(counts)//2]}")
    return num_labels, counts


def check_head(name, head, B=8):
    """Every margin head in HEAD_REGISTRY returns raw LOGITS from forward(),
    not a loss -- SwinWithLoss.forward() applies F.cross_entropy externally
    (see model.py). Mirror that exact usage here rather than assuming a loss
    comes back directly."""
    x = torch.randn(B, IN_FEATURES, requires_grad=True)
    y = torch.randint(0, head.num_classes, (B,))
    logits = head(x, y)
    assert logits.shape == (B, head.num_classes), f"{name}: unexpected logits shape {logits.shape}"
    assert torch.isfinite(logits).all(), f"{name}: non-finite logits on random init"
    loss = F.cross_entropy(logits, y)
    assert torch.isfinite(loss), f"{name}: non-finite loss: {loss}"
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), f"{name}: non-finite/missing gradient"
    # Inference path (labels=None) too, since SwinWithLoss's eval-time forward uses it.
    with torch.no_grad():
        inf_logits = head(x, labels=None)
    assert inf_logits.shape == (B, head.num_classes), f"{name}: unexpected inference logits shape"
    assert torch.isfinite(inf_logits).all(), f"{name}: non-finite inference logits"
    log(f"{name}: forward+backward OK, loss={loss.item():.4f}, inference OK")


def main():
    num_labels, counts = real_class_counts()

    log("=== kappaface ===")
    head = build_head("kappaface", IN_FEATURES, num_labels, class_counts=counts, s=30.0, m0=0.30)
    check_head("kappaface", head)
    # Exercise the epoch-update path directly: update_buffer() already ran inside
    # forward() above (called for every class in check_head's batch), so at least
    # a few classes have dir_initialized=True; recompute_margins() must not crash
    # with a small, uneven number of observed classes -- exactly the real state
    # after epoch 1 of a real run, not the synthetic all-classes-seen case.
    head.recompute_margins()
    assert torch.isfinite(head.per_class_margin).all(), "kappaface: non-finite per_class_margin after recompute"
    log(f"kappaface: recompute_margins() OK, "
        f"mean per_class_margin={head.per_class_margin.mean().item():.4f} "
        f"(m0={head.m0}, seen={int(head.dir_initialized.sum())}/{num_labels} classes)")

    log("=== x2softmax ===")
    head = build_head("x2softmax", IN_FEATURES, num_labels, s=30.0)
    check_head("x2softmax", head)

    log("=== dbm ===")
    head = build_head("dbm", IN_FEATURES, num_labels, class_counts=counts, s=30.0)
    check_head("dbm", head)

    log("=== arcface (loss_ablation's own ArcMarginHead, not SwinWithArcFace) ===")
    head = build_head("arcface", IN_FEATURES, num_labels, s=30.0, m=0.30)
    check_head("arcface", head)

    log("=== ECCEpochUpdateCallback (the actual production callback, not just recompute_similarity() directly) ===")
    # Regression check for the exact bug the first real ArcFace+ECC run hit: 2h19m
    # of healthy training, then AttributeError: 'ModuleDict' object has no
    # attribute 'get' the instant epoch 1 ended and this callback fired for the
    # first time. loss_registry_smoke.py's own earlier ecc checks below call
    # recompute_similarity() directly on a bare ECCAux, which never exercised the
    # callback's aux_modules access pattern at all -- that gap is exactly why this
    # bug shipped past the smoke test the first time. Import and run the REAL
    # production callback class against a REAL nn.ModuleDict this time.
    sys.path.insert(0, ".")
    from SWIN_finetuning_advanced import ECCEpochUpdateCallback
    import torch.nn as nn

    class _FakeModelWithAux(nn.Module):
        def __init__(self, ecc_module):
            super().__init__()
            self.aux_modules = nn.ModuleDict({"ecc": ecc_module})

    class _FakeState:
        epoch = 1.0

    _fake_model = _FakeModelWithAux(AUX_REGISTRY["ecc"](in_features=IN_FEATURES, num_classes=num_labels))
    ECCEpochUpdateCallback().on_epoch_end(args=None, state=_FakeState(), control=None, model=_fake_model)
    log("ECCEpochUpdateCallback.on_epoch_end() OK against a real nn.ModuleDict")

    log("=== ecc aux (on top of a pooled feature + labels) ===")
    ecc = AUX_REGISTRY["ecc"](in_features=IN_FEATURES, num_classes=num_labels)
    x = torch.randn(8, IN_FEATURES, requires_grad=True)
    y = torch.randint(0, num_labels, (8,))
    loss = ecc(x, y)
    assert torch.isfinite(loss), f"ecc: non-finite loss: {loss}"
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "ecc: non-finite/missing gradient"
    log(f"ecc: forward+backward OK, loss={loss.item():.4f}")
    # Same epoch-update exercise as kappaface above: only a handful of classes have
    # been observed (this batch's 8 labels), recompute_similarity() must not crash
    # on that sparse, uneven real-world state.
    ecc.recompute_similarity()
    assert torch.isfinite(ecc.sim_value).all(), "ecc: non-finite sim_value after recompute"
    # The whole point of the fix: sim_index must move OFF its self-referential
    # init value (torch.arange(num_classes)) for at least the classes just seen,
    # or the "push away from most similar OTHER class" term is pushing a sample
    # away from its own center instead.
    seen = y.unique()
    still_self = (ecc.sim_index[seen] == seen).all()
    assert not still_self, ("ecc: sim_index still self-referential for every observed class "
                             "after recompute_similarity() -- the fix did not take effect")
    log(f"ecc: recompute_similarity() OK, sim_index moved off self-reference for observed classes")

    log("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
