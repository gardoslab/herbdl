"""Head-swap 2x2: is DBM's macro-F1 advantage in the classifier or in the features?

The geometry study says DBM's species clouds are *less* separated than ArcFace's, yet
DBM scores higher macro-F1.  The gap between those two facts is the classifier: the
head decides where the boundaries fall, given the features.  This script separates the
two by retraining classifiers on frozen features.

    bodies  dbm | arcface      the two finetuned SWIN-B 384 checkpoints
    heads   dbm | arcface | softmax

Each body's pooled features are extracted once over train_2022 by the eval-only configs
in configs_advanced/phase3_geom_train/ (do_train: false, one forward pass per image).
Every head is then trained on a body's frozen train features with the same protocol
(AdamW, 30 epochs, batch 4096, OneCycle) and scored on the same cached val_2022 features
with the project's macro-F1 (mean per-class F1 over classes present in labels or
predictions -- phase3_per_species.py's definition).

Reading the result:
  diagonal cells        a fresh head on its own body: does retraining reproduce the run?
  off-diagonal cells    the retrained swap: DBM head on ArcFace features and vice versa
  zero-shot rows        the trained classifier of one body applied to the other body's
                        features with no retraining -- an intervention on the readout,
                        not a probe, and the cleaner test of head vs body
  original rows         the checkpoint's own head, the anchor (must match the paper numbers)
  softmax rows          neutral linear probe: no angular margin at all

If the DBM head wins on both bodies, the advantage is the head.  If the DBM body wins
under both heads, it is the features.  If only the matched pairs win, it is co-adaptation.

Inputs (SCC paths):
  PHASE3_GEOM_TRAIN/<body>/features.npy      train pooled features, fp16
  PHASE3_GEOM_TRAIN/<body>/predictions.npz   train labels, same order
  PHASE3_GEOM/<body>/features.npy            val pooled features (already cached)
  PHASE3_GEOM/<body>/predictions.npz         val labels

Output: PHASE3_GEOM/analysis/head_swap.json

Usage: python loss_ablation/head_swap.py [--epochs 30] [--lr 1e-3] [--bodies dbm arcface]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import losses  # noqa: E402

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
GEOM = f"{OUT}/PHASE3_GEOM"
TRAIN = f"{OUT}/PHASE3_GEOM_TRAIN"
N = 15501
BODY_RUN = {  # checkpoint directory, and the tensor name class_directions expects
    "dbm": ("SWIN_BASE_DBM_384_PILOT", "registry"),
    "arcface": ("SWIN_BASE_ARCFACE_384", "arcface"),
}
HEADS = ("dbm", "arcface", "softmax")


def macro_f1(y, pred, n_classes=N):
    """Mean per-class F1 over classes present in labels OR predictions (project metric)."""
    # boundary_scores returns its prediction array as float32; bincount needs integers.
    pred = np.asarray(pred, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    tp = np.bincount(y[pred == y], minlength=n_classes).astype(np.float64)
    n_true = np.bincount(y, minlength=n_classes).astype(np.float64)
    n_pred = np.bincount(pred, minlength=n_classes).astype(np.float64)
    present = (n_true + n_pred) > 0
    f1 = 2 * tp[present] / (n_true[present] + n_pred[present])
    return float(f1.mean())


def scores(y, pred):
    pred = np.asarray(pred, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    return {"accuracy": float(np.mean(pred == y)), "macro_f1": macro_f1(y, pred),
            "n_images": int(len(y))}


@torch.no_grad()
def eval_head(head, feats, y, device, batch=4096):
    preds = []
    for lo in range(0, len(y), batch):
        x = torch.as_tensor(np.asarray(feats[lo:lo + batch], dtype=np.float32), device=device)
        preds.append(head(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_head(name, feats, y, epochs, lr, batch, device, seed=42):
    """Fit one classifier on frozen features; identical protocol for every head."""
    torch.manual_seed(seed)
    kwargs = {}
    if name == "dbm":
        kwargs["class_counts"] = np.bincount(y, minlength=N)
    head = losses.build_head(name, feats.shape[1], N, **kwargs).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(y) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        total = 0.0
        for lo in range(0, len(order), batch):
            idx = order[lo:lo + batch]
            x = torch.as_tensor(np.asarray(feats[idx], dtype=np.float32), device=device)
            t = torch.as_tensor(y[idx], device=device)
            loss = F.cross_entropy(head(x, t), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        print(f"__HEADSWAP__   {name} epoch {epoch + 1}/{epochs} "
              f"loss={total / len(y):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", nargs="*", default=list(BODY_RUN))
    ap.add_argument("--heads", nargs="*", default=list(HEADS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    from phase3_geometry import class_directions  # needs safetensors; val reference only
    from boundary_margins import boundary_scores  # reuse the same cosine-argmax math

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"__HEADSWAP__ device={device} epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch}", flush=True)

    results = {"protocol": {"epochs": args.epochs, "lr": args.lr, "batch": args.batch,
                            "optimizer": "AdamW wd=0.01, OneCycle",
                            "train_split": "train_2022 (eval-only feature pass)",
                            "eval_split": "val_2022 (167,955 images)"},
               "reference": {}, "zero_shot": {}, "cells": {}}

    for body in args.bodies:
        tf = f"{TRAIN}/{body}/features.npy"
        tp = f"{TRAIN}/{body}/predictions.npz"
        vf = f"{GEOM}/{body}/features.npy"
        vp = f"{GEOM}/{body}/predictions.npz"
        for path in (tf, tp, vf, vp):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        train_feats = np.load(tf, mmap_mode="r")
        train_y = np.load(tp)["labels"].astype(np.int64)
        val_feats = np.load(vf, mmap_mode="r")
        val_y = np.load(vp)["labels"].astype(np.int64)
        assert len(train_feats) == len(train_y), body
        assert len(val_feats) == len(val_y), body
        print(f"__HEADSWAP__ body {body}: train {train_feats.shape} {train_feats.dtype}, "
              f"val {val_feats.shape}", flush=True)

        # The checkpoint's own head: the published number, computed from the same features.
        run_dir, kind = BODY_RUN[body]
        dirs = class_directions(run_dir, kind)
        val_np = np.asarray(val_feats, dtype=np.float32)
        ref_pred = boundary_scores(val_np, val_y, dirs, device)["predicted"]
        results["reference"][body] = scores(val_y, ref_pred)
        print(f"__HEADSWAP__ original {body} head: "
              f"acc={results['reference'][body]['accuracy']:.4f} "
              f"macro_f1={results['reference'][body]['macro_f1']:.4f}", flush=True)

        # Zero-shot readout swap: apply the OTHER body's trained classifier to these frozen
        # features, with no retraining. This is an intervention on the readout, not a probe,
        # and it is the cleanest test of whether a fixed set of class directions transfers.
        for other in args.bodies:
            if other == body:
                continue
            other_dirs = class_directions(*BODY_RUN[other])
            swap_pred = boundary_scores(val_np, val_y, other_dirs, device)["predicted"]
            key = f"{body}_features/{other}_head"
            results["zero_shot"][key] = scores(val_y, swap_pred)
            print(f"__HEADSWAP__ zero-shot {other} head on {body} features: "
                  f"acc={results['zero_shot'][key]['accuracy']:.4f} "
                  f"macro_f1={results['zero_shot'][key]['macro_f1']:.4f}", flush=True)
            del other_dirs

        for head_name in args.heads:
            head = train_head(head_name, train_feats, train_y, args.epochs, args.lr,
                              args.batch, device)
            pred = eval_head(head, val_feats, val_y, device)
            cell = scores(val_y, pred)
            cell["body"] = body
            cell["head"] = head_name
            results["cells"][f"{body}/{head_name}"] = cell
            print(f"__HEADSWAP__ {body} body + {head_name} head: "
                  f"acc={cell['accuracy']:.4f} macro_f1={cell['macro_f1']:.4f}", flush=True)
            del head
            if device == "cuda":
                torch.cuda.empty_cache()

        del train_feats, val_feats, dirs

    # The 2x2 contrast, written out so the answer does not depend on reading the table.
    cells, ref = results["cells"], results["reference"]
    if all(k in cells for k in ("dbm/dbm", "arcface/dbm", "dbm/arcface", "arcface/arcface")):
        results["contrast"] = {
            "dbm_head_minus_arcface_head_on_dbm_body":
                cells["dbm/dbm"]["macro_f1"] - cells["dbm/arcface"]["macro_f1"],
            "dbm_head_minus_arcface_head_on_arcface_body":
                cells["arcface/dbm"]["macro_f1"] - cells["arcface/arcface"]["macro_f1"],
            "dbm_body_minus_arcface_body_with_dbm_head":
                cells["dbm/dbm"]["macro_f1"] - cells["arcface/dbm"]["macro_f1"],
            "dbm_body_minus_arcface_body_with_arcface_head":
                cells["dbm/arcface"]["macro_f1"] - cells["arcface/arcface"]["macro_f1"],
            "retrained_minus_original_dbm":
                cells["dbm/dbm"]["macro_f1"] - ref["dbm"]["macro_f1"],
            "retrained_minus_original_arcface":
                cells["arcface/arcface"]["macro_f1"] - ref["arcface"]["macro_f1"],
        }
        for k, v in results["contrast"].items():
            print(f"__HEADSWAP__ contrast {k}: {v:+.4f} macro-F1", flush=True)

    os.makedirs(f"{GEOM}/analysis", exist_ok=True)
    out = f"{GEOM}/analysis/head_swap.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"__HEADSWAP__ wrote {out}", flush=True)


if __name__ == "__main__":
    main()
