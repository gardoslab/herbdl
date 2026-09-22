"""The on-disk contract between a SwinWithLoss checkpoint and repgeom's
geometry analysis, WITHOUT repgeom ever importing loss_ablation's Python code.

Why this indirection exists, not a design preference: `repgeom/` and
`finetuning/SWIN/loss_ablation/` live on different git branches in this
project (repgeom descends from the interpretability-geometry root; this
registry descends from the ArcFace-ablation node), so a direct Python import
across them is not reliably available -- whichever branch is checked out at
analysis time may not have the other's code. A plain file contract sidesteps
that entirely: repgeom only ever needs to read a handful of files, never load
a loss_ablation class.

Two separate purposes, two separate files -- this distinction matters and
collapsing it into one file was a real bug caught before it shipped (see
below):

  class_directions.pt   (n_species, C), from model.class_directions(). What
                         geometry comparisons (clouds, tightness, nearest-class
                         distance, detector alignment -- Sections 4-5 of the
                         interpretability report) need: ONE representative
                         direction per class. For sub-center heads (k>1) this
                         is deliberately only the first sub-center -- a fine
                         choice for comparing class identity, since the
                         sub-centers of one class sit close together by
                         construction.

  head_inference_weight.pt   (n_species, sub_centers, C), from
                         model.head_inference_info(). What EXACT logit
                         reconstruction (per-class recall, spatial evidence --
                         Sections 2-3, 6) needs: ALL of a class's directions,
                         because sub-center ArcFace's actual decision is a MAX
                         over them, and using only sub-center 0 there silently
                         reproduces the wrong prediction whenever a different
                         sub-center wins for a given image. Caught by testing
                         against a live model's own forward(), not by
                         inspection -- see model.py's head_inference_info()
                         docstring for the full account. sub_centers is 1 for
                         every head except subcenter_arcface.

Why exact logits were possible at all, once this was worked out: every head in
this registry reduces, at inference (labels=None, no margin/no aux loss), to
exactly one of two formulas (verified directly from each head's own
forward()):

    "linear"  logit_c = w_c . x + b_c              (SoftmaxHead; HACE's leaf rows)
    "cosine"  logit_c = scale * max_j cosine(w_c_j, x)   (every margin head)

So the export also carries `head_bias.pt` (present only for "linear" heads,
which do not exist for cosine heads to begin with) and `head_kind`/`scale`/
`sub_centers` in the meta JSON -- enough for a reader to compute exact logits
with a few lines of plain tensor arithmetic, no per-head Python class required
on the reading side either.
"""

import json
import os

import torch


def save_geometry_export(model, output_dir, backbone_ckpt_name, meta_extra=None):
    """model: a SwinWithLoss instance. Writes:

        swin_state_dict.pt          model.swin's own state_dict -- loadable
                                     into a plain SwinModel/Swinv2Model built
                                     from `backbone_ckpt_name`, nothing
                                     loss_ablation-specific in this file at all
        class_directions.pt         (n_species, C) -- geometry comparisons only
        head_inference_weight.pt    (n_species, sub_centers, C) -- exact logits
        head_bias.pt                 present only when head_kind == "linear"
        geometry_export_meta.json   backbone_ckpt_name, n_species, feature_dim,
                                     head_kind, scale, sub_centers, has_bias,
                                     plus whatever bookkeeping the caller wants
                                     recorded -- head_kind/scale/sub_centers
                                     are NOT informational, a reader needs them
                                     to compute a correct logit
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.swin.state_dict(), os.path.join(output_dir, "swin_state_dict.pt"))

    cd = model.class_directions().detach().cpu()
    torch.save(cd, os.path.join(output_dir, "class_directions.pt"))

    W, bias, head_kind, scale, sub_centers = model.head_inference_info()
    W = W.detach().cpu()
    if W.shape != (cd.shape[0], sub_centers, cd.shape[1]):
        raise RuntimeError(
            f"head_inference_info() returned W of shape {tuple(W.shape)}, expected "
            f"({cd.shape[0]}, {sub_centers}, {cd.shape[1]}) -- internally "
            "inconsistent with class_directions() or the reported sub_centers, "
            "do not export this")
    torch.save(W, os.path.join(output_dir, "head_inference_weight.pt"))
    if bias is not None:
        torch.save(bias.detach().cpu(), os.path.join(output_dir, "head_bias.pt"))

    meta = {
        "backbone_ckpt_name": backbone_ckpt_name,
        "n_species": int(cd.shape[0]),
        "feature_dim": int(cd.shape[1]),
        "head_kind": head_kind,
        "scale": scale,
        "sub_centers": int(sub_centers),
        "has_bias": bias is not None,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(output_dir, "geometry_export_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def is_geometry_export(checkpoint_dir):
    """What repgeom's loader checks first, before falling back to the
    existing HF classifier.weight/arcface.weight path -- see this module's
    docstring for why no Python import is involved on either side."""
    return os.path.exists(os.path.join(checkpoint_dir, "geometry_export_meta.json"))


def load_geometry_export(checkpoint_dir):
    """Returns (swin_state_dict, class_directions, meta). Class-comparison
    geometry only -- see load_inference_weights below for exact-logit
    reconstruction, kept as a separate call since most callers (the existing,
    already-validated geometry-only path) do not need it."""
    with open(os.path.join(checkpoint_dir, "geometry_export_meta.json")) as fh:
        meta = json.load(fh)
    swin_sd = torch.load(os.path.join(checkpoint_dir, "swin_state_dict.pt"),
                         map_location="cpu")
    cd = torch.load(os.path.join(checkpoint_dir, "class_directions.pt"),
                    map_location="cpu")
    if cd.shape != (meta["n_species"], meta["feature_dim"]):
        raise RuntimeError(
            f"class_directions.pt shape {tuple(cd.shape)} does not match "
            f"meta {(meta['n_species'], meta['feature_dim'])} in {checkpoint_dir} "
            "-- the export is internally inconsistent, do not trust it")
    return swin_sd, cd, meta


def load_inference_weights(checkpoint_dir, meta=None):
    """Returns (W, bias_or_None, head_kind, scale, sub_centers) for exact
    logit reconstruction:

        if head_kind == "linear":
            logit = pooled @ W.T + bias                         # W: (n_species, C)
        else:  # "cosine"
            en = normalize(pooled, dim=1)                        # (B, C)
            wn = normalize(W, dim=2)                              # (n_species, sub_centers, C)
            cos = einsum('bc,nkc->bnk', en, wn)
            logit = scale * cos.max(dim=2).values                # (B, n_species)

    `bias` is None whenever meta['has_bias'] is False (every "cosine" head has
    no bias term to begin with), not a caller error. Pass an already-loaded
    `meta` to avoid reading geometry_export_meta.json twice if the caller
    already has it (e.g. from load_geometry_export)."""
    if meta is None:
        with open(os.path.join(checkpoint_dir, "geometry_export_meta.json")) as fh:
            meta = json.load(fh)
    n, c = meta["n_species"], meta["feature_dim"]
    k = meta["sub_centers"]
    W = torch.load(os.path.join(checkpoint_dir, "head_inference_weight.pt"), map_location="cpu")
    if W.shape != (n, k, c):
        raise RuntimeError(
            f"head_inference_weight.pt shape {tuple(W.shape)} does not match "
            f"meta ({n}, {k}, {c}) in {checkpoint_dir} -- the export is "
            "internally inconsistent, do not trust it")
    bias = None
    if meta.get("has_bias"):
        bias = torch.load(os.path.join(checkpoint_dir, "head_bias.pt"), map_location="cpu")
        if bias.shape != (n,):
            raise RuntimeError(
                f"head_bias.pt shape {tuple(bias.shape)} does not match "
                f"meta n_species={n} in {checkpoint_dir} -- "
                "the export is internally inconsistent, do not trust it")
    return W, bias, meta["head_kind"], meta["scale"], k
