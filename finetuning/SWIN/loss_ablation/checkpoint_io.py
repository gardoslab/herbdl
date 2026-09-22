"""The on-disk contract between a SwinWithLoss checkpoint and repgeom's
geometry analysis, WITHOUT repgeom ever importing loss_ablation's Python code.

Why this indirection exists, not a design preference: `repgeom/` and
`finetuning/SWIN/loss_ablation/` live on different git branches in this
project (repgeom descends from the interpretability-geometry root; this
registry descends from the ArcFace-ablation node), so a direct Python import
across them is not reliably available -- whichever branch is checked out at
analysis time may not have the other's code. A plain file contract sidesteps
that entirely: repgeom only ever needs to read three files, never load a
loss_ablation class.

The insight that makes this contract small: repgeom's geometry analysis
(clouds, tightness, nearest-class distance, detector alignment -- Sections
4-5 of the interpretability report) needs exactly two things that do NOT
depend on which head produced them --

    1. pooled features, which come from `model.swin(pixel_values).pooler_output`
       -- a plain backbone forward pass, identical machinery regardless of
       whether an arcface/kappaface/dbm/... head sits on top
    2. the (n_species, C) class-direction matrix, which every head already
       exposes via `class_directions()` -- a single tensor, head-specific to
       PRODUCE but head-agnostic to READ

Neither requires reproducing a head's specific forward/logit math. What this
contract deliberately does NOT cover: per-class recall, error structure, or
spatial evidence (Sections 2-3, 6 of the report), which DO need real logits
and therefore real head-specific inference code -- out of scope here, tracked
as a separate, larger task in Loss Function Ablations.md rather than silently
implied by this file.
"""

import json
import os

import torch


def save_geometry_export(model, output_dir, backbone_ckpt_name, meta_extra=None):
    """model: a SwinWithLoss instance. Writes:

        swin_state_dict.pt        model.swin's own state_dict -- loadable into
                                   a plain SwinModel/Swinv2Model built from
                                   `backbone_ckpt_name`, nothing loss_ablation-
                                   specific in this file at all
        class_directions.pt       the (n_species, C) tensor from
                                   model.class_directions()
        geometry_export_meta.json backbone_ckpt_name, n_species, feature_dim,
                                   plus whatever bookkeeping the caller wants
                                   recorded (head name, hyperparameters, ...)
                                   -- informational only; repgeom's reader
                                   does not need to interpret it
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.swin.state_dict(), os.path.join(output_dir, "swin_state_dict.pt"))
    cd = model.class_directions().cpu()
    torch.save(cd, os.path.join(output_dir, "class_directions.pt"))
    meta = {
        "backbone_ckpt_name": backbone_ckpt_name,
        "n_species": int(cd.shape[0]),
        "feature_dim": int(cd.shape[1]),
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
    """Returns (swin_state_dict, class_directions, meta). The caller (repgeom)
    is responsible for instantiating a plain SwinModel/Swinv2Model from
    meta['backbone_ckpt_name'] and loading swin_state_dict into it -- this
    function does no model construction, only file I/O, so it has no
    dependency on `transformers` at all."""
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
