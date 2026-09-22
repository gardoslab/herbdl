"""Reads checkpoints from the loss-function-ablation registry
(finetuning/SWIN/loss_ablation/, a different git branch) and compares their
geometry using this project's existing geometry.py, unchanged.

Why a separate file format rather than a Python import: repgeom and
loss_ablation live on different branches in this project's tree (repgeom
descends from the interpretability-geometry root; loss_ablation descends from
the ArcFace-ablation node), so whichever is checked out, the other's code may
not be present in the worktree. loss_ablation/checkpoint_io.py writes three
files -- a plain backbone state_dict, the (n_species, C) class_directions
tensor, and a small JSON meta file -- and this module only ever reads those,
never a loss_ablation class. The functions below deliberately re-implement
that tiny read (rather than import checkpoint_io) for the same reason.

Why this is enough for geometry (but not everything repgeom can do): pooled
features come from a plain backbone forward pass -- identical machinery no
matter which head sits on top -- so within-class tightness, nearest-class
distance, and detector alignment never need a head's specific logit math, only
its (n_species, C) class-direction matrix. Per-class recall, error structure
and spatial evidence DO need real head-specific inference and are out of scope
here -- not silently implied, just not built.

Preprocessing, stated explicitly because getting this wrong is exactly the
kind of silent, plausible-looking bug this project has hit twice already
(SizeDict, bf16-autocast scatter): the loss_ablation E2E harness builds its
pixel tensors via a bare `AutoImageProcessor(...)` call -- the SAME "processor"
mode this file's own extract.py calls out as the bicubic-squash pipeline, NOT
the `eval_protocol` transform (which matches a DIFFERENT script,
SWIN_finetuning_advanced.py's _val_transforms). Using eval_protocol here would
silently measure a different set of pooled features than what the exported
backbone actually saw. This module uses AutoImageProcessor directly, to match.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


def load_export(export_dir):
    """The read half of loss_ablation/checkpoint_io.py's contract,
    reimplemented rather than imported -- see module docstring."""
    with open(os.path.join(export_dir, "geometry_export_meta.json")) as fh:
        meta = json.load(fh)
    swin_sd = torch.load(os.path.join(export_dir, "swin_state_dict.pt"), map_location="cpu")
    cd = torch.load(os.path.join(export_dir, "class_directions.pt"), map_location="cpu")
    if cd.shape != (meta["n_species"], meta["feature_dim"]):
        raise RuntimeError(
            f"class_directions.pt shape {tuple(cd.shape)} does not match "
            f"meta {(meta['n_species'], meta['feature_dim'])} in {export_dir} "
            "-- the export is internally inconsistent, do not trust it")
    return swin_sd, cd, meta


class SubsetImages(Dataset):
    """Pixel tensors for the shared subset_manifest.json every loss_ablation
    export in one harness run was trained/held out on."""

    def __init__(self, rows, proc):
        self.rows = rows
        self.proc = proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(row["filename"]).convert("RGB")
        px = self.proc(images=img, return_tensors="pt").pixel_values[0]
        return px, row["species_id"]


@torch.no_grad()
def extract_pooled_features(swin_sd, meta, rows, device, batch_size=16):
    """Plain backbone forward pass -- head-agnostic by construction, which is
    the whole point of this module (see the file docstring)."""
    proc = AutoImageProcessor.from_pretrained(meta["backbone_ckpt_name"])
    backbone = AutoModel.from_pretrained(meta["backbone_ckpt_name"])
    load_report = backbone.load_state_dict(swin_sd, strict=False)
    if load_report.missing_keys or load_report.unexpected_keys:
        raise RuntimeError(
            f"backbone state_dict from export did not load cleanly: "
            f"missing={load_report.missing_keys[:5]} "
            f"unexpected={load_report.unexpected_keys[:5]}")
    backbone = backbone.to(device).eval()

    loader = DataLoader(SubsetImages(rows, proc), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    for px, sp in loader:
        out = backbone(px.to(device))
        feats.append(out.pooler_output.float().cpu())
        labels.append(sp)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def compare_configs(export_root, device):
    """The actual comparison: for every export directory under `export_root`,
    extract pooled features on the shared subset, then run this project's
    existing class_centroids/intra_class_spread/nearest_neighbour against the
    exported class_directions. Returns a list of per-config dicts.
    """
    import geometry  # this project's, unchanged

    with open(os.path.join(export_root, "subset_manifest.json")) as fh:
        manifest = json.load(fh)
    rows = manifest["train_rows"] + manifest["val_rows"]
    n_species = manifest["n_species"]
    print(f"__LAG__: subset_manifest: {len(rows)} images, {n_species} species "
          f"(seed={manifest.get('seed')})")

    config_dirs = sorted(d for d in os.listdir(export_root)
                         if os.path.isdir(os.path.join(export_root, d)))
    results = []
    for cfg_name in config_dirs:
        export_dir = os.path.join(export_root, cfg_name)
        try:
            swin_sd, cd, meta = load_export(export_dir)
        except FileNotFoundError:
            continue  # not an export dir (e.g. a stray directory)
        print(f"\n__LAG__ ===== {cfg_name} " + "=" * max(0, 40 - len(cfg_name)))

        feat, labels = extract_pooled_features(swin_sd, meta, rows, device)
        feat_t = torch.as_tensor(feat)

        # class_centroids already returns `present` as a plain numpy bool
        # array (see geometry.py: `present.cpu().numpy()`), not a tensor --
        # nearest_neighbour and collapse_metrics both expect it in that form.
        mu, _n_val, mu_g, present = geometry.class_centroids(feat_t, labels, n_species, device)
        intra_cos, _intra_l2 = geometry.intra_class_spread(feat_t, labels, mu, n_species, device)
        cen_sim, _cen_idx = geometry.nearest_neighbour(mu, present, device, k=2)
        nn_cos = cen_sim[:, 0]

        W = cd.to(device)
        coll, _dual_present = geometry.collapse_metrics(feat_t, labels, mu, mu_g, W, present, device)

        import numpy as np
        r = {
            "config": cfg_name,
            "n_images": len(rows),
            "intra_class_cos": float(np.nanmean(intra_cos[present])),
            "nearest_class_cos": float(np.nanmean(nn_cos[present])),
            # coll's own aggregate, not recomputed, so there is exactly one
            # source of truth for this number
            "self_duality_cos": coll["self_duality_mean_cos"],
            **{k: v for k, v in coll.items()
               if k not in ("n_classes_used", "self_duality_mean_cos")},
        }
        results.append(r)
        print(f"__LAG__: intra_class_cos={r['intra_class_cos']:.4f}  "
              f"nearest_class_cos={r['nearest_class_cos']:.4f}  "
              f"self_duality_cos={r['self_duality_cos']:.4f}")

    return results
