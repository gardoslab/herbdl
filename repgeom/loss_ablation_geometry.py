"""Reads checkpoints from the loss-function-ablation registry
(finetuning/SWIN/loss_ablation/, a different git branch) and compares them
using this project's existing geometry.py and spatial.py, unchanged.

Why a separate file format rather than a Python import: repgeom and
loss_ablation live on different branches in this project's tree (repgeom
descends from the interpretability-geometry root; loss_ablation descends from
the ArcFace-ablation node), so whichever is checked out, the other's code may
not be present in the worktree. loss_ablation/checkpoint_io.py writes a plain
backbone state_dict, two weight tensors, and a small JSON meta file, and this
module only ever reads those, never a loss_ablation class. The functions below
deliberately re-implement that tiny read (rather than import checkpoint_io)
for the same reason.

Two things this module computes, from two different exported tensors:

  geometry (clouds, tightness, nearest-class distance, detector alignment)
      needs pooled features (a plain backbone forward pass, head-agnostic) and
      `class_directions.pt` -- ONE representative direction per class. Never
      needs a head's specific logit math.

  recall and spatial evidence need REAL logits, which turned out to be
      possible without any head-specific Python code on this side either:
      every head in loss_ablation's registry reduces, at inference (no
      margin/no aux loss), to exactly one of two formulas (see
      finetuning/SWIN/loss_ablation/model.py's head_inference_info()
      docstring for the full derivation and the two real bugs caught proving
      it):

          "linear"  logit_c = w_c . x + b_c
          "cosine"  logit_c = scale * max_j cos(w_c_j, x)

      Uses `head_inference_weight.pt` (n_species, sub_centers, C) for this,
      NOT class_directions.pt -- for sub-center heads those are different
      tensors on purpose (class_directions.pt is deliberately only
      sub-center 0; using it here would silently reproduce the wrong
      prediction whenever a different sub-center wins for a given image).

Spatial evidence under a cosine head, worked out here rather than assumed:
Swin's pooler is `pooled = mean_p(h_p)` over the 49 final tokens, so a LINEAR
logit decomposes exactly as `mean_p(w_c . h_p) + b_c` (this project's original
finding, extract.py). A cosine logit does not decompose that cleanly --
normalisation is nonlinear -- but does up to one caveat: `scale * cos(w_c, x)
= (scale / ||x||) * (w_c_hat . x) = (scale / ||x||) * mean_p(w_c_hat . h_p)`.
The per-position raw contributions `w_c_hat . h_p` are therefore a faithful
DECOMPOSITION up to `scale/||x||`, a single positive scalar constant across
all 49 positions of one image. That constant cancels exactly in
spatial.py's `evidence_share()`, which shifts and renormalises each image's
map to sum to 1 -- a per-image positive multiplier does not change the
relative shares. For a sub-center head, `w_c_hat` is the WINNING sub-center's
direction for that specific (image, class) pair -- whichever one the max
actually selected -- not an arbitrary one.

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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


def load_export(export_dir):
    """The read half of loss_ablation/checkpoint_io.py's class-comparison
    contract, reimplemented rather than imported -- see module docstring."""
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


def load_inference_weights(export_dir, meta):
    """The read half of loss_ablation/checkpoint_io.py's exact-logit
    contract, reimplemented rather than imported. Returns
    (W, bias_or_None, kind, scale, sub_centers). Requires
    meta['head_kind']/['scale']/['sub_centers'] -- exports written before
    those fields existed (this branch's earlier geometry-only runs) do not
    have this data and will KeyError here rather than silently compute
    nothing, which is the correct failure mode."""
    n, c = meta["n_species"], meta["feature_dim"]
    k = meta["sub_centers"]
    W = torch.load(os.path.join(export_dir, "head_inference_weight.pt"), map_location="cpu")
    if W.shape != (n, k, c):
        raise RuntimeError(
            f"head_inference_weight.pt shape {tuple(W.shape)} does not match "
            f"meta ({n}, {k}, {c}) in {export_dir} -- the export is "
            "internally inconsistent, do not trust it")
    bias = None
    if meta.get("has_bias"):
        bias = torch.load(os.path.join(export_dir, "head_bias.pt"), map_location="cpu")
        if bias.shape != (n,):
            raise RuntimeError(
                f"head_bias.pt shape {tuple(bias.shape)} does not match "
                f"meta n_species={n} in {export_dir} -- the export is "
                "internally inconsistent, do not trust it")
    return W, bias, meta["head_kind"], meta["scale"], k


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


def _load_backbone(meta, swin_sd, device):
    proc = AutoImageProcessor.from_pretrained(meta["backbone_ckpt_name"])
    backbone = AutoModel.from_pretrained(meta["backbone_ckpt_name"])
    load_report = backbone.load_state_dict(swin_sd, strict=False)
    if load_report.missing_keys or load_report.unexpected_keys:
        raise RuntimeError(
            f"backbone state_dict from export did not load cleanly: "
            f"missing={load_report.missing_keys[:5]} "
            f"unexpected={load_report.unexpected_keys[:5]}")
    return proc, backbone.to(device).eval()


@torch.no_grad()
def extract_pooled_features(swin_sd, meta, rows, device, batch_size=16):
    """Plain backbone forward pass, pooled output only -- head-agnostic by
    construction, which is the whole point of this module (see the file
    docstring). Used by the geometry-only path."""
    proc, backbone = _load_backbone(meta, swin_sd, device)
    loader = DataLoader(SubsetImages(rows, proc), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    for px, sp in loader:
        out = backbone(px.to(device))
        feats.append(out.pooler_output.float().cpu())
        labels.append(sp)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def _logits_and_spatial(h, pooled, y, W, bias, kind, scale, device):
    """h: (B, 49, C) per-position hidden states. pooled: (B, C) == h.mean(1).
    y: (B,) true labels. Returns (logits (B,n_species), spat_true (B,49),
    spat_pred (B,49)) -- see module docstring for the cosine-head derivation.
    """
    W = W.to(device)
    if kind == "linear":
        Wc = W[:, 0, :]  # sub_centers is always 1 here
        logits = pooled @ Wc.T + bias.to(device)
        pred = logits.argmax(dim=1)
        spat_true = torch.einsum("bpc,bc->bp", h, Wc[y])
        spat_pred = torch.einsum("bpc,bc->bp", h, Wc[pred])
        return logits, spat_true, spat_pred

    # cosine: need the WINNING sub-center per (image, class-of-interest) pair,
    # not an arbitrary one -- see module docstring.
    en = F.normalize(pooled, dim=1)                 # (B, C)
    wn = F.normalize(W, dim=2)                       # (n_species, k, C)
    cos = torch.einsum("bc,nkc->bnk", en, wn)         # (B, n_species, k)
    logits = scale * cos.max(dim=2).values            # (B, n_species)
    winning_k = cos.argmax(dim=2)                      # (B, n_species)
    pred = logits.argmax(dim=1)

    B = pooled.shape[0]
    idx = torch.arange(B, device=device)
    k_true = winning_k[idx, y]                        # (B,)
    k_pred = winning_k[idx, pred]                      # (B,)
    w_true_hat = wn[y, k_true]                          # (B, C)
    w_pred_hat = wn[pred, k_pred]                        # (B, C)
    spat_true = torch.einsum("bpc,bc->bp", h, w_true_hat)
    spat_pred = torch.einsum("bpc,bc->bp", h, w_pred_hat)
    return logits, spat_true, spat_pred


@torch.no_grad()
def extract_predictions_and_spatial(swin_sd, W, bias, kind, scale, meta, rows, device, batch_size=16):
    """Real logits, predictions, and the 49-position spatial decomposition for
    the true and predicted class -- what per-class recall and spatial
    evidence need, unlike the geometry-only path above."""
    proc, backbone = _load_backbone(meta, swin_sd, device)
    W_dev = W.to(device)
    bias_dev = bias.to(device) if bias is not None else None

    loader = DataLoader(SubsetImages(rows, proc), batch_size=batch_size, shuffle=False)
    labels_all, pred_all, rank_all = [], [], []
    spat_true_all, spat_pred_all = [], []
    for px, sp in loader:
        px, y = px.to(device), sp.to(device)
        out = backbone(px)
        h = out.last_hidden_state.float()
        pooled = out.pooler_output.float()
        logits, spat_true, spat_pred = _logits_and_spatial(
            h, pooled, y, W_dev, bias_dev, kind, scale, device)
        tl = logits.gather(1, y[:, None]).squeeze(1)
        rank = (logits > tl[:, None]).sum(dim=1)
        pred = logits.argmax(dim=1)

        labels_all.append(sp)
        pred_all.append(pred.cpu())
        rank_all.append(rank.cpu())
        spat_true_all.append(spat_true.cpu().numpy())
        spat_pred_all.append(spat_pred.cpu().numpy())

    labels = torch.cat(labels_all).numpy()
    pred = torch.cat(pred_all).numpy()
    rank = torch.cat(rank_all).numpy()
    spat_true = np.concatenate(spat_true_all)
    spat_pred = np.concatenate(spat_pred_all)
    return labels, pred, rank, spat_true, spat_pred


def compare_configs(export_root, device):
    """For every export directory under `export_root`: geometry (always), and
    recall + spatial evidence (when the export has the metadata for it --
    exports written before head_inference_weight.pt existed do not, and are
    reported as geometry-only rather than silently skipped or crashing).
    Returns a list of per-config dicts.
    """
    import geometry  # this project's, unchanged
    import spatial   # this project's, unchanged

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

        W_cls = cd.to(device)
        coll, _dual_present = geometry.collapse_metrics(feat_t, labels, mu, mu_g, W_cls, present, device)

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
        print(f"__LAG__: intra_class_cos={r['intra_class_cos']:.4f}  "
              f"nearest_class_cos={r['nearest_class_cos']:.4f}  "
              f"self_duality_cos={r['self_duality_cos']:.4f}")

        has_inference_info = all(k in meta for k in ("head_kind", "scale", "sub_centers"))
        if not has_inference_info:
            print("__LAG__:   (no head_inference_weight.pt metadata in this export -- "
                  "geometry only, recall/spatial evidence skipped for this config)")
            results.append(r)
            continue

        W_inf, bias, kind, scale, sub_centers = load_inference_weights(export_dir, meta)
        y2, pred, rank, spat_true, spat_pred = extract_predictions_and_spatial(
            swin_sd, W_inf, bias, kind, scale, meta, rows, device)
        assert np.array_equal(y2, labels), "label order drifted between the two forward passes"

        correct = (y2 == pred)
        acc = float(correct.mean())
        rec = spatial.per_class_mean(correct.astype(np.float32), y2, n_species)
        macro_recall = float(np.nanmean(rec[present]))
        r["top1_accuracy"] = acc
        r["macro_recall"] = macro_recall
        r["mean_true_rank"] = float(rank.mean())
        print(f"__LAG__: top1_accuracy={acc:.4f}  macro_recall={macro_recall:.4f}  "
              f"mean_true_rank={rank.mean():.1f}  (kind={kind}, scale={scale}, "
              f"sub_centers={sub_centers})")

        share = spatial.evidence_share(spat_true)
        diff = spatial.concentration(share)
        rm_all = spatial.region_mass(share)
        rm_correct = spatial.region_mass(share[correct]) if correct.any() else {}
        rm_errors = spatial.region_mass(share[~correct]) if (~correct).any() else {}
        r["evidence_diffuseness"] = float(diff.mean())
        r["border_ring_mass"] = rm_all.get("border_ring", float("nan"))
        r["bottom_right_2x2_mass"] = rm_all.get("bottom_right_2x2", float("nan"))
        print(f"__LAG__: evidence_diffuseness={diff.mean():.4f}  "
              f"border_ring_mass={rm_all.get('border_ring', float('nan')):.4f} "
              f"(uniform={rm_all.get('uniform_border_ring', float('nan')):.4f})")
        if rm_correct:
            print(f"__LAG__:   border_ring_mass[correct]={rm_correct['border_ring']:.4f}  "
                  f"border_ring_mass[errors]={rm_errors.get('border_ring', float('nan')):.4f}")

        results.append(r)

    return results
