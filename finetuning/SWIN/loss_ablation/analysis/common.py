"""Shared constants and loaders for the loss-ablation analysis scripts.

Holds the SCC paths, the 15,501-species label setup, the rarity bands, the training-count and
taxonomy readers, the feature / prediction loader, the classifier-weight readers and the macro-F1
helpers that per_band.py, geometry.py, margins.py, head_swap.py and leakage.py import.
No computation of its own; torch and safetensors are imported inside the functions that need them,
so the numpy-only scripts (per_band.py ci / species / compare, geometry.py table) run without them.

Classifier readers (they read different checkpoint layouts, so all are kept):
  class_directions(run_dir, kind)  SWIN base runs, (N, k, C) raw rows (arcface / hace / registry head)
  read_directions(rung, loss)      SWIN ladder runs, (N, 1, C) raw rows; the class count comes from the head
  classifier(name, device)         ConvNeXt / DINOv2 base runs, (N, 1, C) unit rows and the tensor name
  ladder_classifier(rung, loss, device)  every ladder model incl. ConvNeXt / DINOv2, (N, 1, C) unit rows

Macro-F1 helpers (all give the mean per-class F1 over classes in the labels OR the predictions):
  macro_f1(tp, n_true, n_pred) from per-class counts (image-weighted counts for the bootstrap);
  macro_f1_labels(y, p) from label and prediction arrays.
"""

import json
import os

import numpy as np

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
ROOT = os.path.dirname(OUT)  # .../finetuning/output
DIAG = f"{OUT}/MARGIN_DIAG"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
VAL_JSON = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
LABEL = "scientificNameEncoded"
N = 15501

PHASE3_EVAL = f"{OUT}/PHASE3_EVAL"
PHASE3_GEOM = f"{OUT}/PHASE3_GEOM"
PHASE3_GEOM_TRAIN = f"{OUT}/PHASE3_GEOM_TRAIN"
PHASE4_GEOM = f"{OUT}/PHASE4_GEOM"
PHASE4_GEOM_TRAIN = f"{OUT}/PHASE4_GEOM_TRAIN"
BACKBONES = f"{ROOT}/BACKBONES"
LADDER_GEOM = f"{OUT}/LADDER_GEOM"
LADDER_GEOM_TRAIN = f"{OUT}/LADDER_GEOM_TRAIN"
LADDER_EVAL = f"{OUT}/LADDER_EVAL"
LADDER_DATA = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"

# Bands of training images per species. Base data: counts run 2..73. The ladder adds species with a
# single training image, so its geometry bands are extended downwards by a "1" band.
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
LADDER_BANDS = [(1, 1), (2, 10), (11, 25), (26, 50), (51, 73)]

# name -> (weights dir, head kind) of the 15.5k SWIN-B 384px base runs
P3_MODELS = {
    "arcface": ("SWIN_BASE_ARCFACE_384", "arcface"),
    "arcface_seed43": ("SWIN_BASE_ARCFACE_384_SEED43", "arcface"),
    "subcenter_arcface": ("SWIN_BASE_SUBCENTER_ARCFACE_384_PILOT", "arcface"),
    "dbm": ("SWIN_BASE_DBM_384_PILOT", "registry"),
    "dbm_seed43": ("SWIN_BASE_DBM_384_PILOT_SEED43", "registry"),
    "kappaface": ("SWIN_BASE_KAPPAFACE_384_PILOT", "registry"),
    "kappaface_seed43": ("SWIN_BASE_KAPPAFACE_384_PILOT_SEED43", "registry"),
    "arcface_tara": ("SWIN_BASE_ARCFACE_TARA_384_PILOT", "registry"),
    "arcface_ecc": ("SWIN_BASE_ARCFACE_ECC_384_PILOT", "registry"),
    "hace": ("SWIN_BASE_HACE_384_PILOT", "hace"),
    "x2softmax": ("SWIN_BASE_X2SOFTMAX_384_PILOT", "registry"),
}
RUNS = dict(P3_MODELS)
RUNS["cosine_m0"] = ("SWIN_BASE_COSINE_M0_384", "registry")
RUNS["arcface_m01"] = ("SWIN_BASE_ARCFACE_M01_384", "registry")

# The eight base-data losses of the geometry and error-taxonomy tables, and their table names.
BASE_MODELS = ["cosine_m0", "dbm", "arcface_m01", "kappaface", "subcenter_arcface", "arcface_tara",
               "arcface", "arcface_ecc"]
NAMES = {"cosine_m0": "NormFace", "dbm": "DBM", "arcface_m01": "ArcFace m=0.1", "kappaface": "KappaFace",
         "subcenter_arcface": "Sub-center", "arcface_tara": "ArcFace+TARA", "arcface": "ArcFace",
         "arcface_ecc": "ArcFace+ECC"}


# ---------------------------------------------------------------- manifests
def load_rows(path):
    """train/val manifests are one JSON object per line."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_train_counts(path=TRAIN_JSON, minlength=N):
    """Training images per species (minlength=0: as many classes as the manifest reaches)."""
    return np.bincount([json.loads(l)[LABEL] for l in open(path) if l.strip()], minlength=minlength)


def load_taxonomy():
    """(train counts, family id, genus id per species) from train_2022.json; ids are sorted-name ranks."""
    fam, gen = {}, {}
    counts = np.zeros(N, np.int64)
    for line in open(TRAIN_JSON):
        if line.strip():
            r = json.loads(line)
            c = r[LABEL]
            counts[c] += 1
            fam[c], gen[c] = r["family"], (r["family"], r["genus"])
    fam_ids = {f: i for i, f in enumerate(sorted(set(fam.values())))}
    gen_ids = {g: i for i, g in enumerate(sorted(set(gen.values())))}
    cls_fam = np.array([fam_ids[fam[c]] for c in range(N)])
    cls_gen = np.array([gen_ids[gen[c]] for c in range(N)])
    return counts, cls_fam, cls_gen


# ---------------------------------------------------------------- features and predictions
def load(path_feat, path_pred, device):
    """L2-normalised features (on device) and labels of one saved model / split."""
    import torch
    import torch.nn.functional as F
    y = np.load(path_pred)["labels"].astype(np.int64)
    feat = np.load(path_feat, mmap_mode="r")
    assert len(feat) == len(y), path_feat
    x = F.normalize(torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device), dim=1)
    return x, torch.as_tensor(y, device=device)


# ---------------------------------------------------------------- classifier weights
def class_directions(run_dir, kind):
    """(N, k, C) classifier directions read by tensor name from the model file."""
    from safetensors import safe_open
    with safe_open(f"{OUT}/{run_dir}/model.safetensors", "pt") as f:
        if kind == "arcface":
            w = f.get_tensor("arcface.weight").float()
            return w.view(N, w.shape[0] // N, w.shape[1])  # row c*k + j = species c, center j
        if kind == "hace":
            return f.get_tensor("hace.linear.weight").float()[:N].unsqueeze(1)
        return f.get_tensor("head.weight").float().unsqueeze(1)


def classifier(name, device):
    """(N, 1, C) unit-norm classifier rows: head.weight (cosine heads) or head.linear.weight (CE)."""
    import torch.nn.functional as F
    from safetensors import safe_open
    with safe_open(f"{BACKBONES}/{name}/model.safetensors", "pt") as f:
        keys = set(f.keys())
        key = "head.weight" if "head.weight" in keys else "head.linear.weight"
        w = f.get_tensor(key).float()
    return F.normalize(w.to(device), dim=1).unsqueeze(1), key


def read_directions(rung, loss):
    """(N, k, C) classifier rows from that rung's checkpoint.

    The class count comes from the head itself: the 15.5k, 32k and 64k rungs
    do not share it, and the highest validation label need not be the last
    class. Ladder runs use k=1.
    """
    from safetensors import safe_open
    ckpt = f"{OUT}/SWIN_BASE_LADDER_{rung.upper()}_{loss.upper()}/model.safetensors"
    with safe_open(ckpt, "pt") as f:
        keys = set(f.keys())
        # The ladder runs all go through the loss registry, so the classifier is
        # head.weight even for ArcFace. The native arcface.weight name is kept
        # as a fallback for checkpoints that use the built-in ArcFace path.
        if "head.weight" in keys:
            return f.get_tensor("head.weight").float().unsqueeze(1)
        if "arcface.weight" in keys:
            w = f.get_tensor("arcface.weight").float()
            return w.view(w.shape[0], 1, w.shape[1])
        raise KeyError(f"{ckpt}: no head.weight or arcface.weight")


BACKBONE_PREFIXES = ("convnext_", "dinov2_")


def split_backbone(loss):
    """('convnext', 'dbm') for 'convnext_dbm'; (None, loss) for the SWIN ladder losses."""
    for p in BACKBONE_PREFIXES:
        if loss.startswith(p):
            return p[:-1], loss[len(p):]
    return None, loss


def ladder_classifier(rung, loss, device):
    """(N, 1, C) unit rows of a ladder model: SWIN, or ConvNeXt / DINOv2 given as <backbone>_<loss>."""
    import torch.nn.functional as F
    from safetensors import safe_open
    bb, l = split_backbone(loss)
    if bb:  # ConvNeXt / DINOv2 ladder models (configs_advanced/backbones/generate_ladder_all.py)
        tag = f"LADDER_{rung.upper()}_{l.upper()}" if bb == "convnext" else f"LADDER_DINOV2_{rung.upper()}_{l.upper()}"
        ckpt = f"{ROOT}/BACKBONES/{tag}/model.safetensors"
    else:
        ckpt = f"{OUT}/SWIN_BASE_LADDER_{rung.upper()}_{loss.upper()}/model.safetensors"
    with safe_open(ckpt, "pt") as f:
        keys = set(f.keys())
        key = next(k for k in ("head.weight", "head.linear.weight", "arcface.weight") if k in keys)
        w = f.get_tensor(key).float()
    return F.normalize(w.to(device), dim=1).unsqueeze(1)


# ---------------------------------------------------------------- macro-F1
def per_class_counts(labels, preds, n_classes, weights=None):
    """tp, true count, predicted count per class (optionally image-weighted)."""
    correct = (labels == preds).astype(np.float64)
    w = np.ones_like(correct) if weights is None else weights
    tp = np.bincount(labels, weights=w * correct, minlength=n_classes)
    n_true = np.bincount(labels, weights=w, minlength=n_classes)
    n_pred = np.bincount(preds, weights=w, minlength=n_classes)
    return tp, n_true, n_pred


def macro_f1(tp, n_true, n_pred):
    """Mean F1 over every class that appears in the labels OR the predictions (sklearn 'macro')."""
    present = (n_true + n_pred) > 0
    return float(np.mean(2 * tp[present] / (n_true[present] + n_pred[present])))


def per_species_f1(tp, n_true, n_pred):
    """F1 per class; NaN for classes with no validation images (not scored per species)."""
    f1 = np.full(len(tp), np.nan)
    has = n_true > 0
    f1[has] = 2 * tp[has] / (n_true[has] + n_pred[has])
    return f1


def macro_f1_labels(y, p, n=None):
    """macro_f1 from label and prediction arrays (n = number of classes, default: highest id + 1)."""
    if n is None:
        n = int(max(y.max(), p.max())) + 1
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    nt, npred = np.bincount(y, minlength=n), np.bincount(p, minlength=n)
    m = (nt + npred) > 0
    return float(np.mean(2 * tp[m] / (nt[m] + npred[m])))


def per_species_f1_labels(y, p, n):
    """F1 of every class from label / prediction arrays; classes absent from both are NaN.

    Also returns the validation count per class (differs from per_species_f1: a class that is only
    predicted gets F1 = 0 here, not NaN).
    """
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    n_true = np.bincount(y, minlength=n).astype(float)
    n_pred = np.bincount(p, minlength=n).astype(float)
    denom = n_true + n_pred
    f1 = np.full(n, np.nan)
    ok = denom > 0
    f1[ok] = 2 * tp[ok] / denom[ok]
    return f1, n_true
