"""Collect the supplement's geometry table: macro-F1, accuracy, compactness angle and bias-corrected
Fisher score on training and validation images, for every model with saved features.

Sources (all on SCC; nothing is recomputed except F1/accuracy for the SWIN base models, which are read
from their saved predictions):
  SWIN-B, 15.5k base   predictions PHASE3_GEOM{,_TRAIN}/<m>/predictions.npz; angle from
                       MARGIN_DIAG/paper_measures.json (val: NormFace, ArcFace m=0.1, DBM, ArcFace),
                       PHASE3_GEOM/analysis/paper_separability.json (val: the rest),
                       MARGIN_DIAG/train_measures.json (train); Fisher from geometry_robustness.json
                       ('corr_pooled')
  SWIN-B ladder        LADDER_GEOM/analysis/ladder_geometry_full.json
  ConvNeXt / DINOv2    MARGIN_DIAG/backbone_geometry.json
Macro-F1 is the mean per-class F1 over classes in the labels or the predictions (as in the paper).
Missing cells are null. Output: MARGIN_DIAG/supplement_table.json. Run on the login node (CPU).
"""
import json
import os

import numpy as np

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
DIAG = f"{OUT}/MARGIN_DIAG"
SWIN_BASE = ["cosine_m0", "dbm", "arcface_m01", "kappaface", "subcenter_arcface", "arcface", "arcface_ecc"]
LADDER = ["cosine_m0", "dbm", "kappaface", "arcface", "ce"]
BACKBONE = ["ce", "ce_la", "cosine_m0", "dbm", "arcface"]


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else {}


def f1_acc(path):
    if not os.path.isfile(path):
        return None, None
    z = np.load(path)
    y, p = z["labels"].astype(np.int64), z["predictions"].astype(np.int64)
    n = int(max(y.max(), p.max())) + 1
    tp = np.bincount(y[y == p], minlength=n).astype(float)
    nt, npred = np.bincount(y, minlength=n), np.bincount(p, minlength=n)
    m = (nt + npred) > 0
    return float(np.mean(2 * tp[m] / (nt[m] + npred[m]))), float(np.mean(y == p))


def cell(f1=None, acc=None, angle=None, fisher=None):
    return {"macro_f1": f1, "accuracy": acc, "angle_deg": angle, "fisher": fisher}


rows = []
pm, ps, tm = load(f"{DIAG}/paper_measures.json"), load(f"{OUT}/PHASE3_GEOM/analysis/paper_separability.json"), \
    load(f"{DIAG}/train_measures.json")
gr = load(f"{DIAG}/geometry_robustness.json")
for m in SWIN_BASE:
    va = pm.get("val", {}).get(m, {}).get("angle_to_weight", {}).get("overall", {}).get("mean_deg") \
        or ps.get(m, {}).get("intra_classifier", {}).get("overall", {}).get("mean_deg")
    ta = tm.get(m, {}).get("angle_to_weight", {}).get("overall", {}).get("mean_deg")
    fv = gr.get(m, {}).get("val", {}).get("fisher", {}).get("corr_pooled", {}).get("all")
    ft = gr.get(m, {}).get("train", {}).get("fisher", {}).get("corr_pooled", {}).get("all")
    rows.append({"setting": "SWIN-B, 15.5k (base)", "loss": m,
                 "train": cell(*f1_acc(f"{OUT}/PHASE3_GEOM_TRAIN/{m}/predictions.npz"), ta, ft),
                 "val": cell(*f1_acc(f"{OUT}/PHASE3_GEOM/{m}/predictions.npz"), va, fv)})

lg = load(f"{OUT}/LADDER_GEOM/analysis/ladder_geometry_full.json")
for rung, label in (("n15k", "15.5k"), ("n32k", "32k"), ("n64k", "64k")):
    for m in LADDER:
        e = lg.get(rung, {}).get(m, {})
        r = {"setting": f"SWIN-B, ladder {label}", "loss": m}
        for s in ("train", "val"):
            c = e.get(s)
            r[s] = cell(c["macro_f1"], 1 - c["error_pct"] / 100, c["angle_deg"]["overall"],
                        c["fisher"]["corrected"]["all"]) if c else cell()
        rows.append(r)

bg = load(f"{DIAG}/backbone_geometry.json")
for bb, label in (("convnext", "ConvNeXt-B, 15.5k (base)"), ("dinov2", "DINOv2-B, 15.5k (base)")):
    for m in BACKBONE:
        e = bg.get(f"{bb}_{m}", {})
        r = {"setting": label, "loss": m}
        for s in ("train", "val"):
            c = e.get(s)
            r[s] = cell(c["macro_f1"], 1 - c["error_pct"] / 100, c["angle_deg"]["overall"],
                        c["fisher"]["corrected"]["all"]) if c else cell()
        rows.append(r)

for bb, label in (("convnext", "ConvNeXt-B"), ("dinov2", "DINOv2-B")):
  for rung, rl in (("n32k", "32k"), ("n64k", "64k")):
   for m in ("cosine_m0", "dbm", "arcface"):  # filled as each model's geometry exists
    e = lg.get(rung, {}).get(f"{bb}_{m}", {})
    r = {"setting": f"{label}, ladder {rl}", "loss": m}
    for s in ("train", "val"):
        c = e.get(s)
        r[s] = cell(c["macro_f1"], 1 - c["error_pct"] / 100, c["angle_deg"]["overall"],
                    c["fisher"]["corrected"]["all"]) if c else cell()
    rows.append(r)

json.dump(rows, open(f"{DIAG}/supplement_table.json", "w"), indent=1)
filled = sum(1 for r in rows for s in ("train", "val") if r[s]["fisher"] is not None)
print(f"__SUPPTABLE__ {len(rows)} rows, {filled} of {2 * len(rows)} split cells filled -> {DIAG}/supplement_table.json")
