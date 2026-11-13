"""Why does the ladder's 15.5k DBM score higher (0.8959 / 0.8436) than the Phase 2 DBM
(0.8443 / 0.8161)? Separate "better model" from "different validation set".

Phase 2: trained on train_2022, scored on val_2022 (PHASE3_GEOM/dbm*/predictions.npz,
val_2022 order). Ladder 15.5k: trained on n15k_train (merged 2021+2022, same 15,501
species), scored on n15k_val (LADDER_EVAL/n15k_dbm/predictions.npz, n15k_val order).

(1) Common clean subset: n15k_val images that are also in val_2022. Neither model trained
    on them, so both can be scored on exactly the same images.
(2) The ladder model on each source of its own validation set: images from val_2022, from
    train_2022, and from Herbarium 2021.
Predictions are compared by species name (the two runs number species independently).
"""

import json

import numpy as np

D = "/projectnb/herbdl/data/kaggle-herbaria"
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def scores(y, p):
    """accuracy and sklearn-style macro-F1 (over classes in labels or predictions)."""
    names = sorted(set(y) | set(p))
    idx = {n: i for i, n in enumerate(names)}
    yi = np.array([idx[n] for n in y]); pi = np.array([idx[n] for n in p])
    k = len(names)
    tp = np.bincount(yi[yi == pi], minlength=k).astype(float)
    nt = np.bincount(yi, minlength=k).astype(float)
    npd = np.bincount(pi, minlength=k).astype(float)
    return float(np.mean(yi == pi)), float(np.mean(2 * tp / (nt + npd)))


def main():
    tr22, va22 = load(f"{D}/train_2022.json"), load(f"{D}/val_2022.json")
    name22 = {r["scientificNameEncoded"]: r["scientificName"] for r in tr22}
    train22_paths = {r["filename"] for r in tr22}
    va22_pos = {r["filename"]: i for i, r in enumerate(va22)}
    lval = load(f"{LAD}/n15k_val.json")
    lname = {}
    for r in load(f"{LAD}/n15k_train.json"):
        lname[r["scientificNameEncoded"]] = r["scientificName"]
    lab = [r["scientificName"] for r in lval]
    lp = np.load(f"{OUT}/LADDER_EVAL/n15k_dbm/predictions.npz")["predictions"]
    lpred = [lname[int(i)] for i in lp]
    a, f = scores(lab, lpred)
    print(f"__LVP__ ladder DBM on its full val ({len(lab):,} images): acc={a:.4f} macroF1={f:.4f}")

    src = np.array(["val_2022" if r["filepath"] in va22_pos else
                    "train_2022" if r["filepath"] in train22_paths else
                    "herbarium_2021" if "herbarium-2021" in r["filepath"] else "other" for r in lval])
    for s in ("val_2022", "train_2022", "herbarium_2021", "other"):
        m = np.where(src == s)[0]
        if len(m) == 0:
            continue
        a, f = scores([lab[i] for i in m], [lpred[i] for i in m])
        print(f"__LVP__ ladder DBM, val images from {s:15s} n={len(m):7,}: acc={a:.4f} macroF1={f:.4f}")

    common = np.where(src == "val_2022")[0]
    y = [lab[i] for i in common]
    a, f = scores(y, [lpred[i] for i in common])
    print(f"__LVP__ COMMON SUBSET ({len(common):,} val_2022 images neither model trained on, "
          f"{len(set(y)):,} species)")
    print(f"__LVP__   ladder 15.5k DBM (merged data):  acc={a:.4f} macroF1={f:.4f}")
    for run in ("dbm", "dbm_seed43", "kappaface", "arcface"):
        pp = np.load(f"{OUT}/PHASE3_GEOM/{run}/predictions.npz")["predictions"]
        p2 = [name22[int(pp[va22_pos[lval[i]["filepath"]]])] for i in common]
        y2 = [va22[va22_pos[lval[i]["filepath"]]]["scientificName"] for i in common]
        assert y2 == y, "label mismatch between the two manifests"
        a, f = scores(y, p2)
        print(f"__LVP__   Phase 2 {run:12s} (2022 data):     acc={a:.4f} macroF1={f:.4f}")


if __name__ == "__main__":
    main()
