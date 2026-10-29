"""Near-duplicates, stated precisely. Reuses the generic-backbone features cached by
near_duplicates.py (run 1eb2694b): NEARDUP/n15k/{train,val}.npy in n15k_train / n15k_val
order, and nearest.npz (each ladder val image's most similar same-species n15k_train image).

Part 1 -- the ladder 15.5k split. Composition of n15k_train by source, and for ladder val
images with a match >= T, which source the matched n15k_train image comes from.

Part 2 -- the Phase 2-3 split exactly. For each val_2022 image, its most similar
same-species image in train_2022 ONLY (the set the Phase 2-3 models trained on), and the
Phase 2 DBM's accuracy with vs without such a match. Only images that survived the
2021+2022 merge were embedded, so coverage is reported.

Usage: python loss_ablation/near_duplicates_precise.py   (CPU)
"""

import json

import numpy as np

D = "/projectnb/herbdl/data/kaggle-herbaria"
LAD = "/projectnb/herbdl/workspaces/faridkar/herbdl/data/ladder"
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
FEAT = f"{OUT}/NEARDUP/n15k"
TS = [0.93, 0.95, 0.98, 0.99]


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    tr22, va22 = load(f"{D}/train_2022.json"), load(f"{D}/val_2022.json")
    t22 = {r["filename"] for r in tr22}
    v22 = {r["filename"] for r in va22}
    ltr, lva = load(f"{LAD}/n15k_train.json"), load(f"{LAD}/n15k_val.json")

    def source(p):
        return "val_2022" if p in v22 else "train_2022" if p in t22 else (
            "herbarium_2021" if "herbarium-2021" in p else "other")

    tsrc = np.array([source(r["filepath"]) for r in ltr])
    print("__NDP__ PART 1: ladder 15.5k training set (n15k_train) =", f"{len(ltr):,} images:",
          ", ".join(f"{s} {np.sum(tsrc == s):,}" for s in ("herbarium_2021", "train_2022", "val_2022", "other")))
    nn = np.load(f"{FEAT}/nearest.npz", allow_pickle=True)
    best, best_j, vsrc = nn["best"], nn["best_j"], nn["val_source"]
    for s in ("herbarium_2021", "train_2022", "val_2022"):
        for t in TS:
            m = (vsrc == s) & (best >= t)
            src = tsrc[best_j[m]]
            print(f"__NDP__   ladder val from {s:15s} with a match >= {t} in n15k_train: {m.sum():6,} "
                  f"({100 * m.sum() / np.sum(vsrc == s):.1f}%); matched image is from "
                  + ", ".join(f"{k} {np.sum(src == k)}" for k in ("herbarium_2021", "train_2022", "val_2022")))

    # ---- Part 2: val_2022 vs train_2022 only --------------------------------------
    ftr = np.load(f"{FEAT}/train.npy", mmap_mode="r")
    fva = np.load(f"{FEAT}/val.npy", mmap_mode="r")
    where = {}
    for i, r in enumerate(ltr):
        where[r["filepath"]] = ("tr", i)
    for i, r in enumerate(lva):
        where[r["filepath"]] = ("va", i)

    def feats(paths):
        idx = [(k, where[p]) for k, p in enumerate(paths) if p in where]
        F = np.zeros((len(idx), ftr.shape[1]), dtype=np.float32)
        for row, (_, (s, i)) in enumerate(idx):
            F[row] = (ftr if s == "tr" else fva)[i]
        return np.array([k for k, _ in idx]), F

    vi, Fv = feats([r["filename"] for r in va22])
    ti, Ft = feats([r["filename"] for r in tr22])
    yv = np.array([va22[k]["scientificNameEncoded"] for k in vi])
    yt = np.array([tr22[k]["scientificNameEncoded"] for k in ti])
    print(f"__NDP__ PART 2: val_2022 vs train_2022 only. Covered (embedded): val_2022 {len(vi):,} of "
          f"{len(va22):,}; train_2022 {len(ti):,} of {len(tr22):,}")
    order = np.argsort(yt, kind="stable")
    starts = np.searchsorted(yt[order], np.arange(yt.max() + 2))
    b2 = np.full(len(vi), np.nan, dtype=np.float32)
    for c in np.unique(yv):
        v = np.where(yv == c)[0]
        t = order[starts[c]:starts[c + 1]]
        if len(t):
            b2[v] = (Fv[v] @ Ft[t].T).max(axis=1)
    z = np.load(f"{OUT}/PHASE3_GEOM/dbm/predictions.npz")
    correct = (z["predictions"] == z["labels"])[vi]
    ok = np.isfinite(b2)
    print(f"__NDP__   Phase 2 DBM accuracy on the covered val_2022 images: {correct[ok].mean():.4f}")
    for t in TS:
        m = ok & (b2 >= t)
        print(f"__NDP__   val_2022 with a same-species train_2022 match >= {t}: {m.sum():6,} "
              f"({100 * m.sum() / ok.sum():.1f}%); Phase 2 DBM acc with {correct[m].mean():.4f} vs "
              f"without {correct[ok & ~m].mean():.4f}")
    np.savez_compressed(f"{FEAT}/val2022_vs_train2022.npz", val_index=vi, best=b2)


if __name__ == "__main__":
    main()
