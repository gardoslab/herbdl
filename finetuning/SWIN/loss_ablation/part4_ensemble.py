"""Part 4 ensembles: does combining models beat the best single model?

Inputs are the saved per-image validation predictions (predictions.npz: top-1 `predictions`,
`top5`, `labels`, in val_2022.json order) of the finished models. No logits were saved, so the
combiners only use each model's top-1 and top-5 answers:

  vote    plurality of the members' top-1 answers; a tie goes to the member listed first
          (members are listed strongest first).
  borda   rank fusion over top-5: a class in a member's top-5 at rank r (0 = best) earns
          5 - r points from that member; the class with the most total points wins, ties go to
          the first-listed member's ordering. A cheap stand-in for averaging probabilities.
  route   two models. Take the secondary model's answer if that answer names a species with
          <= T training images, else the primary's (the Part 3 rule). T is tuned for macro-F1 on
          a random half of the validation images and scored on the other half, both ways round.
  oracle  an image counts as right if ANY member is right. Not a real classifier: the ceiling
          on what any combiner of these members could reach.

Every ensemble is compared with its best single member by a paired bootstrap over validation
images (95% interval on the accuracy and macro-F1 difference). That measures evaluation-set
noise only, not seed-to-seed training noise (about 0.1 points).

Macro-F1 = mean per-class F1 over every class that appears in the labels or the predictions
(what the training script's eval_f1 reports). Usage (CPU, numpy only):
    python loss_ablation/part4_ensemble.py
"""

import itertools
import json
import os

import numpy as np

OUT_ROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
P3, P4 = f"{OUT_ROOT}/PHASE3_EVAL", f"{OUT_ROOT}/PHASE4_EVAL"
TRAIN_JSON = "/projectnb/herbdl/data/kaggle-herbaria/train_2022.json"
VAL_JSON = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
LABEL = "scientificNameEncoded"
OUT_JSON = f"{OUT_ROOT}/PHASE4_EVAL/analysis/part4_ensemble.json"
BANDS = [(2, 10), (11, 25), (26, 50), (51, 73)]
THRESHOLDS = [0, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75]
N_BOOT = 1000

MODELS = {
    "swin_ce": f"{P3}/ce", "swin_ce_la": f"{P3}/ce_la", "swin_arcface": f"{P3}/arcface", "swin_dbm": f"{P3}/dbm",
    "cnx_ce_llindon": f"{P4}/convnext_ce", "cnx_ce_la": f"{P4}/convnext_ce_la",
    "cnx_arcface": f"{P4}/convnext_arcface", "cnx_dbm": f"{P4}/convnext_dbm",
    "dino_ce_llindon": f"{P4}/dinov2_ce", "dino_ce_la": f"{P4}/dinov2_ce_la",
    "dino_arcface": f"{P4}/dinov2_arcface", "dino_dbm": f"{P4}/dinov2_dbm",
    "coatnet_ce_llindon": f"{P4}/coatnet_ce", "effnet_ce_llindon": f"{P4}/efficientnet_ce",
}
# Ensembles to evaluate: name -> members, strongest first.
SETS = {
    "DBM across backbones": ["swin_dbm", "cnx_dbm", "dino_dbm"],
    "ArcFace across backbones": ["swin_arcface", "cnx_arcface", "dino_arcface"],
    "CE+LA across backbones": ["swin_ce_la", "cnx_ce_la", "dino_ce_la"],
    "SWIN: DBM + ArcFace + CE+LA": ["swin_dbm", "swin_arcface", "swin_ce_la"],
    "ConvNeXt: DBM + ArcFace + CE+LA": ["cnx_dbm", "cnx_arcface", "cnx_ce_la"],
    "DINOv2: DBM + ArcFace + CE+LA": ["dino_dbm", "dino_arcface", "dino_ce_la"],
    "SWIN + ConvNeXt DBM": ["swin_dbm", "cnx_dbm"],
    "DBM + ArcFace, all three backbones": ["swin_dbm", "cnx_dbm", "swin_arcface", "cnx_arcface", "dino_dbm", "dino_arcface"],
    "Everything (14 models)": ["swin_dbm", "cnx_dbm", "swin_arcface", "cnx_arcface", "dino_dbm", "dino_arcface",
                               "swin_ce_la", "cnx_ce_la", "dino_ce_la", "swin_ce", "cnx_ce_llindon",
                               "dino_ce_llindon", "coatnet_ce_llindon", "effnet_ce_llindon"],
}
ROUTES = [("swin_dbm", "cnx_dbm"), ("swin_dbm", "dino_dbm"), ("cnx_dbm", "dino_dbm"),
          ("swin_dbm", "cnx_arcface"), ("cnx_dbm", "swin_arcface")]


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def class_counts(y, p, n):
    tp = np.bincount(y[p == y], minlength=n).astype(np.float64)
    return tp, np.bincount(y, minlength=n).astype(np.float64), np.bincount(p, minlength=n).astype(np.float64)


def macro_f1(y, p, n):
    tp, nt, npd = class_counts(y, p, n)
    keep = (nt + npd) > 0
    return float(np.mean(2 * tp[keep] / (nt[keep] + npd[keep])))


def band_f1(y, p, n, train_count):
    """Mean per-species F1 within each band of training images per species."""
    tp, nt, npd = class_counts(y, p, n)
    f1 = np.full(n, np.nan)
    has = nt > 0
    f1[has] = 2 * tp[has] / (nt[has] + npd[has])
    return {f"{lo}-{hi}": float(np.nanmean(f1[(train_count >= lo) & (train_count <= hi)])) for lo, hi in BANDS}


def vote(tops):
    """Plurality of top-1 answers; ties go to the earliest member among the tied classes."""
    m, n = tops.shape
    votes = np.zeros((m, n))
    for i in range(m):
        votes[i] = (tops == tops[i]).sum(axis=0) - 1e-3 * i  # earlier member wins a tie
    return tops[np.argmax(votes, axis=0), np.arange(n)]


def borda(top5s):
    """Rank fusion over each member's top-5 (5 points for rank 0 down to 1 for rank 4)."""
    cand = np.concatenate(top5s, axis=1)  # (N, 5M)
    pts = np.concatenate([np.tile(5.0 - np.arange(5), (len(t), 1)) for t in top5s], axis=1)
    pts = pts - 1e-4 * np.arange(cand.shape[1])  # earlier column wins a tie
    total = np.zeros_like(pts)
    for j in range(cand.shape[1]):
        total[:, j] = ((cand == cand[:, [j]]) * pts).sum(axis=1)
    return cand[np.arange(len(cand)), np.argmax(total, axis=1)]


def route(primary, secondary, train_count, t):
    return np.where(train_count[secondary] <= t, secondary, primary)


def boot_gap(y, p_ens, p_ref, n, rng):
    """95% interval of ensemble - reference for accuracy and macro-F1, paired over images."""
    N = len(y)
    acc, f1 = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, N, N)
        acc.append(np.mean(p_ens[idx] == y[idx]) - np.mean(p_ref[idx] == y[idx]))
        f1.append(macro_f1(y[idx], p_ens[idx], n) - macro_f1(y[idx], p_ref[idx], n))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return q(acc), q(f1)


def main():
    rng = np.random.default_rng(0)
    train = load_rows(TRAIN_JSON)
    n = max(r[LABEL] for r in train) + 1
    train_count = np.bincount([r[LABEL] for r in train], minlength=n)
    y = np.array([r[LABEL] for r in load_rows(VAL_JSON)])
    print(f"__ENS__ classes={n} train_images={len(train)} val_images={len(y)}")

    top1, top5, single = {}, {}, {}
    for name, d in MODELS.items():
        path = f"{d}/predictions.npz"
        if not os.path.isfile(path):
            print(f"__ENS__ MISSING {name}: {path}")
            continue
        z = np.load(path)
        if not np.array_equal(z["labels"].astype(np.int64), y):
            raise SystemExit(f"{name}: saved labels are not val_2022.json in order")
        top1[name], top5[name] = z["predictions"].astype(np.int64), z["top5"].astype(np.int64)
        single[name] = {"accuracy": float(np.mean(top1[name] == y)), "macro_f1": macro_f1(y, top1[name], n),
                        "bands": band_f1(y, top1[name], n, train_count)}
        print(f"__ENS__ single {name}: acc={single[name]['accuracy']:.4f} f1={single[name]['macro_f1']:.4f}")

    results = {"single": single, "sets": {}, "routing": {}, "pairs": {}}

    # Pairwise complementarity: how often the models agree, and the either-correct ceiling.
    names = list(top1)
    for a, b in itertools.combinations(names, 2):
        ca, cb = top1[a] == y, top1[b] == y
        results["pairs"][f"{a}|{b}"] = {"agree": float(np.mean(top1[a] == top1[b])),
                                        "both_right": float(np.mean(ca & cb)),
                                        "either_right": float(np.mean(ca | cb))}

    for sname, members in SETS.items():
        members = [m for m in members if m in top1]
        if len(members) < 2:
            continue
        best = max(members, key=lambda m: single[m]["macro_f1"])
        out = {"members": members, "best_member": best}
        oracle = np.any([top1[m] == y for m in members], axis=0)
        out["oracle_accuracy"] = float(oracle.mean())
        for method, fn in (("vote", lambda: vote(np.stack([top1[m] for m in members]))),
                           ("borda", lambda: borda([top5[m] for m in members]))):
            p = fn()
            acc_ci, f1_ci = boot_gap(y, p, top1[best], n, rng)
            out[method] = {"accuracy": float(np.mean(p == y)), "macro_f1": macro_f1(y, p, n),
                           "vs_best_acc": float(np.mean(p == y) - single[best]["accuracy"]),
                           "vs_best_f1": float(macro_f1(y, p, n) - single[best]["macro_f1"]),
                           "acc_ci": acc_ci, "f1_ci": f1_ci, "bands": band_f1(y, p, n, train_count)}
            print(f"__ENS__ {sname} [{method}] acc={out[method]['accuracy']:.4f} f1={out[method]['macro_f1']:.4f} "
                  f"(best member {best}: {single[best]['accuracy']:.4f}/{single[best]['macro_f1']:.4f}; "
                  f"gap acc {out[method]['vs_best_acc']:+.4f} {acc_ci}, f1 {out[method]['vs_best_f1']:+.4f} {f1_ci})")
        print(f"__ENS__ {sname} [oracle] acc={out['oracle_accuracy']:.4f}")
        results["sets"][sname] = out

    # Routing: threshold tuned on half A, scored on half B (and the reverse).
    perm = rng.permutation(len(y))
    halves = [perm[: len(y) // 2], perm[len(y) // 2:]]
    for prim, sec in ROUTES:
        if prim not in top1 or sec not in top1:
            continue
        gains = []
        for ia, ib in (halves, halves[::-1]):
            f1s = [macro_f1(y[ia], route(top1[prim][ia], top1[sec][ia], train_count, t), n) for t in THRESHOLDS]
            t = THRESHOLDS[int(np.argmax(f1s))]
            pr = route(top1[prim][ib], top1[sec][ib], train_count, t)
            gains.append((t, float(np.mean(pr == y[ib]) - np.mean(top1[prim][ib] == y[ib])),
                          macro_f1(y[ib], pr, n) - macro_f1(y[ib], top1[prim][ib], n)))
        results["routing"][f"{prim}<-{sec}"] = {"thresholds": [g[0] for g in gains],
                                                "acc_gain": float(np.mean([g[1] for g in gains])),
                                                "f1_gain": float(np.mean([g[2] for g in gains]))}
        print(f"__ENS__ route {prim} <- {sec}: T={[g[0] for g in gains]} acc {np.mean([g[1] for g in gains]):+.4f} "
              f"f1 {np.mean([g[2] for g in gains]):+.4f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    print(f"__ENS__ wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
