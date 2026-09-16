"""Controlled label-cardinality sweep on a frozen representation.

Design, and why each control is there:

* Fixed representation. Features come from a frozen backbone cached once, so
  varying N cannot also vary the encoder.

* Fixed images-per-class. Every class contributes exactly k_train train and
  k_eval eval images at every N, so "more classes" never means "more or less
  evidence per class".

* Nested class sampling. For a given seed the eligible classes are shuffled
  once and N-way tasks take the first N. The N=50 label set is therefore a
  subset of the N=200 set, which is a subset of the N=1000 set, and so on.
  This is what makes the headline measurement possible: the *anchor* classes
  (the first 50) keep the same training images and the same eval images at
  every N, so their F1 can only move because of the labels added around them.
  A fresh independent sample per N would confound cardinality with which
  classes happened to be drawn.

* Two decision rules per task. A learned linear probe, and nearest-class-mean
  on the same centroids. NCM has no optimisation in it at all, so if the probe
  and NCM degrade alike the cause is the geometry of the label space rather
  than anything about fitting a bigger head.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod          # noqa: E402
import features as feat_mod      # noqa: E402
import metrics as metric_mod     # noqa: E402


def log(msg):
    print(f"__LCS__: {msg}", flush=True)


def build_task(labels, valid, n_classes, k_train, k_eval, rng_seed, eligible_order):
    """Nested N-way task: first `n_classes` of the shuffled eligible order."""
    chosen = eligible_order[:n_classes]
    # Per-class image permutation is seeded by (seed, class) only -- never by N --
    # so a class's train/eval images are byte-identical across the whole sweep.
    tr_idx, tr_lab, ev_idx, ev_lab = [], [], [], []
    for new_id, cls in enumerate(chosen):
        pool = np.flatnonzero((labels == cls) & valid)
        perm = np.random.default_rng((rng_seed, int(cls))).permutation(pool.size)
        sel = pool[perm]
        tr_idx.append(sel[:k_train])
        ev_idx.append(sel[k_train:k_train + k_eval])
        tr_lab.append(np.full(k_train, new_id, dtype=np.int64))
        ev_lab.append(np.full(min(k_eval, max(0, sel.size - k_train)), new_id, dtype=np.int64))
    return (np.concatenate(tr_idx), np.concatenate(tr_lab),
            np.concatenate(ev_idx), np.concatenate(ev_lab), chosen)


def class_centroids(x, y, n_classes):
    d = x.shape[1]
    cent = torch.zeros(n_classes, d, device=x.device, dtype=torch.float32)
    cent.index_add_(0, y, x.float())
    cnt = torch.bincount(y, minlength=n_classes).clamp(min=1).unsqueeze(1).float()
    cent = cent / cnt
    return torch.nn.functional.normalize(cent, dim=1)


def nearest_other_centroid_distance(cent, chunk=512):
    """Per-class cosine distance to the closest *other* class centroid.

    The balanced-regime analogue of class frequency: with evidence held equal,
    this is what separates classes the model gets right from ones it does not.
    """
    n = cent.shape[0]
    out = torch.empty(n, device=cent.device)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sim = cent[lo:hi] @ cent.T
        sim[torch.arange(hi - lo, device=cent.device), torch.arange(lo, hi, device=cent.device)] = -2.0
        out[lo:hi] = 1.0 - sim.max(dim=1).values
    return out


def ncm_predict(xe, cent, chunk=8192):
    preds = torch.empty(xe.shape[0], dtype=torch.long, device=xe.device)
    for lo in range(0, xe.shape[0], chunk):
        hi = min(lo + chunk, xe.shape[0])
        preds[lo:hi] = (xe[lo:hi] @ cent.T).argmax(dim=1)
    return preds


def train_probe(xt, yt, n_classes, cfg, device):
    """Linear softmax probe with a *constant* optimisation budget.

    Steps are fixed across N, deliberately. Fixing epochs instead makes the
    number of gradient steps grow with N (more classes at fixed images/class =
    a bigger dataset), so the small-N probes end up undertrained and the sweep
    measures optimisation budget rather than label cardinality. Confirmed the
    hard way: with fixed epochs the accuracy curve came out *increasing* in N.
    `train_accuracy` is reported per task so convergence is checkable in the log.
    """
    torch.manual_seed(cfg["seed"])
    head = torch.nn.Linear(xt.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["probe"]["lr"],
                            weight_decay=cfg["probe"]["weight_decay"])
    steps = int(cfg["probe"]["steps"])
    bs = min(int(cfg["probe"]["batch_size"]), xt.shape[0])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["probe"]["lr"],
                                                total_steps=steps, pct_start=0.1)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=cfg["probe"]["label_smoothing"])
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    n = xt.shape[0]
    order, cur = torch.randperm(n, generator=g).to(device), 0
    last = float("nan")
    for _ in range(steps):
        if cur + bs > n:
            order, cur = torch.randperm(n, generator=g).to(device), 0
        idx = order[cur:cur + bs]
        cur += bs
        opt.zero_grad(set_to_none=True)
        loss = lossf(head(xt[idx]), yt[idx])
        loss.backward()
        opt.step()
        sched.step()
        last = loss.item()
    with torch.no_grad():
        train_acc = float((probe_predict(head, xt) == yt).float().mean().item())
    return head, last, train_acc


@torch.no_grad()
def probe_predict(head, xe, chunk=8192):
    preds = torch.empty(xe.shape[0], dtype=torch.long, device=xe.device)
    for lo in range(0, xe.shape[0], chunk):
        hi = min(lo + chunk, xe.shape[0])
        preds[lo:hi] = head(xe[lo:hi]).argmax(dim=1)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print("=" * 72)
    print("LABEL-CARDINALITY SWEEP  (balanced regime, frozen representation)")
    print("=" * 72)
    print(yaml.safe_dump(cfg, sort_keys=False).rstrip())
    print("=" * 72, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device} torch={torch.__version__}")
    if device == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    df, info = data_mod.load_manifest(cfg["data"]["manifest"],
                                      cfg["data"].get("image_column"),
                                      cfg["data"].get("label_column"),
                                      cfg["data"].get("image_root", ""))
    data_mod.census(df, info)

    feats, valid = feat_mod.build_cache(
        df["path"].tolist(), cfg["features"]["cache_root"], cfg["features"]["backbone"],
        cfg["features"]["image_size"], cfg["features"]["batch_size"],
        cfg["features"]["num_workers"], device)

    labels = df["label"].to_numpy()
    k_train, k_eval = cfg["sweep"]["k_train"], cfg["sweep"]["k_eval"]
    need = k_train + k_eval

    counts = np.bincount(labels[valid], minlength=info["n_classes"])
    eligible = np.flatnonzero(counts >= need)
    log(f"k_train={k_train} k_eval={k_eval} -> {len(eligible):,} of {info['n_classes']:,} "
        f"classes have >={need} usable images ({len(eligible) / info['n_classes']:.1%})")
    if len(eligible) < 50:
        raise RuntimeError(f"Only {len(eligible)} eligible classes at k={need}; lower k_train/k_eval.")

    requested = [n for n in cfg["sweep"]["n_classes"] if isinstance(n, int)]
    ladder = sorted({n for n in requested if n <= len(eligible)} | {len(eligible)})
    dropped = [n for n in requested if n > len(eligible)]
    if dropped:
        log(f"NOTE: requested N values {dropped} exceed the {len(eligible):,} eligible classes "
            f"at this k; the ladder tops out at the full eligible pool instead.")
    log(f"N ladder: {ladder}")

    anchor_n = min(cfg["sweep"]["anchor_classes"], ladder[0])
    all_feats = torch.from_numpy(feats).to(device)
    all_feats = torch.nn.functional.normalize(all_feats.float(), dim=1)

    results = []
    for seed in cfg["sweep"]["seeds"]:
        order = np.random.default_rng(seed).permutation(eligible)
        for n_classes in ladder:
            t0 = time.time()
            tr_i, tr_y, ev_i, ev_y, chosen = build_task(
                labels, valid, n_classes, k_train, k_eval, seed, order)

            xt = all_feats[torch.from_numpy(tr_i).to(device)]
            yt = torch.from_numpy(tr_y).to(device)
            xe = all_feats[torch.from_numpy(ev_i).to(device)]
            ev_y_np = ev_y

            cent = class_centroids(xt, yt, n_classes)
            sep = nearest_other_centroid_distance(cent).cpu().numpy()

            probe_cfg = dict(cfg)
            probe_cfg["seed"] = seed
            head, final_loss, train_acc = train_probe(xt, yt, n_classes, probe_cfg, device)

            row = {"seed": seed, "n_classes": n_classes, "k_train": k_train,
                   "k_eval": k_eval, "n_train": int(len(tr_y)), "n_eval": int(len(ev_y)),
                   "probe_final_loss": final_loss, "probe_train_accuracy": train_acc,
                   "mean_nn_centroid_distance": float(sep.mean()),
                   "median_nn_centroid_distance": float(np.median(sep))}

            for name, preds in (("probe", probe_predict(head, xe)),
                                ("ncm", ncm_predict(xe, cent))):
                m, f1 = metric_mod.decompose(ev_y_np, preds.cpu().numpy(),
                                             n_classes, separability=sep)
                for k, v in m.items():
                    if k not in ("n_classes", "n_eval"):
                        row[f"{name}_{k}"] = v
                # Anchor set: the SAME classes at every N. Their F1 moving is
                # the clean cardinality effect -- same images, same features,
                # only the number of competing labels changed.
                row[f"{name}_anchor_macro_f1"] = float(f1[:anchor_n].mean())
                row[f"{name}_anchor_dead_rate"] = float(np.mean(f1[:anchor_n] == 0))

            row["seconds"] = round(time.time() - t0, 1)
            results.append(row)
            log(f"seed={seed} N={n_classes:>6,d} | probe acc={row['probe_accuracy']:.4f} "
                f"macroF1={row['probe_macro_f1']:.4f} anchorF1={row['probe_anchor_macro_f1']:.4f} "
            f"trainAcc={row['probe_train_accuracy']:.3f} "
                f"gini={row['probe_f1_gini']:.3f} Neff={row['probe_n_eff']:.0f} "
                f"dead={row['probe_dead_class_rate']:.3f} | ncm macroF1={row['ncm_macro_f1']:.4f} "
                f"| {row['seconds']}s")

            del xt, xe, cent, head
            if device == "cuda":
                torch.cuda.empty_cache()

    # Validity check on the fixed step budget. The whole sweep rests on 4k steps
    # being enough at every N; if the hardest task (largest N) still moves when
    # given twice the budget, the curve is partly an optimisation artifact and
    # the reader needs to know. One extra probe, seconds of compute.
    if cfg["probe"].get("convergence_check", True):
        top_n, seed = ladder[-1], cfg["sweep"]["seeds"][0]
        order = np.random.default_rng(seed).permutation(eligible)
        tr_i, tr_y, ev_i, ev_y, _ = build_task(labels, valid, top_n, k_train, k_eval, seed, order)
        xt = all_feats[torch.from_numpy(tr_i).to(device)]
        yt = torch.from_numpy(tr_y).to(device)
        xe = all_feats[torch.from_numpy(ev_i).to(device)]
        doubled = dict(cfg)
        doubled["seed"] = seed
        doubled["probe"] = dict(cfg["probe"])
        doubled["probe"]["steps"] = int(cfg["probe"]["steps"]) * 2
        head2, _, tr2 = train_probe(xt, yt, top_n, doubled, device)
        m2, _ = metric_mod.decompose(ev_y, probe_predict(head2, xe).cpu().numpy(), top_n)
        base_rows = [r for r in results if r["n_classes"] == top_n and r["seed"] == seed]
        b = base_rows[0]
        log("convergence check at N={:,} seed={}: {} steps -> macroF1 {:.4f} (trainAcc {:.3f}) | "
            "{} steps -> macroF1 {:.4f} (trainAcc {:.3f}) | delta {:+.4f}".format(
                top_n, seed, cfg["probe"]["steps"], b["probe_macro_f1"], b["probe_train_accuracy"],
                doubled["probe"]["steps"], m2["macro_f1"], tr2,
                m2["macro_f1"] - b["probe_macro_f1"]))
        results.append({"seed": seed, "n_classes": top_n, "tag": "convergence_check_2x_steps",
                        "steps": doubled["probe"]["steps"], "probe_macro_f1": m2["macro_f1"],
                        "probe_train_accuracy": tr2})
        del xt, xe, head2
        if device == "cuda":
            torch.cuda.empty_cache()

    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, cfg["output"]["results_file"])
    with open(out_file, "w") as fh:
        json.dump({"config": cfg, "eligible_classes": int(len(eligible)),
                   "anchor_classes": anchor_n, "results": results}, fh, indent=2)
    log(f"wrote {out_file}")

    summarise(results, ladder, anchor_n)


def summarise(results, ladder, anchor_n):
    import statistics as st

    def agg(n, key):
        vals = [r[key] for r in results if r["n_classes"] == n and key in r and "tag" not in r]
        if not vals:
            return float("nan"), 0.0
        return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

    cols = [
        ("probe acc", "probe_accuracy", 4),
        ("probe macroF1", "probe_macro_f1", 4),
        ("anchor F1 (fixed cls)", "probe_anchor_macro_f1", 4),
        ("F1 gini", "probe_f1_gini", 3),
        ("N_eff/N", "probe_n_eff_frac", 3),
        ("dead frac", "probe_dead_class_rate", 3),
        ("NCM macroF1", "ncm_macro_f1", 4),
    ]
    width = 22
    header = "{:>8} | ".format("N") + " | ".join("{:>{w}}".format(c[0], w=width) for c in cols)
    print()
    print("=" * len(header))
    print("SUMMARY -- mean +/- sd over seeds. Images per class is constant across every row.")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for n in ladder:
        cells = []
        for _, key, prec in cols:
            m, s_ = agg(n, key)
            cells.append("{:>{w}}".format("{:.{p}f} +/-{:.{p}f}".format(m, s_, p=prec), w=width))
        print("{:>8,d} | ".format(n) + " | ".join(cells))
    print("-" * len(header))

    base, top = ladder[0], ladder[-1]
    a0, _ = agg(base, "probe_anchor_macro_f1")
    a1, _ = agg(top, "probe_anchor_macro_f1")
    m0, _ = agg(base, "probe_macro_f1")
    m1, _ = agg(top, "probe_macro_f1")
    n0, _ = agg(base, "ncm_anchor_macro_f1")
    n1, _ = agg(top, "ncm_anchor_macro_f1")
    pct = (100.0 * (a1 - a0) / a0) if a0 else float("nan")
    print("HEADLINE  anchor set = {} fixed classes, identical train and eval images at every N.".format(anchor_n))
    print("          probe anchor macro-F1 {:.4f} at N={:,}  ->  {:.4f} at N={:,}   (delta {:+.4f}, {:+.1f}%)".format(
        a0, base, a1, top, a1 - a0, pct))
    print("          ncm   anchor macro-F1 {:.4f} at N={:,}  ->  {:.4f} at N={:,}   (delta {:+.4f})".format(
        n0, base, n1, top, n1 - n0))
    print("          Nothing about those classes changed except the number of competing labels.")
    print("          Whole-task probe macro-F1 {:.4f} -> {:.4f} over the same range.".format(m0, m1))
    print("=" * len(header))

    top_rows = [r for r in results if r["n_classes"] == top and "tag" not in r]
    if top_rows:
        prof = np.mean([r["probe_f1_decile_profile"] for r in top_rows], axis=0)
        print("PER-CLASS F1 DECILES at N={:,} (sorted by F1; d1 = best tenth of classes):".format(top))
        print("          " + "  ".join("d{}:{:.3f}".format(i + 1, v) for i, v in enumerate(prof)))
        sep_rows = [r["probe_f1_by_separability_decile"] for r in top_rows
                    if "probe_f1_by_separability_decile" in r]
        if sep_rows:
            sp = np.mean(sep_rows, axis=0)
            print("SAME CLASSES ordered by feature-space separability instead of by score:")
            print("          " + "  ".join("d{}:{:.3f}".format(i + 1, v) for i, v in enumerate(sp)))
        corrs = [r["probe_corr_f1_separability"] for r in top_rows
                 if "probe_corr_f1_separability" in r]
        if corrs:
            print("          corr(per-class F1, distance to nearest other class centroid) = {:+.3f}".format(
                float(np.mean(corrs))))
    print("=" * len(header), flush=True)


if __name__ == "__main__":
    main()
