"""Cone test for the Discussion's mechanism paragraph (15.5k base data, eight losses, saved features only).

The paragraph's reading: a large margin forces each species into a narrow cone; the extra tightening
fits the training images rather than the species; and most of the separation a uniform margin adds is
between unrelated species, which classification does not need. Three checks:

(1) Train vs validation geometry by band. DBM-paper angle to the species' classifier weight and
    pairwise Fisher score (paper_measures.measure, unchanged), on training and on validation images,
    averaged within bands of training images per species. Prediction: the large-margin losses lose
    more tightness and separation from training to validation images, in every band.
(2) Far misses. For every validation image, the rank of its true species among the model's scores
    (cosine to each class weight; max over sub-centres) and, for errors, whether the predicted
    species shares the true species' genus or family. Prediction: errors with the true species ranked
    below fifth are mostly outside the genus, and ArcFace has more of them than DBM. A gallery of
    images ArcFace ranks below fifth but DBM gets right is written for inspection.
(3) Sister vs unrelated separation. The pooled-covariance pairwise Fisher matrix J_ij on validation
    features, split into same-genus pairs, same-family-other-genus pairs and other-family pairs, over
    species with validation images; plus each species' nearest neighbour (smallest J) and whether it
    is in the same genus. Prediction: the extra separation of the large-margin losses over DBM is
    concentrated in other-family pairs, and same-genus pairs stay about as close.

Output: lines prefixed __CONE__; MARGIN_DIAG/cone_test.json; MARGIN_DIAG/cone_far_miss_gallery.{png,csv}.
"""

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_measures as pm  # noqa: E402

VAL_JSON = "/projectnb/herbdl/data/kaggle-herbaria/val_2022.json"
MODELS = ["cosine_m0", "dbm", "arcface_m01", "kappaface", "subcenter_arcface", "arcface_tara",
          "arcface", "arcface_ecc"]
NAMES = {"cosine_m0": "NormFace", "dbm": "DBM", "arcface_m01": "ArcFace m=0.1", "kappaface": "KappaFace",
         "subcenter_arcface": "Sub-center", "arcface_tara": "ArcFace+TARA", "arcface": "ArcFace",
         "arcface_ecc": "ArcFace+ECC"}
DIAG = f"{pm.OUT}/MARGIN_DIAG"
P = "__CONE__"
CHUNK = 8192


def load_meta():
    fam, gen, name = {}, {}, {}
    counts = np.zeros(pm.N, np.int64)
    for line in open(pm.TRAIN_JSON):
        if line.strip():
            r = json.loads(line)
            c = r[pm.LABEL]
            counts[c] += 1
            fam[c], gen[c], name[c] = r["family"], (r["family"], r["genus"]), r["scientificName"]
    fam_ids = {f: i for i, f in enumerate(sorted(set(fam.values())))}
    gen_ids = {g: i for i, g in enumerate(sorted(set(gen.values())))}
    cls_fam = np.array([fam_ids[fam[c]] for c in range(pm.N)])
    cls_gen = np.array([gen_ids[gen[c]] for c in range(pm.N)])
    names = [name[c] for c in range(pm.N)]
    val_files = [json.loads(l)["filename"] for l in open(VAL_JSON) if l.strip()]
    return counts, cls_fam, cls_gen, names, val_files


def part1(counts, device):
    out = {}
    for m in MODELS:
        res = {}
        for split in ("train", "val"):
            r = pm.measure(split, m, counts, device)
            res[split] = {"angle": r["angle_to_weight"]["overall"]["mean_deg"],
                          "fisher": r["fisher"]["overall"]["mean"],
                          "bands": {b: {"angle": r["angle_to_weight"]["bands"][b]["mean_deg"],
                                        "fisher": r["fisher"]["bands"][b]["mean"]}
                                    for b in r["angle_to_weight"]["bands"]}}
            if device == "cuda":
                torch.cuda.empty_cache()
        out[m] = res
        parts = []
        for b in res["val"]["bands"]:
            tr, va = res["train"]["bands"][b], res["val"]["bands"][b]
            parts.append(f"{b}: angle {tr['angle']:.1f}->{va['angle']:.1f} (+{va['angle'] - tr['angle']:.1f}), "
                         f"Fisher {tr['fisher']:.0f}->{va['fisher']:.0f} ({100 * (va['fisher'] / tr['fisher'] - 1):+.0f}%)")
        print(f"{P} [1] {NAMES[m]}: overall angle {res['train']['angle']:.1f}->{res['val']['angle']:.1f}, "
              f"Fisher {res['train']['fisher']:.0f}->{res['val']['fisher']:.0f} | " + " | ".join(parts), flush=True)
    return out


def ranks_for(m, device):
    z = np.load(f"{pm.GEOM}/{m}/predictions.npz")
    y = torch.as_tensor(z["labels"].astype(np.int64), device=device)
    saved_pred = z["predictions"].astype(np.int64)
    feat = np.load(f"{pm.GEOM}/{m}/features.npy", mmap_mode="r")
    w = F.normalize(pm.class_directions(*pm.RUNS[m]).float(), dim=-1).to(device)  # (N, k, C)
    nk, k, c = w.shape
    w2 = w.reshape(nk * k, c)
    rank = torch.empty(len(y), dtype=torch.long, device=device)
    pred = torch.empty(len(y), dtype=torch.long, device=device)
    for s in range(0, len(y), CHUNK):
        f = F.normalize(torch.as_tensor(np.asarray(feat[s:s + CHUNK], dtype=np.float32), device=device), dim=1)
        sc = (f @ w2.T).view(len(f), nk, k).max(-1).values
        yy = y[s:s + CHUNK]
        t = sc.gather(1, yy[:, None])
        rank[s:s + CHUNK] = (sc > t).sum(1) + 1
        pred[s:s + CHUNK] = sc.argmax(1)
    pred = pred.cpu().numpy()
    agree = float(np.mean(pred == saved_pred))
    return y.cpu().numpy(), rank.cpu().numpy(), pred, agree


def part2(cls_fam, cls_gen, names, val_files, device):
    out, keep = {}, {}
    for m in MODELS:
        y, rank, pred, agree = ranks_for(m, device)
        wrong = pred != y
        res = {"n": int(len(y)), "pred_matches_saved": agree}
        for lab, sel in (("rank2", rank == 2), ("rank3_5", (rank >= 3) & (rank <= 5)), ("rank_gt5", rank > 5)):
            sel = sel & wrong
            same_gen = cls_gen[pred[sel]] == cls_gen[y[sel]]
            same_fam = cls_fam[pred[sel]] == cls_fam[y[sel]]
            res[lab] = {"pct_of_images": float(100 * sel.mean()),
                        "pred_same_genus_pct": float(100 * same_gen.mean()) if sel.any() else None,
                        "pred_same_family_other_genus_pct": float(100 * (same_fam & ~same_gen).mean()) if sel.any() else None,
                        "pred_other_family_pct": float(100 * (~same_fam).mean()) if sel.any() else None}
        out[m] = res
        keep[m] = (y, rank, pred)
        r2, r5 = res["rank2"], res["rank_gt5"]
        print(f"{P} [2] {NAMES[m]}: pred==saved {agree:.4f} | near misses (rank 2) {r2['pct_of_images']:.2f}% of images: "
              f"same genus {r2['pred_same_genus_pct']:.0f}%, same family other genus {r2['pred_same_family_other_genus_pct']:.0f}%, "
              f"other family {r2['pred_other_family_pct']:.0f}% | far misses (rank >5) {r5['pct_of_images']:.2f}%: "
              f"same genus {r5['pred_same_genus_pct']:.0f}%, same family other genus {r5['pred_same_family_other_genus_pct']:.0f}%, "
              f"other family {r5['pred_other_family_pct']:.0f}%", flush=True)
    # Far misses ArcFace has and DBM does not.
    ya, ra, pa = keep["arcface"]
    yd, rd, pd = keep["dbm"]
    arc_far, dbm_ok = ra > 5, pd == yd
    only = np.flatnonzero(arc_far & dbm_ok)
    both_far = int(np.sum(arc_far & (rd > 5)))
    same_gen = cls_gen[pa[only]] == cls_gen[ya[only]]
    same_fam = cls_fam[pa[only]] == cls_fam[ya[only]]
    out["arcface_far_dbm_correct"] = {
        "n": int(len(only)), "arcface_far_total": int(arc_far.sum()), "both_far": both_far,
        "arcface_pred_same_genus_pct": float(100 * same_gen.mean()),
        "arcface_pred_same_family_other_genus_pct": float(100 * (same_fam & ~same_gen).mean()),
        "arcface_pred_other_family_pct": float(100 * (~same_fam).mean()),
        "dbm_rank_of_true_in_arcface_far_misses_median": float(np.median(rd[arc_far]))}
    o = out["arcface_far_dbm_correct"]
    print(f"{P} [2] ArcFace far misses: {o['arcface_far_total']}; of these DBM gets {o['n']} right and also ranks below "
          f"fifth {both_far}. ArcFace's predictions on the {o['n']}: same genus {o['arcface_pred_same_genus_pct']:.0f}%, "
          f"same family other genus {o['arcface_pred_same_family_other_genus_pct']:.0f}%, other family "
          f"{o['arcface_pred_other_family_pct']:.0f}%", flush=True)
    rows = []
    for i in only:
        rows.append({"filename": val_files[i], "true": names[ya[i]], "arcface_pred": names[pa[i]],
                     "arcface_true_rank": int(ra[i]),
                     "relation": "same genus" if cls_gen[pa[i]] == cls_gen[ya[i]] else
                     ("same family" if cls_fam[pa[i]] == cls_fam[ya[i]] else "other family")})
    os.makedirs(DIAG, exist_ok=True)
    if not rows:
        print(f"{P} no ArcFace-only far misses; gallery skipped", flush=True)
        return out
    with open(f"{DIAG}/cone_far_miss_gallery.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    try:
        make_gallery(rows)
    except Exception as e:  # the numbers do not depend on the picture
        print(f"{P} gallery skipped: {e!r}", flush=True)
    return out


def make_gallery(rows, n=12, seed=0):
    from PIL import Image, ImageDraw
    pick = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    tw, th, cap = 300, 380, 46
    sheet = Image.new("RGB", (4 * tw, 3 * (th + cap)), "white")
    d = ImageDraw.Draw(sheet)
    for j, i in enumerate(pick):
        r = rows[i]
        im = Image.open(r["filename"]).convert("RGB")
        im.thumbnail((tw, th))
        x, yv = (j % 4) * tw, (j // 4) * (th + cap)
        sheet.paste(im, (x + (tw - im.width) // 2, yv))
        d.text((x + 4, yv + th + 2), f"true: {r['true'][:40]}", fill="black")
        d.text((x + 4, yv + th + 16), f"ArcFace: {r['arcface_pred'][:37]}", fill="darkred")
        d.text((x + 4, yv + th + 30), f"{r['relation']}, true rank {r['arcface_true_rank']}", fill="gray")
    sheet.save(f"{DIAG}/cone_far_miss_gallery.png")
    print(f"{P} wrote gallery of {len(pick)} images", flush=True)


def part3(counts, cls_fam, cls_gen, device):
    out = {}
    for m in MODELS:
        z = np.load(f"{pm.GEOM}/{m}/predictions.npz")
        y_np = z["labels"].astype(np.int64)
        feat = np.load(f"{pm.GEOM}/{m}/features.npy", mmap_mode="r")
        x = F.normalize(torch.as_tensor(np.asarray(feat, dtype=np.float32), device=device), dim=1)
        y = torch.as_tensor(y_np, device=device)
        cnt = torch.bincount(y, minlength=pm.N).float()
        mu = torch.zeros(pm.N, x.size(1), device=device).index_add_(0, y, x) / cnt.clamp(min=1)[:, None]
        Sw = torch.zeros(x.size(1), x.size(1), device=device)
        for s in range(0, len(y), 65536):
            cc = x[s:s + 65536] - mu[y[s:s + 65536]]
            Sw += cc.T @ cc
        Sw /= len(y)
        Sw += 1e-3 * torch.trace(Sw) / Sw.size(0) * torch.eye(Sw.size(0), device=device)
        present = cnt > 0
        mu_p = mu[present]
        A = torch.linalg.inv(Sw)
        B = mu_p @ A
        diag = (mu_p * B).sum(1)
        D = diag[:, None] + diag[None, :] - 2.0 * (B @ mu_p.T)
        idx = present.cpu().numpy()
        g = torch.as_tensor(cls_gen[idx], device=device)
        f = torch.as_tensor(cls_fam[idx], device=device)
        same_g = g[:, None] == g[None, :]
        same_f = f[:, None] == f[None, :]
        offd = ~torch.eye(len(g), dtype=torch.bool, device=device)
        groups = {"same_genus": same_g & offd, "same_family_other_genus": same_f & ~same_g,
                  "other_family": ~same_f}
        res = {"n_species": int(len(g))}
        for k, msk in groups.items():
            res[k] = {"n_pairs": int(msk.sum().item() // 2), "mean_J": float(D[msk].mean().item()),
                      "median_J": float(D[msk].float().median().item())}
        Dn = D.masked_fill(~offd, float("inf"))
        nn_val, nn_idx = Dn.min(1)
        nn_same_g = (g[nn_idx] == g).float().mean().item()
        res["nearest_neighbour"] = {"mean_J": float(nn_val.mean().item()),
                                    "same_genus_pct": float(100 * nn_same_g)}
        has_sister = (same_g & offd).any(1)
        Dg = D.masked_fill(~(same_g & offd), float("inf"))
        res["nearest_sister_mean_J"] = float(Dg.min(1).values[has_sister].mean().item())
        res["n_species_with_sister"] = int(has_sister.sum().item())
        out[m] = res
        print(f"{P} [3] {NAMES[m]}: mean pairwise Fisher J | same genus {res['same_genus']['mean_J']:.0f} "
              f"({res['same_genus']['n_pairs']:,} pairs) | same family other genus "
              f"{res['same_family_other_genus']['mean_J']:.0f} | other family {res['other_family']['mean_J']:.0f} | "
              f"nearest neighbour J {res['nearest_neighbour']['mean_J']:.0f} (same genus "
              f"{res['nearest_neighbour']['same_genus_pct']:.0f}%) | nearest sister J {res['nearest_sister_mean_J']:.0f}",
              flush=True)
        del x, mu, Sw, A, B, D, Dn, Dg, same_g, same_f, offd
        if device == "cuda":
            torch.cuda.empty_cache()
    base = out["dbm"]
    for m in ("arcface", "arcface_ecc", "cosine_m0"):
        r = out[m]
        ratios = {k: r[k]["mean_J"] / base[k]["mean_J"]
                  for k in ("same_genus", "same_family_other_genus", "other_family")}
        ratios["nearest_sister"] = r["nearest_sister_mean_J"] / base["nearest_sister_mean_J"]
        out[f"{m}_over_dbm"] = ratios
        print(f"{P} [3] {NAMES[m]} / DBM mean J: same genus x{ratios['same_genus']:.2f}, same family other genus "
              f"x{ratios['same_family_other_genus']:.2f}, other family x{ratios['other_family']:.2f}, "
              f"nearest sister x{ratios['nearest_sister']:.2f}", flush=True)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{P} device={device}", flush=True)
    counts, cls_fam, cls_gen, names, val_files = load_meta()
    result = {"part1_train_vs_val": part1(counts, device),
              "part2_far_misses": part2(cls_fam, cls_gen, names, val_files, device),
              "part3_sister_vs_unrelated": part3(counts, cls_fam, cls_gen, device)}
    os.makedirs(DIAG, exist_ok=True)
    json.dump(result, open(f"{DIAG}/cone_test.json", "w"), indent=1)
    print(f"{P} wrote {DIAG}/cone_test.json", flush=True)


if __name__ == "__main__":
    main()
