"""What the sibling margin looks like on the real training sets, and a smoke test of the head.

For each training manifest (the 15.5k base data and the 32k and 64k ladder rungs) this reads
the species, genus and family of every training image and computes, with the rule in
sibling_margin.py:

  - the margin of every species (m_y, against a genus sibling), as percentiles over species and
    over training images (a common species weighs more per image, because it has more images),
  - how many species and images carry a margin above 0.01, 0.03, 0.05 and 0.10 cosine units,
  - per training-image band: species, images, the margin range, and how many competitors
    receive a margin against one image,
  - the same exposure for other rarity exponents tau (0.25, 0.5, 1).

Then it builds SiblingMarginHead on the real counts and taxonomy of each manifest, runs a real
batch of labels forward and backward, and checks the applied margins against the NumPy rule.

No images are read and no GPU is needed. The last line is RESULT_JSON, for the writeup.

Usage (from finetuning/SWIN):
  python loss_ablation/sibling_margin_distribution.py
  python loss_ablation/sibling_margin_distribution.py --m-max 0.15 --tau 1.0
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss_ablation import sibling_margin as sm  # noqa: E402

ROOT = "/projectnb/herbdl"
MANIFESTS = {
    "base 15.5k": f"{ROOT}/data/kaggle-herbaria/train_2022.json",
    "ladder 32k": f"{ROOT}/workspaces/faridkar/herbdl/data/ladder/n32k_train.json",
    "ladder 64k": f"{ROOT}/workspaces/faridkar/herbdl/data/ladder/n64k_train.json",
}
LABEL = "scientificNameEncoded"
TAUS = (0.25, 0.5, 1.0)


def load_manifest(path):
    """species ids, genus, family of every training image (one JSON object per line)."""
    ids, genus, family = [], [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(int(r[LABEL]))
                genus.append(r["genus"])
                family.append(r["family"])
    return np.array(ids), genus, family


def pct(x):
    return f"{100 * x:.1f}"


def print_dataset(name, d):
    print(f"\n=== {name}: {d['species']:,} species, {d['images']:,} training images "
          f"({d['n_min']} to {d['n_max']} per species)")
    print(f"species with at least one genus sibling: {pct(d['species_with_sibling'])}%; "
          f"genus siblings per species {d['mean_siblings_per_species']:.1f} on average, "
          f"{d['mean_siblings_per_image']:.1f} per training image; other-genus species of the same family "
          f"{d['mean_other_genus_family_per_species']:.0f} per species")
    print(f"\nMargin of a species against a genus sibling, cosine units (m_max {d['m_max']}, tau {d['tau']}):")
    print("| | min | 10% | 25% | median | 75% | 90% | max |")
    print("|---|---|---|---|---|---|---|---|")
    for label, key in (("over species", "species_margin"), ("over training images", "image_margin")):
        print(f"| {label} | " + " | ".join(f"{d[f'{key}_p{q}']:.4f}" for q in (0, 10, 25, 50, 75, 90, 100)) + " |")
    print("\nShare carrying at least this margin (%):")
    print("| margin | species | training images |")
    print("|---|---|---|")
    for t in (0.01, 0.03, 0.05, 0.10):
        print(f"| {t} | {pct(d[f'species_share_margin_ge_{t}'])} | {pct(d[f'image_share_margin_ge_{t}'])} |")
    print("\nBy training images of the species:")
    print("| images | species | % species | training images | % images | margin range | median margin | siblings | other-genus family | competitors with a margin per image |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b in d["bands"]:
        print(f"| {b['band']} | {b['species']:,} | {pct(b['species_share'])} | {b['images']:,} | {pct(b['image_share'])} | "
              f"{b['margin_min']:.4f}-{b['margin_max']:.4f} | {b['margin_median']:.4f} | {b['siblings_per_species']:.0f} | "
              f"{b['other_genus_family_per_species']:.0f} | {b['competitors_with_margin_per_image']:.0f} |")


def smoke(counts, gid, fid, ids, m_max, tau, family_weight, seed=0):
    """Real labels through the head on CPU: finite loss and gradient, margins equal to the rule."""
    import torch
    import torch.nn.functional as F
    from loss_ablation.losses import SiblingMarginHead

    torch.manual_seed(seed)
    C = len(counts)
    head = SiblingMarginHead(1024, C, counts, gid, fid, s=30.0, m_max=m_max, tau=tau,
                             family_weight=family_weight, log_every=0)
    rng = np.random.default_rng(seed)
    y = ids[rng.integers(0, len(ids), size=128)]
    x = torch.randn(128, 1024)
    logits = head(x, torch.as_tensor(y))
    loss = F.cross_entropy(logits, torch.as_tensor(y))
    loss.backward()
    cos = F.linear(F.normalize(x, dim=1), F.normalize(head.weight, dim=1))
    applied = (logits / 30.0 - cos).detach().numpy()
    want = sm.pair_margin(y, sm.species_margin(counts, m_max, tau), gid, fid, 1.0, family_weight)
    ok = bool(np.allclose(applied, want, atol=1e-5)) and bool(torch.isfinite(head.weight.grad).all()) \
        and bool(torch.isfinite(loss))
    return {"loss": float(loss), "matches_rule": ok, "competitors_with_margin_mean": float((applied > 0).sum(1).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m-max", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--family-weight", type=float, default=0.25)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--out", default=f"{ROOT}/workspaces/{os.environ.get('USER', 'unknown')}/herbdl/finetuning/output/SWIN/sibling_margin_distribution")
    args = ap.parse_args(argv)

    results, sensitivity = {}, {}
    for name, path in MANIFESTS.items():
        ids, genus, family = load_manifest(path)
        C = int(ids.max()) + 1
        gid, fid = sm.taxonomy_arrays(ids, genus, family, C)
        cnt = Counter(ids.tolist())
        counts = np.array([cnt[i] for i in range(C)])
        d = sm.distribution(counts, gid, fid, args.m_max, args.tau, args.family_weight)
        results[name] = d
        print_dataset(name, d)
        sensitivity[name] = {str(t): sm.distribution(counts, gid, fid, args.m_max, t, args.family_weight)
                             for t in TAUS}
        if not args.no_smoke:
            d["smoke"] = smoke(counts, gid, fid, ids, args.m_max, args.tau, args.family_weight)
            print(f"smoke test on {name}: {d['smoke']}")
        sys.stdout.flush()

    print("\nSensitivity to the rarity exponent tau (share of training images with margin >= 0.03 / 0.05, %):")
    print("| data | " + " | ".join(f"tau {t}" for t in TAUS) + " |")
    print("|---|" + "---|" * len(TAUS))
    for name, by_tau in sensitivity.items():
        print(f"| {name} | " + " | ".join(
            f"{pct(by_tau[str(t)]['image_share_margin_ge_0.03'])} / {pct(by_tau[str(t)]['image_share_margin_ge_0.05'])}"
            for t in TAUS) + " |")

    os.makedirs(args.out, exist_ok=True)
    payload = {"settings": vars(args), "datasets": results,
               "sensitivity": {n: {t: {k: v for k, v in d.items() if k != "bands"} for t, d in by.items()}
                               for n, by in sensitivity.items()}}
    with open(f"{args.out}/distribution.json", "w") as f:
        json.dump(payload, f, indent=1)
    print("RESULT_JSON " + json.dumps(payload))


if __name__ == "__main__":
    main()
