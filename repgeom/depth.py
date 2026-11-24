"""At what depth does each taxonomic level get resolved?

The parent node established two things about the *final* representation: 68% of
errors are congeneric, and 80% of species have a congener as their nearest
neighbour in feature space -- taxonomic structure the model was never taught.
This asks where in the network that structure appears.

Method, and why it needs no training. A Swin-base has four stages of
progressively coarser, wider feature maps. Mean-pool each stage's map and you
get one vector per image per depth. Then, at each depth, fit class centroids on
half the validation images and classify the other half by nearest centroid.
Nearest-centroid accuracy is a serviceable stand-in for "how linearly available
is this distinction here", and unlike a linear probe it has no optimiser, no
learning rate and no capacity knob to argue about -- which matters when the
whole claim is a comparison *across* depths of different widths.

Running it at family, genus and species level gives a depth x taxonomic-level
table. The expected shape -- coarse levels resolved early, species only at the
end -- would say the network spends most of its depth on botanical structure
that was never in its labels, and leaves the hardest discrimination to the last
stage.

The fit/eval split is per-class alternation, so both halves see every class with
at least two images and no image is ever classified against a centroid it
helped define.
"""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import extract

SHARD = 50_000
CHUNK = 2048


def _pool(hs):
    """Stage output -> (spatial mean, spatial std) per image.

    The mean alone is what the parent node used, and it is a poor summary of an
    early layer: at stage 1 the discriminating content is texture -- leaf
    surface, vein density, indumentum -- which shows up as how strongly a
    channel *varies* across the sheet, not as its average level. Averaging
    removes exactly that. Returning both lets the analysis separate "there is no
    taxonomic structure here" from "the mean threw it away".
    """
    if hs.dim() == 4:        # (B, C, H, W) -- reshaped_hidden_states
        return hs.mean(dim=(2, 3)), hs.flatten(2).std(dim=2)
    if hs.dim() == 3:        # (B, L, C) -- token sequence
        return hs.mean(dim=1), hs.std(dim=1)
    raise ValueError(f"unexpected hidden-state rank {hs.dim()}")


def build_depth_cache(paths, cfg, device):
    """Pooled features at every stage. Returns {stage_name: (N, C) float16}."""
    ck = cfg["model"]["checkpoint"]
    mode = cfg["extract"]["preprocess"]
    cdir = os.path.join(cfg["extract"]["cache_root"],
                        os.path.basename(ck.rstrip("/")) + "__" + mode + "__depth_meanstd")
    os.makedirs(cdir, exist_ok=True)
    n = len(paths)
    n_shards = (n + SHARD - 1) // SHARD
    meta_file = os.path.join(cdir, "depth_manifest.json")

    names = None
    if os.path.exists(meta_file):
        with open(meta_file) as fh:
            meta = json.load(fh)
        if meta.get("n_images") == n and meta.get("checkpoint") == ck and meta.get("preprocess") == mode:
            names = meta["stages"]

    have = names is not None and all(
        os.path.exists(os.path.join(cdir, f"{s}_{d}_{i:05d}.npy"))
        for s in names for d in ("mean", "std") for i in range(n_shards))

    if not have:
        from transformers import AutoImageProcessor, SwinModel
        proc = AutoImageProcessor.from_pretrained(ck)
        size = extract.processor_size(proc)
        tf = extract.build_preprocess(mode, proc, size)
        model = SwinModel.from_pretrained(ck).to(device).eval()
        autocast = torch.bfloat16 if device == "cuda" else torch.float32

        t0, seen = time.time(), 0
        names = None
        for s in range(n_shards):
            lo, hi = s * SHARD, min((s + 1) * SHARD, n)
            loader = DataLoader(extract.ImageList(paths[lo:hi], tf, size),
                                batch_size=int(cfg["extract"]["batch_size"]),
                                num_workers=int(cfg["extract"]["num_workers"]),
                                pin_memory=(device == "cuda"), shuffle=False)
            buf = None
            with torch.no_grad():
                for px, _ok in loader:
                    px = px.to(device, non_blocking=True)
                    with torch.autocast(device_type=device, dtype=autocast,
                                        enabled=(device == "cuda")):
                        out = model(pixel_values=px, output_hidden_states=True)
                    hs = getattr(out, "reshaped_hidden_states", None) or out.hidden_states
                    if names is None:
                        names = ["embed"] + [f"stage{i}" for i in range(1, len(hs))]
                        print("__RG__: depth taps -> " + "  ".join(
                            f"{nm}:{tuple(h.shape[1:])}" for nm, h in zip(names, hs)), flush=True)
                    if buf is None:
                        buf = {(nm, d): [] for nm in names for d in ("mean", "std")}
                    for nm, h in zip(names, hs):
                        mu, sd = _pool(h.float())
                        buf[(nm, "mean")].append(mu.cpu().numpy().astype(np.float16))
                        buf[(nm, "std")].append(sd.cpu().numpy().astype(np.float16))
            for nm in names:
                for d in ("mean", "std"):
                    extract._save_atomic(os.path.join(cdir, f"{nm}_{d}_{s:05d}.npy"),
                                         np.concatenate(buf[(nm, d)]))
            seen = hi
            rate = seen / max(1e-9, time.time() - t0)
            print(f"__RG__: depth shard {s + 1}/{n_shards}  {hi:,}/{n:,}  {rate:.0f} img/s  "
                  f"eta {(n - hi) / max(rate, 1e-9) / 60:.1f} min", flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        with open(meta_file, "w") as fh:
            json.dump({"n_images": n, "checkpoint": ck, "preprocess": mode, "stages": names}, fh)

    feats = {}
    for nm in names:
        for d in ("mean", "std"):
            feats[(nm, d)] = np.concatenate(
                [np.load(os.path.join(cdir, f"{nm}_{d}_{i:05d}.npy")) for i in range(n_shards)])
        print(f"__RG__: {nm:8s} mean {feats[(nm, 'mean')].shape}  "
              f"std {feats[(nm, 'std')].shape}")
    return feats, names


def cosine_concentration(X, device, n_sample=2000, seed=0):
    """Mean/std of pairwise cosine among sampled images.

    The diagnostic that decides whether a nearest-centroid number at this depth
    means anything. If every pooled vector points essentially the same way
    (mean cosine ~1, tiny spread), the argmax over class centroids is resolving
    noise and the resulting accuracy is not interpretable as "this distinction
    is unavailable here" -- it is an artefact of an uncentred representation.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    x = torch.as_tensor(np.ascontiguousarray(X[idx])).to(device).float()
    xn = torch.nn.functional.normalize(x, dim=1)
    s = (xn @ xn.T)
    iu = torch.triu_indices(len(idx), len(idx), offset=1, device=device)
    v = s[iu[0], iu[1]]
    return float(v.mean()), float(v.std())


def prep(X, centre):
    """Optionally remove the global mean direction before any cosine is taken."""
    if not centre:
        return X
    mu = X.astype(np.float32).mean(axis=0, keepdims=True)
    return (X.astype(np.float32) - mu)


def alternating_split(labels, n_classes):
    """Within each class, send images to fit/eval alternately.

    Every class with >=2 images is represented on both sides, so a rare species
    is not silently dropped from the evaluation the way a global random split
    would drop it.
    """
    order = np.argsort(labels, kind="mergesort")
    fit = np.zeros(len(labels), dtype=bool)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and labels[order[j + 1]] == labels[order[i]]:
            j += 1
        idx = order[i:j + 1]
        fit[idx[0::2]] = True          # even positions fit, odd eval
        i = j + 1
    return fit


def nearest_centroid_accuracy(X_fit, y_fit, X_ev, y_ev, n_groups, device):
    """Cosine nearest-centroid accuracy; groups with no fit images are unreachable."""
    C = X_fit.shape[1]
    sums = torch.zeros(n_groups, C, dtype=torch.float32, device=device)
    cnt = torch.zeros(n_groups, dtype=torch.float32, device=device)
    yf = torch.as_tensor(y_fit, dtype=torch.long)
    for lo in range(0, len(X_fit), CHUNK):
        x = torch.as_tensor(np.ascontiguousarray(X_fit[lo:lo + CHUNK])).to(device).float()
        yy = yf[lo:lo + CHUNK].to(device)
        sums.index_add_(0, yy, x)
        cnt.index_add_(0, yy, torch.ones(len(yy), device=device))
    live = cnt > 0
    mu = torch.zeros_like(sums)
    mu[live] = sums[live] / cnt[live][:, None]
    mu = torch.nn.functional.normalize(mu, dim=1)
    mu[~live] = 0.0                     # zero rows can never win an argmax of cosines

    correct = 0
    ye = torch.as_tensor(y_ev, dtype=torch.long)
    for lo in range(0, len(X_ev), CHUNK):
        x = torch.as_tensor(np.ascontiguousarray(X_ev[lo:lo + CHUNK])).to(device).float()
        x = torch.nn.functional.normalize(x, dim=1)
        pred = (x @ mu.T).argmax(1).cpu()
        correct += int((pred == ye[lo:lo + CHUNK]).sum())
    return correct / len(X_ev), int(live.sum())


def congener_neighbour_rate(X, y, n_classes, gen_of_class, device):
    """Share of species whose nearest *other* species centroid is a congener."""
    C = X.shape[1]
    sums = torch.zeros(n_classes, C, dtype=torch.float32, device=device)
    cnt = torch.zeros(n_classes, dtype=torch.float32, device=device)
    yy_all = torch.as_tensor(y, dtype=torch.long)
    for lo in range(0, len(X), CHUNK):
        x = torch.as_tensor(np.ascontiguousarray(X[lo:lo + CHUNK])).to(device).float()
        yy = yy_all[lo:lo + CHUNK].to(device)
        sums.index_add_(0, yy, x)
        cnt.index_add_(0, yy, torch.ones(len(yy), device=device))
    live = (cnt > 0).cpu().numpy()
    mu = torch.zeros_like(sums)
    lv = torch.as_tensor(live, device=device)
    mu[lv] = sums[lv] / cnt[lv][:, None]
    mu = torch.nn.functional.normalize(mu, dim=1)

    same = 0
    total = 0
    for lo in range(0, n_classes, CHUNK):
        hi = min(lo + CHUNK, n_classes)
        s = mu[lo:hi] @ mu.T
        s[:, ~lv] = -2.0
        s[torch.arange(hi - lo, device=device), torch.arange(lo, hi, device=device)] = -2.0
        nn = s.argmax(1).cpu().numpy()
        for i in range(lo, hi):
            if not live[i]:
                continue
            total += 1
            if gen_of_class[i] and gen_of_class[i] == gen_of_class[nn[i - lo]]:
                same += 1
        del s
    return same / max(total, 1), total
