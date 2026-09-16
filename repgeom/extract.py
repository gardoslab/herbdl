"""One forward pass of the fine-tuned classifier over the validation split.

Everything in this line of work reads the cache this builds, so it is written
once and reused: re-running the analysis must never mean re-running 168k
images through a GPU.

What is cached per image, and why:

  feat       pooled 1024-d penultimate vector -> class centroids, manifold geometry
  pred/rank  decision and the rank of the true class -> per-class recall, near-misses
  top-k      the actual competitors -> taxonomic structure of confusions
  spat_*     the 49 per-position logit contributions -> spatial evidence maps

The spatial term deserves a note. `SwinModel`'s pooler is a plain
`AdaptiveAvgPool1d(1)` over the 7x7=49 final tokens, so for classifier row w_c

    logit_c = w_c . mean_p h_p + b_c = mean_p (w_c . h_p) + b_c

exactly. `w_c . h_p` is therefore a *decomposition* of the logit, not a saliency
heuristic: the 49 numbers sum (over the mean) to the logit itself. No extra
backward pass, no smoothing, nothing to tune.
"""

import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SHARD_SIZE = 50_000
# The arrays written per shard. Keeping the list in one place keeps the writer,
# the resume check and the loader from drifting apart.
FIELDS = ("feat", "pred", "valid", "true_logit", "true_rank",
          "topk_idx", "topk_logit", "entropy", "spat_true", "spat_pred")


class ImageList(Dataset):
    """Paths -> preprocessed tensors, tolerant of unreadable scans."""

    def __init__(self, paths, processor, size):
        self.paths = paths
        self.processor = processor
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            px = self.processor(images=img, return_tensors="pt").pixel_values[0]
            return px, 1
        except Exception:
            # A few corrupt files in a 168k-image pool must not kill a job that
            # holds a GPU. They are flagged invalid and dropped from every
            # statistic downstream.
            return torch.zeros(3, self.size, self.size), 0


def cache_dir_for(root, checkpoint):
    return os.path.join(root, os.path.basename(checkpoint.rstrip("/")))


def _p(cdir, field, i):
    return os.path.join(cdir, f"{field}_{i:05d}.npy")


def _save_atomic(path, arr):
    # A killed job must not leave a half-written shard that a resume would trust.
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def build_cache(paths, labels, cfg, device):
    """Extract (or resume) the val-split cache. Returns a dict of arrays."""
    ck = cfg["model"]["checkpoint"]
    cdir = cache_dir_for(cfg["extract"]["cache_root"], ck)
    os.makedirs(cdir, exist_ok=True)
    n = len(paths)
    topk = int(cfg["extract"]["topk"])
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE

    manifest_file = os.path.join(cdir, "cache_manifest.json")
    if os.path.exists(manifest_file):
        with open(manifest_file) as fh:
            prior = json.load(fh)
        if prior.get("n_images") != n or prior.get("checkpoint") != ck or prior.get("topk") != topk:
            raise RuntimeError(
                f"Cache at {cdir} was built for n={prior.get('n_images')} "
                f"checkpoint={prior.get('checkpoint')} topk={prior.get('topk')}, but this run "
                f"has n={n} checkpoint={ck} topk={topk}. Point extract.cache_root elsewhere "
                "rather than mixing pools.")

    done = [s for s in range(n_shards) if all(os.path.exists(_p(cdir, f, s)) for f in FIELDS)]
    print(f"__RG__: cache {cdir}")
    print(f"__RG__: {len(done)}/{n_shards} shards already complete", flush=True)

    if len(done) < n_shards:
        from transformers import AutoImageProcessor, SwinForImageClassification

        proc = AutoImageProcessor.from_pretrained(ck)
        # transformers >=4.57 returns a SizeDict here, which is mapping-like but
        # not a dict subclass, so probe rather than isinstance-check. Only the
        # corrupt-image fallback tensor depends on this; the real preprocessing
        # is whatever the checkpoint's processor does.
        raw = getattr(proc, "size", None)
        size = 224
        if raw is not None:
            try:
                size = int(raw["height"])
            except Exception:
                size = int(getattr(raw, "height", 224) or 224)
        model = SwinForImageClassification.from_pretrained(ck).to(device).eval()
        W = model.classifier.weight.detach()          # (K, C)
        n_classes = W.shape[0]
        autocast = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"__RG__: model loaded, classes={n_classes:,} feat_dim={W.shape[1]} "
              f"image_size={size}", flush=True)

        t0, seen = time.time(), 0
        for s in range(n_shards):
            if s in done:
                continue
            lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, n)
            loader = DataLoader(ImageList(paths[lo:hi], proc, size),
                                batch_size=int(cfg["extract"]["batch_size"]),
                                num_workers=int(cfg["extract"]["num_workers"]),
                                pin_memory=(device == "cuda"), shuffle=False, drop_last=False)
            y_shard = torch.as_tensor(labels[lo:hi], dtype=torch.long)
            buf = {f: [] for f in FIELDS}
            off = 0

            with torch.no_grad():
                for px, ok in loader:
                    bs = px.shape[0]
                    y = y_shard[off:off + bs].to(device)
                    off += bs
                    px = px.to(device, non_blocking=True)
                    with torch.autocast(device_type=device, dtype=autocast,
                                        enabled=(device == "cuda")):
                        out = model.swin(pixel_values=px)
                        h = out.last_hidden_state           # (B, 49, C)
                        pooled = out.pooler_output          # (B, C) == h.mean(1)
                        logits = model.classifier(pooled)   # (B, K)

                    h = h.float()
                    logits = logits.float()

                    tl = logits.gather(1, y[:, None]).squeeze(1)
                    # Rank of the true class: how many classes outscore it.
                    rank = (logits > tl[:, None]).sum(dim=1)
                    tk = logits.topk(topk, dim=1)
                    pred = tk.indices[:, 0]
                    logp = torch.log_softmax(logits, dim=1)
                    ent = -(logp.exp() * logp).sum(dim=1)

                    # Per-position logit contributions for the true and the
                    # predicted class: (B, 49) = h . w_class
                    buf["spat_true"].append(
                        torch.einsum("bpc,bc->bp", h, W[y].float()).cpu().numpy().astype(np.float16))
                    buf["spat_pred"].append(
                        torch.einsum("bpc,bc->bp", h, W[pred].float()).cpu().numpy().astype(np.float16))

                    buf["feat"].append(pooled.float().cpu().numpy().astype(np.float16))
                    buf["pred"].append(pred.cpu().numpy().astype(np.int32))
                    buf["valid"].append(ok.numpy().astype(bool))
                    buf["true_logit"].append(tl.cpu().numpy().astype(np.float32))
                    buf["true_rank"].append(rank.cpu().numpy().astype(np.int32))
                    buf["topk_idx"].append(tk.indices.cpu().numpy().astype(np.int32))
                    buf["topk_logit"].append(tk.values.cpu().numpy().astype(np.float32))
                    buf["entropy"].append(ent.cpu().numpy().astype(np.float32))

            for f in FIELDS:
                _save_atomic(_p(cdir, f, s), np.concatenate(buf[f]))
            seen += hi - lo
            rate = seen / max(1e-9, time.time() - t0)
            print(f"__RG__: shard {s + 1}/{n_shards}  images {hi:,}/{n:,}  "
                  f"{rate:.0f} img/s  eta {(n - hi) / max(rate, 1e-9) / 60:.1f} min", flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        with open(manifest_file, "w") as fh:
            json.dump({"n_images": n, "checkpoint": ck, "topk": topk,
                       "n_classes": int(n_classes)}, fh)

    cache = {f: np.concatenate([np.load(_p(cdir, f, s)) for s in range(n_shards)])
             for f in FIELDS}
    print(f"__RG__: cache loaded, feat={cache['feat'].shape} "
          f"valid={cache['valid'].sum():,}/{n:,} "
          f"({1 - cache['valid'].mean():.4%} unreadable)", flush=True)
    return cache
