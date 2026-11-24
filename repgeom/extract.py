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


def build_preprocess(mode, proc, size):
    """The three candidate val-time pipelines.

    They matter more than they look. A herbarium sheet is a tall rectangle, so
    squashing it to a square and cropping its middle square are genuinely
    different views of the specimen, and bilinear vs bicubic is a real
    difference at a ~5x downsample. Which one the checkpoint was scored under
    decides whether this analysis describes the model that was reported.

      processor      the saved ViTImageProcessor: bicubic squash to 224x224
      eval_protocol  what SWIN_finetuning.py's _val_transforms actually does:
                     Resize((224,224)) -- torchvision defaults to BILINEAR and
                     a 2-tuple squashes -- then a no-op CenterCrop
      shortest_edge  aspect-preserving resize of the short side, then the
                     centre square: the usual ImageNet recipe, kept as a control
    """
    from torchvision import transforms as T

    if mode == "processor":
        return lambda img: proc(images=img, return_tensors="pt").pixel_values[0]

    norm = T.Normalize(mean=proc.image_mean, std=proc.image_std)
    if mode == "eval_protocol":
        t = T.Compose([T.Resize((size, size)), T.CenterCrop((size, size)), T.ToTensor(), norm])
    elif mode == "shortest_edge":
        t = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(size), T.ToTensor(), norm])
    else:
        raise ValueError(f"unknown preprocess mode {mode!r}")
    return t


class ImageList(Dataset):
    """Paths -> preprocessed tensors, tolerant of unreadable scans."""

    def __init__(self, paths, transform, size):
        self.paths = paths
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.transform(Image.open(self.paths[i]).convert("RGB")), 1
        except Exception:
            # A few corrupt files in a 168k-image pool must not kill a job that
            # holds a GPU. They are flagged invalid and dropped from every
            # statistic downstream.
            return torch.zeros(3, self.size, self.size), 0


def load_head(ck, cfg):
    """Read the (K, C) class-direction matrix straight out of the checkpoint file.

    This must not go through `from_pretrained`. An ArcFace checkpoint stores its
    head as `arcface.weight`, so `SwinForImageClassification` finds no
    `classifier.weight`, **randomly initialises one**, and every geometry number
    downstream would describe noise while looking entirely plausible. Reading the
    tensor by name and failing loudly if it is absent is the only safe option.

    Returns (W, bias, head_kind). ArcFace has no bias -- at inference it scores
    by cosine alone.
    """
    from safetensors.torch import load_file

    head = cfg["model"].get("head", "linear")
    want = {"linear": "classifier.weight", "arcface": "arcface.weight"}
    if head not in want:
        raise ValueError(f"model.head must be one of {sorted(want)}, got {head!r}")

    sd = load_file(os.path.join(ck, "model.safetensors"))
    name = want[head]
    if name not in sd:
        present = [k for k in sd if not k.startswith(("swin.", "swinv2.", "backbone."))]
        raise RuntimeError(
            f"model.head={head!r} expects tensor {name!r} in {ck}, which does not "
            f"contain it. Non-backbone tensors present: {present}")
    W = sd[name].float()
    b = sd.get(name.replace("weight", "bias"))
    print(f"__RG__: head={head} tensor={name} shape={tuple(W.shape)} "
          f"bias={'yes' if b is not None else 'none'}")
    return W, (b.float() if b is not None else None), head


def head_logits(pooled, W, b, head, scale):
    """Reproduce the checkpoint's own inference rule.

    linear   : pooled @ W.T + b
    arcface  : scale * cos(pooled, W), i.e. both sides L2-normalised and no bias.
               This mirrors SwinWithArcFace's eval branch exactly (k=1, margin
               applied only during training).
    """
    if head == "arcface":
        wn = torch.nn.functional.normalize(W, dim=1)
        en = torch.nn.functional.normalize(pooled, dim=1)
        return scale * (en @ wn.T)
    out = pooled @ W.T
    return out + b if b is not None else out


def processor_size(proc):
    """transformers >=4.57 returns a SizeDict here: mapping-like, not a dict."""
    raw = getattr(proc, "size", None)
    if raw is None:
        return 224
    try:
        return int(raw["height"])
    except Exception:
        return int(getattr(raw, "height", 224) or 224)


def probe_preprocessing(paths, labels, cfg, device, n_probe):
    """Score a sample under every candidate pipeline and print the comparison.

    The checkpoint ships a reported accuracy; whichever row reproduces it is the
    preprocessing the rest of this analysis must use. Printing all of them makes
    that choice auditable from the log instead of asserted in a comment.
    """
    from transformers import AutoImageProcessor, SwinForImageClassification

    ck = cfg["model"]["checkpoint"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(paths), size=min(n_probe, len(paths)), replace=False)
    sub_paths = [paths[i] for i in idx]
    sub_y = np.asarray(labels)[idx]

    proc = AutoImageProcessor.from_pretrained(ck)
    size = processor_size(proc)
    model = SwinForImageClassification.from_pretrained(ck).to(device).eval()
    # Same reason as load_head: model(...).logits would run the randomly
    # initialised classifier on an ArcFace checkpoint and the probe would
    # compare three flavours of noise.
    W_cpu, b_cpu, head = load_head(ck, cfg)
    W = W_cpu.to(device)
    b = b_cpu.to(device) if b_cpu is not None else None
    scale = float(cfg["model"].get("arcface_scale", 30.0))
    autocast = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"__RG__: preprocessing probe on {len(sub_paths):,} sampled val images "
          f"(head={head})", flush=True)
    results = {}
    for mode in ("processor", "eval_protocol", "shortest_edge"):
        tf = build_preprocess(mode, proc, size)
        loader = DataLoader(ImageList(sub_paths, tf, size),
                            batch_size=int(cfg["extract"]["batch_size"]),
                            num_workers=int(cfg["extract"]["num_workers"]),
                            pin_memory=(device == "cuda"), shuffle=False)
        preds, off = [], 0
        with torch.no_grad():
            for px, _ok in loader:
                px = px.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=autocast,
                                    enabled=(device == "cuda")):
                    pooled = model.swin(pixel_values=px).pooler_output
                logits = head_logits(pooled.float(), W, b, head, scale)
                preds.append(logits.argmax(1).cpu().numpy())
                off += px.shape[0]
        acc = float((np.concatenate(preds) == sub_y).mean())
        results[mode] = acc
        print(f"__RG__:     {mode:16s} top-1 = {acc:.4f}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def cache_dir_for(root, checkpoint, mode):
    # Preprocessing is part of the cache identity: the same checkpoint under a
    # different resize is a different set of features.
    return os.path.join(root, os.path.basename(checkpoint.rstrip("/")) + "__" + mode)


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
    mode = cfg["extract"]["preprocess"]
    cdir = cache_dir_for(cfg["extract"]["cache_root"], ck, mode)
    os.makedirs(cdir, exist_ok=True)
    n = len(paths)
    topk = int(cfg["extract"]["topk"])
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE

    manifest_file = os.path.join(cdir, "cache_manifest.json")
    if os.path.exists(manifest_file):
        with open(manifest_file) as fh:
            prior = json.load(fh)
        if (prior.get("n_images") != n or prior.get("checkpoint") != ck
                or prior.get("topk") != topk or prior.get("preprocess") != mode):
            raise RuntimeError(
                f"Cache at {cdir} was built for n={prior.get('n_images')} "
                f"checkpoint={prior.get('checkpoint')} topk={prior.get('topk')} "
                f"preprocess={prior.get('preprocess')}, but this run has n={n} "
                f"checkpoint={ck} topk={topk} preprocess={mode}. Point extract.cache_root "
                "elsewhere rather than mixing pools.")

    done = [s for s in range(n_shards) if all(os.path.exists(_p(cdir, f, s)) for f in FIELDS)]
    print(f"__RG__: cache {cdir}")
    print(f"__RG__: {len(done)}/{n_shards} shards already complete", flush=True)

    if len(done) < n_shards:
        from transformers import AutoImageProcessor, SwinForImageClassification

        proc = AutoImageProcessor.from_pretrained(ck)
        size = processor_size(proc)
        tf = build_preprocess(mode, proc, size)
        # from_pretrained is used for the backbone only; the head comes from
        # load_head, since an ArcFace checkpoint has no classifier.* to load.
        model = SwinForImageClassification.from_pretrained(ck).to(device).eval()
        W_cpu, b_cpu, head = load_head(ck, cfg)
        W = W_cpu.to(device)
        b = b_cpu.to(device) if b_cpu is not None else None
        scale = float(cfg["model"].get("arcface_scale", 30.0))
        # ArcFace scores with the normalised row, so that is the direction whose
        # per-position contributions are meaningful.
        W_spat = (torch.nn.functional.normalize(W, dim=1) if head == "arcface" else W).float()
        n_classes = W.shape[0]
        autocast = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"__RG__: model loaded, classes={n_classes:,} feat_dim={W.shape[1]} "
              f"image_size={size} preprocess={mode} head={head}"
              + (f" scale={scale}" if head == "arcface" else ""), flush=True)

        t0, seen = time.time(), 0
        for s in range(n_shards):
            if s in done:
                continue
            lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, n)
            loader = DataLoader(ImageList(paths[lo:hi], tf, size),
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

                    h = h.float()
                    logits = head_logits(pooled.float(), W, b, head, scale)

                    tl = logits.gather(1, y[:, None]).squeeze(1)
                    # Rank of the true class: how many classes outscore it.
                    rank = (logits > tl[:, None]).sum(dim=1)
                    tk = logits.topk(topk, dim=1)
                    pred = tk.indices[:, 0]
                    logp = torch.log_softmax(logits, dim=1)
                    ent = -(logp.exp() * logp).sum(dim=1)

                    # Per-position logit contributions for the true and the
                    # predicted class: (B, 49) = h . w_class
                    #
                    # Still exact under ArcFace. Its logit is
                    #   s * (pooled . w_hat) / ||pooled||,  pooled = mean_p h_p
                    # so the sum over positions decomposes the same way and the
                    # leading s/||pooled|| is one scalar per image -- it cancels
                    # when spatial.py converts these to per-position shares.
                    buf["spat_true"].append(
                        torch.einsum("bpc,bc->bp", h, W_spat[y]).cpu().numpy().astype(np.float16))
                    buf["spat_pred"].append(
                        torch.einsum("bpc,bc->bp", h, W_spat[pred]).cpu().numpy().astype(np.float16))

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
                       "preprocess": mode, "n_classes": int(n_classes)}, fh)

    cache = {f: np.concatenate([np.load(_p(cdir, f, s)) for s in range(n_shards)])
             for f in FIELDS}
    print(f"__RG__: cache loaded, feat={cache['feat'].shape} "
          f"valid={cache['valid'].sum():,}/{n:,} "
          f"({1 - cache['valid'].mean():.4%} unreadable)", flush=True)
    return cache
