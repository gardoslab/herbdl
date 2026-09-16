"""One-off frozen-backbone feature cache over the whole image pool.

The N-sweep only makes sense if the *representation is held fixed* while the
label space varies -- otherwise "more classes" also means "a differently
trained encoder" and the two effects are inseparable. So we forward the pool
through a frozen backbone exactly once and every (N, seed, regime) probe reads
the same vectors.

The cache lives outside the run directory on purpose: it is the expensive
artifact and every later node in this line reuses it.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SHARD_SIZE = 50_000


class ImageList(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), 1
        except Exception:
            # A handful of corrupt scans should not kill a multi-hour job; they
            # are recorded as zero vectors and excluded from every sample.
            return torch.zeros(3, self.transform.target_size, self.transform.target_size), 0


class Preprocess:
    def __init__(self, size, mean, std):
        from torchvision import transforms
        self.target_size = size
        self.t = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.t(img)


def cache_dir_for(root, backbone, size):
    slug = backbone.replace("/", "__") + f"__{size}"
    return os.path.join(root, slug)


def _shard_path(cdir, i):
    return os.path.join(cdir, f"feat_{i:05d}.npy")


def build_cache(paths, cache_root, backbone, image_size, batch_size, num_workers, device):
    """Extract (or resume) features for `paths`. Returns (features, valid_mask)."""
    cdir = cache_dir_for(cache_root, backbone, image_size)
    os.makedirs(cdir, exist_ok=True)
    manifest_file = os.path.join(cdir, "cache_manifest.json")
    n = len(paths)
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE

    prior = None
    if os.path.exists(manifest_file):
        with open(manifest_file) as fh:
            prior = json.load(fh)
        if prior.get("n_images") != n or prior.get("backbone") != backbone:
            raise RuntimeError(
                f"Feature cache at {cdir} was built for {prior.get('n_images')} images / "
                f"{prior.get('backbone')}, but this run has {n} / {backbone}. "
                "Point `features.cache_root` somewhere new rather than mixing pools.")

    complete = [i for i in range(n_shards) if os.path.exists(_shard_path(cdir, i))]
    print(f"__LCS__: feature cache {cdir}")
    print(f"__LCS__: {len(complete)}/{n_shards} shards already present")

    if len(complete) < n_shards:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(backbone)
        mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(proc, "image_std", [0.229, 0.224, 0.225])
        tf = Preprocess(image_size, mean, std)

        model = AutoModel.from_pretrained(backbone).to(device).eval()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        valid_parts = []
        t0 = time.time()
        for s in range(n_shards):
            sp = _shard_path(cdir, s)
            vp = sp.replace("feat_", "valid_")
            if os.path.exists(sp):
                continue
            lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, n)
            loader = DataLoader(ImageList(paths[lo:hi], tf), batch_size=batch_size,
                                num_workers=num_workers, pin_memory=(device == "cuda"),
                                shuffle=False, drop_last=False)
            feats, valids = [], []
            with torch.no_grad():
                for px, ok in loader:
                    px = px.to(device, non_blocking=True)
                    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                        out = model(pixel_values=px)
                    emb = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                        else out.last_hidden_state.mean(dim=1)
                    feats.append(emb.float().cpu().numpy().astype(np.float16))
                    valids.append(ok.numpy().astype(bool))
            # atomic: write to .tmp then rename, so a killed job never leaves a
            # half-written shard that a resume would trust
            np.save(sp + ".tmp.npy", np.concatenate(feats))
            os.replace(sp + ".tmp.npy", sp)
            np.save(vp + ".tmp.npy", np.concatenate(valids))
            os.replace(vp + ".tmp.npy", vp)
            done = hi
            rate = done / max(1e-9, time.time() - t0)
            print(f"__LCS__: shard {s + 1}/{n_shards}  images {done:,}/{n:,}  "
                  f"{rate:.0f} img/s  eta {(n - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    feats = np.concatenate([np.load(_shard_path(cdir, i)) for i in range(n_shards)])
    valid = np.concatenate([np.load(_shard_path(cdir, i).replace("feat_", "valid_"))
                            for i in range(n_shards)])
    with open(manifest_file, "w") as fh:
        json.dump({"n_images": n, "backbone": backbone, "image_size": image_size,
                   "dim": int(feats.shape[1]), "n_valid": int(valid.sum())}, fh)
    print(f"__LCS__: features {feats.shape} dtype={feats.dtype}  "
          f"valid={valid.sum():,}/{n:,} ({1 - valid.mean():.4%} unreadable)")
    return feats, valid
