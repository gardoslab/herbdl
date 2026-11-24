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


# --------------------------------------------------------------------------
# Checkpoint discovery.
#
# The fine-tuned SWIN checkpoints live on SCC under a tree this session cannot
# list interactively (SCC SSH requires 2FA). So the run inventories the tree
# itself and prints it before doing anything expensive: one run then both tells
# us what exists and uses it, instead of guessing a path and failing 20 minutes
# into feature extraction.
# --------------------------------------------------------------------------

def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _ckpt_meta(d):
    cfg = _read_json(os.path.join(d, "config.json"))
    if not cfg:
        return None
    n_lab = cfg.get("num_labels")
    if n_lab is None and isinstance(cfg.get("id2label"), dict):
        n_lab = len(cfg["id2label"])
    pre = _read_json(os.path.join(d, "preprocessor_config.json"))
    size = pre.get("size")
    if isinstance(size, dict):
        size = size.get("height") or size.get("shortest_edge")
    has_weights = any(os.path.exists(os.path.join(d, f)) for f in
                      ("model.safetensors", "pytorch_model.bin",
                       "model.safetensors.index.json", "pytorch_model.bin.index.json"))
    return {"path": d, "num_labels": n_lab, "image_size": size,
            "arch": (cfg.get("architectures") or [None])[0],
            "base": cfg.get("_name_or_path"), "has_weights": has_weights}


def inventory_checkpoints(roots, max_depth=3):
    """Walk checkpoint trees and print every directory that looks loadable."""
    found = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"__LCS__: inventory root missing: {root}")
            continue
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.count("/") - base_depth > max_depth:
                dirnames[:] = []
                continue
            m = _ckpt_meta(dirpath)
            if m:
                found.append(m)
                dirnames[:] = []  # don't descend into a checkpoint
    print("__LCS__ checkpoint inventory --------------------------------")
    if not found:
        print("  (nothing with a config.json under: %s)" % ", ".join(roots))
    for m in sorted(found, key=lambda x: (-(x["num_labels"] or 0), x["path"])):
        print("  {:<6} labels={:<7} size={:<5} {:<28} {}".format(
            "OK" if m["has_weights"] else "NOWTS",
            m["num_labels"] if m["num_labels"] is not None else "?",
            m["image_size"] or "?", (m["arch"] or "?")[:28],
            m["path"]))
    print("__LCS__ checkpoint inventory end ----------------------------", flush=True)
    return found


def resolve_backbone(spec, inventory, want_labels=None, tol=0.25, auto_by_labels=False):
    """Turn a config `backbone` value into something from_pretrained can load.

    Accepts, in order: an existing local path; a HuggingFace hub id; or a bare
    run name (e.g. SWIN_BASE_BASELINE) to be matched against the inventory. For
    a bare name that maps to a training output dir rather than a checkpoint, the
    highest-numbered `checkpoint-N` beneath it wins.
    """
    # A list means "try these in order" -- this session cannot list the SCC
    # checkpoint tree interactively, so the config names several plausible run
    # directories rather than betting the job on one guess.
    if isinstance(spec, (list, tuple)):
        errs = []
        for one in spec:
            try:
                return resolve_backbone(one, inventory, want_labels, tol, auto_by_labels=False)
            except FileNotFoundError as e:
                errs.append(str(e).split("\n")[0])
        print("__LCS__: none of the named candidates resolved:")
        for e in errs:
            print("           " + e)
        if auto_by_labels and want_labels:
            near = [m for m in inventory if m["has_weights"] and m["num_labels"]
                    and abs(m["num_labels"] - want_labels) <= tol * want_labels]
            if near:
                pick = sorted(near, key=lambda m: (m["num_labels"], m["path"]))[-1]
                print("__LCS__: AUTO-SELECTED by label count: {} (num_labels={}). "
                      "No configured name matched; this was chosen only because it sits "
                      "within {:.0%} of the requested {:,}-label space."
                      .format(pick["path"], pick["num_labels"], tol, want_labels))
                return pick["path"]
        raise FileNotFoundError(
            "No configured backbone candidate resolved and no inventoried checkpoint "
            "is within {:.0%} of {} labels. See the inventory above.".format(
                tol, want_labels))

    if os.path.isdir(spec):
        return spec
    if "/" in spec and not spec.startswith("/"):
        return spec  # hub id such as microsoft/swin-base-...

    parts_match = lambda m: (os.path.basename(m["path"].rstrip("/")) == spec
                             or spec in m["path"].split("/")
                             or any(spec.lower() == q.lower() for q in m["path"].split("/")))
    cands = [m for m in inventory if parts_match(m)]
    cands = [m for m in cands if m["has_weights"]]
    # want_labels is a HARD constraint, not a preference. "Keep it at the 15k
    # label space" has to mean a 50-class checkpoint is refused rather than
    # quietly accepted because its directory name happened to match.
    if want_labels and cands:
        near = [m for m in cands if m["num_labels"]
                and abs(m["num_labels"] - want_labels) <= tol * want_labels]
        if not near:
            raise FileNotFoundError(
                "Backbone '{}' resolved to {} but its label space is {}, outside "
                "{:.0%} of the required {:,}. Refusing rather than silently "
                "swapping in a different task.".format(
                    spec, cands[0]["path"], cands[0]["num_labels"], tol, want_labels))
        cands = near

    if not cands:
        names = sorted({p for m in inventory for p in m["path"].split("/")
                        if p.startswith("SWIN") or p.startswith("swin")})
        raise FileNotFoundError(
            "Could not resolve backbone '{}'. It is not a directory, not a hub id, "
            "and no inventoried checkpoint matches it.\n"
            "  SWIN-ish names seen in the inventory: {}\n"
            "  Set features.backbone to one of the inventory paths printed above."
            .format(spec, ", ".join(names) or "(none)"))

    def ckpt_num(m):
        b = os.path.basename(m["path"].rstrip("/"))
        return int(b.split("-")[-1]) if b.startswith("checkpoint-") and b.split("-")[-1].isdigit() else -1

    best = sorted(cands, key=lambda m: (ckpt_num(m), m["path"]))[-1]
    print(f"__LCS__: resolved backbone '{spec}' -> {best['path']} "
          f"(num_labels={best['num_labels']}, image_size={best['image_size']})")
    return best["path"]
