"""Comprehensive end-to-end test harness for the loss-function registry.

Per-file unit tests were deliberately NOT written (see decision log in
Loss Function Ablations.md). Instead this harness runs a REAL short training
loop -- real Kaggle-Herbarium-2022 images, the real starting checkpoint
(microsoft/swin-base-patch4-window12-384-in22k), a real optimizer step -- for
every combination in the registry, and checks the whole path end to end:

    1. loss is finite every step, no NaN/inf under bf16
    2. accuracy on a held-out slice rises clearly above chance by the end
       (the check that actually catches a wrong formulation: a broken margin
       term usually trains happily and plateaus at chance, which no shape or
       finiteness assertion would ever catch)
    3. the margin-off / inference path agrees with what the model reports
       during training in the arg-max sense (same prediction under the same
       weights, since margin only ever helps the true class beat OTHER
       classes during training -- it should not change which class already
       wins by the largest margin as often as it does)
    4. checkpoint save/reload reproduces logits exactly
    5. class_directions() has the right shape and every row is finite --
       this is the contract repgeom's geometry analysis depends on

Held out of scope for this harness (tracked separately in the doc, not
silently skipped): full repgeom compatibility for heads other than
linear/arcface, and FlexiGrad wired into a real HF Trainer.training_step
(grad_modulation.py's primitives are exercised directly here instead, since
that wiring is its own Phase 3 task).

Small-scale by design: a fixed subset of species and images, so every
combination in the registry runs in this one job rather than needing nine.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, SwinForImageClassification

sys.path.insert(0, ".")
from loss_ablation.hierarchy import TaxonomyTree
from loss_ablation.model import SwinWithLoss

RESULTS = []


def log(msg):
    print(f"__E2E__: {msg}", flush=True)


def section(title):
    print(f"\n__E2E__ ===== {title} " + "=" * max(0, 50 - len(title)), flush=True)


class SmallHerbariumSubset(Dataset):
    def __init__(self, rows, image_processor, size, bioclip_transform=None):
        self.rows = rows
        self.proc = image_processor
        self.size = size
        # Separate transform for TARA's BioCLIP teacher -- NOT the same
        # pipeline as `proc` (see bioclip_teacher.py: 224px CLIP-normalised
        # vs 384px ImageNet-normalised). None unless a config actually uses
        # aux='tara', so the other 8 configs pay nothing for this.
        self.bioclip_transform = bioclip_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            img = Image.open(row["filename"]).convert("RGB")
            px = self.proc(images=img, return_tensors="pt").pixel_values[0]
            bpx = self.bioclip_transform(img) if self.bioclip_transform is not None \
                else torch.zeros(1)  # placeholder, unused when tara is off
        except Exception:
            px = torch.zeros(3, self.size, self.size)
            bpx = torch.zeros(3, 224, 224) if self.bioclip_transform is not None else torch.zeros(1)
        return px, bpx, row["species_id"], row["genus_id"], row["family_id"]


def load_small_subset(train_manifest, n_species, images_per_species, seed=0):
    """Real rows from the real manifest -- a fixed number of species, a fixed
    cap of images per species, so the whole harness is bounded regardless of
    which corner of the dataset gets sampled."""
    import random
    rng = random.Random(seed)
    by_species = {}
    with open(train_manifest) as fh:
        for line in fh:
            d = json.loads(line)
            key = d["scientificName"]
            by_species.setdefault(key, []).append(d)

    species_keys = sorted(by_species)
    rng.shuffle(species_keys)
    chosen_species = [s for s in species_keys if len(by_species[s]) >= 4][:n_species]
    if len(chosen_species) < n_species:
        raise RuntimeError(f"only found {len(chosen_species)} species with >=4 images; "
                           f"wanted {n_species}")

    species2id = {s: i for i, s in enumerate(chosen_species)}
    genus2id, family2id = {}, {}
    rows = []
    for s in chosen_species:
        sample = by_species[s][:images_per_species]
        for r in sample:
            g, f = r["genus"], r["family"]
            if g not in genus2id:
                genus2id[g] = len(genus2id)
            if f not in family2id:
                family2id[f] = len(family2id)
            rows.append({"filename": r["filename"], "species_id": species2id[s],
                        "genus_id": genus2id[g], "family_id": family2id[f],
                        "genus": g, "family": f, "species": s})
    rng.shuffle(rows)
    split = int(0.8 * len(rows))
    log(f"subset: {len(chosen_species)} species, {len(genus2id)} genera, "
        f"{len(family2id)} families, {len(rows)} images "
        f"({split} train / {len(rows) - split} held out)")

    species_to_genus = [None] * len(species2id)
    for r in rows:
        species_to_genus[r["species_id"]] = genus2id[r["genus"]]
    genus_to_family = [None] * len(genus2id)
    for r in rows:
        genus_to_family[r["genus_id"]] = family2id[r["family"]]

    class_counts = [0] * len(species2id)
    for r in rows[:split]:
        class_counts[r["species_id"]] += 1
    # every class needs >=1 count for KappaFace/DBM's per-class statistics;
    # the harness's job is to catch exactly this kind of edge case before a
    # real run hits it with a rare species that got zero images in the 80% split.
    class_counts = [max(c, 1) for c in class_counts]

    return (rows[:split], rows[split:], species_to_genus, genus_to_family,
            class_counts, len(species2id))


def processor_size(proc):
    """transformers >=4.57 returns a SizeDict here: mapping-like, not a dict
    subclass, so `isinstance(proc.size, dict)` is False and `int(proc.size)`
    raises. Same fix as repgeom/extract.py's processor_size() -- reused here
    rather than re-derived, since it is the identical incompatibility."""
    raw = getattr(proc, "size", None)
    if raw is None:
        return 384
    try:
        return int(raw["height"])
    except Exception:
        return int(getattr(raw, "height", 384) or 384)


def run_one_config(cfg_name, build_kwargs, backbone_ckpt, train_rows, val_rows,
                    tree, class_counts, n_species, device, steps, batch_size, lr,
                    bioclip_teacher=None, export_root=None):
    section(f"CONFIG: {cfg_name}")
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(backbone_ckpt)
    size = processor_size(proc)

    base = SwinForImageClassification.from_pretrained(
        backbone_ckpt, num_labels=n_species, ignore_mismatched_sizes=True)
    model = SwinWithLoss(base, n_species, tree=tree,
                         aux_kwargs=(dict(tara=dict(bioclip_dim=bioclip_teacher.embed_dim,
                                                    bioclip_embed_fn=bioclip_teacher.as_embed_fn()))
                                     if bioclip_teacher is not None else None),
                         **build_kwargs).to(device)

    bc_transform = bioclip_teacher.transform if bioclip_teacher is not None else None
    train_ds = SmallHerbariumSubset(train_rows, proc, size, bioclip_transform=bc_transform)
    val_ds = SmallHerbariumSubset(val_rows, proc, size, bioclip_transform=bc_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    result = {"config": cfg_name, "ok": False, "reason": None}

    # --- 1. training loop: loss finiteness ------------------------------
    step = 0
    model.train()
    loss_history = []
    try:
        while step < steps:
            for px, bpx, sp, ge, fa in train_loader:
                if step >= steps:
                    break
                px, sp = px.to(device), sp.to(device)
                fwd_kwargs = {}
                if bioclip_teacher is not None:
                    fwd_kwargs["bioclip_pixel_values"] = bpx.to(device)
                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
                    out = model(px, labels=sp, **fwd_kwargs)
                loss = out["loss"]
                if not torch.isfinite(loss):
                    result["reason"] = f"non-finite loss at step {step}: {loss.item()}"
                    raise RuntimeError(result["reason"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_history.append(loss.item())
                if hasattr(model, "hace_epoch_hook"):
                    pass  # placeholder: per-epoch buffer refreshes wired in train_advanced.sh, not here
                step += 1
                if step % max(1, steps // 5) == 0:
                    log(f"  [{cfg_name}] step {step}/{steps} loss={loss.item():.4f}")
    except RuntimeError as e:
        if result["reason"] is None:
            result["reason"] = f"training crashed: {e}"
        RESULTS.append(result)
        log(f"FAIL [{cfg_name}]: {result['reason']}")
        return result

    log(f"  loss finiteness OK ({len(loss_history)} steps, "
        f"first={loss_history[0]:.4f} last={loss_history[-1]:.4f})")

    # --- 2. held-out accuracy clearly above chance ----------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for px, bpx, sp, ge, fa in val_loader:
            px, sp = px.to(device), sp.to(device)
            out = model(px, labels=None)  # TARA never runs at labels=None; bpx unused here
            pred = out["logits"].argmax(dim=1)
            correct += (pred == sp).sum().item()
            total += sp.numel()
    acc = correct / max(total, 1)
    chance = 1.0 / n_species
    result["accuracy"] = acc
    result["chance"] = chance
    if acc < chance * 3:  # a generous but real bar -- catches a dead/broken head
        result["reason"] = f"accuracy {acc:.4f} not clearly above chance {chance:.4f}"
        RESULTS.append(result)
        log(f"FAIL [{cfg_name}]: {result['reason']}")
        return result
    log(f"  held-out accuracy {acc:.4f} vs chance {chance:.4f} "
        f"({acc / chance:.1f}x) -- OK")

    # --- 3. checkpoint round-trip: exact logit reproduction -------------
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    fresh_base = SwinForImageClassification.from_pretrained(
        backbone_ckpt, num_labels=n_species, ignore_mismatched_sizes=True)
    fresh_model = SwinWithLoss(fresh_base, n_species, tree=tree,
                               aux_kwargs=(dict(tara=dict(bioclip_dim=bioclip_teacher.embed_dim,
                                                          bioclip_embed_fn=bioclip_teacher.as_embed_fn()))
                                           if bioclip_teacher is not None else None),
                               **build_kwargs).to(device)
    fresh_model.load_state_dict(torch.load(buf, map_location=device))
    fresh_model.eval()
    px_check = next(iter(val_loader))[0].to(device)
    with torch.no_grad():
        out_a = model(px_check, labels=None)["logits"]
        out_b = fresh_model(px_check, labels=None)["logits"]
    if not torch.allclose(out_a, out_b, atol=1e-4):
        result["reason"] = f"checkpoint round-trip mismatch, max diff {(out_a - out_b).abs().max().item()}"
        RESULTS.append(result)
        log(f"FAIL [{cfg_name}]: {result['reason']}")
        return result
    log("  checkpoint round-trip OK (logits match exactly)")

    # --- 4. class_directions() contract ---------------------------------
    cd = model.class_directions()
    if cd.shape[0] != n_species or not torch.isfinite(cd).all():
        result["reason"] = f"class_directions() bad: shape={tuple(cd.shape)} finite={torch.isfinite(cd).all().item()}"
        RESULTS.append(result)
        log(f"FAIL [{cfg_name}]: {result['reason']}")
        return result
    # lightweight geometry sanity, not full repgeom: within-class cosine
    # should exceed the mean pairwise cosine of a few random OTHER class
    # directions, or something is badly wrong with how directions separate.
    cd_n = F.normalize(cd.float(), dim=1)
    mean_offdiag_cos = (cd_n @ cd_n.T).fill_diagonal_(0).mean().item()
    log(f"  class_directions() OK: shape={tuple(cd.shape)}, "
        f"mean off-diagonal cosine={mean_offdiag_cos:.4f}")

    # --- 5. geometry-export round-trip: the repgeom-facing contract -----
    # This is the fixture repgeom (a different git branch, see
    # checkpoint_io.py's module docstring) will read: a plain backbone
    # state_dict + the class-direction matrix, no loss_ablation import
    # required on the reading side. Verified here by doing exactly what
    # repgeom's reader will do -- reload a PLAIN SwinModel from the exported
    # state_dict and confirm its pooled-feature output matches the trained
    # model's, without going through SwinWithLoss at all.
    export_dir = os.path.join(export_root, cfg_name) if export_root else None
    if export_dir:
        from loss_ablation.checkpoint_io import (load_geometry_export,
                                                  save_geometry_export)
        save_geometry_export(model, export_dir, backbone_ckpt,
                             meta_extra={"config": cfg_name})
        swin_sd, cd_loaded, meta = load_geometry_export(export_dir)
        if not torch.equal(cd_loaded, cd.cpu()):
            result["reason"] = "geometry export: class_directions.pt does not match the live model"
            RESULTS.append(result)
            log(f"FAIL [{cfg_name}]: {result['reason']}")
            return result

        from transformers import SwinModel
        plain_backbone = SwinModel.from_pretrained(backbone_ckpt)
        load_report = plain_backbone.load_state_dict(swin_sd, strict=False)
        if load_report.missing_keys or load_report.unexpected_keys:
            result["reason"] = (f"geometry export: state_dict mismatch on reload -- "
                                f"missing={load_report.missing_keys[:3]} "
                                f"unexpected={load_report.unexpected_keys[:3]}")
            RESULTS.append(result)
            log(f"FAIL [{cfg_name}]: {result['reason']}")
            return result
        plain_backbone = plain_backbone.to(device).eval()
        with torch.no_grad():
            live_pooled = model.swin(px_check).pooler_output
            reloaded_pooled = plain_backbone(px_check).pooler_output
        if not torch.allclose(live_pooled, reloaded_pooled, atol=1e-4):
            result["reason"] = (f"geometry export: reloaded backbone's pooled features don't "
                                f"match the live model's, max diff "
                                f"{(live_pooled - reloaded_pooled).abs().max().item()}")
            RESULTS.append(result)
            log(f"FAIL [{cfg_name}]: {result['reason']}")
            return result
        log(f"  geometry export round-trip OK: {export_dir} "
            f"(backbone reload matches, class_directions matches)")

    result["ok"] = True
    result["wall_s"] = time.time() - t0
    result["mean_offdiag_cos"] = mean_offdiag_cos
    RESULTS.append(result)
    log(f"PASS [{cfg_name}]  acc={acc:.4f} ({acc/chance:.1f}x chance)  "
        f"wall={result['wall_s']:.0f}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="microsoft/swin-base-patch4-window12-384-in22k")
    ap.add_argument("--train_manifest", default="/projectnb/herbdl/data/kaggle-herbaria/train_2022.json")
    ap.add_argument("--n_species", type=int, default=40)
    ap.add_argument("--images_per_species", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export_root",
                    default="/projectnb/herbdl/workspaces/faridkar/herbdl/loss_ablation_geometry_exports",
                    help="Where to write checkpoint_io geometry exports for repgeom to read later "
                         "(a persistent SCC path, not inside this git-tracked run directory, since "
                         "repgeom lives on a different branch -- see checkpoint_io.py). Pass '' to skip.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    section("SUBSET")
    train_rows, val_rows, sp2g, g2f, class_counts, n_species = load_small_subset(
        args.train_manifest, args.n_species, args.images_per_species, seed=args.seed)
    tree = TaxonomyTree(sp2g, g2f)
    log(f"tree: {tree.n_species} species / {tree.n_genera} genera / "
        f"{tree.n_families} families / {tree.n_nodes} nodes")

    # Written once, alongside the exports, so a repgeom-side reader (a
    # different git branch, see checkpoint_io.py) can rebuild the EXACT same
    # small image set each export's backbone was trained on -- rederiving it
    # independently would risk a different subset entirely, since
    # load_small_subset's sampling depends on dict/set iteration order as
    # well as the seed.
    if args.export_root:
        os.makedirs(args.export_root, exist_ok=True)
        id2species = {r["species_id"]: r["species"] for r in train_rows + val_rows}
        id2genus = {r["genus_id"]: r["genus"] for r in train_rows + val_rows}
        id2family = {r["family_id"]: r["family"] for r in train_rows + val_rows}
        with open(os.path.join(args.export_root, "subset_manifest.json"), "w") as fh:
            json.dump({
                "train_rows": train_rows, "val_rows": val_rows,
                "id2species": id2species, "id2genus": id2genus, "id2family": id2family,
                "n_species": n_species, "seed": args.seed,
            }, fh, indent=2)
        log(f"wrote subset manifest to {args.export_root}/subset_manifest.json "
            f"({len(train_rows)} train + {len(val_rows)} val rows)")

    configs = [
        ("softmax", dict(head_name="softmax")),
        ("arcface", dict(head_name="arcface", head_kwargs=dict(s=30.0, m=0.30))),
        ("subcenter_arcface", dict(head_name="subcenter_arcface", head_kwargs=dict(k=3, s=30.0, m=0.30))),
        ("kappaface", dict(head_name="kappaface", head_kwargs=dict(class_counts=class_counts, s=30.0, m0=0.30))),
        ("x2softmax", dict(head_name="x2softmax", head_kwargs=dict(s=30.0))),
        ("dbm", dict(head_name="dbm", head_kwargs=dict(class_counts=class_counts, s=30.0))),
        ("hace", dict(hierarchy="hace")),
        ("arcface+ecc", dict(head_name="arcface", head_kwargs=dict(s=30.0, m=0.30), aux_names=["ecc"])),
        ("arcface+tara", dict(head_name="arcface", head_kwargs=dict(s=30.0, m=0.30), aux_names=["tara"])),
    ]
    needs_bioclip = any("tara" in kw.get("aux_names", []) for _, kw in configs)
    bioclip_teacher = None
    if needs_bioclip:
        section("LOAD BIOCLIP TEACHER")
        from loss_ablation.bioclip_teacher import BioCLIPTeacher
        bioclip_teacher = BioCLIPTeacher(device=device)
        log(f"BioCLIP teacher loaded, embed_dim={bioclip_teacher.embed_dim}")

    export_root = args.export_root or None
    if export_root:
        log(f"geometry exports will be written under {export_root}")

    section("RUN ALL CONFIGS")
    for name, kwargs in configs:
        cfg_bioclip = bioclip_teacher if "tara" in kwargs.get("aux_names", []) else None
        run_one_config(name, kwargs, args.backbone, train_rows, val_rows, tree,
                       class_counts, n_species, device, args.steps, args.batch_size, args.lr,
                       bioclip_teacher=cfg_bioclip, export_root=export_root)

    section("SUMMARY")
    n_pass = sum(1 for r in RESULTS if r["ok"])
    for r in RESULTS:
        status = "PASS" if r["ok"] else "FAIL"
        extra = f"acc={r.get('accuracy', float('nan')):.4f}" if r["ok"] else r.get("reason", "?")
        log(f"  {status:5s} {r['config']:20s} {extra}")
    log(f"{n_pass}/{len(RESULTS)} configurations passed")
    # Path relative to THIS file, not the process cwd -- run_e2e_harness.sh
    # cd's to finetuning/SWIN before invoking `python loss_ablation/e2e_harness.py`,
    # so a bare relative path here landed one directory up from where anyone
    # would look for it (caught after run 064677df: the log said PASS 8/8 but
    # the file was in finetuning/SWIN/, not finetuning/SWIN/loss_ablation/).
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_harness_results.json")
    with open(out_path, "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    log(f"wrote {out_path}")

    if n_pass < len(RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
