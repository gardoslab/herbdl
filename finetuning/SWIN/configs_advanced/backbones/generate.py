"""Part 4 training configs: llindon's committed CE config per backbone + a registry loss.
Run from anywhere:  python finetuning/SWIN/configs_advanced/backbones/generate.py
Hyperparameters are his (lr, epochs, warmup, weight decay, schedule). The effective batch size is
kept on 2 GPUs: ConvNeXt 128x1x2=256, DINOv2 64x1x2=128. Cluster outputs go to output/BACKBONES/<name>."""
import os, subprocess, yaml

TOP = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/BACKBONES"
BACKBONES = {
    "convnext": ("finetuning/ConvNeXt/configs/convnext_base_224_unfrozen.yml", dict(per_device_train_batch_size=128, gradient_accumulation_steps=1)),
    "dinov2": ("finetuning/DinoV2/configs_advanced/dinov2_base_baseline.yml", dict(per_device_train_batch_size=64, gradient_accumulation_steps=1)),
    # CoAtNet-3 (timm, 224px): 32/GPU x 2 GPUs = his effective batch 64
    "coatnet": ("finetuning/CoAtNet/configs_advanced/coatnet_base_baseline.yml", dict(per_device_train_batch_size=32, gradient_accumulation_steps=1)),
    # EfficientNet-B6 (528px): 16/GPU x 8 accumulation x 2 GPUs = his effective batch 256
    "efficientnet": ("finetuning/EfficientNet_B6/configs_advanced/efficientnet_b6_enhanced_workers4.yml",
                     dict(per_device_train_batch_size=16, gradient_accumulation_steps=8, per_device_eval_batch_size=16)),
}
LOSSES = {
    "ce": dict(loss_registry=dict(enabled=True, head="softmax", head_kwargs={}, aux=[]), long_tail=dict(logit_adjustment=False)),
    "ce_la": dict(loss_registry=dict(enabled=True, head="softmax", head_kwargs={}, aux=[]), long_tail=dict(logit_adjustment=True, tau=1.0)),
    "arcface": dict(loss_registry=dict(enabled=True, head="arcface", head_kwargs=dict(s=30.0, m=0.3), aux=[], margin_warmup_epochs=5),
                    long_tail=dict(logit_adjustment=False)),
    "dbm": dict(loss_registry=dict(enabled=True, head="dbm", head_kwargs=dict(s=30.0), aux=[]), long_tail=dict(logit_adjustment=False)),
}
for bb, (src, batch) in BACKBONES.items():
    for loss, extra in LOSSES.items():
        name = f"{bb}_{loss}"
        c = yaml.safe_load(open(f"{TOP}/{src}"))
        t = c["training"]
        t.update(batch)
        t.update(output_dir=f"{OUT}/{name}", logging_dir=f"{OUT}/{name}", do_train=True, do_eval=True,
                 save_total_limit=3, resume_from_checkpoint="newest", overwrite_output_dir=False,
                 dataloader_num_workers=16)
        t["per_device_eval_batch_size"] = batch.get("per_device_eval_batch_size", 64)
        c["data"]["max_eval_samples"] = None
        c.setdefault("custom", {}).update(run_group=f"PART4_{bb.upper()}", run_name=f"PART4_{name.upper()}", run_id=f"part4_{name}",
                                       run_notes=f"Part 4: {bb} with the {loss} loss, llindon recipe, Part 1 data.")
        c["wandb"] = dict(enabled=True, entity="gardoslab", project="herbdl", resume="allow")
        for k in ("arcface", "multi_task", "hace", "ema"):
            c[k] = dict(enabled=False)
        c.update(extra)
        head = f"# Part 4: {bb} + {loss}. Base recipe: {src} (llindon). See Loss Function Ablations.md.\n"
        open(f"{HERE}/{name}.yml", "w").write(head + yaml.safe_dump(c, sort_keys=False))
        print(name)
