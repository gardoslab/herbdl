"""Eval-only configs that re-score llindon's finished plain-CE checkpoints on the full
val_2022 set and save per-image predictions. Run from the repo root:
    python finetuning/SWIN/configs_advanced/backbone_eval/generate.py
Each config is his committed training config with do_train off, W&B off, output_dir moved to
PHASE4_EVAL/<name>, and evaluation.init_weights pointing at his final model.safetensors."""
import os
import subprocess
import yaml

EVAL_ROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE4_EVAL_FULL"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = f"{HERE}/full"
os.makedirs(OUTDIR, exist_ok=True)
# name, committed config (repo-relative), his finished-model folder
RUNS = [
    ("convnext_ce", "finetuning/ConvNeXt/configs/convnext_base_224_unfrozen.yml"),
    ("dinov2_ce", "finetuning/DinoV2/configs_advanced/dinov2_base_baseline.yml"),
    ("coatnet_ce", "finetuning/CoAtNet/configs_advanced/coatnet_base_baseline.yml"),
    ("efficientnet_ce", "finetuning/EfficientNet_B6/configs_advanced/efficientnet_b6_enhanced_workers4.yml"),
]
for name, cfg in RUNS:
    c = yaml.safe_load(open(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip() + "/" + cfg))
    t = c["training"]
    src = t["output_dir"]
    t.update(do_train=False, do_eval=True, output_dir=f"{EVAL_ROOT}/{name}", logging_dir=f"{EVAL_ROOT}/{name}",
             report_to="none", per_device_eval_batch_size=32, resume_from_checkpoint=None, overwrite_output_dir=False)
    c["data"]["max_eval_samples"] = None  # his configs cap eval at 50,000 images; score the full set
    c.setdefault("wandb", {})["enabled"] = False
    c["evaluation"] = {"init_weights": f"{src}/model.safetensors",
                       "predictions_path": f"{EVAL_ROOT}/{name}/predictions.npz"}
    head = (f"# GENERATED eval-only re-score of llindon's finished CE run -- do not train with this.\n"
            f"# Source: {cfg}\n# Weights: {src}/model.safetensors\n")
    open(f"{OUTDIR}/{name}.yml", "w").write(head + yaml.safe_dump(c, sort_keys=False))
    print(name, "<-", src)
