"""Eval-only configs for the finished Part 4 training runs (generate.py's configs with
do_train off, W&B off, init_weights = the run's model.safetensors, per-image predictions saved).
Run from anywhere: python finetuning/SWIN/configs_advanced/backbones/generate_eval.py"""
import os, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output"
NAMES = [f"{b}_{l}" for b in ("convnext", "dinov2") for l in ("arcface", "dbm", "ce_la")]
os.makedirs(f"{HERE}/eval", exist_ok=True)
for name in NAMES:
    c = yaml.safe_load(open(f"{HERE}/{name}.yml"))
    t = c["training"]
    t.update(do_train=False, do_eval=True, report_to="none", resume_from_checkpoint=None,
             output_dir=f"{OUT}/SWIN/PHASE4_EVAL/{name}", logging_dir=f"{OUT}/SWIN/PHASE4_EVAL/{name}")
    c["wandb"]["enabled"] = False
    c["evaluation"] = {"init_weights": f"{OUT}/BACKBONES/{name}/model.safetensors",
                       "predictions_path": f"{OUT}/SWIN/PHASE4_EVAL/{name}/predictions.npz"}
    open(f"{HERE}/eval/{name}.yml", "w").write(f"# GENERATED eval-only re-score of Part 4 run {name}\n" + yaml.safe_dump(c, sort_keys=False))
    print(name)
