import subprocess, yaml
RUNS = [  # name, branch, config file (on that branch)
 ("arcface",            "orx/arcface-at-384px-matched-epoch-comparison-agains",   "swin_base_384_arcface_ablation.yml"),
 ("subcenter_arcface",  "orx/sub-center-arcface-k-3-at-384px-real-scale-pilot",  "swin_base_384_subcenter_arcface_pilot.yml"),
 ("hace",               "orx/hace-at-384px-real-scale-pilot-2-gpu-50-epochs",    "swin_base_384_hace_pilot.yml"),
 ("dbm",                "orx/dbm-at-384px-real-scale-pilot-2-gpu-50-epochs",     "swin_base_384_dbm_pilot.yml"),
 ("x2softmax",          "orx/x2-softmax-at-384px-real-scale-pilot-2-gpu-50-ep",  "swin_base_384_x2softmax_pilot.yml"),
 ("kappaface",          "orx/kappaface-at-384px-real-scale-pilot-1-gpu-50-epo",  "swin_base_384_kappaface_pilot.yml"),
 ("arcface_ecc",        "orx/arcface-ecc-at-384px-real-scale-pilot-1-gpu-50-e",  "swin_base_384_arcface_ecc_pilot.yml"),
 ("arcface_tara",       "orx/arcface-tara-at-384px-real-scale-pilot-2-gpu-50",   "swin_base_384_arcface_tara_pilot.yml"),
]
EVAL_ROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/PHASE3_EVAL"
for name, branch, cfg in RUNS:
    text = subprocess.check_output(["git", "show", f"{branch}:finetuning/SWIN/configs_advanced/{cfg}"], text=True)
    c = yaml.safe_load(text)
    t = c["training"]
    src = t["output_dir"]
    t.update(do_train=False, do_eval=True, output_dir=f"{EVAL_ROOT}/{name}", logging_dir=f"{EVAL_ROOT}/{name}",
             report_to="none", per_device_eval_batch_size=64, resume_from_checkpoint=None, overwrite_output_dir=False)
    c.setdefault("wandb", {})["enabled"] = False
    c["evaluation"] = {"init_weights": f"{src}/model.safetensors",
                       "predictions_path": f"{EVAL_ROOT}/{name}/predictions.npz"}
    head = (f"# GENERATED for Phase 3 eval-only re-evaluation -- do not train with this.\n"
            f"# Source: {branch}:finetuning/SWIN/configs_advanced/{cfg}\n"
            f"# Changes: do_train false, W&B off, output_dir -> PHASE3_EVAL/{name},\n"
            f"# evaluation.init_weights = that run's final model.safetensors, predictions saved as .npz.\n")
    open(f"finetuning/SWIN/configs_advanced/phase3_eval/{name}.yml", "w").write(head + yaml.safe_dump(c, sort_keys=False))
    print(name, "<-", src)
