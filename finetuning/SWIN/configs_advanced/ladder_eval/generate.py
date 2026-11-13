"""Make ladder_eval/<rung>_<loss>.yml: eval-only copies of the ladder training configs that
load the finished model and save per-image predictions for the routing test.
Run from finetuning/SWIN/configs_advanced/:  python3 ladder_eval/generate.py <loss> [<loss> ...]"""
import re
import sys

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
for loss in sys.argv[1:] or ["dbm"]:
    for rung in ("n15k", "n32k", "n64k"):
        s = open(f"swin_base_384_ladder_{rung}_{loss}.yml").read()
        train_dir = re.search(r'(?m)^  output_dir: "([^"]+)"', s).group(1)
        name = f"{rung}_{loss}"
        sub = {  # pattern -> (replacement, expected number of matches)
            r'(?m)^  do_train: true': ("  do_train: false", 1),
            r'(?m)^  (output|logging)_dir: "[^"]+"': (rf'  \1_dir: "{OUT}/LADDER_EVAL/{name}"', 2),
            r'(?m)^  report_to: "wandb"': ('  report_to: "none"', 1),
            r'(?m)^  resume_from_checkpoint: newest\n': ("", 1),
        }
        for pat, (rep, want) in sub.items():
            s, n = re.subn(pat, rep, s)
            assert n == want, (name, pat, n)
        s = s.replace("wandb:\n  enabled: true", "wandb:\n  enabled: false")
        s = s.rstrip("\n") + (f"\n\nevaluation:\n  init_weights: {train_dir}/model.safetensors\n"
                              f"  predictions_path: {OUT}/LADDER_EVAL/{name}/predictions.npz\n")
        s = (f"# Eval-only re-score of the {loss} model trained on the {rung} rung (made by\n"
             f"# ladder_eval/generate.py from swin_base_384_ladder_{rung}_{loss}.yml).\n") + s
        open(f"ladder_eval/{name}.yml", "w").write(s)
        print(name, "<-", train_dir)
