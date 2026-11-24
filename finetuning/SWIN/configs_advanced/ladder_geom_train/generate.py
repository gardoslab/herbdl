"""Make ladder_geom_train/<rung>_<loss>.yml: eval-only passes over a ladder rung's TRAINING images
that save pooled features and predictions (training-image geometry for the supplement).
Copies ladder_eval/<rung>_<loss>.yml and changes only: validation_file -> <rung>_train.json,
output/logging dir, predictions and features -> LADDER_GEOM_TRAIN/<rung>_<loss>, eval batch 64 -> 128.
Run from finetuning/SWIN/configs_advanced/:  python3 ladder_geom_train/generate.py"""
import re

OUT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN"
for loss in ("cosine_m0", "dbm", "kappaface", "arcface", "ce"):
    for rung in ("n15k", "n32k", "n64k"):
        name = f"{rung}_{loss}"
        s = open(f"ladder_eval/{name}.yml").read()
        dst = f"{OUT}/LADDER_GEOM_TRAIN/{name}"
        sub = {  # pattern -> (replacement, expected number of matches)
            rf'(?m)^  validation_file: "([^"]+)/{rung}_val\.json"': (rf'  validation_file: "\1/{rung}_train.json"', 1),
            r'(?m)^  (output|logging)_dir: "[^"]+"': (rf'  \1_dir: "{dst}"', 2),
            r'(?m)^  per_device_eval_batch_size: 64': ("  per_device_eval_batch_size: 128", 1),
            r'(?m)^  predictions_path: .*$': (f"  predictions_path: {dst}/predictions.npz", 1),
            r'(?m)^  features_path: .*\n': ("", None),
        }
        for pat, (rep, want) in sub.items():
            s, n = re.subn(pat, rep, s)
            assert want is None or n == want, (name, pat, n)
        s = s.rstrip("\n") + f"\n  features_path: {dst}/features.npy\n"
        s = f"# Training-image feature pass of {name} (ladder_geom_train/generate.py from ladder_eval/{name}.yml).\n" + s
        open(f"ladder_geom_train/{name}.yml", "w").write(s)
        print(name)
