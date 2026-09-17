"""Emit one SWIN fine-tuning config per ladder rung.

Every config starts from the SAME checkpoint the 15k arm used
(finetuning/output/SWIN/SWIN_BASE_BASELINE) so the only thing varying across
jobs is the size of the label space. `ignore_mismatched_sizes: true` lets the
15,501-way classifier head be replaced by an N-way one while the backbone
weights carry over.

Budget note, learned the hard way from the probe sweep: fixing EPOCHS gives the
big rungs far more gradient steps than the small ones (more classes at 10
images each = a bigger dataset), and fixing STEPS does the reverse. Here we fix
max_steps so every rung gets identical optimisation compute, and record the
resulting epoch count per rung in a comment so the imbalance is visible rather
than hidden.
"""
import json, os, sys

MAN = sys.argv[1]
OUT = sys.argv[2]
CKPT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/SWIN_BASE_BASELINE"
OUTROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/LADDER"
MAX_STEPS = 20000
BATCH = 128
MIN_N = 1000   # below this, 10 images/class makes full fine-tuning pure overfitting

man = json.load(open(MAN))
os.makedirs(OUT, exist_ok=True)
made = []
for e in man["rungs"]:
    n = e["n_classes"]
    if n < MIN_N:
        print(f"skip N={n}: below {MIN_N}, full fine-tuning would just memorise")
        continue
    epochs = MAX_STEPS * BATCH / e["n_train_images"]
    name = f"ladder_n{n}"
    cfg = f"""# SWIN fine-tune, ladder rung N={n:,}
# Resumes from the SAME checkpoint as the 15k arm; only the label space changes.
# {MAX_STEPS:,} steps x batch {BATCH} = {MAX_STEPS*BATCH:,} samples seen
#   = {epochs:.1f} epochs over this rung's {e['n_train_images']:,} train images.
model:
  model_name_or_path: "{CKPT}"
  config_name: null
  cache_dir: null
  model_revision: "main"
  image_processor_name: null
  token: null
  trust_remote_code: false
  ignore_mismatched_sizes: true   # 15,501-way head -> {n:,}-way head

data:
  dataset_name: null
  dataset_config_name: null
  data_file: null
  data_dir: null
  train_file: "{e['train_file']}"
  validation_file: "{e['validation_file']}"
  image_column_name: "filename"
  label_column_name: "scientificNameEncoded"
  max_seq_length: 15
  max_train_samples: null
  max_eval_samples: null
  overwrite_cache: false
  preprocessing_num_workers: null
  train_val_split: null           # explicit val file; do not re-split

training:
  output_dir: "{OUTROOT}/{name}"
  logging_dir: "{OUTROOT}/{name}"
  do_train: true
  do_eval: true
  per_device_train_batch_size: {BATCH}
  per_device_eval_batch_size: 64
  learning_rate: 0.0002
  max_steps: {MAX_STEPS}
  warmup_steps: 0.05
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  lr_scheduler_type: "cosine"
  lr_scheduler_kwargs:
    eta_min: 0.000001
  logging_strategy: "steps"
  logging_steps: 200
  save_strategy: "steps"
  save_steps: 5000
  save_total_limit: 2
  eval_strategy: "steps"
  eval_steps: 5000
  report_to: "wandb"
  bf16: true
  dataloader_num_workers: 16
  remove_unused_columns: false
  overwrite_output_dir: false
  seed: 42

custom:
  lr_type: "cosine"
  frozen: false
  frozen_type: "none"
  run_group: "LabelLadder"
  run_name: "ladder_N{n}"
  run_id: "ladder_n{n}_from_swin_base_baseline"

wandb:
  entity: "gardoslab"
  project: "herbdl"
  resume: "allow"
"""
    path = os.path.join(OUT, name + ".yml")
    open(path, "w").write(cfg)
    made.append((n, path, e["n_train_images"], epochs))

print("\n{:>8} {:>12} {:>9}  {}".format("N", "train imgs", "epochs", "config"))
for n, p, ni, ep in made:
    print("{:>8,} {:>12,} {:>9.1f}  {}".format(n, ni, ep, p))
print(f"\n{len(made)} configs written to {OUT}")
