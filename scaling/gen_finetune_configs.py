"""Emit one SWIN fine-tuning config per ladder rung.

Every config starts from the SAME checkpoint the 15k arm used
(finetuning/output/SWIN/SWIN_BASE_BASELINE) so the only thing varying across
jobs is the size of the label space. `ignore_mismatched_sizes: true` lets the
15,501-way classifier head be replaced by an N-way one while the backbone
weights carry over.

Budget note, learned the hard way from the probe sweep: fixing EPOCHS gives the
big rungs far more gradient steps than the small ones (more classes at 10
images each = a bigger dataset), and fixing STEPS does the reverse. We want
identical optimisation compute at every rung.

SWIN_finetuning_advanced.py builds TrainingArguments from a fixed key list in
which `num_train_epochs` is mandatory and a step cap is never read. An earlier
version of this generator set a step cap instead, and all five jobs died with
KeyError: 'num_train_epochs' before a single batch trained. So the equal-step
budget is expressed THROUGH the epoch knob: num_train_epochs is computed per
rung as TARGET_STEPS * BATCH / n_train_images, landing every rung on the same
~TARGET_STEPS optimiser steps while using the key the trainer accepts.

Keys the trainer does not read are omitted rather than left in to look
meaningful (step caps, save/logging step intervals, overwrite_output_dir).
"""
import json, os, sys

MAN = sys.argv[1]
OUT = sys.argv[2]
CKPT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/SWIN_BASE_BASELINE"
OUTROOT = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/LADDER"
TARGET_STEPS = 20000   # identical optimiser steps at every rung
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
    epochs = TARGET_STEPS * BATCH / e["n_train_images"]
    name = f"ladder_n{n}"
    cfg = f"""# SWIN fine-tune, ladder rung N={n:,}
# Resumes from the SAME checkpoint as the 15k arm; only the label space changes.
# {TARGET_STEPS:,} steps x batch {BATCH} = {TARGET_STEPS*BATCH:,} samples seen
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
  num_train_epochs: {epochs:.4f}   # = {TARGET_STEPS:,} steps at batch {BATCH}
  warmup_steps: 0.05
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  lr_scheduler_type: "cosine"
  lr_scheduler_kwargs:
    eta_min: 0.000001
  logging_strategy: "steps"
  save_strategy: "steps"
  save_total_limit: 2
  eval_strategy: "steps"
  eval_steps: 5000
  report_to: "wandb"
  bf16: true
  dataloader_num_workers: 16
  remove_unused_columns: false
  seed: 42

custom:
  lr_type: "cosine"
  frozen: false
  frozen_type: "none"
  run_group: "LabelLadder"
  run_name: "ladder_N{n}"
  run_id: "ladder_n{n}_from_swin_base_baseline"
  run_notes: "Label-cardinality ladder rung N={n}. Resumed from SWIN_BASE_BASELINE; only the label space differs across rungs. {TARGET_STEPS:,} optimiser steps at batch {BATCH}."

# Only consulted when advanced augmentation is enabled (it is not), but the
# trainer subscripts this section directly, so omitting it risks a KeyError.
randaugment:
  num_ops: 2
  magnitude: 9

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

# --- self-check -----------------------------------------------------------
# Five jobs were once lost because this generator emitted a key the trainer
# never reads (max_steps) and omitted ones it subscripts directly
# (num_train_epochs, run_notes). Rather than trust a hand-kept list, derive the
# requirement from the trainer source and fail loudly before anything is queued.
TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "finetuning", "SWIN", "SWIN_finetuning_advanced.py")
TRAINER = os.environ.get("LADDER_TRAINER_SRC", TRAINER)
if os.path.exists(TRAINER):
    import re, collections, yaml
    src = open(TRAINER).read()
    hard = collections.defaultdict(set)
    for a, b in re.findall(r"""config\[['\"](\w+)['\"]\]\[['\"](\w+)['\"]\]""", src):
        hard[a].add(b)
    soft = collections.defaultdict(set)
    for a, b in re.findall(r"""config\[['\"](\w+)['\"]\]\.get\(['\"](\w+)['\"]""", src):
        soft[a].add(b)
    problems = []
    for _, path, _, _ in made:
        cfg = yaml.safe_load(open(path))
        for sec in hard:
            for key in sorted(hard[sec] - soft[sec]):
                if key not in (cfg.get(sec) or {}):
                    problems.append(f"{os.path.basename(path)}: missing {sec}.{key}")
    if problems:
        print("\nCONFIG VALIDATION FAILED -- these would KeyError at startup:")
        for q in problems:
            print("  " + q)
        sys.exit(1)
    print(f"\nvalidated against {TRAINER}: all required keys present")
else:
    print(f"\nWARNING: trainer source not found at {TRAINER}; skipped validation")

print(f"\n{len(made)} configs written to {OUT}")
