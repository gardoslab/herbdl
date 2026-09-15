This is a project for finetuning models on Herbaria images. 

## SWIN training script

`finetuning/SWIN/SWIN_finetuning_advanced.py` is the only SWIN training script —
a single config-driven script (YAML in, HF `Trainer` underneath) that supports every
technique (multi-task family/genus/species heads, ArcFace, EMA, multi-crop TTA,
advanced augmentation, logit adjustment, curriculum, etc.) toggled via config sections.
The older `SWIN_finetuning.py` (simple baseline) and `SWIN_finetuning_arcface.py`
(ArcFace-only variant) have been removed — their functionality is a strict subset of
what this script already does.

`finetuning/SWIN/train_advanced.sh` is the sole `qsub` entry point for it (the older
`train.sh` was consolidated into it). Submit with `CONFIG_FILE` set via `qsub -v`:

```bash
qsub -l h_rt=24:00:00 -P herbdl -l gpus=1 -l gpu_c=8.0 -l gpu_memory=48G \
     -v CONFIG_FILE=configs_advanced/swin_baseline_augmented.yml \
     -m beas -M faridkar@bu.edu -N SWIN_RUN train_advanced.sh
```

Set `NPROC_PER_NODE=<n>` (also via `qsub -v`) for multi-GPU DDP via `torchrun`.
`submit_concrete.sh`, `submit_curriculum.sh`, `submit_swinv2l384_ablation.sh`, and
`launch_multiple_jobs.sh` all wrap `qsub ... train_advanced.sh` for specific runs.

## Environment setup

`train_advanced.sh` defaults to SCC's shared `academic-ml/fall-2026` module's
`fall-2026-pyt` conda env (Python 3.13, torch 2.13, transformers 5.16.1) — the
current-semester module, so it doesn't need per-semester script updates. It does
three things, each working around a limitation of the shared env:

- Loads `academic-ml/fall-2026` and activates `fall-2026-pyt`.
- Loads `cuda/13.2` (matching fall-2026-pyt's torch build) **before** activating the
  conda env. `fall-2026-pyt` doesn't set `CUDA_HOME`, so anything that needs to
  build/JIT CUDA extensions (deepspeed, flash-attn, etc.) fails with a missing-`CUDA_HOME`
  error even though `torch.cuda.is_available()` is `True`. Without this, batch jobs using
  those features will fail on the exec node even if an interactive shell looked fine.
- Sets `PYTHONUSERBASE` and `PIP_CACHE_DIR` to a project-directory path instead of the
  defaults under `$HOME` (10GB-quota'd). `fall-2026-pyt` is read-only, so any additional
  packages (e.g. `evaluate`, `wandb`, `pytorch_metric_learning`, none of which ship with
  it) must go through `pip install --user`, which otherwise lands in `$HOME/.local`.

For interactive/manual runs (not through `train_advanced.sh`), `source
finetuning/SWIN/env_fall2026.sh` sets up the same three things in your shell.

Batch (`qsub`) scripts using this env must `module load miniconda` before
`module load academic-ml/fall-2026` — unlike an interactive shell, a fresh batch job
doesn't have miniconda's module functions (e.g. `conda activate`) preloaded, and skips
straight to a `CondaError: Run 'conda init' before 'conda activate'` otherwise.

`transformers>=5.16` (as shipped by `fall-2026-pyt`) dropped the `logging_dir`,
`overwrite_output_dir`, and `warmup_ratio` `TrainingArguments` kwargs that
`SWIN_finetuning_advanced.py` used to pass unconditionally under older transformers.
The script now detects kwarg support at runtime via `inspect.signature`, so it works
under any transformers version — no pinning needed.

### Fallback: herb_env

`herb_env` (`conda activate herb_env`, no module load needed) is the previous standard
env and still works as a fallback if `fall-2026-pyt` ever breaks or gets retired at a
semester turnover — swap the `module load`/`conda activate` lines in `train_advanced.sh`
or `env_fall2026.sh` accordingly. It doesn't need the `cuda/13.2` or
`PYTHONUSERBASE`/`PIP_CACHE_DIR` workarounds since it's a normal user-owned env with all
dependencies already installed into it directly.
