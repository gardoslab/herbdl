This is a project for finetuning models on Herbaria images. 

## Environment setup

Use `conda activate herb_env` when running any scripts here.

### Alternate environment: fall-2026-pyt (SCC academic-ml module)

The SWIN scripts (`finetuning/SWIN/SWIN_finetuning.py`, `SWIN_finetuning_advanced.py`,
`SWIN_finetuning_arcface.py`) also run under SCC's shared `academic-ml/fall-2026`
module's `fall-2026-pyt` conda env (Python 3.13, torch 2.13, transformers 5.16.1),
as an alternative to `herb_env`. `finetuning/SWIN/env_fall2026.sh` sets this up:

```bash
source finetuning/SWIN/env_fall2026.sh
```

This does three things, each working around a limitation of the shared env:

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

Batch (`qsub`) scripts using this env must `module load miniconda` before
`module load academic-ml/fall-2026` — unlike an interactive shell, a fresh batch job
doesn't have miniconda's module functions (e.g. `conda activate`) preloaded, and skips
straight to a `CondaError: Run 'conda init' before 'conda activate'` otherwise. See
`finetuning/SWIN/smoke_fall2026_batch.sh` for a working example.

`transformers>=5.16` (as shipped by `fall-2026-pyt`) dropped the `logging_dir`,
`overwrite_output_dir`, and `warmup_ratio` `TrainingArguments` kwargs that the SWIN
scripts used unconditionally under `herb_env`'s older transformers. The scripts now
detect kwarg support at runtime via `inspect.signature`, so they work under both
environments — no version pinning needed.
