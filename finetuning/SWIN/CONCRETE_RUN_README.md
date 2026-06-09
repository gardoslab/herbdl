# Concrete next run — SWIN-L 384 (Tier 1 & 2)

Implements the "Putting it together — a concrete next run" recipe from
`SWIN_training_setup_summary.md` (§10 recommendations, §11 step-by-step), as a single
strong single-model candidate plus the code features it needs.

## What this run is

| Lever | Choice | Section |
|-------|--------|---------|
| Backbone | `microsoft/swin-large-patch4-window12-384-in22k` (stay in SWIN) | 1.1 / 1.2 |
| Resolution | 384 (processor-driven, no resize code) | 1.2 |
| Loss | balanced-softmax CE + multi-task family/genus/species heads | 1.3-A |
| Schedule | 100 epochs, 5% warmup, cosine, **EMA on** | 2.6 |
| Augmentation | MEDIUM (RandAug mag 7, mild mixup/erasing) — cold-start safe | 2.5 |
| Inference | multi-crop + flip TTA wired, enabled only for final prediction | 2.7 |

Config: [`configs_advanced/swin_large_384_concrete.yml`](configs_advanced/swin_large_384_concrete.yml)

## Code features added to `SWIN_finetuning_advanced.py`

All are config-gated and default **off**, so existing configs behave exactly as before.

1. **Balanced softmax / logit adjustment (Tier 1.3-A)** — new `long_tail` section.
   A per-class `log_prior` (log training frequency, in the species/CE head's index
   space) is added to the species logits **during training only** (`logits + tau*log_prior`),
   then plain argmax at inference. Down-weights head classes to lift macro-F1 on the long
   tail. Applied in `MixupTrainer.compute_loss` for single-task and multi-task, in both the
   mixup and non-mixup paths. Not applied to ArcFace.
   ```yaml
   long_tail:
     logit_adjustment: true
     tau: 1.0          # strength; 1.0 = standard balanced softmax
   ```

2. **Weight EMA (Tier 2.6)** — new `ema` section + `EMACallback`.
   Maintains a shadow average of the parameters (`shadow = decay*shadow + (1-decay)*param`
   every step) and copies it into the model at train end, so the final `evaluate()` and
   `save_model()` reflect EMA weights. **Keep `load_best_model_at_end: false`** — the
   best-checkpoint reload would otherwise be overwritten by the EMA copy.
   ```yaml
   ema:
     enabled: true
     decay: 0.9998
   ```

3. **Horizontal-flip TTA (Tier 2.7)** — `multi_crop.flip`.
   `build_multi_crop_transforms(..., flip=True)` also emits a flipped variant of each crop,
   so logits average over crops × {orig, flip}. Leave `multi_crop.enabled: false` during
   training; enable it for the final/leaderboard prediction only.

4. **Gradient-checkpointing passthrough** — `MultiTaskSwinModel` / `SwinWithArcFace` now
   forward `gradient_checkpointing_enable/disable` to the backbone, so
   `training.gradient_checkpointing: true` works for the wrapped models (needed to fit
   SWIN-L @384 on one GPU).

## Environment setup (one-time)

Jobs run via `train_advanced.sh`, which loads:

```bash
module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt
```

`spring-2026-pyt` already provides torch 2.9.1, transformers 4.57.3 (≥4.52, required),
datasets, accelerate, safetensors, torchvision, scikit-learn, pillow, pyyaml, numpy. Two
packages it does **not** include are needed by the trainer — install them once into your
user-site:

```bash
module load miniconda && conda activate spring-2026-pyt
pip install --user evaluate wandb
```

Notes:
- `evaluate` is required (accuracy / macro-F1); `wandb` is needed because the configs use
  `report_to: wandb`. Set `--set training.report_to=none` (or `wandb.enabled: false`) to skip W&B.
- If `import wandb` fails with `cannot import name 'validate_core_schema' from 'pydantic_core'`,
  the `--user` install shadowed the env's `pydantic_core`. Remove the duplicate so the env's
  copy is used again:
  `rm -rf ~/.local/lib/python3.12/site-packages/pydantic_core ~/.local/lib/python3.12/site-packages/pydantic_core-*.dist-info`
- `evaluate.load(...)` downloads its metric script from the HF hub on first use and caches it
  under `~/.cache/huggingface`. Run the smoke test (below) once from a login node to warm the
  cache if your compute nodes can't reach the hub.
- The PyTorch env requires `gpu_c >= 7.0`; the submit scripts request `gpu_c=8.0` (A100), so OK.

Sanity check the env:
```bash
python -c "import torch, transformers, datasets, evaluate, wandb; print('env OK')"
```

### Weights & Biases (logs to gardoslab / herbdl)

The trainer calls `wandb.init(entity="gardoslab", project="herbdl", name=run_name,
group=run_group, id=run_id, ...)` straight from the config (see
`SWIN_finetuning_advanced.py`), so no code change is needed — you only need team membership
+ a valid API key.

1. **Be a member of the `gardoslab` team.** Open <https://wandb.ai/gardoslab> while signed in.
   If you can't see it, ask the team owner to invite your W&B username. `entity="gardoslab"`
   fails with a permission error until you're a member — being logged in is not enough.

2. **Authenticate on SCC** (login node; `~/.netrc` is shared, so compute-node jobs reuse it —
   no per-job login). Grab your key from <https://wandb.ai/authorize>:
   ```bash
   module load miniconda && conda activate spring-2026-pyt
   wandb login --relogin        # paste key; --relogin replaces a stale key
   ```

3. **Verify** (the stored key can be stale even though `~/.netrc` exists):
   ```bash
   wandb login --verify
   python -c "import wandb; v=wandb.Api().viewer; print(v.username, '| teams:', v.teams)"
   ```
   `gardoslab` should appear in `teams`.

Notes:
- Alternative to `~/.netrc`: `export WANDB_API_KEY=<key>` in your shell profile (keeps the
  key out of any committed script).
- The seed loop in `submit_concrete.sh` sets a distinct `run_id`/`run_name` per seed, so seeds
  appear as separate runs grouped under `SWIN_L_384_Concrete`.
- To skip W&B for a run: `--set training.report_to=none` (or `wandb.enabled: false`).
- If a compute node can't reach W&B: `export WANDB_MODE=offline`, then `wandb sync <run_dir>` later.

## How to launch (you run this — nothing is auto-submitted)

Single run (seed 0):
```bash
cd finetuning/SWIN
SEEDS="0" bash submit_concrete.sh
```

3- or 5-seed ensemble:
```bash
bash submit_concrete.sh                 # seeds 0 1 2
SEEDS="0 1 2 3 4" bash submit_concrete.sh
```

Each job requests 1 A100-80G GPU on `herbdl` for 48h and writes to
`finetuning/output/SWIN/SWIN_L_384_CONCRETE_SEED<seed>/` (in your own workspace). Set
`EMAIL=you@bu.edu` for job notifications (it otherwise defaults to the script author).
**A single 1-GPU job will not finish 100 epochs in 48h — see "Runtime" below for the
multi-GPU and resume options.**

### Smoke test first (recommended)
Verify the pipeline end-to-end cheaply before committing 48h jobs:
```bash
qsub -l h_rt=2:00:00 -pe omp 8 -P herbdl -l gpus=1 -l gpu_c=8.0 -l gpu_memory=80G \
     -N SWINL384_SMOKE \
     -v CONFIG_FILE=configs_advanced/swin_large_384_concrete.yml \
-v SET_ARGS="--set data.max_train_samples=2000 --set data.max_eval_samples=2000 --set training.num_train_epochs=1 --set training.output_dir=/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN/SMOKE --set training.overwrite_output_dir=true --set wandb.enabled=false" \
     train_advanced.sh
```

## Runtime: 48h wall limit, multi-GPU (DDP), and resuming

The full run is **100 epochs × ~671,817 images** at effective batch 128 = **524,900
optimization steps**. On **1 A100-80G** it settles at ~1.2 s/step ≈ **~175h (~7 days)** — far
past the **48h** job wall limit (`-l h_rt=48:00:00`), so a single 1-GPU job only reaches
~epoch 27 before SGE kills it. Three options, combinable:

**1. Resume across jobs (no changes).** A checkpoint is saved every epoch
(`save_strategy: epoch`, ~1.8h on 1 GPU) and the trainer auto-resumes from the latest one when
you resubmit to the same `output_dir` (`overwrite_output_dir: false`). Just rerun the same
command after each 48h job ends — ~4 sequential jobs reach 100 epochs.

**2. Multi-GPU / DDP — recommended.** `submit_concrete.sh` honors `NGPUS`, which sets
`NPROC_PER_NODE` so `train_advanced.sh` launches via `torchrun`. Wall-time scales ~linearly:

| GPUs | grad_accum for eff. batch 128 | ~wall time (100 ep) |
|------|-------------------------------|---------------------|
| 1    | 8 (default)                   | ~175h → needs resume |
| 2    | 4                             | ~88h → needs resume  |
| 4    | 2                             | ~44h → ≈ one 48h job |

⚠️ **DDP multiplies the effective batch by `NGPUS`** (each GPU runs its own
`per_device_batch × grad_accum`). The base config targets effective batch **128**, so the
4-GPU run needs `gradient_accumulation_steps: 2` (= `16 × 2 × 4`). A dedicated config —
[`configs_advanced/swin_large_384_concrete_4gpu.yml`](configs_advanced/swin_large_384_concrete_4gpu.yml)
— is identical to the base run but with `grad_accum: 2`, so the launcher path just works:

**4-GPU run via the launcher** (no edits needed):
```bash
cd finetuning/SWIN
EMAIL=tgardos@bu.edu NGPUS=4 RUN_PREFIX=SWIN_L_384_CONCRETE_4GPU \
  CONFIG=configs_advanced/swin_large_384_concrete_4gpu.yml SEEDS="0" \
  bash submit_concrete.sh
```
`NGPUS=4` requests `gpus=4`, `omp 32`, and triggers torchrun (4 processes, 1 per GPU).

**2-GPU run** (often schedules faster — more nodes have 2 free GPUs than 4; ~88h, one resume).
Uses [`configs_advanced/swin_large_384_concrete_2gpu.yml`](configs_advanced/swin_large_384_concrete_2gpu.yml)
(`grad_accum: 4` → `16 × 4 × 2 = 128`):
```bash
cd finetuning/SWIN
EMAIL=tgardos@bu.edu NGPUS=2 RUN_PREFIX=SWIN_L_384_CONCRETE_2GPU \
  CONFIG=configs_advanced/swin_large_384_concrete_2gpu.yml SEEDS="0" \
  bash submit_concrete.sh
```

**4-GPU run via direct qsub** (grad-accum passed inline, no config edit):
```bash
cd finetuning/SWIN
OUT=/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN/SWIN_L_384_CONCRETE_SEED0
qsub -l h_rt=48:00:00 -pe omp 32 -P herbdl -l gpus=4 -l gpu_c=8.0 -l gpu_memory=80G \
     -m beas -M tgardos@bu.edu -N SWINL384_S0 \
     -v CONFIG_FILE=configs_advanced/swin_large_384_concrete.yml,NPROC_PER_NODE=4,\
SET_ARGS="--set training.gradient_accumulation_steps=2 --set training.seed=0 --set training.output_dir=$OUT --set training.logging_dir=$OUT" \
     train_advanced.sh
```

**3. Fewer epochs.** 100 is generous — the 224-px curriculum plateaued by ~50 epochs. Trim with
`--set training.num_train_epochs=50` for a faster first result; resume-extend later.

> Effective-batch note: DDP at 4 GPUs with `grad_accum=2` keeps the global batch at
> `16 × 2 × 4 = 128`, matching the 1-GPU recipe. If you intentionally let it grow (e.g.
> `grad_accum=8` on 4 GPUs → batch 512), scale the LR up accordingly and expect different
> convergence.

### Checking GPU availability (why a job sits in `qw`)

A single-node DDP job (`-pe omp`) needs all its GPUs **free on one host**. SCC has scheduler
job-info collection off, so there's no "queue position" — instead, check live availability:

- `qstat -u $USER` — your jobs (`qw` = waiting, `r` = running).
- `qgpus` — cluster-wide free GPUs by type (A100-80G, H200, …).
- Per-node free count matching this run's needs (`cc≥8.0`, `80G`) — change `f>=2` to the GPU
  count you requested (`f>=4` for the 4-GPU run, `f>=1` for 1-GPU):

```bash
qhost -F gpu_compute_capability,gpu_memory,gpus | awk '
/^scc-/{h=$1;cc=m=f="";next}
/compute_capability=/{s=$0;sub(/.*=/,"",s);cc=s+0}
/gpu_memory=/{s=$0;sub(/.*=/,"",s);m=s+0}
/hc:gpus=/{s=$0;sub(/.*=/,"",s);f=s+0; if(cc>=8 && m>=80 && f>=2) printf "%-12s cc=%-3s mem=%dG free=%d\n",h,cc,m,f}'
```

`hc:gpus` is the per-node *available* count, so a host only appears if it can satisfy your
single-node request right now. No rows for `f>=4` but several for `f>=2` is exactly why the
4-GPU run queues longer than the 2-GPU run. Knobs: `cc>=8` (8.0 = A100, 9.0 = H200), `m>=80`
(min GB), `f>=N` (free GPUs on one node).

## Output paths auto-relocate to your workspace

Most configs in this repo (inherited from faridkar's) hardcode `output_dir`/`logging_dir`
under `/projectnb/herbdl/workspaces/faridkar/herbdl/...`. The trainer rewrites any
`.../workspaces/<author>/herbdl` prefix to the repo you actually run from, preserving the
trailing run name — so a `tgardos` checkout writes to
`/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN/<NAME>` automatically,
with no YAML edits. It logs the rewrite (`__CUSTOM__: Relocated output path ...`). Set
`HERBDL_NO_RELOCATE=1` to disable (e.g. to write somewhere else via an explicit path).

## Warm-start (Tier 2.5 — recommended once a 384 checkpoint exists)

Cold-from-in22k is the dependency-free default. The curriculum finding is that chaining a
hard change from a converged checkpoint beats cold-starting it. Once you have a converged
SWIN-L 384 run, chain from it (keep `config_name`/`image_processor_name` on the 384 arch)
and raise `augmentation.randaugment.magnitude` to 9:
```bash
CKPT=/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN/SWIN_L_384_CONCRETE_SEED0 \
    SEEDS="1" bash submit_concrete.sh
```

## OOM / memory tuning

SWIN-L @384 is heavy. If a job OOMs, lower the per-device batch and raise grad-accum to
keep the effective batch (~128) constant, e.g. via `--set`:
```
--set training.per_device_train_batch_size=8 --set training.gradient_accumulation_steps=16
```
`gradient_checkpointing: true` is already on.

## Final prediction with TTA

For the leaderboard/final eval, enable TTA on the trained checkpoint:
```yaml
multi_crop:
  enabled: true
  crop_sizes: [400, 416, 448, 480, 512]
  target_size: 384
  flip: true
```
The trainer runs `multi_crop_evaluate` after the standard eval and prints averaged
accuracy + macro-F1 (`__CUSTOM__: Multi-crop eval ...`). Mirror the same crops/flip in
`prediction.py` / `kaggle_submission.py` so the submission matches the eval.

## Metrics

Both top-1 accuracy and macro-F1 are reported every epoch (`eval_accuracy` /
`eval_species_f1` for multi-task). Macro-F1 over the long tail is the number to watch
(Tier 0).

## Remote monitoring from phone / MacBook (Claude Code Remote Control)

To babysit a run (check `qstat`, read logs, tweak configs) from an iPhone or MacBook, use
Claude Code **Remote Control** — the `claude` process keeps running on the SCC login node
(full `/projectnb` + `qsub` access), and your phone/browser are just remote windows into it.
This is different from *Claude Code on the web*, whose cloud sandbox has **no** SCC access.

### Updating Claude Code on SCC (needed: ≥ 2.1.51 for Remote Control)

Claude Code here is installed as an npm **prefix** install and run via a shell alias:
```bash
alias claude='npx --prefix ~/claude-code claude'
```
Because of that, `claude update` does **not** work — it targets npm's global prefix, which
is the read-only shared module dir (`/share/pkg.8/.../spring-2026-pyt`). Update the copy the
alias actually uses instead:
```bash
module load miniconda && conda activate spring-2026-pyt   # for a consistent node/npm
npm install --prefix ~/claude-code @anthropic-ai/claude-code@latest
npx --prefix ~/claude-code claude --version               # confirm >= 2.1.51
```
Re-run that `npm install --prefix` line whenever you want to upgrade (don't use `claude update`).

### Starting a Remote Control session

Remote Control requires a **claude.ai subscription login (Pro/Max/Team/Enterprise) — API keys
are not supported**. On the SCC login node:
```bash
unset ANTHROPIC_API_KEY          # if set, it blocks Remote Control
claude /login                    # choose the claude.ai option (not a Console API key)

tmux new -s claude-hpc           # persistent: survives SSH disconnects
# inside tmux:
cd /projectnb/herbdl/workspaces/tgardos/herbdl
claude remote-control --name "HerbDL SWIN-L 384"
```
It prints a session URL and offers a QR code (press space). Detach with `Ctrl-b d`; Claude
keeps running.

- **iPhone:** Claude app → **Code** tab → pick "HerbDL SWIN-L 384" (or scan the QR).
- **MacBook:** open the session URL, or go to **claude.ai/code** and pick the session. For a
  local terminal instead: `ssh -t scc1.bu.edu "tmux attach -t claude-hpc"`.

Notes:
- Keep Claude on the **login node** (lightweight coordinator); GPU training stays in `qsub`
  jobs on compute nodes. Don't run training directly under Claude.
- Remote Control can **push a phone notification** when a long task finishes (enable via `/config`).
- Text commands (`/context`, `/usage`) work from mobile; interactive pickers (`/resume`, `/mcp`)
  only from the local terminal.

## Deferred (next ensemble members)

Per the chosen scope, these are intentionally **not** in this run and remain available to
add later as additional ensemble members: domain-pretrained backbone swap (Tier 1.1
timm/open_clip loader), warmed-up ArcFace rescue (Tier 2.4), class-balanced sampler /
two-stage cRT (Tier 1.3-B), and +2021 data (Tier 2.8).
