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
`finetuning/output/SWIN/SWIN_L_384_CONCRETE_SEED<seed>/`. Adjust the `-M` email in
`submit_concrete.sh` if needed.

### Smoke test first (recommended)
Verify the pipeline end-to-end cheaply before committing 48h jobs:
```bash
qsub -l h_rt=2:00:00 -pe omp 8 -P herbdl -l gpus=1 -l gpu_c=8.0 -l gpu_memory=80G \
     -N SWINL384_SMOKE \
     -v CONFIG_FILE=configs_advanced/swin_large_384_concrete.yml,\
SET_ARGS="--set data.max_train_samples=2000 --set data.max_eval_samples=2000 --set training.num_train_epochs=1 --set training.output_dir=/projectnb/herbdl/workspaces/tgardos/herbdl/finetuning/output/SWIN/SMOKE --set training.overwrite_output_dir=true --set wandb.enabled=false" \
     train_advanced.sh
```

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

## Deferred (next ensemble members)

Per the chosen scope, these are intentionally **not** in this run and remain available to
add later as additional ensemble members: domain-pretrained backbone swap (Tier 1.1
timm/open_clip loader), warmed-up ArcFace rescue (Tier 2.4), class-balanced sampler /
two-stage cRT (Tier 1.3-B), and +2021 data (Tier 2.8).
