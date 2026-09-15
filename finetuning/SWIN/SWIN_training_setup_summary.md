# SWIN Herbaria Classification — Training Setup Summary

*Summary of Farid's SWIN fine-tuning work in `finetuning/SWIN/`. Prepared 2026-06-03. Read-only review — nothing in his repo was modified.*

Source: `/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/SWIN/`

---

## 1. Goal

Fine-tune SWIN / SWIN-V2 Transformers on the **Kaggle Herbaria 2022** dataset for plant **species** classification. Primary metric is top-1 accuracy (and macro-F1 in the local eval / Kaggle leaderboard). All runs log to **Weights & Biases** under `gardoslab/herbdl`.

---

## 2. Data

| Item | Value |
|------|-------|
| Train file | `/projectnb/herbdl/data/kaggle-herbaria/train_2022.json` (~671,800 images) |
| Val file | `/projectnb/herbdl/data/kaggle-herbaria/val_2022.json` (~168,000 images) |
| Image column | `filename` |
| Label column | `scientificNameEncoded` (integer class index) |
| # Classes (2022) | ~15,500 species (`label_mapping_15k.json`) |
| Merged 2021+2022 | ~64,000 species (`label_mapping_64k.json`, `train.json`/`val.json`) — available but the 2022 set is the working dataset |

The data loader (HuggingFace `datasets`) reads JSON, optionally filters rare species, and for multi-task runs derives `family`/`genus`/`species` label encodings on the fly. Rare classes (count ≤ `min_species_samples`, default 2) are dropped for multi-task training.

---

## 3. Models

Backbones are HuggingFace checkpoints (defined in `constants.py`):
- `microsoft/swin-base-patch4-window7-224-in22k` (SWIN-B 224)
- `microsoft/swin-base-patch4-window12-384-in22k` (SWIN-B 384)
- `microsoft/swin-large-patch4-window7-224-in22k` / `...-window12-384-in22k` (SWIN-L)
- `microsoft/swinv2-base-patch4-window12-192-22k`, `...-large-...` (SWIN-V2)

All initialized from **ImageNet-22k pretrained** weights and fine-tuned full (`frozen: false`) by default. There is also a layer-freezing path (`frozen_type` = v1/v3/v4) that progressively unfreezes only the top SWIN stages + classifier head — used for the linear-probe / frozen baselines in `configs/`.

---

## 4. The training engine — `SWIN_finetuning_advanced.py`

A single config-driven script (YAML in, HF `Trainer` underneath) that supports a stack of optional techniques, each toggled by a config section. This is the heart of the setup. (`SWIN_finetuning.py` is the older/simpler baseline version; `SWIN_finetuning_arcface.py` is a variant.)

Key custom components:

- **`MultiTaskSwinModel`** — shared SWIN encoder with three linear heads (family / genus / species). Loss = `species + 0.3·genus + 0.2·family` cross-entropy. Auxiliary supervision for hierarchical taxonomy.
- **`SwinWithArcFace`** + **`SubCenterArcMarginProduct`** — SWIN backbone → embedding (512-d) + BatchNorm → SubCenter ArcFace head (k=3 sub-centers, scale=30, margin=0.5). Optionally blends a CE head (**hybrid loss**, `hybrid_ce_weight`) and/or keeps the family/genus aux heads (ArcFace + multi-task). Has logic to overlay non-backbone weights from a prior checkpoint so ArcFace→Hybrid or 224→384 chaining preserves the trained heads.
- **`MixupCutmixCollator`** — per-batch Mixup (α=0.8) or CutMix (α=1.0), applied with 50% probability, with label smoothing; handles hierarchical labels too.
- **`MixupTrainer`** — `Trainer` subclass that computes the mixed-label loss, routes multi-task / ArcFace losses, and uses a **batch-wise custom evaluation loop** to avoid OOM on the ~15k-class logits.
- **Multi-crop TTA** (`multi_crop_evaluate`) — 5-crop test-time augmentation at inference, averaging logits. Configured but typically left off during training.

### Augmentation pipeline (`augmentation.use_advanced: true`)
RandomResizedCrop (bicubic) + HFlip + **RandAugment** (num_ops=2, mag=9) + optional ColorJitter + ToTensor + Normalize + optional **RandomErasing** (p=0.25), plus Mixup/CutMix and label smoothing (0.1). Validation uses plain Resize→CenterCrop→Normalize.

### Config overrides
The script takes `--config X.yml` plus repeatable `--set key.path=value` dotted overrides, which is how the seed sweeps vary `seed` / `output_dir` / `run_id` from one shared base config.

---

## 5. Config organization

Configs are YAML with sections `model` / `data` / `training` / `custom` / `wandb` (+ optional `augmentation` / `multi_task` / `arcface` / `multi_crop`).

- **`configs/`** — canonical reproducible runs: baselines, frozen-vN linear probes, SWIN/SWIN-V2 base/large at 15k/21k/50k class settings.
- **`configs_advanced/`** — the technique-stacked experiments: `*_enhanced` (aug + high LR), `*_augmented`, `*_multitask`, `*_arcface`, the **curriculum** chain (`swin_curriculum_s1/s2/s3/...`), and the final **pretrained-384 seed ensemble** (`swin_pretrained_384_seed0..4`).
- **`hyperparameter_configs/`** — a 12-cell LR sweep: {cosine, linear} × {1e-4, 2e-4, 5e-4} × {warmup, no-warmup}.

### Standard training hyperparameters (representative)
| Setting | Baseline | Pretrained-384 seed runs |
|---------|----------|--------------------------|
| Batch size (train/eval per device) | 128 / 32 | 64 / 16 (grad accum 2) |
| Learning rate | 2e-4 (enhanced: 5e-4) | 1e-4 |
| Epochs | 50 | 100 |
| Warmup | 5% of steps (`warmup_steps: 0.05`) | 5% |
| Scheduler | cosine (eta_min 1e-6) | linear |
| Weight decay | 0.01 | 0.05 |
| Precision | bf16 | bf16 |
| Eval | every 8964 steps (~1 epoch) | same, + `eval_on_start` |
| Seed | 42 | 0–4 (ensemble) |

> Note on `warmup_steps`: an int = absolute steps, a float in [0,1) = fraction of total steps. There is no `warmup_ratio` (removed in transformers ≥4.52).

---

## 6. Job submission (SCC / SGE)

Jobs run via `qsub` on the `herbdl` project, GPU partition, using `train_advanced.sh` (loads `miniconda` + `academic-ml/fall-2025`, activates the `herb_env` conda env, then runs the python script with `CONFIG_FILE` / `SET_ARGS` env vars).

- **Single job:** `qsub -l h_rt=48:00:00 -pe omp 8 -P herbdl -l gpus=1 -l gpu_c=7.0 ... -v CONFIG_FILE=... train_advanced.sh`
- **Sweeps:** `launch_sweep.py` takes a base config + a sweep YAML listing per-job overrides and fires one `qsub` per experiment (used for LR sweeps and seed ensembles).
- **Curriculum / seed batches:** dedicated shells — `submit_pretrained_seeds.sh` (5 seeds on A100-80G), `submit_curriculum.sh`, `submit_curriculum_pretrained.sh`.
- Defaults: 48h wall time, `omp 8`, 1 GPU, gpu_c 8.0; SGE caps job names at 15 chars.

---

## 7. The experimental arc (what he's actually found)

Two documents lay out the strategy and results: `ADVANCED_TECHNIQUES.md` (the plan, modeled on a competition-winning solution) and `CURRICULUM_REPORT.md` (what actually happened).

**The plan** (`ADVANCED_TECHNIQUES.md`) targets an incremental climb roughly mirroring a winning Kaggle solution: baseline 0.772 → multi-task → higher LR → multi-crop → SubCenter ArcFace → hybrid loss → better aug → 384 res → layer freezing → SWIN-V2, aiming for ~0.86.

**Actual curriculum results** (`CURRICULUM_REPORT.md`, local val F1) tell a more sobering story:

| Stage | Technique | Peak F1 | Δ |
|-------|-----------|---------|---|
| Baseline | CE, no aug | 0.7454 | — |
| Aug (cold) | heavy aug from scratch | 0.6118 | **−0.034** |
| S1 | mild aug warm-up (from baseline ckpt) | 0.7214 | — |
| S2 | medium aug | 0.7421 | +0.021 |
| S3 | heavy aug | 0.7510 | +0.009 |
| S3-cont | LR restart | 0.7510 | +0.000 |
| MultiTask | family/genus aux heads | 0.7523 | +0.001 |
| ArcFace | metric-learning loss | 0.7376 | **−0.015** |

**Key findings he's drawn:**
1. **Curriculum ordering is critical** — heavy augmentation applied cold *destroys* performance (0.61); applied progressively after warming up the backbone, it exceeds baseline (0.751 vs 0.745).
2. The 224px + CE + augmentation combo appears **structurally capped around 0.750–0.752**.
3. **Multi-task** gives only marginal gain (+0.001) — more a regularizer than an accuracy booster here.
4. **ArcFace regressed** under a 60-epoch budget — it needs a long cold-start recovery (random embedding/weight init) before it can beat CE; the budget was too short.
5. Remaining ~5-point gap to ~0.80 is being attacked with **384px resolution**, **SWIN-V2**, and a **seed ensemble** (the current `swin_pretrained_384_seed0..4` runs on SWIN-L, full technique stack, 100 epochs).

---

## 8. Inference / submission side

- `prediction.py` + `run_prediction.sh` — generate predictions; `analyze_predictions.py`, `visualize_plots.py` for analysis (outputs in `prediction_results/`, `PLOTS/`, `TRAINING_ANALYSIS/`).
- `kaggle_submission.py` + `run_kaggle_submission.sh` — produce `submission.csv` for the leaderboard.

---

## 9. Where the current frontier is

The active runs (uncommitted/modified in git) are the **pretrained SWIN-L 384 seed ensemble** and SL224 linear-seed variants — i.e. he has moved past the 224 curriculum ceiling and is now pushing resolution + ensembling. The `*_advanced.py` script and `configs_advanced/` are the live code; `configs/` and `hyperparameter_configs/` are the earlier baseline/sweep foundation.

### Quick file map
| Path | Role |
|------|------|
| `SWIN_finetuning_advanced.py` | Main config-driven trainer (multi-task, ArcFace, mixup, TTA) |
| `train_advanced.sh` | qsub entrypoint (env + launch) |
| `launch_sweep.py` | Fan out qsub jobs from base config + override list |
| `configs/`, `configs_advanced/`, `hyperparameter_configs/` | YAML experiment definitions |
| `ADVANCED_TECHNIQUES.md` | Planned technique ladder (target ~0.86) |
| `CURRICULUM_REPORT.md` | Actual stage-by-stage results |
| `MULTITASK_LEARNING.md` | Multi-task feature docs |
| `CLAUDE.md` | His project spec / conventions |
| `prediction.py`, `kaggle_submission.py` | Inference & leaderboard submission |

---

## 10. Recommendations — closing the gap to 0.86

His current ceiling is ~0.75; the target is ~0.86. The recommendations below are ordered by **expected leverage per unit effort**, and each is tied either to his own findings or to standard fine-grained-recognition (FGVC) practice. The first one is a prerequisite — do it before trusting the size of the "gap."

### Tier 0 — Make sure the gap is real (do this first, ~0 cost)

- **Confirm metric and split parity.** The 0.86 target comes from `ADVANCED_TECHNIQUES.md`, where it's labelled "accuracy / leaderboard," while his curriculum results are **macro-F1 on a local validation split**. Herbarium-style FGVC leaderboards are typically scored on **macro-F1**, which over ~15k classes with a long tail is much harsher than top-1 accuracy and is dominated by rare-species performance. Two action items:
  1. Report **both** top-1 accuracy and macro-F1 on the same split, so it's clear which number 0.86 refers to.
  2. Make sure the local val split tracks the leaderboard — if his 0.2 random `train_val_split` leaks near-duplicate herbarium sheets (same specimen, multiple scans) across train/val, local F1 will be optimistic and progress will be miscalibrated. Verify the split is grouped by specimen/collection, not by image.

  *Why it matters:* if the gap is partly a metric mismatch, the priorities below shift toward long-tail handling; if it's real, they still hold.

### Tier 1 — Highest-leverage levers

1. **Switch the pretrained backbone from ImageNet-22k to an iNaturalist-pretrained model.** This is likely the single biggest lever and is underused in the current setup. ImageNet-22k has almost no botanical fine structure; iNat21-pretrained backbones (e.g. SWIN/ViT/BEiT checkpoints trained on iNaturalist) start with features tuned for exactly this kind of species discrimination. Competition-winning FGVC solutions almost always start from iNat or domain-pretrained weights rather than ImageNet. *Effort: low (swap `model_name_or_path`). Expected: large.*

2. **Resolution is confirmed to help — push it and finish the 384 runs.** His own planned ladder shows 224→384 ≈ +1.5%, and fine structures (leaf venation, floral parts, pubescence) that separate species are simply not resolvable at 224. He's already on this. Consider **448/512** for the final model/ensemble member if memory allows (grad-accumulation keeps effective batch size up).

3. **Treat the long tail explicitly — probably *the* lever for macro-F1.** With ~15.5k classes and a heavy tail, uniform-sampling CE under-serves rare species, which is precisely what macro-F1 penalizes. High-value, well-established techniques:
   - **Two-stage decoupled training (cRT / LWS):** train the representation normally, then *re-train only the classifier head* with class-balanced sampling. Cheap and reliably moves macro-F1.
   - **Balanced/logit-adjusted loss** (balanced softmax, logit adjustment by log class prior, or class-balanced focal loss) as a drop-in for plain CE.
   - **Class-balanced or square-root sampling** during fine-tuning.
   - Note his multi-task family/genus heads already help tail species indirectly (shared coarse signal) — that's the right instinct; pairing it with balanced-classifier retraining should compound.

### Tier 2 — Fix what regressed, and cheap reliable wins

4. **Rescue ArcFace from its cold-start regression.** His finding (ArcFace −0.015, "needs ~40 epochs just to recover") is a classic warm-start problem, not evidence ArcFace is wrong for the task — it's strong for fine-grained, many-class retrieval-style problems. Fixes, in order:
   - **Margin warm-up:** ramp `m` from 0 → 0.5 over the first several epochs instead of starting at full margin.
   - **Head warm-up:** freeze the backbone and train only the embedding + ArcFace head for a few epochs, then unfreeze (mirrors his own curriculum insight — *don't introduce a hard new objective cold*).
   - **Initialize the ArcFace weight matrix from CE class-mean embeddings** rather than `xavier_uniform_`, so it doesn't start random.
   - Give it a real budget (his 60 epochs were too short).
   - Then the **hybrid CE+ArcFace** path he already implemented becomes the natural endpoint.

5. **Apply the curriculum lesson to every hard change.** His clearest, most reproducible finding is *cold heavy-augmentation/objective changes destroy performance; warmed-up they help.* Generalize it: when moving to 384, to ArcFace, or to a new backbone, **initialize from the previous best checkpoint and warm up**, never restart cold. Conversely, he already showed that **restarting the cosine LR to break a plateau does nothing** (S3-cont, +0.000) — stop doing that; change the *model/data*, not the schedule.

6. **Weight EMA** (exponential moving average of model weights). One of the most consistent "free" +0.2–0.5% in FGVC, trivial to add to the trainer.

7. **Turn on the multi-crop TTA he already built — for inference.** The `multi_crop_evaluate` path exists but is disabled. Enable 5-crop + horizontal-flip TTA for the *final submission* (it's the cheapest +0.5–1% he has lying in the codebase). Keep it off during training.

8. **Use the 2021 data he already has on disk.** `train_2021.json` and the merged 64k mapping are present. Either (a) pretrain on combined 2021+2022 then fine-tune on 2022, or (b) add 2021 images for species shared with 2022 as extra training signal — especially valuable for the rare classes driving macro-F1.

### Tier 3 — Ensembling for the final push

9. **Diversify the ensemble beyond seeds.** Seed ensembles (his current 5×) give the smallest diversity-per-member. Bigger gains come from averaging across **different resolutions (384/448), architectures (SWIN-V2, ConvNeXt, BEiT/EVA), and loss types (CE + warmed-up ArcFace)**. Average softmax probabilities (or logits). This is how leaderboard tops are typically reached after a strong single model.

10. **Optimize the decision rule for macro-F1.** Once probabilities are calibrated, macro-F1 can be improved at *inference time* by per-class threshold / prior adjustment rather than plain argmax — relevant because macro-F1 weights every class equally regardless of frequency. Tune on the (specimen-grouped) validation split.

### Suggested sequencing

1. (Tier 0) Pin down the metric, fix the val split if it leaks. → know the true gap.
2. (Tier 1) iNat-pretrained backbone + finish 384 + add balanced-classifier retraining. → this combination alone should recover most of the gap if the leaderboard is macro-F1.
3. (Tier 2) Warm-started ArcFace/hybrid, EMA, TTA, +2021 data. → squeeze the single-model ceiling.
4. (Tier 3) Diverse ensemble + macro-F1 decision-rule tuning. → final points to ~0.86.

*A realistic read: items 1–3 (iNat pretraining, resolution, long-tail handling) are where the bulk of the 11-point gap should come from; ArcFace, EMA, TTA, and ensembling are the polish that takes a strong single model to a leaderboard-topping submission.*

---

## 11. Step-by-step implementation guide (Tier 1 & Tier 2)

These are concrete recipes against his actual codebase. Throughout: configs live in `configs_advanced/`, the trainer is `SWIN_finetuning_advanced.py`, jobs go out via `train_advanced.sh` / `launch_sweep.py`, and any `key.path=value` can be overridden at submit time with `--set` (no need to clone a config to change one field).

> **Convention used below:** "config" = a YAML edit only; "**code**" = a Python edit in `SWIN_finetuning_advanced.py`. I flag the code-required steps explicitly because two of his existing mechanisms (the layer-freeze logic and the ArcFace head) don't compose the way you'd expect — see the call-outs.

---

### Tier 1.1 — Domain-pretrained backbone *(code, not config — see finding below)*

> **Finding (verified against the HuggingFace Hub, June 2026):** there is **no iNaturalist-pretrained SWIN checkpoint** worth using. timm's entire SWIN/SWIN-V2 line is ImageNet-1k/22k only; the lone SWIN+iNat hit is a throwaway AutoTrain model on iNat2018. So the domain-pretraining lever **cannot be taken inside the SWIN family** — it requires switching the backbone architecture. The good news: excellent domain-pretrained checkpoints exist in adjacent families. Shortlist, best-fit first for herbaria species:
>
> | Checkpoint | Arch / res | Why | Loader |
> |-----------|-----------|-----|--------|
> | `imageomics/bioclip-2` | ViT-L/14 | Trained on **TreeOfLife-10M** (plants/fungi/animals incl. herbarium-type images); paper reports **+16–17% absolute** on fine-grained bio classification. Most domain-matched. | `open_clip` |
> | `imageomics/bioclip` | ViT-B/16 | Smaller/older BioCLIP; same idea, lighter. | `open_clip` |
> | `timm/eva02_large_patch14_clip_336.merged2b_ft_inat21` | EVA-02-L @336 | Strongest general iNat21 backbone; iNat21 is plant-heavy. | `timm` |
> | `timm/convnextv2_base.inat21_384` | ConvNeXt-V2-B @384 | **Easiest integration** (timm-native, already 384); good first test of the domain-pretrain hypothesis. | `timm` |
> | `timm/vit_large_patch14_clip_336.laion2b_ft_in12k_in1k_inat21` | ViT-L @336 | Alternative iNat21 ViT. | `timm` |

All of these are **non-SWIN and non-`AutoModelForImageClassification`**, so this is a code task, not a config swap. Steps:

1. **Pick a target.** Pragmatic ladder: prove the hypothesis cheaply with **`timm/convnextv2_base.inat21_384`** (drop-in via timm, already 384), then chase the ceiling with **BioCLIP-2** (most domain-matched) and/or **EVA-02-iNat21**.
2. **Add a loader branch (code).** The model is currently built with `AutoModelForImageClassification.from_pretrained`. Add a branch that builds the backbone with `timm.create_model(name, pretrained=True, num_classes=0)` (timm checkpoints) or `open_clip.create_model_and_transforms` + the visual tower (BioCLIP). Expose `.config.hidden_size` (= timm `num_features`) so the existing head code keeps working.
3. **Adapt the wrappers (code).** `SwinWithArcFace` / `MultiTaskSwinModel` key on `base_model.swin`/`swinv2` and read `outputs.pooler_output`. Generalize to: call the backbone to get a pooled feature vector (timm with `num_classes=0` returns it directly; for ViT/CLIP use the CLS/pooled embedding), then feed that into the existing embedding/ArcFace/CE/multi-task heads. The heads themselves don't change.
4. **Match preprocessing (config/code).** Use the backbone's own normalization and input size (BioCLIP/EVA use CLIP mean/std and 224/336/384; not the SWIN processor's). Pull mean/std/size from the timm/open_clip data config rather than `AutoImageProcessor`.
5. **Keep `ignore_mismatched_sizes`-equivalent behavior:** there's no pretrained 15k head to reuse — the species head is trained fresh on top of the frozen-then-unfrozen domain features. Consider a short head-warm-up (freeze backbone first), as in Tier 2.4.
6. **Smoke-test** with `--set training.max_train_samples=2000 --set training.num_train_epochs=1`, then launch and compare val macro-F1 + top-1 against the ImageNet-22k SWIN baseline at equal epochs.

> **If you want to stay in SWIN with zero code changes:** there's no domain-pretrained option — the best available remains `microsoft/swin-large-patch4-window12-384-in22k` (his current direction). In that case skip this item and lean harder on Tier 1.2 (resolution), Tier 1.3 (long-tail), and Tier 2.8 (pretrain on the 2021 data, which is itself a domain-pretraining step you fully control).

---

### Tier 1.2 — Higher resolution (384, then 448/512) *(config)*

1. **384 is already wired** — it's just the 384 checkpoint (`microsoft/swin-base-patch4-window12-384-in22k` or the SWIN-L 384) in `model.model_name_or_path`. The script reads the input size straight from the image processor (`image_processor.size`), so no resize code to touch. His `swin_pretrained_384_seed*` configs already do this; finish those runs first.
2. **Keep effective batch size constant.** At 384, set `per_device_train_batch_size: 64` and `gradient_accumulation_steps: 2` (his configs already do); for 448/512 drop to 32 / accum 4. Effective batch ≈ 128.
3. **For 448/512 (beyond the checkpoint's native size):** SWIN-v1's window/relative-position-bias is tied to the pretrained resolution, so going above 384 needs position-bias interpolation. **Prefer SWIN-V2 here** — it's designed for resolution transfer (log-spaced continuous position bias). Use a `swinv2` checkpoint from `constants.py` and override the processor size. If you must push SWIN-v1 past 384, that's a **code** change (interpolate `relative_position_bias_table`); not worth it — use V2 instead.
4. **Watch GPU memory / wall time:** 448+ on SWIN-L may need the 80G A100 queue (`-l gpu_memory=80G`, as in `submit_pretrained_seeds.sh`) and possibly `gradient_checkpointing`.
5. **Reserve the highest resolution for a final ensemble member**, not every experiment — it's the most expensive lever.

---

### Tier 1.3 — Long-tail handling *(start config-ish, then code)*

Do these in increasing order of effort; the first alone often moves macro-F1 the most.

**Option A — Logit adjustment / balanced softmax (lowest effort, ~15 lines of code).**
1. Compute the per-class training frequency once (a `Counter` over `dataset["train"][label_column_name]` — the script already imports `Counter`). Convert to a log-prior tensor `log_prior` of shape `[num_classes]`.
2. In `MixupTrainer.compute_loss` (the single-task branch), add the log-prior to the logits before cross-entropy: `logits = logits + tau * log_prior` (start `tau=1.0`). This is **balanced softmax** — it down-weights head classes during training.
3. **At inference, do *not* add the prior** (or subtract it) so rare classes aren't suppressed at prediction time.
4. Gate it behind a new `long_tail.logit_adjustment` config flag so it's reproducible.

**Option B — Two-stage decoupled training (cRT/LWS) (medium effort).**
1. **Stage 1:** train the representation normally (his current full fine-tune).
2. **Stage 2:** start from the Stage-1 checkpoint (`model_name_or_path` = Stage-1 `output_dir`), **freeze the backbone**, and re-train *only the classifier head* for a few epochs with **class-balanced sampling**.
   - ⚠️ **Code gotcha:** his freeze logic in `SWIN_finetuning_advanced.py` (the `frozen_type` block) is hardcoded to **swinv2** parameter names (`"swinv2.layernorm"`, `"swinv2.encoder.layers.3"`) and only keeps params with `'classifier'` in the name trainable. For a SWIN-**v1** model or the ArcFace wrapper this won't behave as intended. For plain CRT on a swinv2 classifier it's fine; otherwise the freeze rule needs a small **code** fix to match the actual head names.
3. **Class-balanced sampling needs code:** HF `Trainer` uses a random sampler by default. Override `get_train_dataloader` (or pass a `WeightedRandomSampler` with weights ∝ 1/class_count) in a `Trainer` subclass. ~20 lines.

**Option C — Class-balanced loss weighting (low effort, partial).**
- Pass `weight=` (inverse-frequency or effective-number weights) into the `CrossEntropyLoss` in `compute_loss`. Cheaper than B but generally weaker than A or B for extreme tails.

Recommended path: **A first** (cheap, reversible), then **B** if macro-F1 is still tail-limited.

---

### Tier 2.4 — Rescue ArcFace from cold-start *(code + config)*

His ArcFace regressed because the embedding + margin head start random and the margin is full-strength from step 0. Apply in this order:

1. **Margin warm-up (code, biggest fix).** In `SubCenterArcMarginProduct`, make `m` settable and ramp it 0 → target over the first N epochs via a `TrainerCallback` (`on_epoch_begin` sets `model.arcface.m` and recomputes `cos_m/sin_m/th/mm`). Starting at `m=0` means ArcFace ≈ plain cosine-softmax early, so there's no cold-start cliff.
2. **Head warm-up (code, not config).** You want the backbone frozen but the **embedding + bn + arcface** head trainable for the first few epochs. ⚠️ The existing `frozen_type` logic keeps only `'classifier'`-named params trainable, so it would freeze the ArcFace head too. Add a freeze branch (e.g. `frozen_type: "arcface_head"`) that freezes `swin.*` but leaves `embedding`, `bn`, `arcface` (and `*_classifier`) trainable. Then unfreeze for the main run.
3. **Initialize the ArcFace weight from CE class means (code, optional but strong).** After loading a CE-trained checkpoint, set each class's `arcface.weight` sub-centers to the normalized mean embedding of that class's training samples instead of `xavier_uniform_`. Removes most of the "40 epochs to recover" cost he observed.
4. **Chain from a warm CE checkpoint (config).** Set `model_name_or_path` to a converged CE/multi-task `output_dir` — his loader already **overlays non-backbone weights** from a checkpoint dir (the `_non_backbone` overlay block), so the embedding/CE heads survive the chain.
5. **Give it budget (config):** ≥100 epochs, not 60.
6. **Then enable the hybrid path (config):** `arcface.hybrid_ce_weight: 0.2–0.3` — his code already blends ArcFace + CE softmax. This stabilizes training and usually beats pure ArcFace.

---

### Tier 2.5 — Curriculum discipline *(process + config)*

1. **Always chain, never cold-start a hard change.** To introduce 384, ArcFace, or a new backbone, set `model.model_name_or_path` to the **previous stage's best `output_dir`** and warm up (low LR, ramped aug). This is his single most reproducible finding (cold heavy-aug → 0.61; warmed → 0.751).
2. **Step augmentation up in stages,** as his S1→S2→S3 did: RandAugment magnitude 4 → 7 → 9, RandomErasing p 0.1 → 0.15 → 0.25, lowering LR each stage. The knobs are all in the `augmentation` config block.
3. **Stop using cosine-LR restarts to break plateaus** — he proved S3-cont gave +0.000. When a stage plateaus, change the *model or data regime* (resolution, backbone, sampling), not the schedule.
4. **Keep a stage ledger** (start ckpt → change → peak F1), like `CURRICULUM_REPORT.md`, so each lever's marginal value is visible.

---

### Tier 2.6 — Weight EMA *(code)*

1. Add an EMA tracker — simplest is `torch.optim.swa_utils.AveragedModel` wrapping the model, or a hand-rolled shadow-weights dict.
2. Update it in a `TrainerCallback.on_step_end` (decay 0.999–0.9999).
3. **Evaluate and save the EMA weights**, not the raw weights, at the end (swap them in before `trainer.evaluate()` / `save_model()`).
4. Gate behind an `ema.enabled` config flag. Expect a steady +0.2–0.5%, essentially free.

---

### Tier 2.7 — Multi-crop + flip TTA at inference *(config + tiny code)*

1. **It's already built** — `multi_crop_evaluate` + `build_multi_crop_transforms` exist and run when `multi_crop.enabled: true`. For a 384 model set:
   ```yaml
   multi_crop:
     enabled: true
     crop_sizes: [400, 416, 448, 480, 512]
     target_size: 384
   ```
2. **Add horizontal-flip TTA (tiny code):** in `build_multi_crop_transforms`, also emit flipped variants (append `RandomHorizontalFlip(p=1.0)` versions), so logits are averaged over crops × {orig, flipped}. Cheap extra robustness.
3. **Use it only for the final/leaderboard prediction**, not during training (leave it off in the seed runs). Mirror the same TTA in `kaggle_submission.py` / `prediction.py` so the submission matches the eval.

---

### Tier 2.8 — Use the 2021 data *(config; optional code for merge)*

He already has `train_2021.json`, `val_2021.json`, and the merged 64k mapping on disk in `/projectnb/herbdl/data/kaggle-herbaria/`.

**Option A — Pretrain on 2021, fine-tune on 2022 (chaining, config-only).**
1. Run a stage with `data.train_file` = the 2021 (or merged) JSON and the 64k label space.
2. Then fine-tune on 2022 by setting `model_name_or_path` to that checkpoint's `output_dir` and `train_file` back to `train_2022.json`, `ignore_mismatched_sizes: true` (label spaces differ).

**Option B — Add 2021 images for shared species (small code/data step).**
1. Build a combined JSON that appends 2021 rows whose species exist in the 2022 label space (a short offline script — produces a new file, doesn't touch his repo).
2. Point `data.train_file` at the combined file. Most useful for the **rare** 2022 species that gain extra examples — directly targets macro-F1.

Validate either against the same specimen-grouped 2022 val split so the comparison is clean.

---

### Putting it together — a concrete next run

A single strong single-model candidate combining the cheap-but-high-value levers:
- **Backbone:** iNat-pretrained (Tier 1.1) if available, else SWIN-L 384 in22k.
- **Resolution:** 384 (Tier 1.2).
- **Loss:** balanced-softmax CE (Tier 1.3-A) + multi-task aux heads (already implemented).
- **Schedule:** warm-started from his best 384 checkpoint, staged aug (Tier 2.5), 100 epochs, EMA on (Tier 2.6).
- **Inference:** multi-crop + flip TTA (Tier 2.7).

Then branch into warmed-up ArcFace (Tier 2.4) and +2021 data (Tier 2.8) as additional ensemble members, and combine per Tier 3.
