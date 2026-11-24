# Advanced Training Configurations

This directory contains enhanced YAML configurations implementing **Phase 1 Quick Wins** from the herbaria competition winning solution.

## What's Different from Standard Configs?

### 1. Higher Learning Rate
- Standard: 2e-4 (0.0002)
- Enhanced: **5e-4 (0.0005)**
- **Expected gain: +1.54%**

### 2. Advanced Data Augmentation
Standard configs use basic augmentation:
- RandomResizedCrop
- RandomHorizontalFlip
- Normalize

Enhanced configs add:
- **RandAugment** (2 ops, magnitude 9)
- **Mixup** (alpha=0.8) - Mixes two images and their labels
- **CutMix** (alpha=1.0) - Cuts and pastes patches between images
- **Label Smoothing** (0.1)
- **Random Erasing** (0.25 probability)
- **Color Jitter** (brightness, contrast, saturation, hue)

**Expected gain: +0.95%**

### 3. Total Expected Improvement
**~2-3% accuracy improvement** over baseline with minimal code changes!

## Available Configurations

### swin_base_224_enhanced.yml
- Model: SWIN Base @ 224 resolution
- Batch size: 128 per GPU
- Learning rate: 5e-4
- All Phase 1 augmentations enabled
- **Best for**: Quick experiments, lower memory requirements

### swin_base_384_enhanced.yml
- Model: SWIN Base @ 384 resolution
- Batch size: 64 per GPU (with grad accumulation=2)
- Learning rate: 5e-4
- All Phase 1 augmentations enabled
- **Additional expected gain: +1.53% over 224 resolution**
- **Best for**: Maximum accuracy (when you have memory)

### swinv2_base_192_enhanced.yml
- Model: SWIN V2 Base @ 192 resolution
- Batch size: 128 per GPU
- Learning rate: 5e-4
- All Phase 1 augmentations enabled
- **Expected gain: +0.29% from SWIN V2 architecture**
- **Best for**: Newest architecture with improvements

## How to Use

### 1. Update train_advanced.sh

Edit the `CONFIG_FILE` variable:
```bash
CONFIG_FILE="configs_advanced/swin_base_224_enhanced.yml"
```

### 2. Launch Training

For multi-GPU training (maintains HF Trainer's distributed training):
```bash
qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=48G -m beas -M faridkar@bu.edu train_advanced.sh
```

The HuggingFace Trainer automatically handles:
- Multi-GPU distributed training
- Gradient synchronization
- Model parallelism
- Mixed precision training (bf16)

### 3. Monitor in WandB

All augmentation settings and config parameters are automatically logged to WandB.

## What Gets Applied

### During Training:
✅ Higher learning rate (5e-4)
✅ RandAugment on images
✅ ColorJitter on images
✅ Random Erasing on images
✅ Mixup/CutMix (random per batch)
✅ Label smoothing

### During Validation:
- Standard transforms only (Resize → CenterCrop → Normalize)
- No augmentation applied to validation set

## Customizing Augmentation

To modify augmentation parameters, edit the `augmentation` section in your config:

```yaml
augmentation:
  use_advanced: true  # Set to false to disable all advanced augmentations

  randaugment:
    num_ops: 2  # Number of operations to apply (try 1-3)
    magnitude: 9  # Strength of augmentations (0-10)

  mixup:
    enabled: true
    alpha: 0.8  # Mixup strength (try 0.2-1.0)

  cutmix:
    enabled: true
    alpha: 1.0  # CutMix strength (try 0.5-1.5)

  mixup_cutmix_prob: 0.5  # Probability of applying mixup/cutmix to a batch

  label_smoothing: 0.1  # Label smoothing (try 0.0-0.2)

  random_erasing:
    enabled: true
    probability: 0.25
    min_area: 0.02
    max_area: 0.33

  color_jitter:
    enabled: true
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4
    hue: 0.1
```

## Multi-Crop Testing (Phase 2)

Multi-crop testing is configured but not yet enabled:

```yaml
multi_crop:
  enabled: false  # Will be used in future for inference
  crop_sizes: [256, 288, 320, 384, 448]
  target_size: 224
```

This will be implemented in a separate inference script for **+0.64% gain**.

## Technical Details

### Mixup/CutMix Implementation
- Uses custom `MixupCutmixCollator` data collator
- Randomly chooses between Mixup and CutMix per batch
- Applied with 50% probability (configurable)
- Custom `MixupTrainer` handles mixed label loss computation

### Multi-GPU Support
The HuggingFace Trainer handles all multi-GPU training automatically:
- Uses `DistributedDataParallel` (DDP) when multiple GPUs detected
- Gradient synchronization across GPUs
- Proper batch size scaling
- Works with `qsub` multi-GPU jobs

### Memory Optimization
- 384 resolution uses smaller batch size (64) with gradient accumulation (2)
- Effective batch size remains 128 per GPU
- bf16 mixed precision reduces memory by ~50%

## Comparison with Standard Training

| Feature | Standard Config | Enhanced Config |
|---------|----------------|-----------------|
| Learning Rate | 2e-4 | **5e-4** |
| Augmentation | Basic | **RandAugment + Mixup/CutMix** |
| Label Smoothing | 0.0 | **0.1** |
| Random Erasing | No | **Yes** |
| Color Jitter | No | **Yes** |
| Expected Improvement | Baseline | **+2-3%** |

## Next Steps (Phase 2)

After Phase 1, consider:
1. **Multi-crop testing** (+0.64%) - Easy to implement
2. **Multi-task learning** (+1%) - Requires family/genus labels
3. **SubCenter ArcFace** (+1.4%) - Complex implementation

See `ADVANCED_TECHNIQUES.md` in the main directory for detailed implementation guides.

## Troubleshooting

### Out of Memory
- Use 224 resolution instead of 384
- Reduce batch size and increase gradient_accumulation_steps
- Reduce mixup/cutmix probability

### Training Unstable
- Lower learning rate to 3e-4
- Reduce RandAugment magnitude (try 6-7)
- Disable Mixup/CutMix temporarily

### Augmentation Too Aggressive
- Set `use_advanced: false` to disable all
- Or selectively disable specific augmentations by setting `enabled: false`

## Questions?

Check `ADVANCED_TECHNIQUES.md` for more details on all techniques.

## Loss-ablation configs

Three example configs cover the kinds of run in the loss-ablation work (code in `../loss_ablation/`).
Every other run was a copy of one of these with a few fields changed, as listed below.

| Example | What it shows |
|---|---|
| `swin_base_384_dbm_pilot.yml` | SWIN-Base 384 on the 15.5k Herbarium 2022 data with a registry head (DBM), 50 epochs, 2 GPUs |
| `swin_base_384_ladder_n32k_cosine_m0.yml` | The species-count ladder (32k species), fixed step budget, ArcFace head with margin 0 (NormFace) |
| `convnext_base_224_dbm.yml` | The same loss on a different backbone (ConvNeXt-Base), with that backbone's own learning rate and batch size |

### Structure of a config

Each config has the same top-level sections; `SWIN_finetuning_advanced.py` reads them.

| Section | What it controls |
|---|---|
| `model` | Backbone checkpoint (`model_name_or_path`) |
| `data` | Training and validation manifests, image and label columns |
| `training` | Output folder, batch size, learning rate, epochs or steps, evaluation schedule, seed, resuming |
| `custom` | Run name, id and notes (used for logging) |
| `augmentation`, `ema`, `multi_crop` | Augmentation and training extras; the loss ablation used only a random crop and flip |
| `loss_registry` | The classifier head and its settings (see below) |
| `arcface`, `hace`, `multi_task`, `long_tail` | Older native heads, HACE, family/genus/species heads, logit adjustment |
| `wandb` | Weights & Biases logging |
| `evaluation` | Eval-only scoring (see below) |

### Fields to change

Always give a copy its own `training.output_dir` and `custom.run_name` / `custom.run_id`; otherwise it
resumes from, or overwrites, the original run's checkpoints.

- **Loss.** `loss_registry.head` is one of `softmax`, `arcface`, `subcenter_arcface`, `kappaface`,
  `x2softmax`, `dbm`, `sibling`; its settings go in `loss_registry.head_kwargs`: the logit scale `s` for
  every margin head, the margin `m` for `arcface` (`m: 0.0` is NormFace), `k` sub-centres for
  `subcenter_arcface`, the base margin `m0` for `kappaface`, `K` and `tau` for `dbm`.
  Plain cross-entropy is `head: softmax`; adding `long_tail.logit_adjustment: true` (with `tau`) gives
  CE + logit adjustment. For HACE, set `hace.enabled: true` and `loss_registry.enabled: false`.
  `loss_registry.aux: [ecc]` adds the ECC auxiliary loss.
- **Backbone.** `model.model_name_or_path`, e.g. `microsoft/swin-base-patch4-window12-384-in22k`,
  `facebook/convnext-base-224-22k`, `facebook/dinov2-base`. Each backbone keeps its own learning rate,
  batch size and schedule; start from the ConvNeXt example for a non-SWIN backbone.
- **Dataset.** `data.train_file`, `data.validation_file`, `data.image_column_name`,
  `data.label_column_name`. The ladder manifests are `data/ladder/{n15k,n32k,n64k}_{train,val}.json`
  (image column `filepath`); see the ladder example.
- **Training length.** `training.num_train_epochs` on the base data. The ladder fixes the budget with
  `training.max_steps: 131250` (50 epochs of the 15.5k data) at every rung.
- **Seed.** `training.seed` (the paper's second seed is 43).
- **GPUs.** `NPROC_PER_NODE` at launch; keep the effective batch (per-device batch x
  `gradient_accumulation_steps` x GPUs) unchanged. KappaFace and ECC run on 1 GPU only.

### Scoring a finished model (eval only)

Copy the model's training config and set:

```yaml
training:
  do_train: false
  output_dir: <new folder>
evaluation:
  init_weights: <training output_dir>/model.safetensors
  predictions_path: <new folder>/predictions.npz   # per-image argmax, top-5, label
  features_path: <new folder>/features.npy          # optional: pooled features for the geometry analyses
wandb:
  enabled: false
```

To score training images instead (for the training-image geometry and the head swap), also set
`data.validation_file` to the training manifest. Run it on 1 GPU:
`HERBDL_NO_RELOCATE=1 CONFIG_FILE=configs_advanced/<copy>.yml bash train_advanced.sh`.
Check that the log reports the full validation set (`Num examples = 167955` on the base data).
