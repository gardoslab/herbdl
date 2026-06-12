# timm_finetune — domain-pretrained backbone fine-tuning

Generic image-classification fine-tuner for **timm domain-pretrained backbones** (iNaturalist /
TreeOfLife features), as a self-contained module separate from `finetuning/SWIN/`. It reuses the
same recipe as the SWIN-L 384 "concrete" run — **multi-task family/genus/species heads on the
full 15,501-class `scientificNameEncoded` target** (== the Kaggle macro-F1 metric), balanced
softmax, weight EMA, mixup/cutmix + medium augmentation, and multi-crop+flip TTA wired for
inference — but swaps the ImageNet-22k SWIN backbone for an **iNat21-pretrained** one.

`train.py` is forked from `SWIN/SWIN_finetuning_advanced.py`; the backbone-agnostic machinery is
identical, with a timm loader (`build_backbone`) and timm preprocessing added. Pick the backbone
in YAML via `model.backbone_type: timm` + `model.model_name_or_path: <timm or hf-hub repo>`.

## Why this experiment
ImageNet-22k has little botanical fine structure; iNat21-pretrained backbones start with features
tuned for species discrimination. This is the highest-leverage untaken lever in the plan. Compare
`eval_species_f1` (full 15.5k species) head-to-head against the SWIN-L 384 full-species run at
equal epochs — the iNat-vs-ImageNet22k test.

## Backbones (what actually loads)
The plan's `timm/convnextv2_base.inat21_384` does **not** exist in a loadable form
(`BBracke/convnextv2_base.inat21_384` → 404). The established, working iNat21 timm checkpoints,
loaded via timm's hf-hub mechanism, are:

| Config | Backbone | Res | feat dim | Notes |
|--------|----------|-----|---------|-------|
| `convnext_large_inat_384_2gpu.yml` (default) | `timm/convnext_large_mlp.laion2b_ft_augreg_inat21` | 384 | 1536 | ConvNeXt-L, iNat21; closest to the ConvNeXt/384 intent |
| `eva02_large_inat_336_2gpu.yml` | `timm/eva02_large_patch14_clip_336.merged2b_ft_inat21` | 336 | 1024 | EVA-02-L, iNat21; strongest general iNat backbone (~300M ViT) |

Any other timm name (registry tag or `owner/repo` hf-hub id) works by editing
`model.model_name_or_path`.

## Setup (one-time)
```bash
module load miniconda && module load academic-ml/spring-2026 && conda activate spring-2026-pyt
pip install --user timm
python -c "import timm; print(timm.__version__)"   # 1.0.27+
```
(W&B: same `gardoslab`/`herbdl` setup as the SWIN module — `wandb login` once.)

## Smoke test (recommended before the real job)
```bash
cd finetuning/timm_finetune
python train.py --config configs/convnext_large_inat_384_2gpu.yml \
  --set data.max_train_samples=2000 --set data.max_eval_samples=2000 \
  --set training.num_train_epochs=1 --set wandb.enabled=false \
  --set training.output_dir=/tmp/cnxl_smoke --set training.logging_dir=/tmp/cnxl_smoke
```
Watch for: `timm backbone '...' loaded — num_features=1536, input=384x384`; `Species head target
column: scientificNameEncoded`; `Multi-task model created with 272 families, 2564 genera, 15500
species`; a non-zero `eval_species_f1`; and a saved `config.json` with `backbone_type: timm`.

## Launch (2-GPU DDP)
```bash
cd finetuning/timm_finetune
EMAIL=tgardos@bu.edu NGPUS=2 RUN_PREFIX=CONVNEXT_L_INAT_384 \
  CONFIG=configs/convnext_large_inat_384_2gpu.yml SEEDS="0" bash submit.sh
# EVA-02 alternative:
EMAIL=tgardos@bu.edu NGPUS=2 RUN_PREFIX=EVA02_L_INAT_336 \
  CONFIG=configs/eva02_large_inat_336_2gpu.yml SEEDS="0" bash submit.sh
```
Writes to `<workspace>/herbdl/finetuning/output/timm/<RUN_PREFIX>_SEED<seed>/`. Effective batch
128 (16 × grad-accum 4 × 2 GPUs). ~24h jobs; per-epoch checkpoint + auto-resume (rerun the same
command after each job). See the SWIN `CONCRETE_RUN_README.md` "Runtime/GPU availability" notes —
they apply here too (these backbones fit **48G**, so `GPU_MEM=48G` opens far more idle nodes).

## Notes / caveats
- **Full fine-tune** from step 0 (heads are fresh; iNat features transfer well). No freeze code.
- **DDP + gradient checkpointing**: the timm passthrough calls `backbone.set_grad_checkpointing()`;
  `gradient_checkpointing_kwargs.use_reentrant` is HF-only and ignored. If a 2-GPU job hits
  "marked ready twice", disable grad checkpointing (`--set training.gradient_checkpointing=false`)
  — ConvNeXt-L likely fits 48G at batch 16 without it.
- **Inference/submission is not wired for timm yet** — `SWIN/prediction.py` / `kaggle_submission.py`
  are SWIN-specific. The saved `config.json` carries `backbone_name`/`image_size`/`image_mean`/
  `image_std` so a small timm predictor can rebuild the model; that's a follow-up.
- timm + **single-task or ArcFace** is intentionally not wired (the trainer raises a clear error);
  use `multi_task.enabled: true`.
