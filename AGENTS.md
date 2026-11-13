# AGENTS.md

Project guide for coding agents (Codex, Claude Code, others) working in this repository. The repository map is
`README.md`; the loss-ablation code and trainer options are in `finetuning/SWIN/loss_ablation/README.md`.

## Project Overview

This repository contains experiments in unsupervised learning of herbarium specimens, focusing on plant species classification using deep learning models including SWIN Transformers, CLIP, and BioCLIP.

## Environment & Setup

This project runs on Boston University's Shared Computing Cluster (SCC). Key paths:
- Base project path: `/projectnb/herbdl/`
- Image data: `/projectnb/herbdl/data/kaggle-herbaria/`
- Model checkpoints: `/projectnb/herbdl/workspaces/<username>/herbdl/finetuning/output/`

### Virtual Environment Setup

Create and activate a virtual environment on SCC:
```bash
# Create virtualenv
virtualenv venv
source venv/bin/activate

# Install dependencies for finetuning
pip install -r finetuning/requirements.txt

# Install dependencies for clustering visualization
pip install -r clustering_viz/requirements.txt
```

### SCC training environment (`fall-2026-pyt`)

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

#### Fallback: herb_env

`herb_env` (`conda activate herb_env`, no module load needed) is the previous standard
env and still works as a fallback if `fall-2026-pyt` ever breaks or gets retired at a
semester turnover — swap the `module load`/`conda activate` lines in `train_advanced.sh`
or `env_fall2026.sh` accordingly. It doesn't need the `cuda/13.2` or
`PYTHONUSERBASE`/`PIP_CACHE_DIR` workarounds since it's a normal user-owned env with all
dependencies already installed into it directly.

## Core Architecture

### 1. Dataset Management (`datasets/`)

**HerbariaClassificationDataset** (`dataset.py`) is the primary dataset class:
- Loads CSV annotations with columns: `image_id`, `filename`, `caption`, `scientificName`, `family`, `genus`, `species`, `scientificNameEncoded`
- Integrates with HuggingFace's `AutoImageProcessor` for preprocessing
- Supports flexible label columns (e.g., `scientificNameEncoded` for species-level classification)
- Key paths defined in `constants.py`:
  - Kaggle 2021 dataset: `KAGGLE_HERBARIUM_21_TRAIN_CSV`, `KAGGLE_HERBARIUM_21_TRAIN`
  - Kaggle 2022 dataset: `KAGGLE_HERBARIUM_22_TRAIN_CSV`, `KAGGLE_HERBARIUM_22_TRAIN`

### 2. Model Finetuning (`finetuning/`)

Three main model architectures are explored:

**SWIN Transformer** (`finetuning/SWIN/`):
- The only training script: `SWIN_finetuning_advanced.py` (YAML config in, HF `Trainer` underneath), launched through `train_advanced.sh`.
  Every technique is a config section: multi-task family/genus/species heads, ArcFace, EMA, multi-crop TTA, advanced augmentation,
  logit adjustment, curriculum. The older `SWIN_finetuning.py` and `SWIN_finetuning_arcface.py` were removed (strict subsets).
- `train_advanced.sh` is the sole `qsub` entry point: `qsub -v CONFIG_FILE=configs_advanced/<cfg>.yml train_advanced.sh`;
  set `NPROC_PER_NODE=<n>` for multi-GPU DDP via `torchrun`
- Runs are configs in `configs_advanced/`; the loss-ablation code (losses, trainer options, analysis scripts) is in `loss_ablation/` (see its `README.md`)
- WandB: project `herbdl`, entity `gardoslab` (set per config under `wandb:`)
- Other backbones (ConvNeXt, DINOv2, CoAtNet, EfficientNet) run through the same script via `loss_ablation/backbones.py`

**BioCLIP** (`finetuning/BioCLIP/`):
- Zero-shot evaluation for biological domain
- Scripts: `zero_shot.py`, `train_evaluation.py`

**SWIN-CLIP Hybrid** (`finetuning/SWIN-CLIP/`):
- Combines SWIN visual features with CLIP text-image alignment
- Training: `train.py`, `trainer.py`, `train_baseline.py`
- Modular architecture: `modular_model.py`

### 3. Zero-Shot Evaluation (`CLIP/`)

CLIP zero-shot experiments on herbarium specimens:
- Primary notebook: `CLIP_0shot.ipynb`
- Tests species identification with/without text labels visible
- Explores phenology detection (flowers, buds, leaves)
- Key finding: CLIP performs OCR on specimen labels, affecting zero-shot accuracy

### 4. Clustering & Visualization (`clustering_viz/`)

**Purpose**: Visualize learned representations using dimensionality reduction and detect outliers

**Workflow** (`kaggle22_clustering.ipynb`):
1. Load fine-tuned SWIN checkpoint
2. Extract feature vectors from model using parallel preprocessing
3. Apply dimensionality reduction (PCA/t-SNE)
4. Generate interactive Plotly visualization as JSON
5. Optionally detect outliers using Euclidean distance or Mahalanobis distance

**Key Scripts**:
- `kaggle22_clustering.ipynb`: Main workflow for feature extraction and t-SNE/PCA visualization
- `asteraceae_outliers.ipynb`: Outlier detection using Euclidean distance from cluster centroids
- `outlier_detection/asteraceae_outliers.ipynb`: Advanced outlier detection using Mahalanobis distance on PCA-reduced features
- `generate_thumbnails.py`: Pre-generates optimized thumbnails for faster hover preview loading in the web interface

**Interactive Visualization** (`index.html`):
- Click points to view herbarium specimens (stacks up to 3 images)
- "Keep sample" toggle prevents images from being replaced
- Hover preview shows thumbnail on mouse-over (uses pre-generated thumbnails for fast loading)
- Search/filter controls to show/hide specific species clusters
- Axis locking option to maintain consistent zoom level
- Update the JSON filepath in the `<script>` section to switch between different plots
- Requires SCC access to image paths in `/projectnb/herbdl/data/`

**Example Checkpoints**:
- Asteraceae top 50 species: `/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/astera_50/checkpoint-1300`
- All Kaggle 2022 species: `/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/checkpoint-139125`

### 5. Utilities (`utils/`)

- `image_utils.py`: Image preprocessing helpers
- `resize_images.py`: Batch image resizing
- `image_install_parallel.py`: Parallel image downloading
- `notifications.py`: Job notification system

## Common Commands

### Running SWIN Finetuning

```bash
cd finetuning/SWIN
qsub -l h_rt=12:00:00 -P herbdl -l gpus=2 -l gpu_c=8.0 -l gpu_memory=48G \
     -v NPROC_PER_NODE=2,CONFIG_FILE=configs_advanced/swin_base_384_dbm_pilot.yml train_advanced.sh
```

Eval-only re-scoring of a finished model: a config with `do_train: false`, `evaluation.init_weights` and
`evaluation.predictions_path` (see `configs_advanced/phase3_eval/`).

### Running BioCLIP Evaluation

```bash
cd finetuning/BioCLIP
bash job.sh  # Submit SCC job for zero-shot evaluation
```

### Creating Clustering Visualizations

1. Set up environment:
```bash
cd clustering_viz
source venv/bin/activate  # or your virtualenv
pip install -r requirements.txt
```

2. Run the notebook to generate feature vectors and apply t-SNE:
```bash
jupyter notebook kaggle22_clustering.ipynb
```

3. In the notebook, configure:
   - `CHECKPOINT_PATH`: Path to your fine-tuned SWIN checkpoint
   - `VAL_IMAGE_FILE`: Path to validation dataset JSON file

4. The notebook generates:
   - `asteraceae_tsne_plot_*.json`: t-SNE visualization data
   - `asteraceae_pca_*.csv`: PCA-reduced features for outlier detection
   - `asteraceae_tsne_*.csv`: t-SNE coordinates with labels

5. Generate thumbnails for faster hover preview (recommended):
```bash
python generate_thumbnails.py asteraceae_tsne_plot_astera_50_checkpoint-1300.json
```
   This creates optimized 200px thumbnails in the `thumbnails/` directory, improving hover performance by 10-20x.

6. Update `index.html` line 80 with the generated JSON filepath:
```javascript
fetch("your_generated_file.json")
```

7. View visualization via SCC OnDemand:
   - Navigate to `/projectnb/herbdl/workspaces/<username>/herbdl/clustering_viz/index.html`
   - Or access directly: `https://scc-ondemand1.bu.edu/pun/sys/dashboard/files/fs//projectnb/herbdl/workspaces/<username>/herbdl/clustering_viz/index.html`

### Outlier Detection Workflow

For identifying mislabeled or anomalous specimens:

1. Run `kaggle22_clustering.ipynb` to generate PCA and t-SNE coordinates
2. Use `asteraceae_outliers.ipynb` for basic outlier detection:
   - Calculates Euclidean distance from cluster centroids
   - Flags points beyond 2 standard deviations
3. Use `outlier_detection/asteraceae_outliers.ipynb` for advanced detection:
   - Applies Mahalanobis distance on PCA-reduced features
   - Uses percentile thresholds (e.g., 95th percentile)
   - Overlays outliers on t-SNE plots with different markers
4. Visualize outliers in `index.html` by loading the `*_outliers_*.json` file

## Data Format

**Training JSON** (e.g., `train_22_scientific.json`):
```json
{"image": "/path/to/image.jpg", "caption": "scientific_name_or_encoded_label"}
```

**Validation JSON** (e.g., `val_astera_50.json`):
Used by clustering notebooks for feature extraction. Each line is a JSON object:
```json
{"filename": "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_images/xyz.jpg", "family": "Asteraceae", "scientific_name": "Ageratina jucunda"}
```

**CSV Annotations**:
- Required columns: `image_id`, `filename`, label column (e.g., `scientificNameEncoded`)
- Optional: `scientificName`, `family`, `genus`, `species`, `caption`

## Model Checkpoints

Checkpoints are saved in `finetuning/output/` with subdirectories by model type:
- `SWIN/kaggle22/`: SWIN models trained on Kaggle 2022 dataset
- Structure follows HuggingFace checkpoint format (config.json, pytorch_model.bin, trainer_state.json)

## SCC Job Submission

Most training scripts expect GPU resources. Example job parameters:
```bash
#$ -l gpus=1
#$ -l gpu_c=7.0  # Compute capability
```

See `finetuning/command.sh` for checkpoint resumption logic using `jq` to parse `trainer_state.json`.

## Lessons learned (loss ablations, scaling ladder, backbone repeat)

Hard-won; read before launching or analysing anything. Results and numbers live in the project's
"Loss Function Ablations" document (kept outside this repo).

### Evaluation: count the images, then trust the number
- **Always check how many validation images were scored.** `val_2022` has 167,955. Two silent bugs changed results:
  2-GPU runs were scored on rank 0's half only (fixed: predictions are gathered across ranks; macro-F1 was ~1 point
  low), and a collaborator's configs capped evaluation at `data.max_eval_samples: 50000` (accuracy barely moves but
  macro-F1 is 1.8-2.3 points lower on a subset, because species with no validation image but some predictions score F1 = 0).
  Look for `Num examples = 167955` in the log.
- An eval-only re-score (`evaluation.init_weights` + `evaluation.predictions_path`, `do_train: false`) must reproduce the
  training run's own final numbers; use it to recover a score when a job died right after saving its model, and to save
  per-image predictions for later analysis.
- Macro-F1 = mean per-class F1 over classes in the labels or the predictions. Compare models only on the same images.
  "Points" means percentage points (0.8161 -> 0.8125 is 0.36 points).
- Seed noise is about 0.1 points (two seeds of three losses only). Bootstrap intervals over validation images measure
  evaluation noise, not training noise.
- **Same-specimen copies:** about 1.3-1.7% of `val_2022` images have a copy in `train_2022` (cosine of a generic SWIN
  feature >= 0.93 flags 1.68%; by eye ~1.25%, because the 0.93-0.95 band is mostly look-alike scans). Models get ~98% of
  them right; removing them lowers scores by a few tenths and changes no ranking. The ladder's validation set is easier
  (Herbarium 2021 images) and not comparable with the ablation.
- **Geometry metrics:** the separation gap subtracts an image-to-centroid cosine from a centroid-to-centroid cosine, so it
  depends on how many images each centroid has. Compare by number of training images per species and with an equal number
  of validation images per species (the paper uses the DBM-paper angle and Fisher score per band instead; see
  `loss_ablation/README.md`). The overall average is dominated by common species
  (53% of species have 51-73 training images).

### Training and configs
- The native cross-entropy path applies label smoothing 0.1 by default (the collator default) and ignores logit adjustment;
  the loss registry heads bypass it. Run plain cross-entropy through the registry `softmax` head. Logit adjustment works on
  that head only.
- The loss ablation uses **only a random crop and flip**. Heavy augmentation (RandAugment, mixup/CutMix, label smoothing)
  exists in many older configs; applied from the first step it hurt badly (0.62 vs 0.78); ramp it in
  (`augmentation.warmup_epochs`) or start from a trained checkpoint.
- bf16: margin heads must take `acos` in fp32 (a bf16 clamp epsilon is a no-op, so `acos(>1)` gives NaN). Grep logs for
  non-finite warnings before relaunching.
- Keep the **effective batch** (per-device x accumulation x GPUs) when changing GPU count. Use
  `resume_from_checkpoint: newest`; never leave `overwrite_output_dir` on. KappaFace and ECC are 1-GPU only.
- Check a new config against the script's required keys before launching (e.g. `custom.run_notes`); a missing key kills
  the job in seconds. Compare losses within a backbone only; each backbone keeps its own tuned recipe.

### Cluster and job operations
- SGE stops jobs at 12 hours ("job died on the exec node"). Relaunch from the newest checkpoint. Elapsed time under 12 hours
  with exit status 137 means a node loss or kill: relaunch the same way. A job ending right at the limit can save the model
  but not the final summary: re-score it.
- After every launch run `orx supervise <run> --restart` and confirm the run leaves "starting" (the launcher prints "Could
  not spawn `orx supervise`" and otherwise no status updates arrive). Do not trust the run list alone: stale "running"
  records hid dead jobs for two days. Check `qstat` and checkpoint modification times. Logs of running jobs lag; read the
  job's log file on the cluster.
- The first evaluation of a run is slow (cold file reads, up to an hour), later ones take minutes. One-off NCCL watchdog
  timeouts and Weights & Biases 404s on resume happen; relaunching from the checkpoint fixes them.
- Keep at most 4 jobs queued or running unless told otherwise.

### Git
- Each experiment is an `orx/...` branch off the previous one; all are merged into `loss-ablations`. Shared single-file
  configs (`repgeom/configs/repgeom.yml`, the SWIN-L hybrid yml) were overwritten by each experiment: keep one file per
  variant instead.
