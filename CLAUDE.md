# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- Primary script: `SWIN_finetuning.py` - Full HuggingFace Trainer-based training with WandB logging
- Alternative: `train.py` - Custom training loop with partial layer freezing support
- Evaluation: `eval.py`
- Base model: `microsoft/swin-base-patch4-window12-384`
- WandB integration configured with project: `herbdl`, entity: `bu-spark-ml`
- Environment variables control training behavior:
  - `FROZEN`: Enable layer freezing (true/false)
  - `FROZEN_TYPE`: Freezing strategy (v1, etc.)
  - `LR_TYPE`: Learning rate schedule type
  - `RUN_GROUP`, `RUN_NAME`, `RUN_ID`: WandB run identification

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

**Interactive Visualization** (`index.html`):
- Click points to view herbarium specimens (stacks up to 3 images)
- "Keep sample" toggle prevents images from being replaced
- Hover preview shows thumbnail on mouse-over
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

Standard training with HuggingFace Trainer:
```bash
cd finetuning/SWIN
python SWIN_finetuning.py \
    --output_dir ../output/SWIN/kaggle22/ \
    --model_name_or_path "microsoft/swin-base-patch4-window12-384" \
    --train_file ../datasets/train_22_scientific.json \
    --do_train --do_eval \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3
```

Custom training with layer freezing:
```bash
cd finetuning/SWIN
python train.py  # Freezes all layers except classifier and final layernorm
```

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

5. Update `index.html` line 80 with the generated JSON filepath:
```javascript
fetch("your_generated_file.json")
```

6. View visualization via SCC OnDemand:
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
