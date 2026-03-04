# herbdl

Explorations in unsupervised learning of herbaria samples using deep learning models for plant species classification.

## Project Overview

This repository contains experiments on herbarium specimen classification using SWIN Transformers, CLIP, and BioCLIP models. The project runs on Boston University's Shared Computing Cluster (SCC).

## Repository Structure

### `datasets/`
Dataset management and preprocessing utilities.

- `dataset.py` - `HerbariaClassificationDataset` class for loading and preprocessing herbarium images
- `constants.py` - Path definitions for Kaggle 2021 and 2022 herbarium datasets
- `merge_datasets.py` / `merge_datasets.ipynb` - Tools for combining multiple datasets
- Supports flexible label columns (species, family, genus) and integrates with HuggingFace AutoImageProcessor

**Key Datasets:**
- Kaggle Herbarium 2021 dataset
- Kaggle Herbarium 2022 dataset

### `finetuning/`
Model training and evaluation scripts for multiple architectures.

#### `SWIN/`
SWIN Transformer model training and evaluation.

- `SWIN_finetuning.py` - Primary training script using HuggingFace Trainer with WandB logging
- `train.py` - Custom training loop with layer freezing support
- `eval.py` - Model evaluation utilities
- Base model: `microsoft/swin-base-patch4-window12-384`

#### `BioCLIP/`
BioCLIP zero-shot evaluation for biological domain.

- `zero_shot.py` - Zero-shot evaluation on herbarium data
- `train_evaluation.py` - Training set evaluation

#### `SWIN-CLIP/`
Hybrid model combining SWIN visual features with CLIP text-image alignment.

- `train.py` - Main training script
- `modular_model.py` - Modular architecture implementation
- `trainer.py` - Custom trainer implementation
- `train_baseline.py` - Baseline model training

**Configuration:**
- Model checkpoints saved in `output/SWIN/kaggle22/`
- WandB integration: project `herbdl`, entity `bu-spark-ml`
- Environment variables control freezing, learning rate schedules, and run identification

### `CLIP/`
Zero-shot evaluation experiments using OpenAI CLIP.

- `CLIP_0shot.ipynb` - Primary evaluation notebook
- Tests species identification with/without visible text labels
- Explores phenology detection (flowers, buds, leaves)
- Documents CLIP's OCR behavior on specimen labels

### `clustering_viz/`
Interactive visualization of learned representations and outlier detection.

**Main Workflows:**
- `kaggle22_clustering.ipynb` - Feature extraction, PCA/t-SNE dimensionality reduction, and visualization generation
- `asteraceae_outliers.ipynb` - Euclidean distance-based outlier detection
- `outlier_detection/asteraceae_outliers.ipynb` - Advanced Mahalanobis distance-based outlier detection
- `generate_thumbnails.py` - Pre-generates optimized thumbnails for fast hover previews
- `index.html` - Interactive Plotly-based web interface with filtering, search, and image preview

**Features:**
- Click points to view herbarium specimens (stacks up to 3 images)
- Hover preview with optimized thumbnails (10-20x faster loading)
- Search/filter controls for species clusters
- Axis locking for consistent zoom levels
- Outlier visualization with different markers

### `descriptions/`
Text description generation and web scraping utilities.

- `scrape_ncsu.py` - Scrape plant descriptions from NCSU database
- `wikipedia_scrape.py` - Extract plant information from Wikipedia
- `generate_conv.py` - Generate conversational descriptions
- `playground.ipynb` - Experimentation notebook

### `kaggle_eval/`
Kaggle competition evaluation scripts and results.

- `evaluation.py` - Evaluation metrics computation
- `CLIP_explain.ipynb` - CLIP model interpretability analysis
- `evaluation_result/` - JSON files with family, genus, and species-level results

### `utils/`
General utility scripts.

- `resize_images.py` - Batch image resizing
- `image_install_parallel.py` - Parallel image downloading
- `notifications.py` - Job notification system
- `compression.sh` - Image compression utilities
- `labeling.ipynb` - Data labeling tools

## Quick Start

### Environment Setup

```bash
# Create and activate virtual environment
virtualenv venv
source venv/bin/activate

# Install dependencies for finetuning
pip install -r finetuning/requirements.txt

# Install dependencies for clustering visualization
pip install -r clustering_viz/requirements.txt
```

### Running SWIN Finetuning

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

### Creating Clustering Visualizations

```bash
cd clustering_viz
jupyter notebook kaggle22_clustering.ipynb
# Configure checkpoint path and validation dataset
# Generate thumbnails: python generate_thumbnails.py <plot_json_file>
# Update index.html with JSON filepath
# View via SCC OnDemand
```

## Data Paths (SCC)

- Base project: `/projectnb/herbdl/`
- Image data: `/projectnb/herbdl/data/kaggle-herbaria/`
- Model checkpoints: `/projectnb/herbdl/workspaces/<username>/herbdl/finetuning/output/`

## Additional Documentation

- `CLAUDE.md` - Detailed project guide for Claude Code
- Subdirectory READMEs in `datasets/`, `utils/`, `CLIP/`, and `finetuning/output/`

## License

See [LICENSE](LICENSE) file for details.
