# Model Prediction Script

This script loads a trained SWIN model and runs predictions on an evaluation dataset.

## Features

- Loads model from a checkpoint directory
- Uses the same config files as training
- Supports both single-task and multi-task models
- Saves predictions with metadata (image paths, true labels, predicted labels)
- Optionally saves raw logits for further analysis
- Calculates and reports accuracy metrics

## Usage

### Basic Usage

```bash
python prediction.py \
    --config configs/swinv2_base_unfrozen_15k.yml \
    --checkpoint /path/to/checkpoint/directory \
    --use_validation
```

### Full Options

```bash
python prediction.py \
    --config <path_to_config.yml> \
    --checkpoint <path_to_checkpoint> \
    --output <output_file.json> \
    --batch_size 64 \
    --use_validation \
    --save_logits
```

### Arguments

- `--config`: (Required) Path to the YAML configuration file used for training
- `--checkpoint`: (Required) Path to the model checkpoint directory
- `--output`: Output JSON file path (default: `predictions_{checkpoint_name}.json`)
- `--batch_size`: Batch size for prediction (default: 32)
- `--use_validation`: Use the validation file specified in the config
- `--data_file`: Override with a custom data file path
- `--save_logits`: Save raw logits in addition to predictions (warning: creates large files)

**Note:** The script automatically respects the `max_eval_samples` parameter from the config file. If set, only that many samples will be evaluated (randomly shuffled with the config's seed).

## Output Format

The script produces a JSON file with the following structure:

### Single-Task Model

```json
{
  "config": "path/to/config.yml",
  "checkpoint": "path/to/checkpoint",
  "num_samples": 1000,
  "accuracy": 0.85,
  "predictions": [
    {
      "image_path": "/path/to/image1.jpg",
      "true_label": 42,
      "predicted_label": 42
    },
    ...
  ],
  "logits": [...] // Only if --save_logits is used
}
```

### Multi-Task Model

```json
{
  "config": "path/to/config.yml",
  "checkpoint": "path/to/checkpoint",
  "num_samples": 1000,
  "species_accuracy": 0.85,
  "genus_accuracy": 0.90,
  "family_accuracy": 0.95,
  "predictions": [
    {
      "image_path": "/path/to/image1.jpg",
      "family_true": "Asteraceae",
      "genus_true": "Helianthus",
      "species_true": "Helianthus annuus",
      "family_pred": "Asteraceae",
      "genus_pred": "Helianthus",
      "species_pred": "Helianthus annuus"
    },
    ...
  ],
  "logits": {
    "species": [...],
    "genus": [...],
    "family": [...]
  } // Only if --save_logits is used
}
```

## Examples

### 1. Run predictions on validation set

```bash
python prediction.py \
    --config configs/swinv2_base_unfrozen_15k.yml \
    --checkpoint output/SWIN/SWINV2_BASE_UF/checkpoint-50000 \
    --use_validation \
    --batch_size 64
```

### 2. Run predictions on custom test set

```bash
python prediction.py \
    --config configs/swinv2_base_unfrozen_15k.yml \
    --checkpoint output/SWIN/SWINV2_BASE_UF/checkpoint-50000 \
    --data_file /path/to/test_2022.json \
    --output predictions_test.json \
    --batch_size 64
```

### 3. Save raw logits for analysis

```bash
python prediction.py \
    --config configs/swinv2_base_unfrozen_15k.yml \
    --checkpoint output/SWIN/SWINV2_BASE_UF/checkpoint-50000 \
    --use_validation \
    --save_logits \
    --output predictions_with_logits.json
```

## Batch Prediction Script

You can also use `run_prediction.sh` as a template. Edit the script to set your paths:

```bash
# Edit run_prediction.sh to set your paths
vim run_prediction.sh

# Run the script
./run_prediction.sh
```

## Analysis

After running predictions, you can analyze the results using Python:

```python
import json

# Load predictions
with open('predictions_output.json', 'r') as f:
    results = json.load(f)

# Print overall accuracy
print(f"Accuracy: {results['accuracy']:.4f}")

# Find incorrect predictions
incorrect = [p for p in results['predictions']
             if p['true_label'] != p['predicted_label']]
print(f"Number of errors: {len(incorrect)}")

# Analyze specific examples
for pred in incorrect[:5]:
    print(f"Image: {pred['image_path']}")
    print(f"  True: {pred['true_label']}, Predicted: {pred['predicted_label']}")
```

## Notes

- The script automatically detects whether the model is single-task or multi-task based on the config
- GPU is used automatically if available
- For multi-task models, all three taxonomy levels (family, genus, species) are predicted and evaluated
- The script preserves the exact same data preprocessing as used during training
- The `max_eval_samples` config parameter is automatically applied to limit evaluation dataset size
- Dataset is shuffled (using the seed from config) before applying the sample limit to ensure random selection
