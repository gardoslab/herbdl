# Multi-Task Learning for SWIN

This document explains how to use the multi-task learning feature for hierarchical plant classification.

## Overview

Multi-task learning enables the model to learn from multiple related tasks simultaneously. In this implementation, the model predicts:
1. **Family** (coarse-grained)
2. **Genus** (medium-grained)
3. **Species** (fine-grained) - Primary task

By learning these hierarchical relationships, the model achieves better feature representations and improved accuracy, especially for species with limited training examples.

## Expected Performance Gain

According to the ADVANCED_TECHNIQUES.md document, multi-task learning provides approximately **+1% accuracy improvement** over baseline single-task training.

## Dataset Requirements

Your dataset (CSV or JSON file) must contain the following columns:

- `image_id` or similar identifier
- `filepath`: Path to the image file
- `scientificName`: Full scientific name (optional, can be a combination of family+genus+species)
- `family`: Family name (e.g., "Asteraceae")
- `genus`: Genus name (e.g., "Helianthus")
- `species`: Species name (e.g., "Helianthus annuus")

### Example JSON entry:
```json
{
  "image_id": "12345",
  "filepath": "/path/to/image.jpg",
  "scientificName": "Asteraceae Helianthus annuus",
  "family": "Asteraceae",
  "genus": "Helianthus",
  "species": "Helianthus annuus"
}
```

### Example CSV entry:
```csv
image_id,filepath,scientificName,family,genus,species
12345,/path/to/image.jpg,Asteraceae Helianthus annuus,Asteraceae,Helianthus,Helianthus annuus
```

## Configuration

To enable multi-task learning, add the following section to your YAML config file:

```yaml
multi_task:
  enabled: true
  min_species_samples: 2  # Filter out species with <= 2 samples
  family_weight: 0.2      # Weight for family classification loss
  genus_weight: 0.3       # Weight for genus classification loss
  species_weight: 1.0     # Weight for species classification loss (primary)
```

### Configuration Parameters

- **enabled**: Set to `true` to enable multi-task learning
- **min_species_samples**: Minimum number of samples required for a species to be included in training. Species with fewer samples are filtered out to avoid overfitting on rare classes.
- **family_weight**: Loss weight for family classification (default: 0.2)
- **genus_weight**: Loss weight for genus classification (default: 0.3)
- **species_weight**: Loss weight for species classification (default: 1.0)

The total loss is computed as:
```
total_loss = species_weight * species_loss + genus_weight * genus_loss + family_weight * family_loss
```

## Model Architecture

The multi-task SWIN model (`MultiTaskSwinModel`) consists of:

1. **Shared Encoder**: The base SWIN transformer encoder extracts features from images
2. **Three Classification Heads**:
   - Family classifier: `Linear(hidden_size, num_families)`
   - Genus classifier: `Linear(hidden_size, num_genera)`
   - Species classifier: `Linear(hidden_size, num_species)`

All three classifiers share the same feature representation from the encoder, enabling transfer of knowledge across taxonomy levels.

## Usage Example

### 1. Prepare your dataset

Ensure your dataset has the required columns (family, genus, species).

### 2. Create a configuration file

Use the provided example config:
```bash
configs_advanced/swin_base_224_multitask.yml
```

Or modify an existing config by adding the `multi_task` section.

### 3. Run training

```bash
python SWIN_finetuning_advanced.py --config configs_advanced/swin_base_224_multitask.yml
```

## Implementation Details

### Data Filtering

The implementation automatically filters out species with insufficient samples:
- Species with `<= min_species_samples` are removed from both training and validation sets
- This helps prevent overfitting on rare species and improves model generalization

### Label Encoding

The system automatically creates separate label encodings for each taxonomy level:
- `family2id`: Maps family names to integer IDs
- `genus2id`: Maps genus names to integer IDs
- `species2id`: Maps species names to integer IDs

### Loss Computation

During training, the model computes separate cross-entropy losses for each task:
```python
family_loss = CrossEntropyLoss(family_logits, family_labels)
genus_loss = CrossEntropyLoss(genus_logits, genus_labels)
species_loss = CrossEntropyLoss(species_logits, species_labels)

total_loss = species_weight * species_loss + genus_weight * genus_loss + family_weight * family_loss
```

### Compatibility with Data Augmentation

Multi-task learning is fully compatible with advanced data augmentation techniques:
- **Mixup/CutMix**: The collator handles hierarchical labels correctly
- **RandAugment**: Works seamlessly with multi-task training
- **Label Smoothing**: Applied to all three classification tasks

## Metrics

The training logs report the following metrics:
- `accuracy`: Overall species classification accuracy (primary metric)
- `species_accuracy`: Species-level accuracy
- `species_f1`: Species-level F1 score

Future enhancements may include separate accuracy metrics for family and genus predictions.

## Tips for Best Performance

1. **Balance Loss Weights**: The default weights (0.2, 0.3, 1.0) work well for most cases, but you can tune them based on your dataset:
   - Increase `family_weight` if families are hard to distinguish
   - Increase `genus_weight` if genus-level features are important

2. **Data Quality**: Ensure taxonomy labels are accurate and consistent in your dataset

3. **Sample Threshold**: Adjust `min_species_samples` based on your dataset size:
   - Smaller datasets: Use 2-3 samples
   - Larger datasets: Can use 5-10 samples for more robust training

4. **Combine with Other Techniques**: Multi-task learning works best when combined with:
   - Higher learning rate (5e-4 instead of 2e-4)
   - Advanced augmentation (RandAugment, Mixup, CutMix)
   - Higher resolution images (384 instead of 224)

## Troubleshooting

### Error: "Missing required columns"
Your dataset doesn't have the `family`, `genus`, or `species` columns. Add these columns to your data files.

### Too many species filtered out
Reduce the `min_species_samples` parameter if too many species are being filtered. Check the console output to see how many samples were filtered.

### Poor family/genus accuracy
Try increasing the corresponding loss weights (`family_weight` or `genus_weight`) to give more importance to these auxiliary tasks.

## References

- Implementation based on ADVANCED_TECHNIQUES.md, Section 1: Multi-Task Learning
- Expected improvement: ~1% accuracy gain over baseline
- This technique was used in the herbaria competition winning solution
