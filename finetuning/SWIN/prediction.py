#!/usr/bin/env python
# coding=utf-8
"""
Prediction script for SWIN models
Loads a trained model and runs predictions on the evaluation set
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    set_seed,
)

logger = logging.getLogger(__name__)


class MultiTaskSwinModel(nn.Module):
    """
    Multi-task SWIN model with separate classification heads for family, genus, and species.
    """
    def __init__(self, base_model, num_families, num_genera, num_species):
        super().__init__()
        # Extract the base SWIN encoder (works for both swin and swinv2)
        if hasattr(base_model, 'swinv2'):
            self.swin = base_model.swinv2
        elif hasattr(base_model, 'swin'):
            self.swin = base_model.swin
        else:
            raise ValueError("Base model must have 'swin' or 'swinv2' attribute")

        # Get hidden size from config
        hidden_size = base_model.config.hidden_size

        # Three separate classification heads
        self.family_classifier = nn.Linear(hidden_size, num_families)
        self.genus_classifier = nn.Linear(hidden_size, num_genera)
        self.species_classifier = nn.Linear(hidden_size, num_species)

    def forward(self, pixel_values, **kwargs):
        outputs = self.swin(pixel_values)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        family_logits = self.family_classifier(pooled_output)
        genus_logits = self.genus_classifier(pooled_output)
        species_logits = self.species_classifier(pooled_output)

        return {
            'logits': species_logits,  # Primary output is species
            'species_logits': species_logits,
            'genus_logits': genus_logits,
            'family_logits': family_logits
        }


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run predictions on evaluation set")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--output', type=str, default=None, help='Output file path (default: predictions.json)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--use_validation', action='store_true', help='Use validation file from config')
    parser.add_argument('--use_training', action='store_true', help='Use training file from config')
    parser.add_argument('--data_file', type=str, default=None, help='Override data file to run predictions on')
    parser.add_argument('--save_logits', action='store_true', help='Save raw logits in addition to predictions')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader worker processes for image loading')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config_from_yaml(args.config)

    # Extract parameters from config
    model_config = config['model']
    data_config = config['data']
    custom_config = config.get('custom', {})
    multi_task_config = config.get('multi_task', {})

    use_multi_task = multi_task_config.get('enabled', False)
    min_species_samples = multi_task_config.get('min_species_samples', 2)

    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)

    # Determine which data file to use
    if args.data_file:
        data_file = args.data_file
        logger.info(f"Using provided data file: {data_file}")
    elif args.use_training:
        data_file = data_config['train_file']
        logger.info(f"Using training file from config: {data_file}")
    elif args.use_validation:
        data_file = data_config['validation_file']
        logger.info(f"Using validation file from config: {data_file}")
    else:
        # Default to validation file if available, otherwise train file
        data_file = data_config.get('validation_file') or data_config.get('train_file')
        logger.info(f"Using data file: {data_file}")

    if not data_file or not os.path.exists(data_file):
        raise ValueError(f"Data file not found: {data_file}")

    # Load dataset
    logger.info(f"Loading dataset from {data_file}")
    extension = data_file.split(".")[-1]
    dataset = load_dataset(extension, data_files={'eval': data_file})
    eval_dataset = dataset['eval']

    logger.info(f"Loaded {len(eval_dataset)} samples")

    # Apply max_eval_samples limit if specified in config
    max_eval_samples = data_config.get('max_eval_samples')
    if max_eval_samples is not None and max_eval_samples > 0:
        logger.info(f"Limiting evaluation to {max_eval_samples} samples (from config)")
        eval_dataset = eval_dataset.shuffle(seed=seed).select(range(min(max_eval_samples, len(eval_dataset))))
        logger.info(f"Using {len(eval_dataset)} samples after applying max_eval_samples")

    # Get column names
    image_column_name = data_config['image_column_name']
    label_column_name = data_config['label_column_name']

    # Filter species with insufficient samples (for multi-task learning)
    if use_multi_task:
        from collections import Counter

        logger.info(f"Filtering species with <={min_species_samples} samples for multi-task learning")

        # Check if multi-task columns exist
        required_columns = ['family', 'genus', 'species']
        missing_columns = [col for col in required_columns if col not in eval_dataset.column_names]
        if missing_columns:
            raise ValueError(
                f"Multi-task learning enabled but missing required columns: {missing_columns}. "
                f"Available columns: {eval_dataset.column_names}"
            )

        # Get unique values for each taxonomy level
        unique_families = sorted(set(eval_dataset["family"]))
        unique_genera = sorted(set(eval_dataset["genus"]))
        unique_species = sorted(set(eval_dataset["species"]))

        # Create label-to-id mappings
        family2id = {f: i for i, f in enumerate(unique_families)}
        genus2id = {g: i for i, g in enumerate(unique_genera)}
        species2id = {s: i for i, s in enumerate(unique_species)}

        # Create id-to-label mappings
        id2family = {i: f for f, i in family2id.items()}
        id2genus = {i: g for g, i in genus2id.items()}
        id2species = {i: s for s, i in species2id.items()}

        num_families = len(unique_families)
        num_genera = len(unique_genera)
        num_species = len(unique_species)

        logger.info(f"Found {num_families} families, {num_genera} genera, {num_species} species")
    else:
        # For single-task, load the mapping from encoded labels to species names
        # The model was trained with scientificNameEncoded as direct class indices
        # We need to load the mapping that was created during training

        # Try to load the saved label mapping from the checkpoint directory
        label_mapping_path = os.path.join(args.checkpoint, "label_mapping.json")

        if os.path.exists(label_mapping_path):
            logger.info(f"Loading label mapping from {label_mapping_path}")
            with open(label_mapping_path, 'r') as f:
                json_encoded_to_species = json.load(f)
            # Convert string keys back to integers
            encoded_to_species = {int(k): v for k, v in json_encoded_to_species.items()}
            logger.info(f"Loaded mapping for {len(encoded_to_species)} encoded species labels")
        else:
            # Fallback: create mapping from eval dataset (may be incorrect if different encoding)
            logger.warning(
                f"Label mapping file not found at {label_mapping_path}. "
                f"Creating mapping from eval dataset. This may be incorrect if the eval dataset "
                f"uses a different encoding scheme than the training dataset!"
            )

            if 'scientificName' not in eval_dataset.column_names:
                raise ValueError(
                    f"scientificName column not found in dataset. "
                    f"Available columns: {eval_dataset.column_names}"
                )

            encoded_to_species = {}
            for item in eval_dataset:
                encoded_val = item[label_column_name]
                species_name = item['scientificName']
                if encoded_val not in encoded_to_species:
                    encoded_to_species[encoded_val] = species_name
                elif encoded_to_species[encoded_val] != species_name:
                    logger.warning(
                        f"Inconsistent mapping: encoded {encoded_val} maps to both "
                        f"'{encoded_to_species[encoded_val]}' and '{species_name}'"
                    )
            logger.info(f"Created mapping for {len(encoded_to_species)} encoded species labels")

        logger.info("Single-task mode: model will predict scientificNameEncoded values, "
                   "which will be decoded to species names")

    # Load image processor
    logger.info("Loading image processor")
    image_processor = AutoImageProcessor.from_pretrained(
        model_config['image_processor_name'] or model_config['model_name_or_path'],
        cache_dir=model_config['cache_dir'],
        revision=model_config['model_revision'],
        token=model_config['token'],
        trust_remote_code=model_config['trust_remote_code'],
    )

    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")

    if use_multi_task:
        # Load base model first
        config_obj = AutoConfig.from_pretrained(
            args.checkpoint,
            num_labels=num_species,
        )
        base_model = AutoModelForImageClassification.from_pretrained(
            args.checkpoint,
            config=config_obj,
            ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', False),
        )

        # Wrap in multi-task model
        model = MultiTaskSwinModel(
            base_model,
            num_families=num_families,
            num_genera=num_genera,
            num_species=num_species
        )
        logger.info("Loaded multi-task model")
    else:
        # Load model with its original configuration (don't override num_labels)
        model = AutoModelForImageClassification.from_pretrained(
            args.checkpoint,
            ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', False),
        )
        logger.info(f"Loaded single-task model with {model.config.num_labels} output classes")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"Model moved to {device}")

    # Define transforms
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])

    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )

    transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    # Build a PyTorch Dataset so DataLoader can prefetch with multiple workers
    class ImageDataset(Dataset):
        def __init__(self, hf_dataset, image_col, label_col, transform, multi_task=False,
                     family2id=None, genus2id=None, species2id=None):
            self.data = hf_dataset
            self.image_col = image_col
            self.label_col = label_col
            self.transform = transform
            self.multi_task = multi_task
            self.family2id = family2id
            self.genus2id = genus2id
            self.species2id = species2id

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data[idx]
            img = Image.open(row[self.image_col]).convert("RGB")
            pixel_values = self.transform(img)
            if self.multi_task:
                return {
                    'pixel_values': pixel_values,
                    'image_path': row[self.image_col],
                    'family_label': self.family2id[row['family']],
                    'genus_label': self.genus2id[row['genus']],
                    'species_label': self.species2id[row['species']],
                }
            else:
                return {
                    'pixel_values': pixel_values,
                    'image_path': row[self.image_col],
                    'true_label': row[self.label_col],
                }

    if use_multi_task:
        torch_dataset = ImageDataset(
            eval_dataset, image_column_name, label_column_name, transforms,
            multi_task=True, family2id=family2id, genus2id=genus2id, species2id=species2id,
        )
    else:
        torch_dataset = ImageDataset(eval_dataset, image_column_name, label_column_name, transforms)

    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Run predictions
    logger.info(f"Running predictions with batch_size={args.batch_size}, num_workers={args.num_workers}...")
    all_predictions = []
    all_labels = []
    all_logits = [] if args.save_logits else None
    all_metadata = []

    # For multi-task, also track hierarchical predictions
    if use_multi_task:
        all_family_predictions = []
        all_genus_predictions = []
        all_species_predictions = []
        all_family_labels = []
        all_genus_labels = []
        all_species_labels = []
        if args.save_logits:
            all_family_logits = []
            all_genus_logits = []
            all_species_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            pixel_values = batch['pixel_values'].to(device)

            # Run inference
            outputs = model(pixel_values=pixel_values)

            if use_multi_task:
                species_logits = outputs['species_logits']
                genus_logits = outputs['genus_logits']
                family_logits = outputs['family_logits']

                species_preds = torch.argmax(species_logits, dim=-1).cpu().numpy()
                genus_preds = torch.argmax(genus_logits, dim=-1).cpu().numpy()
                family_preds = torch.argmax(family_logits, dim=-1).cpu().numpy()

                all_species_predictions.extend(species_preds.tolist())
                all_genus_predictions.extend(genus_preds.tolist())
                all_family_predictions.extend(family_preds.tolist())

                family_labels = batch['family_label'].tolist()
                genus_labels = batch['genus_label'].tolist()
                species_labels = batch['species_label'].tolist()

                all_family_labels.extend(family_labels)
                all_genus_labels.extend(genus_labels)
                all_species_labels.extend(species_labels)

                if args.save_logits:
                    all_species_logits.extend(species_logits.cpu().numpy().tolist())
                    all_genus_logits.extend(genus_logits.cpu().numpy().tolist())
                    all_family_logits.extend(family_logits.cpu().numpy().tolist())

                for j in range(len(batch['image_path'])):
                    all_metadata.append({
                        'image_path': batch['image_path'][j],
                        'family_true': family_labels[j],
                        'genus_true': genus_labels[j],
                        'species_true': species_labels[j],
                        'family_pred': int(family_preds[j]),
                        'genus_pred': int(genus_preds[j]),
                        'species_pred': int(species_preds[j]),
                    })
            else:
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()

                true_labels = batch['true_label'].tolist()
                all_predictions.extend(predictions.tolist())
                all_labels.extend(true_labels)

                if args.save_logits:
                    all_logits.extend(logits.cpu().numpy().tolist())

                for j in range(len(batch['image_path'])):
                    all_metadata.append({
                        'image_path': batch['image_path'][j],
                        'true_label': true_labels[j],
                        'predicted_label': int(predictions[j]),
                    })

    # Calculate accuracy
    if use_multi_task:
        species_correct = sum([1 for i in range(len(all_species_predictions))
                              if all_species_predictions[i] == all_species_labels[i]])
        genus_correct = sum([1 for i in range(len(all_genus_predictions))
                            if all_genus_predictions[i] == all_genus_labels[i]])
        family_correct = sum([1 for i in range(len(all_family_predictions))
                             if all_family_predictions[i] == all_family_labels[i]])

        total = len(all_species_predictions)
        species_accuracy = species_correct / total
        genus_accuracy = genus_correct / total
        family_accuracy = family_correct / total

        logger.info(f"Species Accuracy: {species_accuracy:.4f} ({species_correct}/{total})")
        logger.info(f"Genus Accuracy: {genus_accuracy:.4f} ({genus_correct}/{total})")
        logger.info(f"Family Accuracy: {family_accuracy:.4f} ({family_correct}/{total})")

        # Prepare results
        results = {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'num_samples': total,
            'species_accuracy': species_accuracy,
            'genus_accuracy': genus_accuracy,
            'family_accuracy': family_accuracy,
            'predictions': all_metadata,
        }

        if args.save_logits:
            results['logits'] = {
                'species': all_species_logits,
                'genus': all_genus_logits,
                'family': all_family_logits,
            }
    else:
        correct = sum([1 for i in range(len(all_predictions))
                      if all_predictions[i] == all_labels[i]])
        total = len(all_predictions)
        accuracy = correct / total

        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        # Prepare results
        results = {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'num_samples': total,
            'accuracy': accuracy,
            'predictions': all_metadata,
        }

        if args.save_logits:
            results['logits'] = all_logits

    # Save results
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).name
        args.output = f"predictions_{checkpoint_name}.json"

    logger.info(f"Saving predictions to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Save label mappings to a separate file for decoding
    mappings_output = args.output.replace('.json', '_mappings.json')
    logger.info(f"Saving label mappings to {mappings_output}")

    if use_multi_task:
        mappings = {
            'type': 'multi_task',
            'id2family': id2family,
            'id2genus': id2genus,
            'id2species': id2species,
        }
    else:
        mappings = {
            'type': 'single_task',
            'id2label': encoded_to_species,
        }

    with open(mappings_output, 'w') as f:
        json.dump(mappings, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
