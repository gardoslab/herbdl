#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import wandb

# Import pytorch-metric-learning for ArcFace loss
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity

os.environ['WANDB_DISABLED'] = 'false'


""" Fine-tuning a 🤗 Transformers model for image classification with ArcFace loss"""

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class MultiTaskArcFaceModel(nn.Module):
    """
    Multi-task model with ArcFace loss for family, genus, and species classification.
    Uses pytorch-metric-learning for ArcFace implementation.
    """
    def __init__(self, base_model, num_families, num_genera, num_species,
                 embedding_size=512, arcface_s=30.0, arcface_m=0.50, arcface_k=3):
        super().__init__()
        # Store config from base model - required by Trainer
        self.config = base_model.config

        # Extract the base encoder (works for both swin and swinv2)
        if hasattr(base_model, 'swinv2'):
            self.encoder = base_model.swinv2
        elif hasattr(base_model, 'swin'):
            self.encoder = base_model.swin
        else:
            raise ValueError("Base model must have 'swin' or 'swinv2' attribute")

        # Get hidden size from config
        hidden_size = base_model.config.hidden_size

        # Embedding layers for each taxonomy level
        self.family_embedding = nn.Linear(hidden_size, embedding_size)
        self.family_bn = nn.BatchNorm1d(embedding_size)

        self.genus_embedding = nn.Linear(hidden_size, embedding_size)
        self.genus_bn = nn.BatchNorm1d(embedding_size)

        self.species_embedding = nn.Linear(hidden_size, embedding_size)
        self.species_bn = nn.BatchNorm1d(embedding_size)

        # Classifier heads for each taxonomy level (for inference)
        self.family_classifier = nn.Linear(embedding_size, num_families)
        self.genus_classifier = nn.Linear(embedding_size, num_genera)
        self.species_classifier = nn.Linear(embedding_size, num_species)

        # ArcFace loss functions from pytorch-metric-learning
        # Using SubCenterArcFace for robustness to noisy labels
        self.family_arcface_loss = losses.SubCenterArcFaceLoss(
            num_classes=num_families,
            embedding_size=embedding_size,
            margin=arcface_m,
            scale=arcface_s,
            sub_centers=arcface_k
        )

        self.genus_arcface_loss = losses.SubCenterArcFaceLoss(
            num_classes=num_genera,
            embedding_size=embedding_size,
            margin=arcface_m,
            scale=arcface_s,
            sub_centers=arcface_k
        )

        self.species_arcface_loss = losses.SubCenterArcFaceLoss(
            num_classes=num_species,
            embedding_size=embedding_size,
            margin=arcface_m,
            scale=arcface_s,
            sub_centers=arcface_k
        )

        # Loss weights
        self.species_weight = 1.0
        self.genus_weight = 0.3
        self.family_weight = 0.2

    def forward(self, pixel_values, family_labels=None, genus_labels=None, species_labels=None, **kwargs):
        outputs = self.encoder(pixel_values)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # Get embeddings for each taxonomy level
        family_emb = self.family_bn(self.family_embedding(pooled_output))
        genus_emb = self.genus_bn(self.genus_embedding(pooled_output))
        species_emb = self.species_bn(self.species_embedding(pooled_output))

        # Get logits from classifier heads (for inference)
        family_logits = self.family_classifier(family_emb)
        genus_logits = self.genus_classifier(genus_emb)
        species_logits = self.species_classifier(species_emb)

        loss = None
        if family_labels is not None and genus_labels is not None and species_labels is not None:
            # Compute ArcFace losses for each taxonomy level
            family_loss = self.family_arcface_loss(family_emb, family_labels)
            genus_loss = self.genus_arcface_loss(genus_emb, genus_labels)
            species_loss = self.species_arcface_loss(species_emb, species_labels)

            # Weighted combination (species is most important, then genus, then family)
            loss = (self.species_weight * species_loss +
                   self.genus_weight * genus_loss +
                   self.family_weight * family_loss)

        return {
            'loss': loss,
            'logits': species_logits,  # Primary output is species
            'species_logits': species_logits,
            'genus_logits': genus_logits,
            'family_logits': family_logits
        }


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_file: Optional[str] = field(default=None, metadata={"help": "The input data file (a jsonlines or CSV file)."})
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column_name: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    label_column_name: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    train_val_split: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "The proportion of the train set used as validation set in case there's no validation split"
        },
    )
    min_species_samples: Optional[int] = field(
        default=2,
        metadata={"help": "Minimum number of samples per species for multi-task learning"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    # ArcFace specific parameters
    embedding_size: int = field(
        default=512,
        metadata={"help": "Embedding size for ArcFace"}
    )
    arcface_s: float = field(
        default=30.0,
        metadata={"help": "ArcFace scale parameter"}
    )
    arcface_m: float = field(
        default=0.50,
        metadata={"help": "ArcFace margin parameter"}
    )
    arcface_k: int = field(
        default=3,
        metadata={"help": "Number of sub-centers for SubCenter ArcFace"}
    )


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse command line arguments for config file
    arg_parser = argparse.ArgumentParser(description="SWIN Fine-tuning with ArcFace and Multi-task Learning")
    arg_parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = arg_parser.parse_args()

    # Load YAML config
    config = load_config_from_yaml(args.config)

    # Extract custom parameters
    learning_rate_type = config['custom']['lr_type']
    frozen = config['custom']['frozen']
    frozen_type = config['custom']['frozen_type']
    run_group = config['custom']['run_group']
    run_name = config['custom']['run_name']
    run_id = config['custom']['run_id']

    # Extract ArcFace parameters
    arcface_config = config.get('arcface', {})
    use_arcface = arcface_config.get('enabled', True)
    embedding_size = arcface_config.get('embedding_size', 512)
    arcface_s = arcface_config.get('scale', 30.0)
    arcface_m = arcface_config.get('margin', 0.50)
    arcface_k = arcface_config.get('num_subcenters', 3)

    # Multi-task is required for this script
    min_species_samples = config.get('multi_task', {}).get('min_species_samples', 2)

    print(f"__CUSTOM__: Learning rate type: {learning_rate_type}")
    print(f"__CUSTOM__: Frozen: {frozen}")
    print(f"__CUSTOM__: Frozen type: {frozen_type}")
    print(f"__CUSTOM__: ArcFace enabled: {use_arcface}")
    print(f"__CUSTOM__: ArcFace embedding size: {embedding_size}")
    print(f"__CUSTOM__: ArcFace scale: {arcface_s}, margin: {arcface_m}, k: {arcface_k}")
    print(f"__CUSTOM__: Multi-task learning: ENABLED (required for ArcFace multi-task)")
    print(f"__CUSTOM__: Min species samples: {min_species_samples}")

    # Create ModelArguments from config
    model_args = ModelArguments(
        model_name_or_path=config['model']['model_name_or_path'],
        config_name=config['model']['config_name'],
        cache_dir=config['model']['cache_dir'],
        model_revision=config['model']['model_revision'],
        image_processor_name=config['model']['image_processor_name'],
        token=config['model']['token'],
        trust_remote_code=config['model']['trust_remote_code'],
        ignore_mismatched_sizes=config['model']['ignore_mismatched_sizes'],
        embedding_size=embedding_size,
        arcface_s=arcface_s,
        arcface_m=arcface_m,
        arcface_k=arcface_k,
    )

    # Create DataTrainingArguments from config
    data_args = DataTrainingArguments(
        dataset_name=config['data']['dataset_name'],
        dataset_config_name=config['data']['dataset_config_name'],
        data_file=config['data']['data_file'],
        data_dir=config['data']['data_dir'],
        train_file=config['data']['train_file'],
        validation_file=config['data']['validation_file'],
        image_column_name=config['data']['image_column_name'],
        label_column_name=config['data']['label_column_name'],
        max_seq_length=config['data']['max_seq_length'],
        max_train_samples=config['data']['max_train_samples'],
        max_eval_samples=config['data']['max_eval_samples'],
        overwrite_cache=config['data']['overwrite_cache'],
        preprocessing_num_workers=config['data']['preprocessing_num_workers'],
        train_val_split=config['data']['train_val_split'],
        min_species_samples=min_species_samples,
    )

    # Create TrainingArguments from config
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        logging_dir=config['training']['logging_dir'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_train_epochs=config['training']['num_train_epochs'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        logging_strategy=config['training']['logging_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        report_to=config['training']['report_to'],
        bf16=config['training']['bf16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        seed=config['training']['seed'],
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Initialize wandb with complete config
    wandb_config = {
        # Model config
        "model_name": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,
        "ignore_mismatched_sizes": model_args.ignore_mismatched_sizes,
        # Data config
        "train_file": data_args.train_file,
        "validation_file": data_args.validation_file,
        "image_column_name": data_args.image_column_name,
        "label_column_name": data_args.label_column_name,
        "max_train_samples": data_args.max_train_samples,
        "max_eval_samples": data_args.max_eval_samples,
        "train_val_split": data_args.train_val_split,
        # Training config
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "warmup_steps": training_args.warmup_steps,
        "weight_decay": training_args.weight_decay,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "bf16": training_args.bf16,
        "seed": training_args.seed,
        # Custom config
        "frozen": frozen,
        "frozen_type": frozen_type,
        "learning_rate_type": learning_rate_type,
        # ArcFace config
        "use_arcface": use_arcface,
        "embedding_size": embedding_size,
        "arcface_s": arcface_s,
        "arcface_m": arcface_m,
        "arcface_k": arcface_k,
        "multi_task": True,
        "min_species_samples": min_species_samples,
    }

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        resume=config['wandb']['resume'],
        name=run_name,
        group=run_group,
        id=run_id,
        config=wandb_config
    )

    # Set the learning rate scheduler parameters from config
    if 'lr_scheduler_kwargs' in config['training'] and config['training']['lr_scheduler_kwargs']:
        training_args.learning_rate_kwargs = config['training']['lr_scheduler_kwargs']

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if hasattr(data_args, 'test_file') and data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        print(dataset)
        print(dataset["train"].features)

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )

    # Check for required multi-task columns
    required_columns = ['family', 'genus', 'species']
    missing_columns = [col for col in required_columns if col not in dataset_column_names]
    if missing_columns:
        raise ValueError(
            f"Multi-task ArcFace requires the following columns: {missing_columns}. "
            f"Available columns: {dataset_column_names}"
        )

    # Filter species with insufficient samples
    print(f"__CUSTOM__: Filtering species with <={min_species_samples} samples")
    species_counts = Counter(dataset["train"]["species"])
    valid_species = {s for s, c in species_counts.items() if c > min_species_samples}

    original_size = len(dataset["train"])
    dataset["train"] = dataset["train"].filter(lambda x: x["species"] in valid_species)
    filtered_size = len(dataset["train"])
    print(f"__CUSTOM__: Filtered {original_size - filtered_size} samples, kept {filtered_size} samples")

    # Also filter validation set if it exists
    if "validation" in dataset:
        dataset["validation"] = dataset["validation"].filter(lambda x: x["species"] in valid_species)
        print(f"__CUSTOM__: Validation set filtered to {len(dataset['validation'])} samples")

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        family_labels = torch.tensor([example["family_label"] for example in examples])
        genus_labels = torch.tensor([example["genus_label"] for example in examples])
        species_labels = torch.tensor([example["species_label"] for example in examples])
        return {
            "pixel_values": pixel_values,
            "family_labels": family_labels,
            "genus_labels": genus_labels,
            "species_labels": species_labels,
            "labels": species_labels  # For compatibility with Trainer
        }

    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        counts = Counter(dataset["train"][data_args.label_column_name])
        valid_labels = {l for l, c in counts.items() if c > 1}
        filtered = dataset["train"].filter(lambda x: x[data_args.label_column_name] in valid_labels)

        dataset['train'] = filtered
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=42, shuffle=False)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
        print(f"Split the dataset into train and validation with proportions {1 - data_args.train_val_split} and {data_args.train_val_split}.")
        print(f"Training split has {len(dataset['train'])} examples and validation split has {len(dataset['validation'])} examples.")

    # Create hierarchical label mappings
    print("__CUSTOM__: Creating hierarchical label mappings for multi-task ArcFace")

    # Get unique values for each taxonomy level
    unique_families = sorted(dataset["train"].unique("family"))
    unique_genera = sorted(dataset["train"].unique("genus"))
    unique_species = sorted(dataset["train"].unique("species"))

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

    print(f"__CUSTOM__: Found {num_families} families, {num_genera} genera, {num_species} species")

    # Store mappings
    hierarchical_mappings = {
        'family2id': family2id,
        'genus2id': genus2id,
        'species2id': species2id,
        'id2family': id2family,
        'id2genus': id2genus,
        'id2species': id2species,
        'num_families': num_families,
        'num_genera': num_genera,
        'num_species': num_species
    }

    # Save id-to-label mappings to output directory
    import json as _json
    os.makedirs(training_args.output_dir, exist_ok=True)
    label_mappings_path = os.path.join(training_args.output_dir, "label_mappings_multitask.json")
    with open(label_mappings_path, "w") as f:
        _json.dump({
            "id2family": {str(k): v for k, v in id2family.items()},
            "id2genus":  {str(k): v for k, v in id2genus.items()},
            "id2species": {str(k): v for k, v in id2species.items()},
        }, f, indent=2)
    print(f"__CUSTOM__: Saved label mappings to {label_mappings_path}")

    # Prepare label mappings for compatibility
    labels = dataset["train"].unique(data_args.label_column_name)
    num_labels = len(labels)
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}

    # Load the accuracy metric
    accuracy_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Load f1 score metric
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)

    def compute_metrics(p):
        """Computes accuracy for species classification"""
        if len(p.predictions.shape) > 1:
            predictions = np.argmax(p.predictions, axis=1)
        else:
            predictions = p.predictions
        accuracy = accuracy_metric.compute(predictions=predictions, references=p.label_ids)["accuracy"]
        f1_score = f1_metric.compute(predictions=predictions, references=p.label_ids, average="weighted")["f1"]

        return {
            "accuracy": accuracy,
            "f1": f1_score
        }

    # Create ArcFace multi-task model
    print("__CUSTOM__: Creating multi-task ArcFace SWIN model")

    # First load a base model
    config_obj = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_species,  # Use species count for base config
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    base_model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config_obj,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Wrap it in multi-task ArcFace model
    model = MultiTaskArcFaceModel(
        base_model,
        num_families=hierarchical_mappings['num_families'],
        num_genera=hierarchical_mappings['num_genera'],
        num_species=hierarchical_mappings['num_species'],
        embedding_size=model_args.embedding_size,
        arcface_s=model_args.arcface_s,
        arcface_m=model_args.arcface_m,
        arcface_k=model_args.arcface_k
    )

    print(f"__CUSTOM__: Multi-task ArcFace model created with:")
    print(f"  - {hierarchical_mappings['num_families']} families")
    print(f"  - {hierarchical_mappings['num_genera']} genera")
    print(f"  - {hierarchical_mappings['num_species']} species")
    print(f"  - Embedding size: {model_args.embedding_size}")
    print(f"  - ArcFace s={model_args.arcface_s}, m={model_args.arcface_m}, k={model_args.arcface_k}")

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if frozen:
        print("__CUSTOM__: Freezing model according to the frozen type: ", frozen_type)
        for name, param in model.named_parameters():
            # Updated freezing logic to work with both swin and swinv2
            encoder_name = "swinv2" if hasattr(base_model, 'swinv2') else "swin"

            if frozen_type == "v1":
                # Only train classifier heads and final layernorm
                if not any(kw in name for kw in ['family_', 'genus_', 'species_', f'{encoder_name}.layernorm']):
                    param.requires_grad = False
            elif frozen_type == "v4":
                # Train last 3 layers + heads
                if not any(kw in name for kw in ['family_', 'genus_', 'species_', f'{encoder_name}.layernorm',
                                                   f'{encoder_name}.encoder.layers.3', f'{encoder_name}.encoder.layers.2',
                                                   f'{encoder_name}.encoder.layers.1']):
                    param.requires_grad = False
            elif frozen_type == "v3":
                # Train last 2 layers + heads
                if not any(kw in name for kw in ['family_', 'genus_', 'species_', f'{encoder_name}.layernorm',
                                                   f'{encoder_name}.encoder.layers.3', f'{encoder_name}.encoder.layers.2']):
                    param.requires_grad = False
            else:
                # Default: train last layer + heads
                if not any(kw in name for kw in ['family_', 'genus_', 'species_', f'{encoder_name}.layernorm',
                                                   f'{encoder_name}.encoder.layers.3']):
                    param.requires_grad = False

    # Define torchvision transforms to be applied to each image
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(Image.open(pil_img).convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        # Add hierarchical labels
        example_batch["family_label"] = [
            hierarchical_mappings['family2id'][f] for f in example_batch["family"]
        ]
        example_batch["genus_label"] = [
            hierarchical_mappings['genus2id'][g] for g in example_batch["genus"]
        ]
        example_batch["species_label"] = [
            hierarchical_mappings['species2id'][s] for s in example_batch["species"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(Image.open(pil_img).convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        # Add hierarchical labels
        example_batch["family_label"] = [
            hierarchical_mappings['family2id'][f] for f in example_batch["family"]
        ]
        example_batch["genus_label"] = [
            hierarchical_mappings['genus2id'][g] for g in example_batch["genus"]
        ]
        example_batch["species_label"] = [
            hierarchical_mappings['species2id'][s] for s in example_batch["species"]
        ]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        logger.info(f"Number of unique labels in the validation dataset: {len(dataset['validation'].unique(data_args.label_column_name))}")
        dataset["validation"].set_transform(val_transforms)

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision", "arcface", "multi-task"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
