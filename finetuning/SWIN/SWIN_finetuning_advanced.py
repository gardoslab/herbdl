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
import random

import evaluate
import numpy as np
import torch
import torch.nn as nn
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
    ColorJitter,
    RandomErasing,
    RandAugment,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import wandb

os.environ['WANDB_DISABLED'] = 'false'

""" Fine-tuning a 🤗 Transformers model for image classification with advanced augmentations"""

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


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


class MultiTaskSwinModel(nn.Module):
    """
    Multi-task SWIN model with separate classification heads for family, genus, and species.
    """
    def __init__(self, base_model, num_families, num_genera, num_species):
        super().__init__()
        # Store config from base model - required by Trainer
        self.config = base_model.config

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

    def forward(self, pixel_values, family_labels=None, genus_labels=None, species_labels=None, **kwargs):
        outputs = self.swin(pixel_values)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        family_logits = self.family_classifier(pooled_output)
        genus_logits = self.genus_classifier(pooled_output)
        species_logits = self.species_classifier(pooled_output)

        loss = None
        if family_labels is not None and genus_labels is not None and species_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            family_loss = loss_fct(family_logits, family_labels)
            genus_loss = loss_fct(genus_logits, genus_labels)
            species_loss = loss_fct(species_logits, species_labels)

            # Weighted combination (species is most important, then genus, then family)
            loss = species_loss + 0.3 * genus_loss + 0.2 * family_loss

        return {
            'loss': loss,
            'logits': species_logits,  # Primary output is species
            'species_logits': species_logits,
            'genus_logits': genus_logits,
            'family_logits': family_logits
        }


class MixupCutmixCollator:
    """
    Collator that applies Mixup and/or Cutmix augmentation.
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, label_smoothing=0.1, num_classes=1000, multi_task=False):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.multi_task = multi_task

    def __call__(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])

        # Check if "label" key exists (training) or use direct label access (validation)
        # During training, transforms set "label" key; during validation, they don't
        if "label" in examples[0]:
            labels = torch.tensor([example["label"] for example in examples])
        else:
            # This is for validation/evaluation - no mixup/cutmix should be applied
            # Just return the basic batch
            result = {"pixel_values": pixel_values}

            if self.multi_task and "family_label" in examples[0]:
                result.update({
                    "family_labels": torch.tensor([example["family_label"] for example in examples]),
                    "genus_labels": torch.tensor([example["genus_label"] for example in examples]),
                    "species_labels": torch.tensor([example["species_label"] for example in examples])
                })

            return result

        # For multi-task learning, also extract hierarchical labels
        if self.multi_task:
            family_labels = torch.tensor([example["family_label"] for example in examples])
            genus_labels = torch.tensor([example["genus_label"] for example in examples])
            species_labels = torch.tensor([example["species_label"] for example in examples])

        batch_size = pixel_values.size(0)

        # Decide whether to apply mixup/cutmix
        if random.random() > self.prob:
            # No mixup/cutmix
            result = {"pixel_values": pixel_values, "labels": labels}
            if self.multi_task:
                result.update({
                    "family_labels": family_labels,
                    "genus_labels": genus_labels,
                    "species_labels": species_labels
                })
            return result

        # Decide between mixup and cutmix
        use_cutmix = random.random() < 0.5

        # Get random permutation for mixing
        indices = torch.randperm(batch_size)

        if use_cutmix and self.cutmix_alpha > 0:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            _, _, h, w = pixel_values.shape

            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)

            cx = np.random.randint(w)
            cy = np.random.randint(h)

            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)

            pixel_values[:, :, bby1:bby2, bbx1:bbx2] = pixel_values[indices, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        elif self.mixup_alpha > 0:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            pixel_values = lam * pixel_values + (1 - lam) * pixel_values[indices]
        else:
            lam = 1.0

        # Create mixed labels (soft labels)
        labels_a = labels
        labels_b = labels[indices]

        result = {
            "pixel_values": pixel_values,
            "labels": labels_a,
            "labels_b": labels_b,
            "lam": lam
        }

        # For multi-task, mix all hierarchical labels
        if self.multi_task:
            result.update({
                "family_labels": family_labels,
                "family_labels_b": family_labels[indices],
                "genus_labels": genus_labels,
                "genus_labels_b": genus_labels[indices],
                "species_labels": species_labels,
                "species_labels_b": species_labels[indices],
            })

        return result


class MixupTrainer(Trainer):
    """
    Custom Trainer that handles Mixup/Cutmix loss computation and batch-wise evaluation.
    """
    def __init__(self, *args, multi_task=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_task = multi_task

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels_a = inputs.pop("labels")
        labels_b = inputs.pop("labels_b", None)
        lam = inputs.pop("lam", 1.0)

        # Handle multi-task learning
        if self.multi_task:
            family_labels = inputs.pop("family_labels")
            genus_labels = inputs.pop("genus_labels")
            species_labels = inputs.pop("species_labels")
            family_labels_b = inputs.pop("family_labels_b", None)
            genus_labels_b = inputs.pop("genus_labels_b", None)
            species_labels_b = inputs.pop("species_labels_b", None)

            # Pass hierarchical labels to model
            inputs["family_labels"] = family_labels
            inputs["genus_labels"] = genus_labels
            inputs["species_labels"] = species_labels

            outputs = model(**inputs)

            # If we have mixup/cutmix, we need to manually compute the mixed loss
            if family_labels_b is not None:
                loss_fct = nn.CrossEntropyLoss()

                # Get logits for each taxonomy level
                family_logits = outputs.get("family_logits")
                genus_logits = outputs.get("genus_logits")
                species_logits = outputs.get("species_logits")

                # Compute mixed losses
                family_loss = lam * loss_fct(family_logits, family_labels) + (1 - lam) * loss_fct(family_logits, family_labels_b)
                genus_loss = lam * loss_fct(genus_logits, genus_labels) + (1 - lam) * loss_fct(genus_logits, genus_labels_b)
                species_loss = lam * loss_fct(species_logits, species_labels) + (1 - lam) * loss_fct(species_logits, species_labels_b)

                # Combined loss with same weighting as the model
                loss = species_loss + 0.3 * genus_loss + 0.2 * family_loss
            else:
                # Model already computed the loss
                loss = outputs.get("loss")

            return (loss, outputs) if return_outputs else loss
        else:
            # Standard single-task training
            # For standard models, only pass pixel_values (labels handled separately)
            outputs = model(pixel_values=inputs["pixel_values"])
            logits = outputs.get("logits")

            if labels_b is not None:
                # Mixup/Cutmix loss
                loss_fct = nn.CrossEntropyLoss()
                loss = lam * loss_fct(logits, labels_a) + (1 - lam) * loss_fct(logits, labels_b)
            else:
                # Standard loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels_a)

            return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Custom evaluation loop that computes metrics batch-wise to avoid OOM.
        """
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        model.eval()

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")

        # Initialize accumulators for batch-wise metric computation
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        total_batches = 0

        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)

            labels_a = inputs.pop("labels", None)
            labels_b = inputs.pop("labels_b", None)
            lam = inputs.pop("lam", 1.0)

            # Initialize labels variable
            labels = None

            # For multi-task, also handle hierarchical labels
            if self.multi_task:
                if "family_labels" in inputs:
                    family_labels = inputs.pop("family_labels")
                if "genus_labels" in inputs:
                    genus_labels = inputs.pop("genus_labels")
                if "species_labels" in inputs:
                    species_labels = inputs.pop("species_labels")
                    # Use species labels as the primary labels for metrics
                    labels = species_labels
            else:
                # For single-task, use labels_a as the primary labels
                labels = labels_a

            with torch.no_grad():
                if self.multi_task:
                    # For multi-task, pass all labels to model
                    outputs = model(
                        pixel_values=inputs["pixel_values"],
                        family_labels=family_labels if "family_labels" in locals() else None,
                        genus_labels=genus_labels if "genus_labels" in locals() else None,
                        species_labels=labels
                    )
                else:
                    outputs = model(pixel_values=inputs["pixel_values"])
                    logits = outputs.get("logits")

                    if labels_b is not None:
                        # Mixup/Cutmix loss
                        loss_fct = nn.CrossEntropyLoss()
                        loss = lam * loss_fct(logits, labels_a) + (1 - lam) * loss_fct(logits, labels_b)
                    elif labels_a is not None:
                        # Standard loss
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits, labels_a)

                    if labels_a is not None:
                        total_loss += loss.item()
                        total_batches += 1

                # Get logits (primary output for predictions)
                logits = outputs.get("logits")
                if logits is None:
                    logits = outputs.get("species_logits")

            # Compute predictions for this batch
            predictions = torch.argmax(logits, dim=-1)

            # Move to CPU and store (batch-wise to save memory)
            all_predictions.append(predictions.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())

        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy() if all_labels else None

        # Compute metrics
        metrics = {}
        if self.compute_metrics is not None and all_labels is not None:
            # Create a mock EvalPrediction object
            from transformers.trainer_utils import EvalPrediction

            # For compute_metrics, we need to pass logits in the right shape
            # Since we've already computed predictions, we'll use one-hot encoding
            num_labels = logits.shape[-1]
            mock_logits = np.zeros((len(all_predictions), num_labels))
            mock_logits[np.arange(len(all_predictions)), all_predictions] = 1.0

            eval_pred = EvalPrediction(predictions=mock_logits, label_ids=all_labels)
            metrics = self.compute_metrics(eval_pred)

        # Add loss to metrics
        if total_batches > 0:
            metrics[f"{metric_key_prefix}_loss"] = total_loss / total_batches

        # Prefix all keys with metric_key_prefix
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return transformers.trainer_utils.EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_examples,
        )


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse command line arguments for config file
    arg_parser = argparse.ArgumentParser(description="SWIN Fine-tuning with advanced augmentations")
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

    # Extract augmentation parameters
    aug_config = config.get('augmentation', {})
    use_advanced_aug = aug_config.get('use_advanced', False)

    # Extract multi-task learning parameters
    multi_task_config = config.get('multi_task', {})
    use_multi_task = multi_task_config.get('enabled', False)
    min_species_samples = multi_task_config.get('min_species_samples', 2)
    family_weight = multi_task_config.get('family_weight', 0.2)
    genus_weight = multi_task_config.get('genus_weight', 0.3)
    species_weight = multi_task_config.get('species_weight', 1.0)

    print(f"__CUSTOM__: Learning rate type: {learning_rate_type}")
    print(f"__CUSTOM__: Frozen: {frozen}")
    print(f"__CUSTOM__: Frozen type: {frozen_type}")
    print(f"__CUSTOM__: Advanced augmentation: {use_advanced_aug}")
    print(f"__CUSTOM__: Multi-task learning: {use_multi_task}")
    if use_multi_task:
        print(f"__CUSTOM__: Min species samples: {min_species_samples}")
        print(f"__CUSTOM__: Loss weights - Family: {family_weight}, Genus: {genus_weight}, Species: {species_weight}")

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
        label_smoothing_factor=aug_config.get('label_smoothing', 0.0) if use_advanced_aug else 0.0,
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
        # Augmentation config
        "use_advanced_augmentation": use_advanced_aug,
        "augmentation_config": aug_config if use_advanced_aug else None,
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

    def collate_fn(examples):
        # For standard collation without mixup/cutmix
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Filter species with insufficient samples (for multi-task learning)
    if use_multi_task:
        from collections import Counter

        print(f"__CUSTOM__: Filtering species with <={min_species_samples} samples")

        # Check if multi-task columns exist
        required_columns = ['family', 'genus', 'species']
        missing_columns = [col for col in required_columns if col not in dataset_column_names]
        if missing_columns:
            raise ValueError(
                f"Multi-task learning enabled but missing required columns: {missing_columns}. "
                f"Available columns: {dataset_column_names}"
            )

        # Filter by species count
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

    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        from collections import Counter

        counts = Counter(dataset["train"][data_args.label_column_name])
        valid_labels = {l for l, c in counts.items() if c > 1}
        filtered = dataset["train"].filter(lambda x: x[data_args.label_column_name] in valid_labels)

        dataset['train'] = filtered
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=42, shuffle=False)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
        print(f"Split the dataset into train and validation with proportions {1 - data_args.train_val_split} and {data_args.train_val_split}.")
        print(f"Training split has {len(dataset['train'])} examples and validation split has {len(dataset['validation'])} examples.")

    # Prepare label mappings
    labels = dataset["train"].unique(data_args.label_column_name)
    num_labels = len(labels)
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {str(i): i for i in range(num_labels)}

    # For multi-task learning, create hierarchical label mappings
    if use_multi_task:
        print("__CUSTOM__: Creating hierarchical label mappings for multi-task learning")

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

        # Store mappings for later use
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

    # Load the accuracy metric
    accuracy_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Load f1 score metric
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)

    # Define compute_metrics based on whether multi-task learning is enabled
    if use_multi_task:

        def preprocess_logits_for_metrics(logits, labels):
            """
            For multi-task learning, logits is a tuple of (species_logits, genus_logits, family_logits)
            We return the tuple as-is for the compute_metrics function to process
            """
            return logits

        def compute_metrics(p):
            """Computes accuracy for all taxonomy levels in multi-task learning"""
            # For multi-task, predictions will be tuples of (species, genus, family) logits
            # p.predictions is typically a tuple when model returns multiple outputs
            if isinstance(p.predictions, tuple):
                species_logits, genus_logits, family_logits = p.predictions[0], p.predictions[1], p.predictions[2]
            else:
                # If model returns a single tensor, assume it's species logits (primary task)
                species_logits = p.predictions

            species_predictions = np.argmax(species_logits, axis=1) if len(species_logits.shape) > 1 else species_logits

            # Compute species accuracy (primary metric)
            species_accuracy = accuracy_metric.compute(predictions=species_predictions, references=p.label_ids)["accuracy"]
            species_f1 = f1_metric.compute(predictions=species_predictions, references=p.label_ids, average="weighted")["f1"]

            metrics = {
                "accuracy": species_accuracy,  # Primary accuracy is species
                "species_accuracy": species_accuracy,
                "species_f1": species_f1,
            }

            # If we have genus and family logits, compute their accuracies too
            if isinstance(p.predictions, tuple) and len(p.predictions) >= 3:
                genus_predictions = np.argmax(genus_logits, axis=1) if len(genus_logits.shape) > 1 else genus_logits
                family_predictions = np.argmax(family_logits, axis=1) if len(family_logits.shape) > 1 else family_logits

                # Note: For these we'd need separate label_ids, but for now just compute from species
                # This is a simplified version - in production you'd want to track these separately
                metrics["genus_predictions_available"] = True
                metrics["family_predictions_available"] = True

            return metrics
    else:
        def preprocess_logits_for_metrics(logits, labels):
            """
                logits: batch_size x num_classes
                labels: batch_size
            """
            predictions = torch.argmax(logits, dim=1)
            return predictions

        def compute_metrics(p):
            """Computes accuracy on a batch of predictions"""
            if len(p.predictions.shape) > 1: # Predictions contain logits
                predictions = np.argmax(p.predictions, axis=1)
            else: # Predictions contain label indices
                predictions = p.predictions
            accuracy = accuracy_metric.compute(predictions=predictions, references=p.label_ids)["accuracy"]
            f1_score = f1_metric.compute(predictions=predictions, references=p.label_ids, average="weighted")["f1"]

            return {
                "accuracy": accuracy,
                "f1": f1_score
            }

    # Create model based on whether multi-task learning is enabled
    if use_multi_task:
        print("__CUSTOM__: Creating multi-task SWIN model")

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

        # Wrap it in multi-task model
        model = MultiTaskSwinModel(
            base_model,
            num_families=hierarchical_mappings['num_families'],
            num_genera=hierarchical_mappings['num_genera'],
            num_species=hierarchical_mappings['num_species']
        )

        print(f"__CUSTOM__: Multi-task model created with {hierarchical_mappings['num_families']} families, "
              f"{hierarchical_mappings['num_genera']} genera, {hierarchical_mappings['num_species']} species")
    else:
        config_obj = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config_obj,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
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
            if frozen_type == "v1":
                if 'classifier' not in name and "swinv2.layernorm" not in name:
                    param.requires_grad = False
            elif frozen_type == "v4":
                if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3") and not name.startswith("swinv2.encoder.layers.2") and not name.startswith("swinv2.encoder.layers.1"):
                    param.requires_grad = False
            elif frozen_type == "v3":
                if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3") and not name.startswith("swinv2.encoder.layers.2"):
                    param.requires_grad = False
            else:
                if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3"):
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

    # Build augmentation transforms based on config
    if use_advanced_aug:
        print("__CUSTOM__: Using advanced augmentation pipeline")

        train_transform_list = [
            RandomResizedCrop(size, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
        ]

        # Add RandAugment if configured
        if aug_config.get('randaugment', {}).get('num_ops', 0) > 0:
            train_transform_list.append(
                RandAugment(
                    num_ops=aug_config['randaugment']['num_ops'],
                    magnitude=aug_config['randaugment']['magnitude']
                )
            )

        # Add ColorJitter if configured
        if aug_config.get('color_jitter', {}).get('enabled', False):
            cj = aug_config['color_jitter']
            train_transform_list.append(
                ColorJitter(
                    brightness=cj.get('brightness', 0.4),
                    contrast=cj.get('contrast', 0.4),
                    saturation=cj.get('saturation', 0.4),
                    hue=cj.get('hue', 0.1)
                )
            )

        train_transform_list.extend([ToTensor(), normalize])

        # Add RandomErasing if configured (after ToTensor)
        if aug_config.get('random_erasing', {}).get('enabled', False):
            re = aug_config['random_erasing']
            train_transform_list.append(
                RandomErasing(
                    p=re.get('probability', 0.25),
                    scale=(re.get('min_area', 0.02), re.get('max_area', 0.33)),
                )
            )

        _train_transforms = Compose(train_transform_list)
    else:
        print("__CUSTOM__: Using standard augmentation pipeline")
        _train_transforms = Compose([
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])

    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(Image.open(pil_img).convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        # Keep the label for mixup collator
        example_batch["label"] = example_batch[data_args.label_column_name]

        # Add hierarchical labels for multi-task learning
        if use_multi_task:
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

        # Keep the label for the collator/trainer
        example_batch["label"] = example_batch[data_args.label_column_name]

        # Add hierarchical labels for multi-task learning
        if use_multi_task:
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

    # Decide whether to use mixup/cutmix
    use_mixup_cutmix = (
        use_advanced_aug and
        (aug_config.get('mixup', {}).get('enabled', False) or
         aug_config.get('cutmix', {}).get('enabled', False))
    )

    # Determine number of classes for data collator
    if use_multi_task:
        collator_num_classes = hierarchical_mappings['num_species']  # Use species count
    else:
        collator_num_classes = num_labels

    if use_mixup_cutmix or use_multi_task:
        # Use mixup collator (can handle both mixup/cutmix and multi-task)
        if use_mixup_cutmix:
            print("__CUSTOM__: Using Mixup/CutMix data collator" + (" with multi-task support" if use_multi_task else ""))
        else:
            print("__CUSTOM__: Using multi-task data collator")

        data_collator = MixupCutmixCollator(
            mixup_alpha=aug_config.get('mixup', {}).get('alpha', 0.8) if use_mixup_cutmix else 0,
            cutmix_alpha=aug_config.get('cutmix', {}).get('alpha', 1.0) if use_mixup_cutmix else 0,
            prob=aug_config.get('mixup_cutmix_prob', 0.5) if use_mixup_cutmix else 0,
            label_smoothing=aug_config.get('label_smoothing', 0.1),
            num_classes=collator_num_classes,
            multi_task=use_multi_task
        )

        # Use custom trainer for mixup/cutmix loss or multi-task learning
        trainer = MixupTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if training_args.do_train else None,
            eval_dataset=dataset["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=image_processor,
            data_collator=data_collator,
            multi_task=use_multi_task,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
    else:
        # Standard trainer for single-task learning without mixup/cutmix
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if training_args.do_train else None,
            eval_dataset=dataset["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=image_processor,
            data_collator=collate_fn,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
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
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
