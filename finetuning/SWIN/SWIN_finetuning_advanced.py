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
import re
import sys
import yaml
from dataclasses import dataclass, field
from typing import Optional
import random

import math

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
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import wandb

os.environ['WANDB_DISABLED'] = 'false'  # may be overridden by config

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

    def gradient_checkpointing_enable(self, **kwargs):
        # Passthrough so HF Trainer's gradient_checkpointing flag reaches the backbone.
        self.swin.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.swin.gradient_checkpointing_disable()

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


class SubCenterArcMarginProduct(nn.Module):
    """SubCenter ArcFace margin head (k sub-centers per class for robustness to label noise)."""
    def __init__(self, in_features, out_features, k=3, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight).view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine, dim=2)  # [B, num_classes]

        if labels is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm) if not self.easy_margin else torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


class SwinWithArcFace(nn.Module):
    """
    SWIN backbone + SubCenter ArcFace species head.
    Optionally adds CE auxiliary heads for family/genus (multi-task).
    Optionally blends a CE species head with ArcFace (hybrid loss).
    """
    def __init__(self, base_model, num_species, embedding_size=512, scale=30.0, margin=0.50,
                 num_subcenters=3, num_families=None, num_genera=None,
                 family_weight=0.2, genus_weight=0.3, hybrid_ce_weight=0.0):
        super().__init__()
        self.config = base_model.config
        if hasattr(base_model, 'swinv2'):
            self.swin = base_model.swinv2
        elif hasattr(base_model, 'swin'):
            self.swin = base_model.swin
        else:
            raise ValueError("Base model must have 'swin' or 'swinv2' attribute")

        hidden_size = base_model.config.hidden_size
        self.num_species = num_species
        self.family_weight = family_weight
        self.genus_weight = genus_weight
        self.hybrid_ce_weight = hybrid_ce_weight

        self.embedding = nn.Linear(hidden_size, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.arcface = SubCenterArcMarginProduct(embedding_size, num_species, k=num_subcenters, s=scale, m=margin)

        if hybrid_ce_weight > 0:
            self.ce_classifier = nn.Linear(hidden_size, num_species)

        self.use_multi_task = num_families is not None and num_genera is not None
        if self.use_multi_task:
            self.family_classifier = nn.Linear(hidden_size, num_families)
            self.genus_classifier = nn.Linear(hidden_size, num_genera)

    def gradient_checkpointing_enable(self, **kwargs):
        # Passthrough so HF Trainer's gradient_checkpointing flag reaches the backbone.
        self.swin.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.swin.gradient_checkpointing_disable()

    def forward(self, pixel_values, labels=None, family_labels=None, genus_labels=None, species_labels=None, **kwargs):
        pooled = self.swin(pixel_values).pooler_output
        embeddings = self.bn(self.embedding(pooled))
        arc_labels = species_labels if species_labels is not None else labels

        if arc_labels is not None:
            arc_logits = self.arcface(embeddings, arc_labels)
            arc_loss = F.cross_entropy(arc_logits, arc_labels)

            if self.hybrid_ce_weight > 0:
                ce_logits = self.ce_classifier(pooled)
                ce_loss = F.cross_entropy(ce_logits, arc_labels)
                w = self.hybrid_ce_weight
                loss = (1 - w) * arc_loss + w * ce_loss
                logits = torch.log((1 - w) * F.softmax(arc_logits, dim=1) + w * F.softmax(ce_logits, dim=1) + 1e-8)
            else:
                loss = arc_loss
                logits = arc_logits

            if self.use_multi_task and family_labels is not None and genus_labels is not None:
                loss = loss + self.family_weight * F.cross_entropy(self.family_classifier(pooled), family_labels)
                loss = loss + self.genus_weight * F.cross_entropy(self.genus_classifier(pooled), genus_labels)
        else:
            # Inference: cosine similarity, no margin
            weight = F.normalize(self.arcface.weight, p=2, dim=1)
            emb = F.normalize(embeddings, p=2, dim=1)
            cosine = F.linear(emb, weight).view(-1, self.num_species, self.arcface.k)
            cosine, _ = torch.max(cosine, dim=2)
            arc_logits = cosine * self.arcface.s

            if self.hybrid_ce_weight > 0:
                ce_logits = self.ce_classifier(pooled)
                w = self.hybrid_ce_weight
                logits = torch.log((1 - w) * F.softmax(arc_logits, dim=1) + w * F.softmax(ce_logits, dim=1) + 1e-8)
            else:
                logits = arc_logits
            loss = None

        result = {'loss': loss, 'logits': logits}
        if self.use_multi_task:
            result['family_logits'] = self.family_classifier(pooled)
            result['genus_logits'] = self.genus_classifier(pooled)
        return result


class MixupCutmixCollator:
    """
    Collator that applies Mixup and/or Cutmix augmentation.
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, label_smoothing=0.1, num_classes=1000, multi_task=False, label_column_name="label"):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.multi_task = multi_task
        self.label_column_name = label_column_name

    def __call__(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])

        # Check if "label" key exists (training) or use direct label access (validation)
        # During training, transforms set "label" key; during validation, they don't
        if "label" in examples[0]:
            labels = torch.tensor([example["label"] for example in examples])
        else:
            # Validation/evaluation — no mixup/cutmix, just collate cleanly
            result = {
                "pixel_values": pixel_values,
                "labels": torch.tensor([example[self.label_column_name] for example in examples]),
            }

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
    def __init__(self, *args, multi_task=False, arcface=False,
                 logit_adjustment=False, log_prior=None, logit_adjustment_tau=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_task = multi_task
        self.arcface = arcface
        # Balanced-softmax / logit adjustment (Tier 1.3-A). log_prior is a
        # [num_species] tensor of log class frequencies; added to the species
        # logits during TRAINING only (never at inference) to down-weight head
        # classes and lift macro-F1 on the long tail.
        self.logit_adjustment = logit_adjustment
        self.log_prior = log_prior
        self.logit_adjustment_tau = logit_adjustment_tau

    def _adjust_logits(self, logits):
        """Add tau * log_prior to species logits (balanced softmax). No-op if disabled."""
        if self.log_prior is None:
            return logits
        return logits + self.logit_adjustment_tau * self.log_prior.to(logits.device, logits.dtype)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels_a = inputs.pop("labels")
        labels_b = inputs.pop("labels_b", None)
        lam = inputs.pop("lam", 1.0)

        smoothing = getattr(self.data_collator, 'label_smoothing', 0.0)

        # ArcFace: model computes its own loss internally (no mixup/cutmix with ArcFace)
        if self.arcface:
            if self.multi_task:
                family_labels = inputs.pop("family_labels", None)
                genus_labels = inputs.pop("genus_labels", None)
                species_labels = inputs.pop("species_labels", None)
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    species_labels=species_labels,
                    family_labels=family_labels,
                    genus_labels=genus_labels,
                )
            else:
                outputs = model(pixel_values=inputs["pixel_values"], labels=labels_a)
            loss = outputs.get("loss")
            return (loss, outputs) if return_outputs else loss

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
                loss_fct = nn.CrossEntropyLoss(label_smoothing=smoothing)

                # Get logits for each taxonomy level
                family_logits = outputs.get("family_logits")
                genus_logits = outputs.get("genus_logits")
                # Balanced softmax: adjust species logits only (the long-tailed target)
                species_logits = self._adjust_logits(outputs.get("species_logits"))

                # Compute mixed losses
                family_loss = lam * loss_fct(family_logits, family_labels) + (1 - lam) * loss_fct(family_logits, family_labels_b)
                genus_loss = lam * loss_fct(genus_logits, genus_labels) + (1 - lam) * loss_fct(genus_logits, genus_labels_b)
                species_loss = lam * loss_fct(species_logits, species_labels) + (1 - lam) * loss_fct(species_logits, species_labels_b)

                # Combined loss with same weighting as the model
                loss = species_loss + 0.3 * genus_loss + 0.2 * family_loss
            elif self.logit_adjustment:
                # No mixup this batch, but balanced softmax is on: recompute the
                # combined loss in-trainer so the species logits get the log-prior
                # (the model's internal loss does not apply it).
                loss_fct = nn.CrossEntropyLoss(label_smoothing=smoothing)
                species_logits = self._adjust_logits(outputs.get("species_logits"))
                species_loss = loss_fct(species_logits, species_labels)
                genus_loss = loss_fct(outputs.get("genus_logits"), genus_labels)
                family_loss = loss_fct(outputs.get("family_logits"), family_labels)
                loss = species_loss + 0.3 * genus_loss + 0.2 * family_loss
            else:
                # Model already computed the loss
                loss = outputs.get("loss")

            return (loss, outputs) if return_outputs else loss
        else:
            # Standard single-task training
            outputs = model(pixel_values=inputs["pixel_values"])
            logits = self._adjust_logits(outputs.get("logits"))  # balanced softmax (no-op if disabled)
            loss_fct = nn.CrossEntropyLoss(label_smoothing=smoothing)

            if labels_b is not None:
                # Mixup/Cutmix loss
                loss = lam * loss_fct(logits, labels_a) + (1 - lam) * loss_fct(logits, labels_b)
            else:
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
                if self.arcface:
                    # ArcFace inference: no labels → cosine similarity logits (no margin)
                    outputs = model(pixel_values=inputs["pixel_values"])
                elif self.multi_task:
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


class EMACallback(TrainerCallback):
    """
    Exponential Moving Average of model weights (Tier 2.6).

    Keeps a shadow copy of the trainable parameters, updated every optimizer
    step as shadow = decay*shadow + (1-decay)*param. At the end of training the
    EMA weights are copied into the model, so the final `trainer.evaluate()` and
    `trainer.save_model()` both reflect the averaged weights (typically a steady
    +0.2-0.5% for ~free). Only parameters are averaged; buffers (e.g. BN running
    stats) are left as-is.

    Note: do not combine with `load_best_model_at_end: true` — the best-checkpoint
    reload happens before this callback and would be overwritten by the EMA copy.
    """
    def __init__(self, decay=0.9998):
        self.decay = decay
        self.shadow = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.shadow = {
            n: p.detach().clone().float()
            for n, p in model.named_parameters() if p.requires_grad
        }
        print(f"__CUSTOM__: EMA enabled (decay={self.decay}); tracking {len(self.shadow)} parameter tensors")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.shadow is None:
            return
        d = self.decay
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def copy_to_model(self, model):
        if self.shadow is None:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        print("__CUSTOM__: Copying EMA weights into model for final eval/save")
        self.copy_to_model(model)


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_multi_crop_transforms(crop_sizes, target_size, image_mean, image_std, flip=False):
    """
    Returns Compose transforms for multi-crop TTA (Tier 2.7).

    One transform per crop size; if `flip` is True, also emit a horizontally
    flipped variant of each crop, so logits are averaged over crops x {orig, flip}.
    """
    norm = Normalize(mean=image_mean, std=image_std)
    transforms = []
    for crop_size in crop_sizes:
        base = [Resize(crop_size), CenterCrop(target_size)]
        transforms.append(Compose(base + [ToTensor(), norm]))
        if flip:
            transforms.append(Compose(base + [RandomHorizontalFlip(p=1.0), ToTensor(), norm]))
    return transforms


def multi_crop_evaluate(model, filepaths, labels, crop_transforms, device, compute_metrics_fn):
    """Runs multi-crop (TTA) inference on the validation set and prints accuracy/F1."""
    from transformers.trainer_utils import EvalPrediction
    model.eval()
    batch_size = 32
    all_predictions = []

    for i in range(0, len(filepaths), batch_size):
        batch_fps = filepaths[i:i + batch_size]
        images = [Image.open(fp).convert("RGB") for fp in batch_fps]

        summed_logits = None
        with torch.no_grad():
            for transform in crop_transforms:
                pixel_values = torch.stack([transform(img) for img in images]).to(device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.get("logits")
                if logits is None:
                    logits = outputs.get("species_logits")
                summed_logits = logits if summed_logits is None else summed_logits + logits

        avg_logits = summed_logits / len(crop_transforms)
        preds = torch.argmax(avg_logits, dim=-1).cpu().numpy()
        all_predictions.extend(preds.tolist())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(labels)

    eval_pred = EvalPrediction(predictions=all_predictions, label_ids=all_labels)
    metrics = compute_metrics_fn(eval_pred)

    print("__CUSTOM__: Multi-crop eval results:")
    for k, v in metrics.items():
        print(f"__CUSTOM__: Multi-crop eval {k}: {v}")

    return metrics


def _resolve_num_workers(n):
    """Return n, or all scheduler-allocated CPUs when n == -1."""
    if n == -1:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 8
    return n


def _relocate_output_dir(path):
    """
    Re-root an output/logging path to the workspace this script is actually
    running from, instead of whoever authored the config.

    Configs in this repo hardcode paths like
    /projectnb/herbdl/workspaces/<author>/herbdl/finetuning/output/SWIN/<NAME>.
    When a different user runs the same config, rewrite the
    `.../workspaces/<author>/herbdl` prefix to this checkout's repo root so the
    run is written under the runner's own workspace rather than the author's.
    The trailing run name (.../output/SWIN/<NAME>) is preserved. No-op if the
    path doesn't match that layout or is already under this repo. Set
    HERBDL_NO_RELOCATE=1 to disable (e.g. to write elsewhere on purpose).
    """
    if not isinstance(path, str) or not path or os.environ.get('HERBDL_NO_RELOCATE'):
        return path
    # repo root = three levels up from this file: .../herbdl/finetuning/SWIN/<file>
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    m = re.match(r'^(.*/workspaces/[^/]+/herbdl)(/.*)?$', path)
    if not m:
        return path
    relocated = repo_root + (m.group(2) or '')
    if relocated != path:
        print(f"__CUSTOM__: Relocated output path to current workspace:\n"
              f"             from {path}\n               to {relocated}")
    return relocated


def main():
    # Parse command line arguments for config file
    arg_parser = argparse.ArgumentParser(description="SWIN Fine-tuning with advanced augmentations")
    arg_parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    arg_parser.add_argument('--set', metavar='KEY=VALUE', action='append', default=[],
                            help='Override a config value using dotted key notation, e.g. training.seed=42')
    args = arg_parser.parse_args()

    # Load YAML config
    config = load_config_from_yaml(args.config)

    # Apply --set overrides
    def _coerce(v):
        for cast in (int, float):
            try: return cast(v)
            except ValueError: pass
        if v.lower() in ('true', 'false'):
            return v.lower() == 'true'
        return v

    for override in args.set:
        key, _, value = override.partition('=')
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = _coerce(value)
        print(f"Config override: {key} = {d[parts[-1]]!r}")

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

    # Extract multi-crop testing parameters
    multi_crop_config = config.get('multi_crop', {})
    multi_crop_enabled = multi_crop_config.get('enabled', False)
    multi_crop_sizes = multi_crop_config.get('crop_sizes', [256, 288, 320, 384, 448])
    multi_crop_target_size = multi_crop_config.get('target_size', 224)
    multi_crop_flip = multi_crop_config.get('flip', False)

    # Extract long-tail (balanced softmax / logit adjustment) parameters — Tier 1.3-A
    long_tail_config = config.get('long_tail', {})
    use_logit_adjustment = long_tail_config.get('logit_adjustment', False)
    logit_adjustment_tau = long_tail_config.get('tau', 1.0)

    # Extract EMA parameters — Tier 2.6
    ema_config = config.get('ema', {})
    use_ema = ema_config.get('enabled', False)
    ema_decay = ema_config.get('decay', 0.9998)

    # Extract multi-task learning parameters
    multi_task_config = config.get('multi_task', {})
    use_multi_task = multi_task_config.get('enabled', False)
    min_species_samples = multi_task_config.get('min_species_samples', 2)
    family_weight = multi_task_config.get('family_weight', 0.2)
    genus_weight = multi_task_config.get('genus_weight', 0.3)
    species_weight = multi_task_config.get('species_weight', 1.0)

    # Extract ArcFace parameters
    arcface_config = config.get('arcface', {})
    use_arcface = arcface_config.get('enabled', False)
    arcface_embedding_size = arcface_config.get('embedding_size', 512)
    arcface_scale = arcface_config.get('scale', 30.0)
    arcface_margin = arcface_config.get('margin', 0.50)
    arcface_num_subcenters = arcface_config.get('num_subcenters', 3)
    arcface_hybrid_ce_weight = arcface_config.get('hybrid_ce_weight', 0.0)

    print(f"__CUSTOM__: Learning rate type: {learning_rate_type}")
    print(f"__CUSTOM__: Frozen: {frozen}")
    print(f"__CUSTOM__: Frozen type: {frozen_type}")
    print(f"__CUSTOM__: Advanced augmentation: {use_advanced_aug}")
    print(f"__CUSTOM__: Multi-task learning: {use_multi_task}")
    print(f"__CUSTOM__: ArcFace: {use_arcface}")
    print(f"__CUSTOM__: Logit adjustment (balanced softmax): {use_logit_adjustment} (tau={logit_adjustment_tau})")
    print(f"__CUSTOM__: Weight EMA: {use_ema} (decay={ema_decay})")
    if use_multi_task:
        print(f"__CUSTOM__: Min species samples: {min_species_samples}")
        print(f"__CUSTOM__: Loss weights - Family: {family_weight}, Genus: {genus_weight}, Species: {species_weight}")
    if use_arcface:
        print(f"__CUSTOM__: ArcFace embedding_size={arcface_embedding_size}, scale={arcface_scale}, margin={arcface_margin}, k={arcface_num_subcenters}, hybrid_ce_weight={arcface_hybrid_ce_weight}")

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
        output_dir=_relocate_output_dir(config['training']['output_dir']),
        logging_dir=_relocate_output_dir(config['training']['logging_dir']),
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        learning_rate=float(config['training']['learning_rate']),
        num_train_epochs=config['training']['num_train_epochs'],
        warmup_steps=config['training'].get('warmup_steps', 0),
        weight_decay=config['training']['weight_decay'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        logging_strategy=config['training']['logging_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training'].get('eval_steps', None),  # only used when eval_strategy == "steps"
        report_to=config['training']['report_to'] if config['wandb'].get('enabled', True) else 'none',
        bf16=config['training']['bf16'],
        dataloader_num_workers=_resolve_num_workers(config['training']['dataloader_num_workers']),
        dataloader_pin_memory=config['training'].get('dataloader_pin_memory', True),
        remove_unused_columns=config['training']['remove_unused_columns'],
        seed=config['training']['seed'],
        label_smoothing_factor=aug_config.get('label_smoothing', 0.0) if use_advanced_aug else 0.0,
        eval_on_start=config['training'].get('eval_on_start', False),
        torch_compile=config['training'].get('torch_compile', False),
        gradient_checkpointing=config['training'].get('gradient_checkpointing', False),
        gradient_checkpointing_kwargs=config['training'].get('gradient_checkpointing_kwargs', None),
        load_best_model_at_end=config['training'].get('load_best_model_at_end', False),
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Initialize wandb with complete config
    if config['wandb'].get('enabled', True):
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
            config=wandb_config,
            notes=config['custom']['run_notes']
        )
    else:
        os.environ['WANDB_DISABLED'] = 'true'

    # Set the learning rate scheduler parameters from config
    if 'lr_scheduler_kwargs' in config['training'] and config['training']['lr_scheduler_kwargs']:
        training_args.lr_scheduler_kwargs = config['training']['lr_scheduler_kwargs']

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
    overwrite_output_dir = config['training'].get('overwrite_output_dir', False)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Set overwrite_output_dir: true in your config to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `output_dir` or set `overwrite_output_dir: true` in your config to train from scratch."
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
            family_predictions = np.argmax(family_logits, axis=1) if 'family_logits' in locals() and len(family_logits.shape) > 1 else None
            genus_predictions = np.argmax(genus_logits, axis=1) if 'genus_logits' in locals() and len(genus_logits.shape) > 1 else None

            # Compute species accuracy (primary metric)
            species_accuracy = accuracy_metric.compute(predictions=species_predictions, references=p.label_ids)["accuracy"]
            species_f1 = f1_metric.compute(predictions=species_predictions, references=p.label_ids, average="macro")["f1"]

            family_accuracy = accuracy_metric.compute(predictions=family_predictions, references=p.label_ids)["accuracy"] if family_predictions is not None else None
            genus_accuracy = accuracy_metric.compute(predictions=genus_predictions, references=p.label_ids)["accuracy"] if genus_predictions is not None else None

            family_f1 = f1_metric.compute(predictions=family_predictions, references=p.label_ids, average="macro")["f1"] if family_predictions is not None else None
            genus_f1 = f1_metric.compute(predictions=genus_predictions, references=p.label_ids, average="macro")["f1"] if genus_predictions is not None else None

            metrics = {
                "accuracy": species_accuracy,  # Primary accuracy is species
                "species_accuracy": species_accuracy,
                "species_f1": species_f1,
                "family_accuracy": family_accuracy,
                "genus_accuracy": genus_accuracy,
                "family_f1": family_f1,
                "genus_f1": genus_f1
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
            f1_score = f1_metric.compute(predictions=predictions, references=p.label_ids, average="macro")["f1"]

            return {
                "accuracy": accuracy,
                "f1": f1_score
            }

    # Long-tail: per-class log-prior for balanced softmax (Tier 1.3-A).
    # Computed once over the (already filtered/split) training set, in the SAME
    # class-index space the species/CE head outputs, so it lines up with the logits.
    log_prior = None
    if use_logit_adjustment:
        from collections import Counter
        if use_multi_task:
            sp2id = hierarchical_mappings['species2id']
            cnt = Counter(sp2id[s] for s in dataset["train"]["species"])
            n_cls = hierarchical_mappings['num_species']
        else:
            cnt = Counter(dataset["train"][data_args.label_column_name])
            n_cls = num_labels
        freq = np.array([cnt.get(i, 0) for i in range(n_cls)], dtype=np.float64)
        freq = freq / max(freq.sum(), 1.0)
        freq = np.clip(freq, 1e-12, None)  # floor empty classes so log() is finite
        log_prior = torch.tensor(np.log(freq), dtype=torch.float32)
        print(f"__CUSTOM__: Balanced softmax log-prior built over {n_cls} classes "
              f"(min/max log-prior = {log_prior.min().item():.3f}/{log_prior.max().item():.3f})")

    # Create model based on which objectives are enabled
    if use_arcface:
        print("__CUSTOM__: Creating SwinWithArcFace model")
        arc_num_species = hierarchical_mappings['num_species'] if use_multi_task else num_labels
        config_obj = AutoConfig.from_pretrained(
            model_args.config_name or model_args.model_name_or_path,
            num_labels=arc_num_species,
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
        model = SwinWithArcFace(
            base_model,
            num_species=arc_num_species,
            embedding_size=arcface_embedding_size,
            scale=arcface_scale,
            margin=arcface_margin,
            num_subcenters=arcface_num_subcenters,
            num_families=hierarchical_mappings['num_families'] if use_multi_task else None,
            num_genera=hierarchical_mappings['num_genera'] if use_multi_task else None,
            family_weight=family_weight,
            genus_weight=genus_weight,
            hybrid_ce_weight=arcface_hybrid_ce_weight,
        )
        print(f"__CUSTOM__: SwinWithArcFace created — num_species={arc_num_species}, "
              f"multi_task={use_multi_task}, hybrid_ce_weight={arcface_hybrid_ce_weight}")
        # Overlay non-backbone weights from checkpoint (preserves embedding/arcface/CE heads
        # when chaining ArcFace→Hybrid or ArcFace→384, since AutoModelForImageClassification
        # only maps swin.* keys and discards the custom heads).
        _ckpt_dir = model_args.model_name_or_path
        if os.path.isdir(_ckpt_dir):
            _st_path = os.path.join(_ckpt_dir, 'model.safetensors')
            _bin_path = os.path.join(_ckpt_dir, 'pytorch_model.bin')
            if os.path.exists(_st_path) or os.path.exists(_bin_path):
                try:
                    if os.path.exists(_st_path):
                        from safetensors.torch import load_file as _load_st
                        _ckpt_sd = _load_st(_st_path)
                    else:
                        _ckpt_sd = torch.load(_bin_path, map_location='cpu')
                    # Only load keys that are NOT backbone (swin.*) — backbone already loaded
                    # and handles window-size mismatches (e.g. 224→384) via ignore_mismatched_sizes.
                    _non_backbone = {k: v for k, v in _ckpt_sd.items() if not k.startswith('swin.')}
                    _res = model.load_state_dict(_non_backbone, strict=False)
                    _loaded = len(_non_backbone) - len(_res.missing_keys)
                    print(f"__CUSTOM__: Overlaid {_loaded}/{len(_non_backbone)} non-backbone weights from checkpoint")
                except Exception as _e:
                    print(f"__CUSTOM__: Could not overlay non-backbone weights: {_e}")

    elif use_multi_task:
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
        # Save raw filepaths/labels before set_transform for multi-crop eval
        if multi_crop_enabled:
            eval_filepaths = list(dataset["validation"][data_args.image_column_name])
            eval_labels = list(dataset["validation"][data_args.label_column_name])
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

    if use_mixup_cutmix or use_multi_task or use_arcface or use_logit_adjustment:
        # Use mixup collator (can handle mixup/cutmix, multi-task, and arcface).
        # Also used for plain single-task + balanced softmax (mixup disabled), since
        # the logit adjustment lives in MixupTrainer.compute_loss.
        if use_mixup_cutmix:
            print("__CUSTOM__: Using Mixup/CutMix data collator" + (" with multi-task support" if use_multi_task else ""))
        elif use_arcface:
            print("__CUSTOM__: Using ArcFace data collator" + (" with multi-task support" if use_multi_task else ""))
        elif use_multi_task:
            print("__CUSTOM__: Using multi-task data collator")
        else:
            print("__CUSTOM__: Using data collator (balanced softmax, no mixup)")

        data_collator = MixupCutmixCollator(
            mixup_alpha=aug_config.get('mixup', {}).get('alpha', 0.8) if use_mixup_cutmix else 0,
            cutmix_alpha=aug_config.get('cutmix', {}).get('alpha', 1.0) if use_mixup_cutmix else 0,
            prob=aug_config.get('mixup_cutmix_prob', 0.5) if use_mixup_cutmix else 0,
            label_smoothing=aug_config.get('label_smoothing', 0.1),
            num_classes=collator_num_classes,
            multi_task=use_multi_task,
            label_column_name=data_args.label_column_name,
        )

        # Use custom trainer for mixup/cutmix loss, multi-task learning, or arcface
        trainer = MixupTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"] if training_args.do_train else None,
            eval_dataset=dataset["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            processing_class=image_processor,
            data_collator=data_collator,
            multi_task=use_multi_task,
            arcface=use_arcface,
            logit_adjustment=use_logit_adjustment,
            log_prior=log_prior,
            logit_adjustment_tau=logit_adjustment_tau,
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
            processing_class=image_processor,
            data_collator=collate_fn,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

    # Weight EMA (Tier 2.6): copies averaged weights into the model at train end,
    # so the final evaluate()/save_model() below reflect the EMA weights.
    if use_ema:
        trainer.add_callback(EMACallback(decay=ema_decay))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # Ensure config.json is always present for custom (non-PreTrainedModel) wrappers
        # so that downstream stages can call AutoConfig.from_pretrained on this directory.
        if hasattr(model, 'config') and not isinstance(model, transformers.PreTrainedModel):
            model.config.save_pretrained(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Multi-crop evaluation (test-time augmentation)
    if multi_crop_enabled and training_args.do_eval:
        print("__CUSTOM__: Running multi-crop (TTA) evaluation")
        crop_transforms = build_multi_crop_transforms(
            crop_sizes=multi_crop_sizes,
            target_size=multi_crop_target_size,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            flip=multi_crop_flip,
        )
        multi_crop_evaluate(
            model=model,
            filepaths=eval_filepaths,
            labels=eval_labels,
            crop_transforms=crop_transforms,
            device=training_args.device,
            compute_metrics_fn=compute_metrics,
        )

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
