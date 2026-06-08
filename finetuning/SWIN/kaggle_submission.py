#!/usr/bin/env python3
"""
Kaggle Herbarium 2022 submission script.

Pipeline:
  model output index -> label_mapping_15k.json -> 'Family Genus species'
                     -> train_metadata categories  -> category_id
                     -> submission CSV (Id, Predicted)

Usage:
  python kaggle_submission.py --checkpoint <path> [--output submission.csv] [--batch_size 64]
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TEST_METADATA  = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/test_metadata.json"
TEST_IMAGE_DIR = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/test_images"
TRAIN_METADATA = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_metadata.json"
LABEL_MAPPING  = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/SWIN/label_mapping_15k.json"


class MultiTaskSwinModel(nn.Module):
    """Mirrors the definition used during training — needed to load multitask checkpoints."""
    def __init__(self, base_model, num_families, num_genera, num_species):
        super().__init__()
        if hasattr(base_model, 'swinv2'):
            self.swin = base_model.swinv2
        elif hasattr(base_model, 'swin'):
            self.swin = base_model.swin
        else:
            raise ValueError("Base model must have 'swin' or 'swinv2' attribute")
        hidden_size = base_model.config.hidden_size
        self.family_classifier = nn.Linear(hidden_size, num_families)
        self.genus_classifier   = nn.Linear(hidden_size, num_genera)
        self.species_classifier = nn.Linear(hidden_size, num_species)

    def forward(self, pixel_values, **kwargs):
        pooled = self.swin(pixel_values).pooler_output
        return {
            'logits':          self.species_classifier(pooled),
            'species_logits':  self.species_classifier(pooled),
            'genus_logits':    self.genus_classifier(pooled),
            'family_logits':   self.family_classifier(pooled),
        }


class TestDataset(Dataset):
    def __init__(self, records, image_dir, transform):
        self.records   = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = os.path.join(self.image_dir, rec['file_name'])
        try:
            img = Image.open(path).convert("RGB")
            pixel_values = self.transform(img)
            failed = False
        except Exception:
            # Return a black image placeholder; will be assigned default category later
            size = self.transform.transforms[1].size  # CenterCrop size
            if isinstance(size, int):
                size = (size, size)
            pixel_values = torch.zeros(3, *size)
            failed = True
        return {
            'pixel_values': pixel_values,
            'image_id': rec['image_id'],
            'failed': failed,
        }


def build_transforms(image_processor):
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    return Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--output',     default='submission.csv', help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--multitask',  action='store_true',
                        help='Set if checkpoint is a MultiTaskSwinModel')
    args = parser.parse_args()

    # ── Label mappings ────────────────────────────────────────────────────────
    logger.info("Building label mappings...")

    with open(LABEL_MAPPING) as f:
        idx_to_name = json.load(f)                     # "0" -> "Family Genus species"

    with open(TRAIN_METADATA) as f:
        train_meta = json.load(f)
    name_to_catid = {
        f"{c['family']} {c['genus']} {c['species']}": c['category_id']
        for c in train_meta['categories']
    }

    # Compose: model output index -> category_id
    idx_to_catid = {
        int(k): name_to_catid[v]
        for k, v in idx_to_name.items()
        if v in name_to_catid
    }
    assert len(idx_to_catid) == 15501, f"Expected 15501 entries, got {len(idx_to_catid)}"
    logger.info(f"Label mapping: {len(idx_to_catid)} classes verified")

    # ── Test metadata ─────────────────────────────────────────────────────────
    with open(TEST_METADATA) as f:
        test_records = json.load(f)      # list of {image_id, file_name, license}
    logger.info(f"Test images: {len(test_records):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.checkpoint}")
    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)

    if args.multitask:
        config = AutoConfig.from_pretrained(args.checkpoint, num_labels=15501)
        base = AutoModelForImageClassification.from_pretrained(
            args.checkpoint, config=config, ignore_mismatched_sizes=True)
        # Dummy counts — actual head weights are loaded from state dict
        model = MultiTaskSwinModel(base, num_families=1, num_genera=1, num_species=15501)
        sd = torch.load(os.path.join(args.checkpoint, 'model.safetensors'),
                        map_location='cpu', weights_only=True)
        model.load_state_dict(sd, strict=False)
    else:
        model = AutoModelForImageClassification.from_pretrained(
            args.checkpoint, ignore_mismatched_sizes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info(f"Model on {device}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    transform   = build_transforms(image_processor)
    dataset     = TestDataset(test_records, TEST_IMAGE_DIR, transform)
    dataloader  = DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference...")
    results  = {}   # image_id -> category_id
    failed   = []   # image_ids that couldn't be loaded
    DEFAULT_CATID = idx_to_catid[0]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(pixel_values=pixel_values)
            logits  = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            preds   = torch.argmax(logits, dim=-1).cpu().tolist()

            for img_id, pred_idx, is_failed in zip(
                    batch['image_id'], preds, batch['failed']):
                if is_failed:
                    results[img_id] = DEFAULT_CATID
                    failed.append(img_id)
                else:
                    results[img_id] = idx_to_catid[pred_idx]

    if failed:
        logger.warning(f"{len(failed)} images failed to load and got default category {DEFAULT_CATID}")

    # ── Write submission CSV ──────────────────────────────────────────────────
    missing = [r['image_id'] for r in test_records if r['image_id'] not in results]
    if missing:
        logger.error(f"{len(missing)} image_ids have no prediction — filling with default")
        for img_id in missing:
            results[img_id] = DEFAULT_CATID

    logger.info(f"Writing {args.output} ({len(test_records):,} rows)...")
    with open(args.output, 'w') as f:
        f.write("Id,Predicted\n")
        for rec in test_records:
            img_id = rec['image_id']
            f.write(f"{img_id},{results[img_id]}\n")

    logger.info(f"Done — {len(results):,} predictions written to {args.output}")


if __name__ == "__main__":
    main()
