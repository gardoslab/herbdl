# Advanced Training Techniques - Implementation Guide

This document explains how to implement the advanced techniques from the herbaria competition winning solution.

## Overview of Techniques

The progression shows incremental improvements:
1. **Baseline**: SWIN-B 224 → 0.77223
2. **Multi-task learning**: +cross-entropy on family/genus/species → 0.7829 (+1.06%)
3. **Higher LR**: 2e-4 → 5e-4 → 0.79444 (+1.54%)
4. **Multi-crop testing**: 5-crop ensemble → 0.80088 (+0.64%)
5. **SubCenter ArcFace**: Replace CE loss → 0.81532 (+1.44%)
6. **Hybrid loss**: CE + ArcFace → 0.82201 (+0.67%)
7. **Better augmentation**: SWIN-Transformer augmentation → 0.8315 (+0.95%)
8. **Higher resolution**: 224 → 384 → 0.8468 (+1.53%)
9. **Multi-crop 384**: 5-crop at higher res → 0.84895 (+0.22%)
10. **Layer freezing**: Gradual unfreezing → 0.85499 (+0.60%)
11. **Square resize**: Better aspect ratio → 0.85563 (+0.06%)
12. **SWIN V2**: Architecture upgrade → 0.85851 (+0.29%)

---

## 1. Multi-Task Learning (Family/Genus/Species)

### What it does:
Instead of only predicting species, the model also learns to predict taxonomic hierarchy (family, genus, species). This provides auxiliary supervision that helps the model learn better features.

### Requirements:
- Your dataset needs `family`, `genus`, and `species` columns
- Need to create label encodings for each taxonomy level
- Modify model architecture to have 3 classification heads

### Implementation:

#### Step 1: Data preparation
Your JSON files need these fields:
```json
{
  "filepath": "/path/to/image.jpg",
  "scientificNameEncoded": 42,
  "family": "Asteraceae",
  "familyEncoded": 5,
  "genus": "Helianthus",
  "genusEncoded": 120,
  "species": "Helianthus annuus",
  "speciesEncoded": 42
}
```

#### Step 2: Create custom model class
You'll need to create a custom SWIN model with multiple heads:

```python
class MultiTaskSwinModel(nn.Module):
    def __init__(self, base_model, num_families, num_genera, num_species):
        super().__init__()
        self.swin = base_model.swin  # or base_model.swinv2

        # Get hidden size
        hidden_size = base_model.config.hidden_size

        # Three separate classification heads
        self.family_classifier = nn.Linear(hidden_size, num_families)
        self.genus_classifier = nn.Linear(hidden_size, num_genera)
        self.species_classifier = nn.Linear(hidden_size, num_species)

    def forward(self, pixel_values, family_labels=None, genus_labels=None, species_labels=None):
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

            # Weighted combination (you can tune these weights)
            loss = species_loss + 0.3 * genus_loss + 0.2 * family_loss

        return {
            'loss': loss,
            'species_logits': species_logits,
            'genus_logits': genus_logits,
            'family_logits': family_logits
        }
```

**Complexity**: Medium - Requires dataset modification and custom model
**Expected gain**: ~1%

---

## 2. Multi-Crop Testing (Test-Time Augmentation)

### What it does:
During inference, create multiple crops of the image at different scales, run inference on each, and average the predictions. This improves robustness.

### For 224 resolution:
- Resize to: 256, 288, 320, 384, 448
- Center crop each to 224
- Average predictions from 5 crops

### For 384 resolution:
- Resize to: 400, 416, 448, 480, 512
- Center crop each to 384
- Average predictions from 5 crops

### Implementation:

```python
def multi_crop_inference(model, image_path, image_processor, crop_sizes, target_size):
    """
    Args:
        model: Trained model
        image_path: Path to image
        image_processor: SWIN image processor
        crop_sizes: List of sizes to resize to (e.g., [256, 288, 320, 384, 448])
        target_size: Final crop size (e.g., 224 or 384)
    """
    from PIL import Image
    import torch

    img = Image.open(image_path).convert('RGB')
    all_logits = []

    for size in crop_sizes:
        # Resize image
        resized = img.resize((size, size), Image.BICUBIC)

        # Center crop to target size
        left = (size - target_size) // 2
        top = (size - target_size) // 2
        cropped = resized.crop((left, top, left + target_size, top + target_size))

        # Process and predict
        inputs = image_processor(cropped, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            all_logits.append(outputs.logits)

    # Average predictions
    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    return avg_logits
```

**Complexity**: Easy - Only affects inference
**Expected gain**: ~0.6-0.9%
**Note**: This is ONLY for testing/evaluation, not training

---

## 3. SubCenter ArcFace Loss

### What it does:
ArcFace is a metric learning loss that learns more discriminative features by adding an angular margin in the feature space. SubCenter variant is more robust to noisy labels.

### Why it helps:
- Standard cross-entropy: Learns to separate classes in output space
- ArcFace: Learns to separate classes with angular margin in embedding space
- Better for fine-grained recognition with many classes

### Implementation:

This is COMPLEX and requires:

1. **Install dependency**:
```bash
pip install pytorch-metric-learning
```

2. **Create ArcFace head**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SubCenterArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, k=3, s=30.0, m=0.50, easy_margin=False):
        """
        Args:
            in_features: Size of input features
            out_features: Number of classes
            k: Number of sub-centers per class
            s: Scale parameter
            m: Margin parameter
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.k = k

        # Weight matrix: [out_features * k, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, in_features]
            labels: [batch_size]
        """
        # Normalize features and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: [batch_size, out_features * k]
        cosine = F.linear(embeddings, weight)

        # Reshape to [batch_size, out_features, k]
        cosine = cosine.view(-1, self.out_features, self.k)

        # Take max over sub-centers
        cosine, _ = torch.max(cosine, dim=2)  # [batch_size, out_features]

        # Add margin only to correct class
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
```

3. **Modify model architecture**:
```python
class SwinWithArcFace(nn.Module):
    def __init__(self, base_model, num_classes, embedding_size=512):
        super().__init__()
        self.swin = base_model.swin
        hidden_size = base_model.config.hidden_size

        # Embedding layer (bottleneck)
        self.embedding = nn.Linear(hidden_size, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        # ArcFace head
        self.arcface = SubCenterArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            k=3,  # 3 sub-centers
            s=30.0,
            m=0.50
        )

    def forward(self, pixel_values, labels=None):
        outputs = self.swin(pixel_values)
        pooled = outputs.pooler_output

        # Get embeddings
        embeddings = self.bn(self.embedding(pooled))

        if labels is not None:
            logits = self.arcface(embeddings, labels)
            loss = F.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits}
        else:
            # For inference, use cosine similarity
            weight = F.normalize(self.arcface.weight, p=2, dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            logits = F.linear(embeddings, weight) * self.arcface.s
            return {'logits': logits}
```

**Complexity**: Hard - Requires significant architecture changes
**Expected gain**: ~1.4%

---

## 4. Dynamic Margins for ArcFace

### What it does:
Adjust the margin `m` based on class difficulty. Classes with fewer samples or harder to distinguish get larger margins.

### Implementation:
```python
def compute_dynamic_margins(label_counts, min_margin=0.3, max_margin=0.7):
    """
    Compute per-class margins based on sample counts
    Classes with fewer samples get larger margins
    """
    margins = []
    max_count = max(label_counts)
    for count in label_counts:
        # Inverse relationship: fewer samples = larger margin
        margin = max_margin - (count / max_count) * (max_margin - min_margin)
        margins.append(margin)
    return torch.tensor(margins)
```

**Complexity**: Medium
**Expected gain**: Additional ~0.5% on top of ArcFace

---

## 5. Hybrid Loss (CE + ArcFace)

### What it does:
Combine cross-entropy and ArcFace losses. ArcFace learns discriminative embeddings, CE provides stable gradients.

### Implementation:
```python
class HybridModel(nn.Module):
    def __init__(self, base_model, num_classes, embedding_size=512):
        super().__init__()
        self.swin = base_model.swin
        hidden_size = base_model.config.hidden_size

        # Embedding for ArcFace
        self.embedding = nn.Linear(hidden_size, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.arcface = SubCenterArcMarginProduct(embedding_size, num_classes)

        # Separate head for cross-entropy
        self.ce_classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values, labels=None):
        outputs = self.swin(pixel_values)
        pooled = outputs.pooler_output

        # ArcFace path
        embeddings = self.bn(self.embedding(pooled))
        arcface_logits = self.arcface(embeddings, labels)

        # CE path
        ce_logits = self.ce_classifier(pooled)

        if labels is not None:
            arcface_loss = F.cross_entropy(arcface_logits, labels)
            ce_loss = F.cross_entropy(ce_logits, labels)

            # Combine losses (tune alpha)
            loss = 0.7 * arcface_loss + 0.3 * ce_loss

            # Combine logits after softmax
            arcface_probs = F.softmax(arcface_logits, dim=1)
            ce_probs = F.softmax(ce_logits, dim=1)
            combined_probs = 0.7 * arcface_probs + 0.3 * ce_probs

            return {'loss': loss, 'logits': torch.log(combined_probs + 1e-8)}
        else:
            arcface_probs = F.softmax(arcface_logits, dim=1)
            ce_probs = F.softmax(ce_logits, dim=1)
            combined_probs = 0.7 * arcface_probs + 0.3 * ce_probs
            return {'logits': torch.log(combined_probs + 1e-8)}
```

**Complexity**: Hard
**Expected gain**: ~0.7% on top of pure ArcFace

---

## 6. SWIN-Transformer Data Augmentation

### What it does:
Use the exact augmentation pipeline from the SWIN paper (RandAugment, Mixup, CutMix, etc.)

### Implementation:
```python
from torchvision.transforms import RandAugment, AutoAugment

def get_swin_augmentation(size=224):
    from torchvision.transforms import (
        RandomResizedCrop, RandomHorizontalFlip,
        RandAugment, ToTensor, Normalize
    )

    return Compose([
        RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=3),  # bicubic
        RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# For Mixup/CutMix, you need to modify the training loop
from timm.data.mixup import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.8,  # Mixup probability
    cutmix_alpha=1.0,  # CutMix probability
    prob=0.5,  # Probability of applying mixup/cutmix
    mode='batch',
    label_smoothing=0.1,
    num_classes=num_classes
)
```

**Complexity**: Easy-Medium
**Expected gain**: ~0.95%

---

## Recommended Implementation Order

Given the complexity and gains, I recommend this order:

### Phase 1: Easy wins (1-2 days)
1. **Increase learning rate** to 5e-4 → +1.54%
2. **Better augmentation** (RandAugment) → +0.95%
3. **Multi-crop testing** → +0.64%

**Total gain**: ~3% with minimal code changes

### Phase 2: Architecture changes (1 week)
4. **Higher resolution** (384) → +1.53%
5. **SWIN V2** → +0.29%
6. **Multi-task learning** (family/genus/species) → +1%

**Total additional gain**: ~2.8%

### Phase 3: Advanced losses (2 weeks)
7. **SubCenter ArcFace** → +1.4%
8. **Hybrid loss** → +0.67%
9. **Dynamic margins** → +0.5%

**Total additional gain**: ~2.6%

---

## Quick Start: Phase 1 Config

I can create a config file right now with the easy wins:

```yaml
# swin_base_384_enhanced.yml
training:
  learning_rate: 0.0005  # 5e-4 instead of 2e-4
  # ... other settings
```

Would you like me to:
1. Create enhanced configs with Phase 1 improvements?
2. Implement multi-crop testing functionality?
3. Create a custom training script with multi-task learning?
4. Implement the full ArcFace solution?

Let me know which direction you'd like to go!
