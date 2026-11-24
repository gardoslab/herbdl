"""Pooled-feature adapters so SwinWithLoss can sit on non-SWIN backbones.

Each adapter takes the stock HuggingFace / timm classification model, keeps only its
encoder, and returns an object with `.pooler_output` (the feature vector the classifier
head would have read). The feature definitions follow the stock classifier of each model
and llindon's multi-task wrappers (finetuning/{ConvNeXt,DinoV2,EfficientNet_B6,CoAtNet}),
so a registry softmax head sees the same input his plain-CE runs did.

    ConvNeXt      ConvNextModel.pooler_output (layer-normed global mean)
    DINOv2        concat(CLS token, mean patch token), as Dinov2ForImageClassification does
    EfficientNet  EfficientNetModel.pooler_output (global pool), flattened
    CoAtNet/timm  forward_head(forward_features(x), pre_logits=True)
"""

from types import SimpleNamespace

import torch
import torch.nn as nn


class PooledEncoder(nn.Module):
    def __init__(self, inner, kind):
        super().__init__()
        self.inner = inner
        self.kind = kind

    def forward(self, pixel_values):
        if self.kind == "timm":
            pooled = self.inner.forward_head(self.inner.forward_features(pixel_values), pre_logits=True)
        elif self.kind == "dinov2":
            seq = self.inner(pixel_values).last_hidden_state
            pooled = torch.cat([seq[:, 0], seq[:, 1:].mean(dim=1)], dim=1)
        else:  # convnext, efficientnet
            pooled = self.inner(pixel_values).pooler_output
        return SimpleNamespace(pooler_output=pooled.flatten(start_dim=1))

    # HF Trainer's gradient_checkpointing flag reaches the encoder when it supports it.
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.inner, "gradient_checkpointing_enable"):
            self.inner.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.inner, "gradient_checkpointing_disable"):
            self.inner.gradient_checkpointing_disable()


def build_pooled_encoder(base_model):
    """Returns (encoder_module, feature_dim) for a non-SWIN stock classification model."""
    cfg = base_model.config
    if hasattr(base_model, "convnext"):
        return PooledEncoder(base_model.convnext, "convnext"), cfg.hidden_sizes[-1]
    if hasattr(base_model, "dinov2"):
        return PooledEncoder(base_model.dinov2, "dinov2"), 2 * cfg.hidden_size
    if hasattr(base_model, "efficientnet"):
        return PooledEncoder(base_model.efficientnet, "efficientnet"), cfg.hidden_dim
    if hasattr(base_model, "timm_model"):
        timm_model = base_model.timm_model
        return PooledEncoder(timm_model, "timm"), timm_model.num_features
    raise ValueError(f"No pooled-feature adapter for {type(base_model).__name__}")
