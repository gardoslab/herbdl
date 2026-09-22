"""SwinWithLoss: composes a SWIN backbone with one entry from each of the four
registries (losses.py's heads, hierarchy.py's HACE, aux.py's ECC/TARA) per the
config schema in Loss Function Ablations.md:

    loss:
      head: arcface            # softmax|arcface|subcenter_arcface|kappaface|x2softmax|dbm
      hierarchy: none           # none|hace
      aux: []                   # any subset of [ecc, tara]
      grad_modulation: none     # none|flexigrad -- NOT wired here, see below

`head` and `hierarchy: hace` are mutually exclusive (HACE owns its own output
layer; see hierarchy.py's HACEHead docstring). grad_modulation is deliberately
absent from forward(): FlexiGrad needs the per-task gradients BEFORE they are
summed into one backward() call, which this module's single scalar loss
cannot express. Wiring FlexiGrad requires a custom Trainer.training_step using
grad_modulation.py directly against a multitask model's per-level losses --
tracked as its own item in Loss Function Ablations.md's Phase 3, not
implemented in this class.
"""

import torch.nn as nn
import torch.nn.functional as F

from .aux import AUX_REGISTRY
from .hierarchy import HACEHead, TaxonomyTree
from .losses import build_head


class SwinWithLoss(nn.Module):
    def __init__(self, base_model, num_species, head_name="softmax", head_kwargs=None,
                 hierarchy=None, tree: TaxonomyTree = None, hace_kwargs=None,
                 aux_names=None, aux_kwargs=None):
        super().__init__()
        self.config = base_model.config
        if hasattr(base_model, "swinv2"):
            self.swin = base_model.swinv2
        elif hasattr(base_model, "swin"):
            self.swin = base_model.swin
        else:
            raise ValueError("base_model must have a 'swin' or 'swinv2' attribute")
        hidden_size = base_model.config.hidden_size

        self.hierarchy = hierarchy
        head_kwargs = head_kwargs or {}
        if hierarchy == "hace":
            if tree is None:
                raise ValueError("hierarchy='hace' requires a TaxonomyTree")
            self.hace = HACEHead(hidden_size, tree, **(hace_kwargs or {}))
            self.head = None
        elif hierarchy in (None, "none"):
            self.head = build_head(head_name, hidden_size, num_species, **head_kwargs)
            self.hace = None
        else:
            raise ValueError(f"Unknown hierarchy {hierarchy!r}; expected None or 'hace'")

        aux_names = aux_names or []
        aux_kwargs = aux_kwargs or {}
        self.aux_modules = nn.ModuleDict({
            name: AUX_REGISTRY[name](in_features=hidden_size,
                                     **({"num_classes": num_species} if name == "ecc" else aux_kwargs.get(name, {})))
            for name in aux_names
        })

    def gradient_checkpointing_enable(self, **kwargs):
        self.swin.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.swin.gradient_checkpointing_disable()

    def set_margin(self, m):
        if self.head is not None:
            self.head.set_margin(m)

    def class_directions(self):
        """The (n_species, C) matrix repgeom's geometry analysis reads,
        regardless of which head/hierarchy is active."""
        return self.hace.class_directions() if self.hace is not None else self.head.class_directions()

    def forward(self, pixel_values, labels=None, **kwargs):
        pooled = self.swin(pixel_values).pooler_output

        if self.hace is not None:
            out = self.hace(pooled, labels)
            if labels is not None:
                loss, logits = out
            else:
                logits, loss = out, None
        else:
            logits = self.head(pooled, labels)
            loss = F.cross_entropy(logits, labels) if labels is not None else None

        if labels is not None:
            for name, mod in self.aux_modules.items():
                if name == "tara":
                    loss = loss + mod(pooled, pixel_values)
                else:  # ecc
                    loss = loss + mod(pooled, labels)

        return {"loss": loss, "logits": logits}
