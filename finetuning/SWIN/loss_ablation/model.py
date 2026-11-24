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
from .losses import ArcMarginHead, SoftmaxHead, build_head


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
            # Not a SWIN: ConvNeXt / DINOv2 / EfficientNet / timm (CoAtNet). The encoder
            # is still stored as self.swin so heads, exports and checkpoints are unchanged.
            from .backbones import build_pooled_encoder
            self.swin, hidden_size = build_pooled_encoder(base_model)
        if hasattr(base_model, "swinv2") or hasattr(base_model, "swin"):
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

    def head_inference_info(self):
        """(W, bias_or_None, kind, scale, sub_centers) -- everything a reader
        needs to reproduce this model's exact inference-time (labels=None)
        logits without any head-specific Python code of its own, because every
        head in this registry reduces to exactly one of two formulas once no
        margin/aux loss is applied:

            "linear"  logit_c = w_c . x + b_c
                      -- SoftmaxHead's own forward, and HACE's leaf rows
                      (leaves have no descendants, so aggregation is the
                      identity there -- see hierarchy.py's HACEHead docstring
                      -- and argmax over softmax is argmax over the raw
                      linear logit, softmax being monotonic). sub_centers is
                      always 1.
            "cosine"  logit_c = scale * max_j cosine(w_c_j, x)   j in [0, sub_centers)
                      -- every margin head (ArcMarginHead, KappaFaceHead,
                      X2SoftmaxHead, DBMHead). Confirmed directly from each
                      head's own forward(), not assumed: every one of their
                      `if labels is None:` branches returns exactly
                      `cosine * self.s` -- margin, per-class adaptation, and
                      the quadratic term only ever apply when labels is given.

        W is ALWAYS (n_species, sub_centers, C), for both kinds -- the
        checkpoint_io.py contract on the other end enforces this shape
        unconditionally (it has no way to know which kind a given export is
        until it reads the same meta this function's caller writes), so a
        2D W for "linear" heads is not a harmless shorthand, it is a shape
        mismatch that fails the export. sub_centers is 1 for every head
        except subcenter_arcface, where it's still just a size-1 middle
        dimension for the linear/single-center cases.

        The max-over-sub_centers is NOT optional bookkeeping: sub-center
        ArcFace's whole mechanism is that max, and `class_directions()`
        deliberately exposes only sub-center 0 (a representative direction
        for geometry comparisons between CLASSES, not a claim that it alone
        determines the logit) -- reusing it here for k>1 silently reproduces
        the WRONG logit whenever a different sub-center wins for a given
        image. Caught by testing against a live model's own forward(), not by
        inspection: for k=1 heads the two are identical by construction, so
        only a real k>1 case (subcenter_arcface, k=3) exposed the gap.

        sub_centers is read from the head's TYPE, not a generic `.k`
        attribute: X2SoftmaxHead also has a `self.k`, but it is an unrelated
        formula hyperparameter (Eq. 4's constant term), not a sub-center
        count, and happens to default to 1.0 -- a `getattr(head, 'k', 1)`
        check would silently work by coincidence at the default and silently
        break the moment that hyperparameter were tuned to anything else.

        ECC/TARA never change any of this: they are training-time-only
        additive losses on top of whichever base head is active, and never
        touch that head's own forward path.
        """
        if self.hace is not None:
            n = self.hace.tree.n_species
            W = self.hace.linear.weight.detach()[:n].unsqueeze(1)
            b = self.hace.linear.bias.detach()[:n]
            return W, b, "linear", 1.0, 1
        h = self.head
        if isinstance(h, SoftmaxHead):
            W = h.class_directions().unsqueeze(1)
            return W, h.linear.bias.detach(), "linear", 1.0, 1
        if isinstance(h, ArcMarginHead):
            W = h.weight.detach().view(h.num_classes, h.k, h.in_features)
            return W, None, "cosine", float(h.s), h.k
        # KappaFaceHead, X2SoftmaxHead, DBMHead: single center, unsqueezed to
        # the same (n_species, sub_centers, C) shape so the reader's formula
        # never has to branch on which of these three it is.
        return h.class_directions().unsqueeze(1), None, "cosine", float(h.s), 1

    def forward(self, pixel_values, labels=None, bioclip_pixel_values=None,
                labels_b=None, lam=1.0, label_smoothing=0.0, **kwargs):
        pooled = self.swin(pixel_values).pooler_output

        if self.hace is not None:
            out = self.hace(pooled, labels)
            if labels is not None:
                loss, logits = out
            else:
                logits, loss = out, None
        else:
            logits = self.head(pooled, labels)
            if labels is None:
                loss = None
            elif labels_b is None and label_smoothing == 0.0:
                loss = F.cross_entropy(logits, labels)
            else:
                # Heavy augmentation: label smoothing and/or mixup/CutMix. The margin is applied
                # per label set, so the head runs once more with the second label set (the backbone
                # runs once): loss = lam * CE(head(f, y_a), y_a) + (1 - lam) * CE(head(f, y_b), y_b).
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
                if labels_b is not None:
                    logits_b = self.head(pooled, labels_b)
                    loss = lam * loss + (1.0 - lam) * F.cross_entropy(
                        logits_b, labels_b, label_smoothing=label_smoothing)

        aux_losses = {}  # detached per-aux values, for logging only (never used in the graph)
        if labels is not None:
            for name, mod in self.aux_modules.items():
                if name == "tara":
                    # NEVER pixel_values here: BioCLIP needs its own 224px
                    # CLIP-normalised preprocessing, not the SWIN backbone's
                    # 384px ImageNet-normalised input -- see aux.py's
                    # TARAAux docstring and bioclip_teacher.py. Reusing
                    # pixel_values would silently train against a
                    # meaningless (but finite, unit-norm) teacher signal, so
                    # this fails loudly instead of falling back to it.
                    if bioclip_pixel_values is None:
                        raise ValueError(
                            "aux='tara' is active but bioclip_pixel_values was not "
                            "provided. It must be a SEPARATE tensor from pixel_values, "
                            "preprocessed with the BioCLIP teacher's own transform "
                            "(see bioclip_teacher.BioCLIPTeacher.transform) -- passing "
                            "the SWIN pixel_values here would silently corrupt the "
                            "teacher signal rather than raise, which is worse than this.")
                    aux_loss = mod(pooled, bioclip_pixel_values)
                else:  # ecc
                    aux_loss = mod(pooled, labels)
                aux_losses[name] = aux_loss.detach()
                loss = loss + aux_loss

        return {"loss": loss, "logits": logits, "aux_losses": aux_losses}
