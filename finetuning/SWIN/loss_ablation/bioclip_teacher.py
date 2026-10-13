"""The frozen BioCLIP teacher TARAAux (aux.py) aligns against.

Loads `hf-hub:imageomics/bioclip` via `open_clip`, the same call the project's
existing `finetuning/BioCLIP/zero_shot.py` uses. Confirmed by loading the real
model (not assumed): 512-d embeddings, and a preprocessing pipeline that is
NOT the SWIN backbone's --

    BioCLIP   Resize(224, bicubic) -> CenterCrop(224) -> ToTensor ->
              Normalize(mean=[0.4815,0.4578,0.4082], std=[0.2686,0.2613,0.2758])
    SWIN      Resize(384, bilinear, squash) -> CenterCrop(384) -> ToTensor ->
              Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

so a caller MUST run each image through `BioCLIPTeacher.transform` separately
from whatever produces the SWIN backbone's `pixel_values` -- see aux.py's
TARAAux docstring for why reusing one tensor for both is a silent-corruption
trap, not a shortcut.

`open_clip_torch` is not part of `fall-2026-pyt`; installed via
`pip install --user open_clip_torch` under the project's PYTHONUSERBASE, same
convention as every other extra package this project needs (see
finetuning/CLAUDE.md).
"""

import torch
import torch.nn as nn


class BioCLIPTeacher:
    """Not an nn.Module on purpose: this has no learnable parameters and must
    never appear in a training checkpoint's state_dict or optimizer param
    groups. `TARAAux` takes an `embed_fn` callable, not a submodule, for
    exactly this reason.
    """

    def __init__(self, device="cuda"):
        import open_clip
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(device)
        self.transform = preprocess_val  # the Dataset applies this, not this class
        self.device = device
        self.embed_dim = 512  # confirmed by loading the real model; see module docstring

    @torch.no_grad()
    def embed(self, bioclip_pixel_values):
        feat = self.model.encode_image(bioclip_pixel_values.to(self.device))
        return feat.float()  # TARAAux does its own normalisation; return raw

    def as_embed_fn(self):
        """What TARAAux(..., bioclip_embed_fn=...) actually wants: a plain
        callable, not this wrapper object, so TARAAux stays agnostic to how
        the teacher is implemented."""
        return self.embed
