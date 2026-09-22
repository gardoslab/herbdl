"""Auxiliary losses: added on top of whatever head is active, computed from
the same pooled feature. Composable with any head in losses.py, and with each
other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECCAux(nn.Module):
    """Exploration of Class Center (ECC).
    Source: https://arxiv.org/abs/2407.04243 (TCSVT 2024), Eq. 1-10. Already
    validated on a Swin-base backbone in the source paper (Table 3,
    Swin-ECC beats plain Swin on CUB/CAR/NAB/AIR), the closest architecture
    match of any of the nine sources.

    Two components, both driven by a running per-class "class center":

      MCC (multiple class-center constraint), Eq. 5 -- pulls a sample toward
      its own class-center FEATURE, pushes it away from the most similar
      OTHER class's center feature, weighted by how similar those two classes
      are (so confusable classes get pushed harder):

          L_MCC = sum_k [ (1 - cos(X_k, F_y_k))
                          + s_{y_k,sim_k} * cos(X_k, F_sim_k) ]

      CLG (class-center label generation), Eq. 6-9 -- soft labels built from
      the running per-class average PREDICTION distribution, then KL-divergence
      to the sample's own prediction:

          L_CLG = sum_k KL( softmax(readout(X_k)) || softmax(L_{y_k}) )

      L_ECC = lambda1 * L_MCC + lambda2 * L_CLG,  added to whatever head loss
      is active (paper: L_final = L_CE + L_ECC; here, L_head + L_ECC).

    lambda1/lambda2 defaults: the paper's ablation-chosen values were not
    recoverable from the extracted PDF text (the ablation table sits past
    where extraction was pursued). Defaulted to 1.0/1.0 here and flagged
    explicitly as NOT a confirmed paper value -- treat as tunable, not as a
    reproduction of their setting.

    CLG's "last fully connected layer" (Eq. 6's f) is a SEPARATE small linear
    readout owned by this module, not the active head's own logits. This is a
    deliberate design choice, not in the paper (which only ever pairs ECC with
    a plain softmax classifier): a margin head's logits have the margin baked
    into the target column, which would bias the softmax CLG computes its KL
    divergence against. A plain, undistorted N-way readout keeps ECC
    composable with every head in the registry.

    Memory at full scale (15,501 species): the class-center feature buffer is
    (15501, 1024) float32 ~ 63MB, trivial. The class-center label-DISTRIBUTION
    buffer and the pairwise similarity matrix are both (15501, 15501) float32,
    ~960MB EACH -- a real cost, not a rounding error, worth watching in the
    harness's memory report before deciding whether ECC runs at full scale or
    only in the small-scale ablation.
    """

    def __init__(self, in_features, num_classes, lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.readout = nn.Linear(in_features, num_classes)  # Eq. 6's f, ECC-private

        self.register_buffer("center_feat", torch.zeros(num_classes, in_features))
        self.register_buffer("center_logit", torch.zeros(num_classes, num_classes))
        self.register_buffer("counter", torch.zeros(num_classes, dtype=torch.long))
        # Cached once per epoch by ECCEpochUpdateCallback (an (N,N) similarity
        # matrix is too expensive to recompute every step at 15.5k classes).
        self.register_buffer("sim_index", torch.arange(num_classes))  # self until first update
        self.register_buffer("sim_value", torch.zeros(num_classes))

    @torch.no_grad()
    def update_buffers(self, embeddings, raw_logits, labels):
        """Eq. 1-2 / Eq. 6's running-mean update -- exact incremental mean per
        class via a counter, not an EMA (the paper's formula has no momentum
        hyperparameter; it is the textbook online-mean recurrence)."""
        emb = embeddings.detach().float()
        logit = raw_logits.detach().float()
        for c in labels.unique():
            mask = labels == c
            n_new = int(mask.sum())
            c_cur = int(self.counter[c])
            batch_feat_sum = emb[mask].sum(dim=0)
            batch_logit_sum = logit[mask].sum(dim=0)
            self.center_feat[c] = (batch_feat_sum + c_cur * self.center_feat[c]) / (c_cur + n_new)
            self.center_logit[c] = (batch_logit_sum + c_cur * self.center_logit[c]) / (c_cur + n_new)
            self.counter[c] += n_new

    @torch.no_grad()
    def recompute_similarity(self):
        """Eq. 3-4. Called once per epoch. Excludes self when finding the most
        similar OTHER class (the paper's Eq. 4 implicitly must, since a class
        is always most similar to itself; not stating that would silently push
        every sample away from its own center)."""
        F_ = F.normalize(self.center_feat, p=2, dim=1)
        S = F_ @ F_.T  # Eq. 3
        S.fill_diagonal_(-2.0)  # exclude self from the argmax below
        sim_value, sim_index = S.max(dim=1)  # Eq. 4
        self.sim_value = sim_value
        self.sim_index = sim_index

    def forward(self, embeddings, labels):
        raw_logits = self.readout(embeddings)
        self.update_buffers(embeddings, raw_logits, labels)

        F_y = self.center_feat[labels]
        sim_idx_y = self.sim_index[labels]
        sim_val_y = self.sim_value[labels]
        F_sim = self.center_feat[sim_idx_y]

        mcc_loss = (1.0 - F.cosine_similarity(embeddings, F_y, dim=1)
                    + sim_val_y * F.cosine_similarity(embeddings, F_sim, dim=1)).mean()

        p = F.log_softmax(raw_logits, dim=1)
        q = F.softmax(self.center_logit[labels], dim=1)
        clg_loss = F.kl_div(p, q, reduction="batchmean")  # Eq. 9, KL(P || Q)

        return self.lambda1 * mcc_loss + self.lambda2 * clg_loss


class TARAAux(nn.Module):
    """Feature-alignment loss against a frozen biological foundation model.

    ADAPTATION, not a reproduction. Source
    (https://www.alphaxiv.org/abs/2603.00431) aligns a large multimodal
    model's representations with a biological foundation model's (BioCLIP2)
    for VQA-style hierarchical recognition -- a different task and a different
    kind of model from ours. What transplants directly is the core mechanism
    (the paper's L_V, its visual alignment term): project the classifier's
    intermediate features into the teacher's embedding space with a learnable
    projector, and maximise cosine similarity, teacher frozen and no gradient
    flowing back into it. Everything else in the source paper (the VQA
    question format, the RFT alternation, the text-alignment term) does not
    apply to a plain image classifier and is not implemented.

    L_TARA = -cos( projector(pooled_features), bioclip_embedding(image) )

    `bioclip_embed_fn` is injected rather than owned here, so this module has
    no opinion on how the teacher embedding is produced or cached -- the
    project already has a BioCLIP integration (finetuning/BioCLIP/) and
    running a second full forward pass through it every step is a real cost
    worth deciding deliberately, not one this module should hide.
    """

    def __init__(self, in_features, bioclip_dim, bioclip_embed_fn):
        super().__init__()
        self.projector = nn.Linear(in_features, bioclip_dim)
        self.bioclip_embed_fn = bioclip_embed_fn  # callable: pixel_values -> (B, bioclip_dim), no_grad

    def forward(self, pooled_features, pixel_values):
        with torch.no_grad():
            teacher = self.bioclip_embed_fn(pixel_values)
            teacher = F.normalize(teacher, p=2, dim=1)
        student = F.normalize(self.projector(pooled_features), p=2, dim=1)
        return -(student * teacher).sum(dim=1).mean()


AUX_REGISTRY = {
    "ecc": ECCAux,
    "tara": TARAAux,
}
