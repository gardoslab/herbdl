"""Loss-function registry for the margin-loss ablation program.

See ../../../artifacts/loss-function-ablations/Loss Function Ablations.md for
the tracked plan. This module exists because the nine requested losses do not
share one intervention point, so they cannot be swapped with a single
`loss.name` flag without misrepresenting four of them:

  head             replaces the classifier: the class-direction matrix and its
                   margin rule. softmax | arcface | subcenter_arcface |
                   kappaface | x2softmax | dbm
  hierarchy        changes the output structure. none | hace
  aux              extra loss terms added on top of the head's loss, computed
                   from the same pooled feature. ecc | tara (composable, a list)
  grad_modulation  backprop-time only; does not touch the loss value at all.
                   none | flexigrad (requires a multi-task model — see
                   grad_modulation.py)

Every head implements the same three-method interface so the existing
`repgeom` geometry analysis (within-class tightness, nearest-class distance,
detector alignment) runs unchanged regardless of which one is active:

    forward(embeddings, labels=None) -> logits    # margin applied only when
                                                    # labels is not None, i.e.
                                                    # only at train time
    class_directions() -> (K, C) tensor            # the matrix repgeom reads
    set_margin(m)                                  # no-op where irrelevant

Grounding (read in full before writing the formulas below; a source with no
equations extractable from the paper text is not implemented from a
paraphrase — see the per-class docstring for exactly which equation numbers
were used):

  ArcFace              https://www.alphaxiv.org/abs/1801.07698
  Sub-center ArcFace   https://doi.org/10.1007/978-3-030-58621-8_43
  KappaFace            https://www.alphaxiv.org/abs/2201.07394   (Eq. 3, 8-13)
  X2-Softmax           https://www.alphaxiv.org/abs/2312.05281   (Eq. 4; best
                       hyperparameters from their ablation: a=-1.0, h=-0.3, k=1.0)
  ECC                  https://arxiv.org/abs/2407.04243          (Eq. 1-10)
  DBM                  https://www.alphaxiv.org/abs/2412.15477   (Eq. 6-9;
                       K=0.1, tau=1 from their Fig. 2 caption)
  HACE                 https://www.alphaxiv.org/abs/2605.06274
  TARA                 https://www.alphaxiv.org/abs/2603.00431   (ADAPTED —
                       see TARAAux docstring)

FlexiGrad lives in grad_modulation.py, not here, because it is not a loss: it
is a gradient-surgery rule applied to the shared backbone's gradient during
backprop and never changes the scalar loss value itself.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================== heads =====

class SoftmaxHead(nn.Module):
    """Plain linear classifier. The baseline every other head is compared
    against — this is what SWIN_BASE_BASELINE and the CE runs already use.
    """

    def __init__(self, in_features, num_classes, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, embeddings, labels=None):
        return self.linear(embeddings)

    def class_directions(self):
        return self.linear.weight.detach()

    def set_margin(self, m):
        pass  # no margin concept


class ArcMarginHead(nn.Module):
    """ArcFace (k=1) and Sub-center ArcFace (k>1) in one class.

    This is SubCenterArcMarginProduct from SWIN_finetuning_advanced.py,
    unmodified in its math — the existing k=1 configuration is the validated
    reference run (SWIN_BASE_ARCFACE_ABLATION, 0.7971/0.7605). Moved here
    unchanged rather than reimplemented, specifically so that run stays
    reproducible byte-for-byte from this registry.

    logit_target   = s * cos(theta_y + m)
    logit_other    = s * cos(theta_j),  j != y
    cos(theta_i)   = max over k sub-centers of cos(embedding, W_i,k)

    At inference (labels=None) no margin is applied — pure s*cosine.
    """

    def __init__(self, in_features, num_classes, k=1, s=30.0, m=0.30, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * k, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self._set_trig(m)

    def _set_trig(self, m):
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_margin(self, m):
        self.m = m
        self._set_trig(m)

    def _cosine(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight).view(-1, self.num_classes, self.k)
        cosine, _ = torch.max(cosine, dim=2)
        return cosine

    def forward(self, embeddings, labels=None):
        cosine = self._cosine(embeddings)
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm) if not self.easy_margin \
            else torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

    def class_directions(self):
        # (num_classes * k, in_features) -> (num_classes, in_features) by
        # taking the first sub-center. repgeom's geometry analysis expects one
        # direction per class; for k>1 the sub-centers are close by
        # construction (that is the point of sub-center ArcFace), so the first
        # is a representative, not an average that would blur them together.
        w = self.weight.detach().view(self.num_classes, self.k, self.in_features)
        return w[:, 0, :]


class KappaFaceHead(nn.Module):
    """Adaptive additive angular margin, per-class margin set by von
    Mises-Fisher concentration and class population (KappaFace, Eq. 3, 8-13).

    Per-class margin:
        kappa_hat_c = r_bar(d - r_bar^2) / (1 - r_bar^2)      Eq. 3 (MLE)
            r_bar = || running mean-direction buffer for class c ||
        kappa_tilde_c = (kappa_hat_c - mean(kappa_hat)) / std(kappa_hat)   Eq. 10
        w_kappa_c = 1 - sigmoid(kappa_tilde_c * T)             Eq. 11
        w_pop_c   = (cos(pi * n_c / K) + 1) / 2, K = max_c n_c Eq. 12 (fixed,
                    computed once from the training-set class counts)
        psi_c     = (1 - gamma) * w_pop_c + gamma * w_kappa_c  Eq. 13
        m_c       = psi_c * m0
    then the standard ArcFace logit with m replaced by m_c (per-class, not
    per-sample) -- same structure as ArcMarginHead.forward with self.m -> m_c.

    Implements the paper's "memory buffer" variant (Sec. III-C), not the
    momentum-encoder variant -- simpler, and the paper reports them as
    interchangeable at inference. The buffer is a per-class running mean
    embedding direction, updated online every step (EMA with `buffer_momentum`)
    rather than collected once per epoch (recollecting 671,817 features once an
    epoch is exactly the cost the paper's Sec. III-C says to avoid). kappa
    itself is recomputed from the buffer once per epoch via
    `KappaEpochUpdateCallback`, matching the paper's stated cadence
    ("concentration values ... recalculated once an epoch").
    """

    def __init__(self, in_features, num_classes, class_counts, s=30.0, m0=0.30,
                 temperature=0.3, gamma=0.5, buffer_momentum=0.99, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m0 = m0
        self.T = temperature
        self.gamma = gamma
        self.buffer_momentum = buffer_momentum
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("KappaFaceHead requires class_counts > 0 for every class "
                              "(w_pop's cos(pi * n_c / K) term is undefined for an absent class)")
        K = counts.max()
        w_pop = (torch.cos(math.pi * counts / K) + 1.0) / 2.0
        self.register_buffer("w_pop", w_pop)  # Eq. 12, fixed for the run

        # Memory buffer: running mean *direction* per class (not renormalised
        # to unit length here -- its norm is r_bar, which Eq. 3 needs directly).
        self.register_buffer("mean_dir", torch.zeros(num_classes, in_features))
        self.register_buffer("dir_initialized", torch.zeros(num_classes, dtype=torch.bool))
        # Per-class margin, recomputed once per epoch; m0 until the first update
        # (kappa is meaningless before the buffer has seen real embeddings).
        self.register_buffer("per_class_margin", torch.full((num_classes,), m0))

    @torch.no_grad()
    def update_buffer(self, embeddings, labels):
        """Called every training step. EMA update of the per-class mean
        direction from L2-normalised embeddings, so r_bar in Eq. 3 reflects
        direction concentration and not feature magnitude drift."""
        emb = F.normalize(embeddings.detach().float(), p=2, dim=1)
        for c in labels.unique():
            mask = labels == c
            batch_mean = emb[mask].mean(dim=0)
            if not self.dir_initialized[c]:
                self.mean_dir[c] = batch_mean
                self.dir_initialized[c] = True
            else:
                mom = self.buffer_momentum
                self.mean_dir[c] = mom * self.mean_dir[c] + (1 - mom) * batch_mean

    @torch.no_grad()
    def recompute_margins(self):
        """Eq. 3 + 8-13. Called once per epoch by KappaEpochUpdateCallback."""
        d = self.in_features
        r_bar = self.mean_dir.norm(dim=1).clamp(max=0.999)  # r_bar in [0,1)
        kappa_hat = r_bar * (d - r_bar.pow(2)) / (1 - r_bar.pow(2)).clamp(min=1e-6)  # Eq. 3

        seen = self.dir_initialized
        if seen.sum() < 2:
            return  # not enough classes observed yet to normalise
        mu_k = kappa_hat[seen].mean()
        sigma_k = kappa_hat[seen].std().clamp(min=1e-6)
        kappa_tilde = (kappa_hat - mu_k) / sigma_k  # Eq. 10
        w_kappa = 1.0 - torch.sigmoid(kappa_tilde * self.T)  # Eq. 11

        psi = (1 - self.gamma) * self.w_pop + self.gamma * w_kappa  # Eq. 13
        new_margin = psi * self.m0
        self.per_class_margin = torch.where(seen, new_margin, self.per_class_margin)

    def _cosine(self, embeddings):
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(emb, w)

    def forward(self, embeddings, labels=None):
        cosine = self._cosine(embeddings)
        if labels is None:
            return cosine * self.s
        self.update_buffer(embeddings, labels)
        m_c = self.per_class_margin[labels]  # (B,), per-SAMPLE lookup of its class's margin
        cos_m, sin_m = torch.cos(m_c), torch.sin(m_c)
        th, mm = torch.cos(math.pi - m_c), torch.sin(math.pi - m_c) * m_c
        target_cos = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        sine = torch.sqrt((1.0 - target_cos.pow(2)).clamp(0, 1))
        phi = target_cos * cos_m - sine * sin_m
        phi = torch.where(target_cos > th, phi, target_cos - mm) if not self.easy_margin \
            else torch.where(target_cos > 0, phi, target_cos)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        logits = cosine.clone()
        logits.scatter_(1, labels.view(-1, 1), phi.unsqueeze(1))
        return logits * self.s

    def class_directions(self):
        return self.weight.detach()

    def set_margin(self, m):
        # Margin-warmup ramps m0, the ceiling every per-class margin scales
        # toward, not a single global value.
        self.m0 = m


class X2SoftmaxHead(nn.Module):
    """X2-Softmax: replaces ArcFace's cos(theta+m) with a quadratic function
    of the raw angle, removing the margin hyperparameter (2312.05281, Eq. 4).

        f(theta) = a*(theta - h)^2 + k      applied to the TARGET class only
        logit_target = s * f(theta_y)
        logit_other  = s * cos(theta_j),  j != y

    Defaults a=-1.0, h=-0.3, k=1.0 are the paper's own best-ablation values
    (Table 2, MS1Mv3/ResNet50). `s` defaults to 30 to stay consistent with
    every other head in this project rather than the paper's face-recognition
    convention of s=64 -- flagged here since it is a real deviation from the
    source, not an oversight.
    """

    def __init__(self, in_features, num_classes, s=30.0, a=-1.0, h=-0.3, k=1.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.a = a
        self.h = h
        self.k = k
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _cosine_and_theta(self, embeddings):
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, w).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        return cosine, theta

    def forward(self, embeddings, labels=None):
        cosine, theta = self._cosine_and_theta(embeddings)
        if labels is None:
            return cosine * self.s
        f_theta = self.a * (theta - self.h).pow(2) + self.k  # Eq. 4
        target_f = f_theta.gather(1, labels.view(-1, 1)).squeeze(1)
        logits = cosine.clone()
        logits.scatter_(1, labels.view(-1, 1), target_f.unsqueeze(1))
        return logits * self.s

    def class_directions(self):
        return self.weight.detach()

    def set_margin(self, m):
        pass  # X2-Softmax has no margin hyperparameter -- that is its point


class DBMHead(nn.Module):
    """Difficulty-aware Balancing Margin (2412.15477, Eq. 6-9).

        rho_y     = n_y / n_min
        m_C       = K * rho_y^(-tau)                         Eq. 6 (class-wise)
        d_I       = (1 - cos(theta_y)) / 2                   Eq. 7 (instance difficulty)
        m_I       = m_C * d_I                                Eq. 8 (instance-wise)
        hard      = 1[argmin_i(theta_i) != y]                a "hard positive":
                    the sample's angularly nearest class is not its own
        logit_y   = s * cos(theta_y + m_C + hard * m_I)       Eq. 9
        logit_j   = s * cos(theta_j),  j != y

    K=0.1, tau=1 are the paper's own figure-caption defaults ("Margins for
    K=0.1 and tau=1"), reported to work well across their benchmarks.
    """

    def __init__(self, in_features, num_classes, class_counts, s=30.0, K=0.1, tau=1.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.K = K
        self.tau = tau
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("DBMHead requires class_counts > 0 for every class")
        rho = counts / counts.min()
        m_C = K * rho.pow(-tau)  # Eq. 6, fixed for the run (class frequency is static)
        self.register_buffer("m_class", m_C)

    def _cosine_and_theta(self, embeddings):
        emb = F.normalize(embeddings, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, w).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        return cosine, theta

    def forward(self, embeddings, labels=None):
        cosine, theta = self._cosine_and_theta(embeddings)
        if labels is None:
            return cosine * self.s

        target_theta = theta.gather(1, labels.view(-1, 1)).squeeze(1)
        target_cos = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        m_C = self.m_class[labels]
        d_I = (1.0 - target_cos) / 2.0  # Eq. 7

        nearest_class = theta.argmin(dim=1)  # smallest angle = most confident match
        hard = (nearest_class != labels).float()  # Eq. 9's indicator
        m_I = m_C * d_I  # Eq. 8
        m_total = m_C + hard * m_I

        target_phi = torch.cos(target_theta + m_total)  # Eq. 9
        logits = cosine.clone()
        logits.scatter_(1, labels.view(-1, 1), target_phi.unsqueeze(1))
        return logits * self.s

    def class_directions(self):
        return self.weight.detach()

    def set_margin(self, m):
        # DBM's margin is derived per-class from frequency (Eq. 6), not a
        # single scalar -- rescale K, which uniformly scales every m_C, so the
        # shared margin-warmup ramp still has a lever to pull.
        self.K = m


HEAD_REGISTRY = {
    "softmax": SoftmaxHead,
    "arcface": lambda in_f, n, **kw: ArcMarginHead(in_f, n, k=1, **kw),
    "subcenter_arcface": lambda in_f, n, k=3, **kw: ArcMarginHead(in_f, n, k=k, **kw),
    "kappaface": KappaFaceHead,
    "x2softmax": X2SoftmaxHead,
    "dbm": DBMHead,
}


def build_head(name, in_features, num_classes, **kwargs):
    if name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head {name!r}. Registered: {sorted(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[name](in_features, num_classes, **kwargs)
