"""FlexiGrad: adaptive gradient modulation for hierarchical fine-grained
classification. https://www.alphaxiv.org/abs/2607.17563, Sec. 3.2-3.4 and its
Algorithm 1 pseudocode.

Not a loss -- it never changes L_total = sum_k L_k. It changes how the
per-task gradients on the SHARED backbone are combined before the optimizer
step, which is why it lives outside losses.py and requires a multi-task model
(hierarchy: multitask in the config schema, not hierarchy: hace -- HACE
already aggregates everything into one scalar loss, so there is nothing here
for FlexiGrad to modulate between).

Per pair of granularity levels (i, j):

    cos(theta_ij) = <g_i, g_j> / (||g_i|| ||g_j||)                     Eq. 4

    if cos(theta_ij) < 0   (conflicting):
        g_i <- g_i - [ <g_i,g_j> / (||g_j||^2 + eps) ] * g_j           Eq. 5
    else                    (cooperative):
        w_ij = (1 + cos(theta_ij)) / 2                                  Eq. 6
        g_i <- w_ij * g_i + (1 - w_ij) * [ <g_i,g_j> / (||g_j||^2 + eps) ] * g_j

    Delta_theta = sum_i g_i (after every pair has been resolved)       Algorithm 1

The paper's Algorithm 1 applies this over ordered pairs (i, j) with j > i,
accumulating in place into g_FG_i -- reproduced here exactly, including that
ordering, since the operation is not symmetric or associative and a different
pairing order is a different algorithm.
"""

import torch


def flexigrad_combine(task_grads, eps=1e-8):
    """task_grads: list of K flat 1-D tensors, one per granularity level's
    gradient on the SAME shared parameters (same shape, same order).
    Returns: one flat tensor, the combined update (Algorithm 1's g_FG_final).
    """
    k = len(task_grads)
    if k < 2:
        return task_grads[0].clone() if task_grads else None

    fg = [g.clone() for g in task_grads]  # g_FG_i initialised to g_i
    for i in range(k):
        for j in range(i + 1, k):
            gi_raw, gj = task_grads[i], task_grads[j]  # cos/dot use the ORIGINAL g_i, g_j
            dot = torch.dot(gi_raw, gj)
            cos_ij = dot / (gi_raw.norm() * gj.norm() + eps)
            proj = dot / (gj.norm().pow(2) + eps) * gj  # the shared projection term
            if cos_ij < 0:
                fg[i] = fg[i] - proj  # Eq. 5
            else:
                w = (1 + cos_ij) / 2  # Eq. 6
                fg[i] = w * fg[i] + (1 - w) * proj

    return torch.stack(fg).sum(dim=0)


def per_task_grads(losses, shared_params):
    """One backward pass per task loss (retain_graph on all but the last),
    returning a flat gradient vector per task over `shared_params`, in the
    same flattened order every time -- required so flexigrad_combine's
    element-wise ops between tasks line up parameter-for-parameter.

    Does not step the optimizer or zero grads; the caller (a custom
    Trainer.training_step) owns that, since HF Trainer's default training_step
    does one backward() on the summed loss and has no notion of "per-task"
    gradients at all -- this is the actual reason FlexiGrad cannot be a
    drop-in loss like the other eight.
    """
    grads_per_task = []
    for idx, loss in enumerate(losses):
        retain = idx < len(losses) - 1
        grads = torch.autograd.grad(loss, shared_params, retain_graph=retain, allow_unused=True)
        flat = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, shared_params)
        ])
        grads_per_task.append(flat)
    return grads_per_task


def apply_combined_grad(combined, shared_params):
    """Unflatten `combined` back onto shared_params[*].grad, in the same
    order per_task_grads used to flatten them."""
    offset = 0
    for p in shared_params:
        n = p.numel()
        p.grad = combined[offset: offset + n].view_as(p).clone()
        offset += n
