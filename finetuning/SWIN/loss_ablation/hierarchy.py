"""Taxonomy tree + HACE (Hierarchy-Aware Cross-Entropy).

Source: https://www.alphaxiv.org/abs/2605.06274, "When Labels Have Structure:
Improving Image Classification with Hierarchy-Aware Cross-Entropy" (Chan,
D'Ascenzo, Cultrera di Montesano).

HACE combines two mechanisms that the paper shows must be used together under
leaf-only supervision (its central point, not an implementation detail):

  prediction aggregation    expand the output layer to every node in the
                             hierarchy, and let a parent's probability be the
                             sum of its own raw softmax mass plus all its
                             descendants'. This gives internal nodes a
                             well-defined prediction, but on its own gives
                             them no gradient, since leaf-only labels put zero
                             target mass on every internal node.
  ancestral label smoothing dilute each leaf's one-hot mass up its ancestor
                             path, geometrically decaying with distance from
                             the leaf. This gives internal nodes a nonzero
                             target, but on its own has no mechanism keeping
                             the model's internal-node predictions consistent
                             with those targets.

Combined: L_HACE = -sum_i p*(i) * log q*(i) over all N tree nodes, both leaf
and internal (the paper's direct-replacement form of cross-entropy, with p and
q both expanded from n leaves to N total nodes).

Our tree for this project: 15,501 species -> 2,564 genera -> 272 families ->
1 root. N = 15,501 + 2,564 + 272 + 1 = 18,338 nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaxonomyTree:
    """species -> genus -> family -> root, as a flat node index space.

    Node ids: [0, n_species) are species (leaves), followed by
    [n_species, n_species+n_genera) genera, then families, then one root node
    at the very end. Species/genus/family ids passed in must already be dense
    0..n-1 integers *within their own level* (exactly what the training
    script's family2id/genus2id/species2id already produce).
    """

    def __init__(self, species_to_genus, genus_to_family):
        """species_to_genus: (n_species,) int array, species id -> genus id.
        genus_to_family:   (n_genera,) int array, genus id -> family id.
        """
        species_to_genus = torch.as_tensor(species_to_genus, dtype=torch.long)
        genus_to_family = torch.as_tensor(genus_to_family, dtype=torch.long)

        self.n_species = int(species_to_genus.shape[0])
        self.n_genera = int(genus_to_family.shape[0])
        self.n_families = int(genus_to_family.max().item()) + 1
        self.n_nodes = self.n_species + self.n_genera + self.n_families + 1

        self.species_offset = 0
        self.genus_offset = self.n_species
        self.family_offset = self.n_species + self.n_genera
        self.root = self.n_nodes - 1

        if species_to_genus.max().item() >= self.n_genera or species_to_genus.min().item() < 0:
            raise ValueError("species_to_genus contains an out-of-range genus id")

        # parent[i] = node id of i's immediate parent; parent[root] = root
        # (self-loop, so "ancestor of" below includes the node itself cleanly).
        parent = torch.full((self.n_nodes,), self.root, dtype=torch.long)
        parent[self.species_offset:self.genus_offset] = self.genus_offset + species_to_genus
        parent[self.genus_offset:self.family_offset] = self.family_offset + genus_to_family
        parent[self.family_offset:self.root] = self.root
        self.parent = parent

    def reachability_matrix(self, dtype=torch.float32):
        """R[i, j] = 1 if j is an ancestor of i, OR j == i (i is its own
        ancestor at distance 0). Row i is a descendant, column j an ancestor,
        so for a batch of raw softmax rows q_hat (B, N), `q_hat @ R` gives
        (B, N) with entry [b, j] = sum over i (with j an ancestor-or-self of i)
        of q_hat[b, i] -- each node's own raw mass plus all its descendants',
        exactly prediction aggregation, Sec. 4.1. (For a single column vector
        q instead of a batch of rows, the equivalent product is `R.T @ q`.)

        Built by walking `parent` at most `depth` times per node (depth=3 for
        species->genus->family->root), not by a generic transitive closure --
        O(N * depth) instead of O(N^2) work, though the stored matrix is still
        O(N^2) memory (18,338^2 floats = ~1.3GB in fp32). For the small-scale
        E2E harness's synthetic tree this is trivial; for the full run this
        matrix should be built once and cached, not rebuilt per step.
        """
        n = self.n_nodes
        R = torch.zeros(n, n, dtype=dtype)
        idx = torch.arange(n)
        cur = idx.clone()
        R[idx, idx] = 1.0
        for _ in range(4):  # species/genus/family/root: at most 3 hops + self
            nxt = self.parent[cur]
            newly_reached = nxt != cur
            R[idx[newly_reached], nxt[newly_reached]] = 1.0
            cur = nxt
        return R

    def ancestor_path(self, leaf_id):
        """[leaf_id, parent, grandparent, ..., root] -- used to build the
        ancestral-smoothing target for one leaf."""
        path = [leaf_id]
        node = leaf_id
        while node != self.root:
            node = int(self.parent[node])
            path.append(node)
        return path


class HACEHead(nn.Module):
    """The HACE loss + its own output layer. This is its own `hierarchy`
    registry entry rather than a head, because it changes the output
    dimension from n_species to tree.n_nodes -- it cannot sit interchangeably
    where a plain head sits.

    At inference, no special handling is needed for the species-level
    prediction: a leaf has no descendants, so its aggregated probability
    equals its raw softmax probability (q*(leaf) = q_hat(leaf) always, R's
    leaf rows are one-hot on themselves plus their ancestors, but the
    ARGMAX over leaves only needs their own raw mass, and the relative order
    of raw softmax over leaves is unchanged by adding ancestor mass to
    OTHER nodes). Concretely: `logits[:, :n_species]` from raw softmax is
    what accuracy is computed from, same as any other head's leaf logits.
    """

    def __init__(self, in_features, tree: TaxonomyTree, label_smoothing=0.0, dilution=0.6):
        super().__init__()
        self.tree = tree
        self.linear = nn.Linear(in_features, tree.n_nodes)
        self.epsilon = label_smoothing  # Sec 4.2 Step 1, horizontal (optional)
        self.d = dilution  # Sec 4.2 Step 2, ancestral dilution parameter, 0 < d <= 1
        if not (0.0 < dilution <= 1.0):
            raise ValueError(f"dilution must be in (0, 1], got {dilution}")
        self.register_buffer("R", tree.reachability_matrix())
        self._target_cache = {}  # leaf_id -> precomputed sparse ancestral target

    def _ancestral_target_row(self, leaf_id):
        """p_tilde diluted up the ancestor path (Sec 4.2 Step 2), for a SINGLE
        leaf identity -- cached, since the tree is fixed for the whole run."""
        if leaf_id in self._target_cache:
            return self._target_cache[leaf_id]
        n = self.tree.n_nodes
        row = torch.zeros(n)
        leaf_mass = 1.0 - self.epsilon
        path = self.tree.ancestor_path(leaf_id)  # [leaf, parent, ..., root]
        remaining = leaf_mass
        for depth, node in enumerate(path[:-1]):
            share = self.d * (1 - self.d) ** depth * leaf_mass
            row[node] = share
            remaining -= share
        row[path[-1]] += remaining  # whatever is left lands on the root
        if self.epsilon > 0:
            row[: self.tree.n_species] += self.epsilon / self.tree.n_species
        self._target_cache[leaf_id] = row
        return row

    def _targets(self, labels):
        rows = torch.stack([self._ancestral_target_row(int(y)) for y in labels])
        return rows.to(labels.device)

    def forward(self, embeddings, labels=None):
        z = self.linear(embeddings)  # (B, N) raw logits over ALL tree nodes
        q_hat = F.softmax(z, dim=1)
        q_star = q_hat @ self.R  # aggregation: (B,N) @ (N,N) -> (B,N), Sec 4.1

        if labels is None:
            return q_star[:, : self.tree.n_species]  # species-level "logits" (probabilities)

        p_star = self._targets(labels)
        loss = -(p_star * torch.log(q_star.clamp_min(1e-12))).sum(dim=1).mean()
        # Species-level probabilities for accuracy/eval, alongside the loss —
        # the training script's forward() wraps this the same way every other
        # head wraps (loss, logits).
        return loss, q_star[:, : self.tree.n_species]

    def class_directions(self):
        # Species rows only, so this stays comparable to every other head's
        # (n_species, C) matrix for repgeom.
        return self.linear.weight.detach()[: self.tree.n_species]

    def set_margin(self, m):
        pass  # no margin concept
