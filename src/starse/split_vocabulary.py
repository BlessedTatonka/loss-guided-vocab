"""A second vocabulary for contested (language, token) pairs.

Every other conditioning mechanism in this repo failed for one measured reason:
the shared table rewrites about 85% of a row's magnitude in 3,000 steps, so it
absorbs any per-language transform a head could learn and leaves the head
redundant.

This does something different. A selected (language, token) pair gets its **own
row**, and a learned gate mixes it with the shared row:

    E'[w, l] = (1 - g) * E_shared[w] + g * E_split[idx(l, w)]

The split row is free capacity, not a correction, so there is nothing for the
shared table to absorb. Split rows start as exact copies of their shared row, so
the model is numerically identical to the unconditioned one at step zero
whatever the gate says, and both paths receive gradient immediately — the
failure mode where a zero-initialised head never leaves zero cannot occur.

An L1 penalty on the gate makes a split row earn its keep. After training the
gate is the answer to the scientific question: which pairs actually needed
separating, and does that agree with the interference measured beforehand.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SplitVocabulary(nn.Module):
    def __init__(self, index_path: str, vocab_size: int, dim: int, n_languages: int,
                 gate_init: float = 0.0) -> None:
        super().__init__()
        data = np.load(index_path)
        language_index = torch.tensor(data["language_index"].astype(np.int64))
        token_index = torch.tensor(data["token_index"].astype(np.int64))
        self.n_rows = int(language_index.numel())

        # (language, token) -> split row, as one flat lookup. -1 means "shared
        # only", which is the vast majority of pairs.
        lookup = torch.full((n_languages, vocab_size), -1, dtype=torch.long)
        lookup[language_index, token_index] = torch.arange(self.n_rows)
        self.register_buffer("lookup", lookup, persistent=False)
        self.register_buffer("row_language", language_index, persistent=False)
        self.register_buffer("row_token", token_index, persistent=False)

        self.rows = nn.Parameter(torch.zeros(self.n_rows, dim))
        self.gate_logit = nn.Parameter(torch.full((self.n_rows,), float(gate_init)))
        self._initialised = False

    @torch.no_grad()
    def seed_from(self, table: torch.Tensor) -> None:
        """Copy each split row from the shared row it specialises."""
        self.rows.copy_(table[self.row_token])
        self._initialised = True

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def penalty(self) -> torch.Tensor:
        """Mean gate opening — an L1 that pushes unused rows back to shared."""
        return self.gate().mean()

    def fusion_penalty(self) -> torch.Tensor:
        """Pull every language's row for a token toward the consensus for it.

        This is the merging mechanism. Training starts with one private row per
        (language, token) — full capacity, no interference — and this term makes
        a row stay distinct only if the contrastive loss pays for it. Sweeping
        its weight walks continuously from one vocabulary per language to a
        single shared one, so the *effective* number of vocabularies is read off
        rather than chosen.

        Fusing toward the per-token mean is the O(L) relaxation of the O(L^2)
        all-pairs fused lasso; rows that need not differ collapse onto the
        centroid, and which languages stay off it together is the clustering.
        """
        token = self.row_token
        uniq, inverse = torch.unique(token, return_inverse=True)
        total = torch.zeros(uniq.numel(), self.rows.shape[1],
                            dtype=self.rows.dtype, device=self.rows.device)
        total.index_add_(0, inverse, self.rows)
        count = torch.zeros(uniq.numel(), dtype=self.rows.dtype, device=self.rows.device)
        count.index_add_(0, inverse, torch.ones_like(inverse, dtype=self.rows.dtype))
        centroid = total / count.clamp(min=1.0).unsqueeze(1)
        # Group lasso over languages: the L2 norm of each deviation, averaged.
        return (self.rows - centroid[inverse]).norm(dim=1).mean()

    def forward(self, token_ids: torch.Tensor, lang_per_token: torch.Tensor,
                shared: torch.Tensor, use_gate: bool = True) -> torch.Tensor:
        row = self.lookup[lang_per_token, token_ids]
        has = row >= 0
        if not bool(has.any()):
            return shared
        safe = row.clamp(min=0)
        if not use_gate:
            # Fusion mode: the private row replaces the shared one outright, and
            # merging is driven by the penalty rather than by a mixing gate.
            return torch.where(has.unsqueeze(1), self.rows[safe], shared)
        g = self.gate()[safe].unsqueeze(1) * has.unsqueeze(1)
        return shared * (1.0 - g) + self.rows[safe] * g

    def extra_repr(self) -> str:
        return f"rows={self.n_rows}"
