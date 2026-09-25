"""K vocabularies, and a learned assignment of languages to them.

The earlier fusion attempt merged *per token*: two languages could share a row
for one word and keep separate rows for the next. That is cell-level sharing,
not vocabulary sharing, and it is why the readout came out as mush — the merge
decision was never coupled across a language's tokens.

Here the decision lives at the language level. Each language holds one
distribution over K vocabularies, used for every token it emits:

    E[l, w] = E_shared[w] + sum_k pi[l, k] * Delta_k[w]

`pi` is L x K — 48 numbers for twelve languages and four vocabularies — so the
grouping is read directly off the parameter instead of being inferred from row
distances. An entropy penalty pushes each language toward one vocabulary, and K
is the explicit axis: K=1 is today's single shared vocabulary, K=L with a hard
assignment is one per language, and where quality stops improving is the answer
to how many a multilingual encoder actually needs.

Offsets start at zero so the model is numerically identical to the shared one at
step zero. The assignment logits carry the small random asymmetry instead: with
identical offsets and a uniform assignment every expert would receive the same
gradient and the K vocabularies would collapse back into one by symmetry.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class VocabAssignment(nn.Module):
    def __init__(self, index_path: str, vocab_size: int, dim: int,
                 n_languages: int, n_vocabs: int = 4, seed: int = 20260727) -> None:
        super().__init__()
        data = np.load(index_path)
        tokens = np.unique(data["token_index"].astype(np.int64))
        self.n_tokens = int(tokens.size)
        self.n_vocabs = int(n_vocabs)

        slot = torch.full((vocab_size,), -1, dtype=torch.long)
        slot[torch.tensor(tokens)] = torch.arange(self.n_tokens)
        self.register_buffer("slot", slot, persistent=False)

        self.offsets = nn.Parameter(torch.zeros(self.n_vocabs, self.n_tokens, dim))
        generator = torch.Generator().manual_seed(seed)
        self.logits = nn.Parameter(
            torch.randn(n_languages, self.n_vocabs, generator=generator) * 0.01)

    def assignment(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def entropy(self) -> torch.Tensor:
        """Mean assignment entropy; penalising it drives each language to one slot."""
        pi = self.assignment()
        return -(pi * torch.log(pi.clamp_min(1e-9))).sum(-1).mean()

    def forward(self, token_ids: torch.Tensor, lang_per_token: torch.Tensor,
                shared: torch.Tensor) -> torch.Tensor:
        index = self.slot[token_ids]
        has = index >= 0
        if not bool(has.any()):
            return shared
        safe = index.clamp(min=0)
        # (K, N, dim) gathered offsets weighted by each token's language row of pi
        gathered = self.offsets[:, safe]
        weights = self.assignment()[lang_per_token]
        delta = torch.einsum("nk,knd->nd", weights, gathered)
        return shared + delta * has.unsqueeze(1)

    def extra_repr(self) -> str:
        return f"tokens={self.n_tokens}, vocabs={self.n_vocabs}"
