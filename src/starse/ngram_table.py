"""Hashed token n-gram rows for a static encoder.

Fifteen representation-side interventions in this repo produced nothing, and the
per-language numbers say why: quality tracks *tokenisation*, not capacity.
Spearman between tokens-per-word and held-out Recall@1 is -0.607 over 57
space-separated languages; the third that tokenise worst average 0.191 against
0.295 for the third that tokenise best. Georgian needs 5.7 tokens per word and
scores 0.050.

Those languages do not lack a good vector for a word — they lack any row that
*means* a word. A Transformer reassembles the pieces with attention; a
mean-pooled static encoder cannot. The only repair available to it is a bigger
lookup unit.

So each adjacent token n-gram inside a sentence gets a row, hashed into a fixed
table, pooled alongside the unigram rows:

    h(x) = ( sum_w E[w] + sum_g F[hash(g)] ) / (|U| + |G|)

Allocation across languages needs no rule: a fragmented language simply emits
more distinct n-grams and takes more of the table, while a language that already
tokenises into words gains little. The budget follows the deficit by itself.

Rows start at zero, so the pooled direction — and therefore every cosine — is
identical to the unconditioned model at step zero, while the rows still receive
gradient from the first batch. Unlike every conditioning head tried here, an
n-gram row is a *new input feature* rather than a second route to an existing
one, so the shared table cannot absorb it: the feature does not otherwise exist.
"""

from __future__ import annotations

import torch
from torch import nn

# Odd 32-bit constants; the product mixes the low bits that adjacent token ids
# share, which matters because ids are correlated within a language.
_MIX_A = 2654435761
_MIX_B = 40503


class NgramTable(nn.Module):
    def __init__(self, buckets: int, dim: int, orders: tuple[int, ...] = (2,)) -> None:
        super().__init__()
        if any(order < 2 for order in orders):
            raise ValueError("n-gram orders must be at least 2")
        self.buckets = int(buckets)
        self.orders = tuple(orders)
        self.rows = nn.Parameter(torch.zeros(self.buckets, dim))

    def extra_repr(self) -> str:
        return f"buckets={self.buckets}, orders={self.orders}"

    def _hash(self, ids: torch.Tensor, order: int) -> torch.Tensor:
        value = torch.zeros_like(ids[0])
        for position in range(order):
            value = value * _MIX_A + ids[position] * _MIX_B + position
        return value.abs() % self.buckets

    def gather(
        self,
        content: torch.Tensor,
        segment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows and their sentence index for every n-gram inside a sentence."""
        vectors: list[torch.Tensor] = []
        segments: list[torch.Tensor] = []
        for order in self.orders:
            if content.numel() <= order:
                continue
            window = [content[i: content.numel() - order + 1 + i] for i in range(order)]
            starts = segment[: segment.numel() - order + 1]
            # An n-gram must lie inside one sentence, so every position in the
            # window has to carry the same segment id.
            same = torch.ones_like(starts, dtype=torch.bool)
            for i in range(1, order):
                same &= segment[i: segment.numel() - order + 1 + i] == starts
            if not bool(same.any()):
                continue
            index = self._hash([w[same] for w in window], order)
            vectors.append(self.rows[index])
            segments.append(starts[same])
        if not vectors:
            empty = content.new_zeros((0,), dtype=torch.long)
            return self.rows.new_zeros((0, self.rows.shape[1])), empty
        return torch.cat(vectors), torch.cat(segments)
