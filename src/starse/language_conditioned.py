"""Language-conditioned static embedding.

The language is carried as a marker token prepended to each text, so the two
sides of a parallel pair can carry different language identities without
changing the trainer's two-column dataset schema.
"""

from __future__ import annotations

import torch
from torch import nn

MODES = (
    "none", "marker", "centroid", "diag", "lowrank", "bilinear", "senses",
    "hard", "gate", "gate_senses", "split", "ngram", "ngram3", "wordpool",
    "ngram_gate", "fuse", "vocabmoe", "collapse", "collapse_shared",
    "collapse_dynamic", "collapse_reclaimed", "collapse_online",
    "collapse_online_compiled", "collapse_recursive", "idfpool",
)

# Language-specific pooling changes the relative numerator composition. One
# shared vocabulary row cannot represent a different weight in every language;
# the positive weighted-mean denominator itself cancels under cosine scoring.
GATING_MODES = ("gate", "gate_senses")


class LanguageConditioner(nn.Module):
    """The conditioning head. Owns every parameter the base table does not."""

    def __init__(
        self,
        *,
        mode: str,
        dim: int,
        vocab_size: int,
        n_languages: int,
        code_dim: int = 16,
        n_senses: int = 64,
        rank: int = 8,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        self.mode = mode
        self.dim = dim
        self.n_languages = n_languages
        self.code_dim = code_dim
        self.n_senses = n_senses
        self.rank = rank

        if mode == "centroid":
            self.centroid = nn.Parameter(torch.zeros(n_languages, dim))
        elif mode == "diag":
            self.scale = nn.Parameter(torch.ones(n_languages, dim))
        elif mode == "lowrank":
            self.left = nn.Parameter(torch.randn(n_languages, dim, rank) * 0.02)
            self.right = nn.Parameter(torch.zeros(n_languages, rank, dim))
        elif mode == "bilinear":
            self.code = nn.EmbeddingBag(vocab_size, code_dim, mode="mean")
            nn.init.normal_(self.code.weight, std=0.02)
            self.readout = nn.Parameter(torch.zeros(n_languages, code_dim, dim))
        if mode in ("gate", "gate_senses"):
            self.gate_code = nn.Embedding(vocab_size, code_dim)
            nn.init.normal_(self.gate_code.weight, std=0.02)
            self.gate_language = nn.Parameter(torch.zeros(n_languages, code_dim))
        if mode in ("senses", "hard", "gate_senses"):
            self.code = nn.Embedding(vocab_size, code_dim)
            nn.init.normal_(self.code.weight, std=0.02)
            self.language = nn.Parameter(torch.ones(n_languages, code_dim))
            self.probe = nn.Parameter(torch.randn(n_senses, code_dim) * 0.02)
            self.senses = nn.Parameter(torch.zeros(n_senses, dim))

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, languages={self.n_languages}, "
                f"code_dim={self.code_dim}, senses={self.n_senses}, rank={self.rank}")

    def token_gate(self, token_ids: torch.Tensor, lang_per_token: torch.Tensor) -> torch.Tensor:
        """Multiplicative weight per token, centred on 1 at initialisation."""
        score = (self.gate_code(token_ids) * self.gate_language[lang_per_token]).sum(-1)
        return 2.0 * torch.sigmoid(score)

    def sense_weights(self, token_ids: torch.Tensor, lang_per_token: torch.Tensor) -> torch.Tensor:
        """Alpha over the sense dictionary, per token, given its language."""
        a = self.code(token_ids)
        v = self.language[lang_per_token]
        logits = (a * v) @ self.probe.T
        alpha = torch.softmax(logits, dim=-1)
        if self.mode == "hard":
            index = alpha.argmax(dim=-1, keepdim=True)
            onehot = torch.zeros_like(alpha).scatter_(-1, index, 1.0)
            alpha = onehot + alpha - alpha.detach()
        return alpha

    def forward(
        self,
        pooled: torch.Tensor,
        language: torch.Tensor,
        token_ids: torch.Tensor,
        segment: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.mode
        if mode in ("none", "marker"):
            return pooled
        if mode == "centroid":
            return pooled - self.centroid[language]
        if mode == "diag":
            return pooled * self.scale[language]
        if mode == "lowrank":
            left = self.left[language]
            right = self.right[language]
            latent = torch.einsum("bd,bdr->br", pooled, left)
            return pooled + torch.einsum("br,brd->bd", latent, right)
        if mode == "bilinear":
            offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=token_ids.device),
                lengths.cumsum(0)[:-1],
            ])
            mean_code = self.code(token_ids, offsets)
            return pooled + torch.einsum("bk,bkd->bd", mean_code, self.readout[language])
        if mode == "gate":
            return pooled

        alpha = self.sense_weights(token_ids, language[segment])
        summed = torch.zeros(pooled.shape[0], self.n_senses,
                             dtype=alpha.dtype, device=alpha.device)
        summed.index_add_(0, segment, alpha)
        mean_alpha = summed / lengths.clamp(min=1).unsqueeze(1).to(summed.dtype)
        return pooled + mean_alpha @ self.senses


def split_markers(
    input_ids: torch.Tensor,
    offsets: torch.Tensor,
    marker_lookup: torch.Tensor,
    keep_marker: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Peel the leading language marker off every sequence.

    Returns content ids, sentence indices, content lengths, per-sentence
    languages, and offsets for the content-only token stream.
    """
    total = input_ids.numel()
    batch = offsets.numel()
    marker_ids = input_ids[offsets]
    language = marker_lookup[marker_ids]

    ends = torch.cat([offsets[1:], torch.tensor([total], device=offsets.device)])
    full_lengths = ends - offsets
    if keep_marker:
        segment = torch.repeat_interleave(
            torch.arange(batch, device=offsets.device), full_lengths)
        return input_ids, segment, full_lengths, language, offsets

    keep = torch.ones(total, dtype=torch.bool, device=input_ids.device)
    keep[offsets] = False
    content = input_ids[keep]
    lengths = (full_lengths - 1).clamp(min=0)
    segment = torch.repeat_interleave(
        torch.arange(batch, device=offsets.device), lengths)
    new_offsets = torch.cat([
        torch.zeros(1, dtype=torch.long, device=offsets.device),
        lengths.cumsum(0)[:-1],
    ])
    return content, segment, lengths, language, new_offsets
