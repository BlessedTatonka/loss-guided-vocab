"""Learned hard adjacent-pair tokenization for a static embedding table."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from starse.collapse import _MIX_A, _MIX_B
from starse.structured_tokenizer import batched_matching, structured_straight_through

if TYPE_CHECKING:
    from starse.collapse import CollapseTable


class OnlineMergeTable(nn.Module):
    """Score every adjacent pair and emit a legal hard MAP token sequence."""

    def __init__(
        self,
        buckets: int,
        dim: int,
        n_languages: int,
        *,
        rank: int = 32,
        bias_init: float = -2.0,
        pair_key_base: int,
        compiled_pair_keys: torch.Tensor | None = None,
        compiled_pair_scores: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if buckets <= 0 or dim <= 0 or n_languages <= 0 or rank <= 0:
            raise ValueError("online tokenizer dimensions must be positive")
        if pair_key_base <= 0:
            raise ValueError("pair_key_base must be positive")
        if (compiled_pair_keys is None) != (compiled_pair_scores is None):
            raise ValueError("compiled pair keys and scores must be provided together")

        self.buckets = int(buckets)
        self.dim = int(dim)
        self.rank = int(rank)
        self.pair_key_base = int(pair_key_base)
        self.compiled = compiled_pair_keys is not None
        keys = torch.as_tensor(
            [] if compiled_pair_keys is None else compiled_pair_keys, dtype=torch.long
        )
        scores = torch.as_tensor(
            [] if compiled_pair_scores is None else compiled_pair_scores,
            dtype=torch.float32,
        )
        if keys.ndim != 1 or scores.ndim != 1 or keys.numel() != scores.numel():
            raise ValueError("compiled pair keys and scores must have the same length")
        if keys.numel() and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("compiled pair keys must be strictly increasing")
        if keys.numel() and (int(keys.min()) < 0 or int(keys.max()) >= pair_key_base**2):
            raise ValueError("compiled pair key is outside pair_key_base")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("compiled pair scores must be finite")
        self.register_buffer("compiled_pair_keys", keys, persistent=True)
        self.register_buffer("compiled_pair_scores", scores, persistent=True)

        rows = keys.numel() if self.compiled else self.buckets
        self.merged = nn.Parameter(
            torch.zeros(rows, self.dim), requires_grad=not self.compiled
        )
        self.language_bias = nn.Parameter(
            torch.full((n_languages,), float(bias_init)), requires_grad=not self.compiled
        )
        if not self.compiled:
            self.score_hash = nn.Parameter(torch.zeros(self.buckets))
            self.left_projection = nn.Parameter(torch.empty(self.rank, self.dim))
            self.right_projection = nn.Parameter(torch.empty(self.rank, self.dim))
            self.interaction_weight = nn.Parameter(torch.zeros(self.rank))
            nn.init.xavier_uniform_(self.left_projection)
            nn.init.xavier_uniform_(self.right_projection)

        self.last_token_count = torch.empty(0)
        self.last_map_edges = torch.empty(0, 0, dtype=torch.bool)
        self._vocabulary_telemetry = None
        self._cache_audit_active = False
        self._cache_phase: str | None = None
        self._cache_records: deque[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = deque()

    def extra_repr(self) -> str:
        return (
            f"compiled={self.compiled}, buckets={self.buckets}, "
            f"exact_pairs={self.compiled_pair_keys.numel()}, dim={self.dim}, rank={self.rank}, "
            f"languages={self.language_bias.numel()}"
        )

    def set_vocabulary_telemetry(self, window) -> None:
        """Attach detached run telemetry without registering module state."""

        self._vocabulary_telemetry = window

    def _hash(self, left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        return ((left_ids * _MIX_A + right_ids * _MIX_B).abs()) % self.buckets

    def start_cache_audit(self) -> None:
        if self._cache_phase is not None or self._cache_records:
            raise RuntimeError("previous online tokenizer cache audit is incomplete")
        self._cache_audit_active = True

    def begin_cache_record(self) -> None:
        if self._cache_audit_active:
            if self._cache_phase is not None:
                raise RuntimeError("online tokenizer cache phase is already active")
            self._cache_phase = "record"

    def begin_cache_replay(self) -> None:
        if self._cache_audit_active:
            if self._cache_phase is not None:
                raise RuntimeError("online tokenizer cache phase is already active")
            self._cache_phase = "replay"

    def end_cache_phase(self) -> None:
        self._cache_phase = None

    def abort_cache_audit(self) -> None:
        self._cache_records.clear()
        self._cache_phase = None
        self._cache_audit_active = False

    def assert_cache_audit_empty(self) -> None:
        if self._cache_phase is not None or self._cache_records:
            raise RuntimeError("online tokenizer cache audit did not replay every record")
        self._cache_audit_active = False

    def _audit_map(
        self,
        content: torch.Tensor,
        segment: torch.Tensor,
        language: torch.Tensor,
        map_edges: torch.Tensor,
    ) -> None:
        if not self._cache_audit_active or self._cache_phase is None:
            return
        values = tuple(
            tensor.detach().clone()
            for tensor in (content, segment, language, map_edges)
        )
        if self._cache_phase == "record":
            self._cache_records.append(values)
            return
        if self._cache_phase != "replay" or not self._cache_records:
            self.abort_cache_audit()
            raise RuntimeError("cache/replay MAP mismatch")
        expected = self._cache_records.popleft()
        if not all(
            observed.shape == wanted.shape and torch.equal(observed, wanted)
            for observed, wanted in zip(values, expected, strict=True)
        ):
            self.abort_cache_audit()
            raise RuntimeError("cache/replay MAP mismatch")

    def seed_from_collapse(self, source: CollapseTable) -> None:
        """Copy the registered Q15 tensors without sharing parameter storage."""

        if self.compiled:
            raise ValueError("cannot seed a compiled online tokenizer from Q15 hashes")
        expected = {
            "merged": tuple(self.merged.shape),
            "score": tuple(self.score_hash.shape),
            "language_bias": tuple(self.language_bias.shape),
        }
        observed = {
            "merged": tuple(source.merged.shape),
            "score": tuple(source.score.shape),
            "language_bias": tuple(source.language_bias.shape),
        }
        if observed != expected:
            raise ValueError(
                f"Q15 source tensors do not fit online table: {observed} != {expected}"
            )
        with torch.no_grad():
            self.merged.copy_(source.merged)
            self.score_hash.copy_(source.score)
            self.language_bias.copy_(source.language_bias)

    def _edge_scores_and_index(
        self,
        left_ids: torch.Tensor,
        right_ids: torch.Tensor,
        language_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (
            left_ids.ndim == right_ids.ndim == language_ids.ndim == 1
            and left_ids.shape == right_ids.shape == language_ids.shape
        ):
            raise ValueError("edge scorer needs aligned one-dimensional tensors")
        if embedding_weight.ndim != 2 or embedding_weight.shape[1] != self.dim:
            raise ValueError("embedding_weight has the wrong shape")
        if left_ids.numel() and (
            int(torch.minimum(left_ids.min(), right_ids.min())) < 0
            or int(torch.maximum(left_ids.max(), right_ids.max()))
            >= embedding_weight.shape[0]
        ):
            raise ValueError("pair token id is outside the embedding table")
        if language_ids.numel() and (
            int(language_ids.min()) < 0
            or int(language_ids.max()) >= self.language_bias.numel()
        ):
            raise ValueError("pair language id is outside the bias table")

        if self.compiled:
            keys = left_ids.to(torch.long) * self.pair_key_base + right_ids.to(torch.long)
            if self.compiled_pair_keys.numel():
                index = torch.searchsorted(self.compiled_pair_keys, keys)
                safe_index = index.clamp(max=self.compiled_pair_keys.numel() - 1)
                found = (index < self.compiled_pair_keys.numel()) & (
                    self.compiled_pair_keys.index_select(0, safe_index) == keys
                )
                global_score = self.compiled_pair_scores.index_select(0, safe_index)
            else:
                safe_index = torch.zeros_like(keys)
                found = torch.zeros_like(keys, dtype=torch.bool)
                global_score = torch.zeros_like(keys, dtype=torch.float32)
            scores = torch.where(
                found,
                global_score
                + self.language_bias.index_select(0, language_ids).float(),
                torch.zeros_like(global_score),
            )
            index = safe_index
        else:
            index = self._hash(left_ids, right_ids)
            device_type = embedding_weight.device.type
            with torch.autocast(device_type=device_type, enabled=False):
                left = embedding_weight.index_select(0, left_ids).float()
                right = embedding_weight.index_select(0, right_ids).float()
                left_code = torch.tanh(F.linear(left, self.left_projection.float()))
                right_code = torch.tanh(F.linear(right, self.right_projection.float()))
                amortized = (
                    (left_code * right_code * self.interaction_weight.float()).sum(dim=1)
                    / math.sqrt(self.rank)
                )
                scores = (
                    self.score_hash.index_select(0, index).float()
                    + self.language_bias.index_select(0, language_ids).float()
                    + amortized
                )
            found = torch.ones_like(index, dtype=torch.bool)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("nonfinite online tokenizer edge score")
        return scores, index, found

    def edge_scores(
        self,
        left_ids: torch.Tensor,
        right_ids: torch.Tensor,
        language_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return float32 occurrence scores for aligned ids and languages."""

        scores, _, _ = self._edge_scores_and_index(
            left_ids, right_ids, language_ids, embedding_weight
        )
        return scores

    @staticmethod
    def _sentence_layout(
        segment: torch.Tensor, n_sentences: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.zeros(
            n_sentences, dtype=torch.long, device=segment.device
        )
        lengths.index_add_(0, segment, torch.ones_like(segment))
        if bool((lengths < 1).any()):
            raise ValueError("online tokenizer requires at least one token per sentence")
        starts = lengths.cumsum(0) - lengths
        return lengths, starts

    def pool(
        self,
        vectors: torch.Tensor,
        content: torch.Tensor,
        segment: torch.Tensor,
        language: torch.Tensor,
        n_sentences: int,
        *,
        embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the literal tokens emitted by hard MAP with marginal gradients."""

        if vectors.ndim != 2 or vectors.shape != (content.numel(), self.dim):
            raise ValueError("vectors must align with content and the table dimension")
        if content.ndim != 1 or segment.ndim != 1 or content.shape != segment.shape:
            raise ValueError("content and segment must be aligned one-dimensional tensors")
        if language.shape != (n_sentences,):
            raise ValueError("language must contain one id per sentence")
        if segment.numel() and (
            int(segment.min()) < 0 or int(segment.max()) >= n_sentences
        ):
            raise ValueError("segment index is outside the sentence batch")
        lengths, starts = self._sentence_layout(segment, n_sentences)

        numerator = torch.zeros(
            n_sentences, self.dim, dtype=vectors.dtype, device=vectors.device
        )
        numerator.index_add_(0, segment, vectors)
        token_count = lengths.to(dtype=vectors.dtype)
        maximum_edges = int((lengths - 1).max().item())
        edge_mask = torch.zeros(
            n_sentences, maximum_edges, dtype=torch.bool, device=vectors.device
        )
        padded_scores = torch.zeros(
            n_sentences, maximum_edges, dtype=torch.float32, device=vectors.device
        )

        if maximum_edges:
            inside = segment[:-1] == segment[1:]
            positions = torch.nonzero(inside, as_tuple=False).squeeze(1)
            edge_segment = segment.index_select(0, positions)
            within = positions - starts.index_select(0, edge_segment)
            left_ids = content.index_select(0, positions)
            right_ids = content.index_select(0, positions + 1)
            edge_language = language.index_select(0, edge_segment)
            scores, pair_index, eligible = self._edge_scores_and_index(
                left_ids, right_ids, edge_language, embedding_weight
            )
            eligible_segment = edge_segment[eligible]
            eligible_within = within[eligible]
            edge_mask[eligible_segment, eligible_within] = True
            padded_scores[eligible_segment, eligible_within] = scores[eligible]
            matching = batched_matching(padded_scores, edge_mask)
            if self.training:
                padded_edges = structured_straight_through(
                    matching.marginals, matching.map_edges, dtype=vectors.dtype
                )
            else:
                padded_edges = matching.map_edges.to(dtype=vectors.dtype)
            edges = padded_edges[eligible_segment, eligible_within]
            pair_vectors = self.merged.index_select(0, pair_index[eligible]).to(vectors.dtype)
            eligible_positions = positions[eligible]
            delta = (
                pair_vectors
                - vectors.index_select(0, eligible_positions)
                - vectors.index_select(0, eligible_positions + 1)
            )
            numerator.index_add_(0, eligible_segment, delta * edges.unsqueeze(1))
            token_count.index_add_(0, eligible_segment, -edges)
            map_edges = matching.map_edges
            telemetry_marginals = matching.marginals
            telemetry_hash_indices = pair_index[eligible]
        else:
            map_edges = edge_mask
            telemetry_marginals = torch.zeros_like(padded_scores)
            telemetry_hash_indices = torch.empty(
                0, dtype=torch.long, device=content.device
            )

        if not bool(torch.isfinite(token_count).all()) or bool((token_count < 1).any()):
            raise FloatingPointError("invalid online tokenizer token count")
        mean = numerator / token_count.unsqueeze(1)
        embeddings = F.normalize(mean, dim=1)
        if not bool(torch.isfinite(embeddings).all()):
            raise FloatingPointError("nonfinite online tokenizer sentence embedding")
        self.last_token_count = token_count.detach()
        self.last_map_edges = map_edges.detach()
        if self._vocabulary_telemetry is not None:
            self._vocabulary_telemetry.observe_online(
                base_tokens=int(content.numel()),
                edge_scores=padded_scores,
                marginals=telemetry_marginals,
                map_edges=map_edges,
                hash_indices=telemetry_hash_indices,
                edge_mask=edge_mask,
            )
        self._audit_map(content, segment, language, map_edges)
        return embeddings, map_edges
