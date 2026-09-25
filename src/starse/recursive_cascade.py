"""Recursive criterion-grown token vocabulary primitives.

This module is intentionally small and CPU-testable.  It establishes the Q32
mechanism contract before the recursive tokenizer is connected to the streaming
trainer: learned tokens are emitted as discrete ids, may parent later tokens,
receive gradients, and resume with exact optimizer state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from starse.collapse import _weighted_index_add_


@dataclass(frozen=True)
class CascadeToken:
    token_id: int
    span: tuple[int, ...]
    generation: int
    left_parent: int | None
    right_parent: int | None


class RecursiveCascadeVocabulary(nn.Module):
    """A monotone DAG vocabulary over canonical base-token spans."""

    def __init__(
        self,
        base_embeddings: torch.Tensor,
        *,
        max_span_length: int = 32,
    ) -> None:
        super().__init__()
        if base_embeddings.ndim != 2 or not base_embeddings.shape[0]:
            raise ValueError("base_embeddings must be a nonempty matrix")
        if not base_embeddings.is_floating_point() or not bool(
            torch.isfinite(base_embeddings).all()
        ):
            raise ValueError("base_embeddings must be finite floating point")
        if max_span_length < 2:
            raise ValueError("max_span_length must be at least two")
        self.dim = int(base_embeddings.shape[1])
        self.max_span_length = int(max_span_length)
        self.base_size = int(base_embeddings.shape[0])
        self.embeddings = nn.ParameterList(
            nn.Parameter(row.detach().clone()) for row in base_embeddings
        )
        self._tokens = tuple(
            CascadeToken(index, (index,), 0, None, None)
            for index in range(self.base_size)
        )

    @property
    def tokens(self) -> tuple[CascadeToken, ...]:
        return self._tokens

    def _span_index(self) -> dict[tuple[int, ...], int]:
        return {token.span: token.token_id for token in self._tokens}

    def tokenize(self, base_ids: Sequence[int]) -> tuple[int, ...]:
        """Greedily emit longest canonical spans with stable id tie-breaking."""

        sequence = tuple(int(value) for value in base_ids)
        if any(value < 0 or value >= self.base_size for value in sequence):
            raise ValueError("tokenize input must contain only base token ids")
        spans = self._span_index()
        by_first: dict[int, list[tuple[tuple[int, ...], int]]] = {}
        for span, token_id in spans.items():
            by_first.setdefault(span[0], []).append((span, token_id))
        for candidates in by_first.values():
            candidates.sort(key=lambda item: (-len(item[0]), item[1]))

        emitted: list[int] = []
        position = 0
        while position < len(sequence):
            chosen_span = (sequence[position],)
            chosen_id = sequence[position]
            for span, token_id in by_first.get(sequence[position], ()):
                if sequence[position : position + len(span)] == span:
                    chosen_span, chosen_id = span, token_id
                    break
            emitted.append(chosen_id)
            position += len(chosen_span)
        return tuple(emitted)

    @staticmethod
    def adjacent_pairs(token_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
        emitted = tuple(int(value) for value in token_ids)
        return tuple(zip(emitted, emitted[1:]))

    def lookup(self, token_ids: Sequence[int]) -> torch.Tensor:
        ids = tuple(int(value) for value in token_ids)
        if not ids:
            raise ValueError("lookup requires at least one token")
        if any(value < 0 or value >= len(self.embeddings) for value in ids):
            raise ValueError("lookup token id is outside the vocabulary")
        return torch.stack([self.embeddings[value] for value in ids])

    def encode_mean(self, base_ids: Sequence[int]) -> tuple[torch.Tensor, tuple[int, ...]]:
        emitted = self.tokenize(base_ids)
        return self.lookup(emitted).mean(dim=0), emitted

    @staticmethod
    def _average_parent_state(
        optimizer: torch.optim.Optimizer,
        left: nn.Parameter,
        right: nn.Parameter,
    ) -> dict[Any, Any]:
        left_state = optimizer.state.get(left, {})
        right_state = optimizer.state.get(right, {})
        if not left_state or set(left_state) != set(right_state):
            raise ValueError("both parents require aligned initialized optimizer state")
        result: dict[Any, Any] = {}
        for name in left_state:
            left_value, right_value = left_state[name], right_state[name]
            if torch.is_tensor(left_value) != torch.is_tensor(right_value):
                raise ValueError(f"optimizer parent state type differs: {name}")
            if not torch.is_tensor(left_value):
                if left_value != right_value:
                    raise ValueError(f"optimizer parent scalar differs: {name}")
                result[name] = left_value
            elif left_value.ndim == 0:
                if not torch.equal(left_value, right_value):
                    raise ValueError(f"optimizer parent step differs: {name}")
                result[name] = left_value.detach().clone()
            else:
                if left_value.shape != left.shape or right_value.shape != right.shape:
                    raise ValueError(f"optimizer parent row shape differs: {name}")
                result[name] = (0.5 * (left_value + right_value)).detach().clone()
        return result

    def promote_pairs(
        self,
        pairs: Iterable[tuple[int, int]],
        *,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[CascadeToken, ...]:
        """Materialize every new canonical span and attach it to Adam in place."""

        requested = tuple((int(left), int(right)) for left, right in pairs)
        if not requested:
            return ()
        if len(set(requested)) != len(requested):
            raise ValueError("promotion pairs must be unique")
        existing = self._span_index()
        additions: list[CascadeToken] = []
        for left_id, right_id in requested:
            if not (0 <= left_id < len(self._tokens) and 0 <= right_id < len(self._tokens)):
                raise ValueError("promotion parent id is outside the vocabulary")
            left_token, right_token = self._tokens[left_id], self._tokens[right_id]
            span = left_token.span + right_token.span
            if len(span) > self.max_span_length:
                raise ValueError("promotion exceeds max_span_length")
            if span in existing:
                raise ValueError("promotion span already exists")
            generation = max(left_token.generation, right_token.generation) + 1
            token_id = len(self._tokens) + len(additions)
            parameter = nn.Parameter(
                0.5
                * (
                    self.embeddings[left_id].detach()
                    + self.embeddings[right_id].detach()
                )
            )
            state = self._average_parent_state(
                optimizer, self.embeddings[left_id], self.embeddings[right_id]
            )
            self.embeddings.append(parameter)
            optimizer.param_groups[0]["params"].append(parameter)
            optimizer.state[parameter] = state
            token = CascadeToken(token_id, span, generation, left_id, right_id)
            additions.append(token)
            existing[span] = token_id
        self._tokens = (*self._tokens, *additions)
        return tuple(additions)

    def snapshot(self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        return {
            "version": 1,
            "protocol": "recursive-cascade-micro-state-v1",
            "base_size": self.base_size,
            "dim": self.dim,
            "max_span_length": self.max_span_length,
            "tokens": [
                {
                    "token_id": token.token_id,
                    "span": list(token.span),
                    "generation": token.generation,
                    "left_parent": token.left_parent,
                    "right_parent": token.right_parent,
                }
                for token in self._tokens
            ],
            # PyTorch state_dict values alias live parameter/optimizer storage.
            # A resume snapshot must be immutable while the source run keeps
            # training, so clone the complete trees at the snapshot boundary.
            "model": copy.deepcopy(self.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }

    @classmethod
    def from_snapshot(
        cls,
        payload: dict[str, Any],
        *,
        optimizer_kwargs: dict[str, Any],
    ) -> tuple[RecursiveCascadeVocabulary, torch.optim.AdamW]:
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or payload.get("protocol") != "recursive-cascade-micro-state-v1"
        ):
            raise ValueError("recursive cascade snapshot identity differs")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or len(tokens) < int(payload["base_size"]):
            raise ValueError("recursive cascade snapshot tokens are invalid")
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("recursive cascade snapshot model is invalid")
        base_rows = torch.stack(
            [model_state[f"embeddings.{index}"] for index in range(int(payload["base_size"]))]
        )
        restored = cls(base_rows, max_span_length=int(payload["max_span_length"]))
        restored_tokens: list[CascadeToken] = []
        for index, row in enumerate(tokens):
            token = CascadeToken(
                int(row["token_id"]),
                tuple(int(value) for value in row["span"]),
                int(row["generation"]),
                None if row["left_parent"] is None else int(row["left_parent"]),
                None if row["right_parent"] is None else int(row["right_parent"]),
            )
            if token.token_id != index:
                raise ValueError("recursive cascade token ids are not contiguous")
            restored_tokens.append(token)
        for index in range(restored.base_size, len(restored_tokens)):
            restored.embeddings.append(
                nn.Parameter(model_state[f"embeddings.{index}"].detach().clone())
            )
        restored._tokens = tuple(restored_tokens)
        restored.load_state_dict(model_state, strict=True)
        optimizer = torch.optim.AdamW(restored.parameters(), **optimizer_kwargs)
        optimizer.load_state_dict(payload["optimizer"])
        return restored, optimizer


def recursive_pair_key(left: int, right: int) -> int:
    """Return a collision-free Cantor identity for two non-negative token ids."""

    left, right = int(left), int(right)
    if left < 0 or right < 0:
        raise ValueError("recursive pair parents must be non-negative")
    total = left + right
    key = total * (total + 1) // 2 + right
    if key > torch.iinfo(torch.int64).max:
        raise OverflowError("recursive pair identity exceeds int64")
    return key


def decode_recursive_pair_key(key: int) -> tuple[int, int]:
    """Invert :func:`recursive_pair_key` exactly using integer arithmetic."""

    key = int(key)
    if key < 0:
        raise ValueError("recursive pair key must be non-negative")
    diagonal = (math.isqrt(8 * key + 1) - 1) // 2
    diagonal_start = diagonal * (diagonal + 1) // 2
    right = key - diagonal_start
    left = diagonal - right
    if recursive_pair_key(left, right) != key:
        raise ValueError("recursive pair key is not canonical")
    return left, right


def _recursive_pair_keys(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.dtype != torch.long or right.dtype != torch.long:
        raise ValueError("recursive token ids must use torch.long")
    if bool((left < 0).any()) or bool((right < 0).any()):
        raise ValueError("recursive token ids must be non-negative")
    total = left + right
    # Cantor pairing is exact while the triangular term fits signed int64.
    if total.numel() and int(total.max()) > 3_037_000_498:
        raise OverflowError("recursive pair identity exceeds int64")
    return total * (total + 1) // 2 + right


def _proposal_hash(
    left: torch.Tensor, right: torch.Tensor, buckets: int
) -> torch.Tensor:
    """Preserve the Q15 proposal-row mapping at the exact Q32 launch."""

    return ((left * 2_654_435_761 + right * 40_503).abs()) % int(buckets)


class ExactRecursiveDiscovery:
    """Unbounded exact pair/language statistics for one finite half-window."""

    def __init__(self, *, n_languages: int) -> None:
        if n_languages <= 0:
            raise ValueError("recursive discovery needs a positive language count")
        self.n_languages = int(n_languages)
        self.records: dict[int, list[float | int]] = {}
        self.observed_occurrences = 0
        self.transferred_records = 0

    @torch.no_grad()
    def observe(
        self,
        pair_key: torch.Tensor,
        language: torch.Tensor,
        probability: torch.Tensor,
        gradient: torch.Tensor,
    ) -> None:
        if not pair_key.numel():
            return
        utility = -probability.detach().float() * gradient.detach().float()
        finite = (
            torch.isfinite(utility)
            & torch.isfinite(probability.detach().float())
            & (language >= 0)
            & (language < self.n_languages)
        )
        if not bool(finite.any()):
            return
        pair_key = pair_key.detach()[finite].to(torch.long)
        language = language.detach()[finite].to(torch.long)
        probability = probability.detach()[finite].float()
        utility = utility[finite]
        self.observed_occurrences += int(pair_key.numel())
        composite = pair_key * self.n_languages + language
        unique, inverse = torch.unique(composite, return_inverse=True)
        utility_sum = torch.zeros(unique.numel(), device=utility.device)
        probability_sum = torch.zeros(unique.numel(), device=utility.device)
        support = torch.zeros(unique.numel(), dtype=torch.long, device=utility.device)
        positive_utility_support = torch.zeros(
            unique.numel(), dtype=torch.long, device=utility.device
        )
        utility_sum.index_add_(0, inverse, utility)
        probability_sum.index_add_(0, inverse, probability)
        support.index_add_(0, inverse, torch.ones_like(inverse))
        positive_utility_support.index_add_(
            0, inverse, (utility > 0).to(torch.long)
        )
        rows = zip(
            unique.cpu().tolist(),
            utility_sum.cpu().tolist(),
            probability_sum.cpu().tolist(),
            support.cpu().tolist(),
            positive_utility_support.cpu().tolist(),
            strict=True,
        )
        for composite_key, value, probability_value, count, positive_count in rows:
            key = int(composite_key)
            row = self.records.setdefault(key, [0.0, 0.0, 0, 0])
            row[0] = float(row[0]) + float(value)
            row[1] = float(row[1]) + float(probability_value)
            row[2] = int(row[2]) + int(count)
            row[3] = int(row[3]) + int(positive_count)
            self.transferred_records += 1

    def snapshot_records(self) -> tuple[dict[str, float | int], ...]:
        result = []
        for composite, row in sorted(self.records.items()):
            pair_key, language = divmod(composite, self.n_languages)
            utility, probability_sum, support, positive_utility_support = (
                float(row[0]), float(row[1]), int(row[2]), int(row[3])
            )
            result.append(
                {
                    "composite_key": composite,
                    "pair_key": pair_key,
                    "language_index": language,
                    "utility": utility,
                    "probability_sum": probability_sum,
                    "captured_support": support,
                    "positive_utility_support": positive_utility_support,
                    "mean_probability": probability_sum / support,
                }
            )
        return tuple(result)


class DenseRecursiveCascade(nn.Module):
    """Batched recursive tokenizer plus one dense trainable learned-token table.

    Accepted rules are applied once per structural generation.  Consequently a
    token born at boundary ``g`` can be emitted and used by discovery during the
    following interval, but a transition cannot recursively consume rows it is
    creating itself.  This makes every structural update an atomic DAG layer.
    """

    def __init__(
        self,
        *,
        base_size: int,
        dim: int,
        n_languages: int,
        residual_buckets: int,
        max_span_length: int = 32,
        restored_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if base_size <= 0 or dim <= 0 or n_languages <= 0:
            raise ValueError("recursive cascade dimensions must be positive")
        if residual_buckets <= 0 or max_span_length < 2:
            raise ValueError("recursive residual capacity/span limit is invalid")
        self.base_size = int(base_size)
        self.dim = int(dim)
        self.n_languages = int(n_languages)
        self.residual_buckets = int(residual_buckets)
        self.max_span_length = int(max_span_length)

        restored = dict(restored_state or {})
        metadata_names = {
            "rule_left", "rule_right", "rule_generation", "span_offsets",
            "span_values", "rule_active", "learned",
        }
        legacy_metadata_names = metadata_names - {"rule_active"}
        if restored and set(restored) not in (metadata_names, legacy_metadata_names):
            raise ValueError("recursive cascade restored state has an invalid schema")
        learned = restored.get("learned", torch.empty((0, self.dim)))
        if learned.ndim != 2 or learned.shape[1] != self.dim:
            raise ValueError("recursive learned table has an invalid shape")
        self.learned = nn.Parameter(learned.detach().clone())
        count = int(learned.shape[0])
        defaults = {
            "rule_left": torch.empty(0, dtype=torch.long),
            "rule_right": torch.empty(0, dtype=torch.long),
            "rule_generation": torch.empty(0, dtype=torch.long),
            "span_offsets": torch.zeros(1, dtype=torch.long),
            "span_values": torch.empty(0, dtype=torch.long),
        }
        for name, default in defaults.items():
            value = restored.get(name, default).detach().clone().to(torch.long)
            self.register_buffer(name, value, persistent=True)
        active = restored.get(
            "rule_active", torch.ones(count, dtype=torch.bool)
        ).detach().clone().to(torch.bool)
        self.register_buffer("rule_active", active, persistent=True)
        if not (
            self.rule_left.numel() == count
            and self.rule_right.numel() == count
            and self.rule_generation.numel() == count
            and self.rule_active.numel() == count
            and self.span_offsets.numel() == count + 1
            and int(self.span_offsets[0]) == 0
            and int(self.span_offsets[-1]) == self.span_values.numel()
        ):
            raise ValueError("recursive rule metadata does not align with learned rows")
        if count and (
            bool((self.rule_generation <= 0).any())
            or bool((self.rule_generation[1:] < self.rule_generation[:-1]).any())
            or int(self.rule_left.max()) >= self.base_size + count
            or int(self.rule_right.max()) >= self.base_size + count
        ):
            raise ValueError("recursive rule DAG metadata is invalid")
        self._validate_metadata()

        self.proposal_merged = nn.Parameter(torch.zeros(self.residual_buckets, self.dim))
        self.proposal_score = nn.Parameter(torch.zeros(self.residual_buckets))
        self.language_bias = nn.Parameter(torch.full((self.n_languages,), -2.0))
        self.discovery: ExactRecursiveDiscovery | None = None
        self.usage_counts: torch.Tensor | None = None

    @property
    def learned_count(self) -> int:
        return int(self.learned.shape[0])

    @property
    def generation(self) -> int:
        return int(self.rule_generation[-1]) if self.rule_generation.numel() else 0

    @property
    def active_count(self) -> int:
        return int(self.rule_active.sum())

    def configure_discovery(self) -> None:
        self.discovery = ExactRecursiveDiscovery(n_languages=self.n_languages)
        self.usage_counts = torch.zeros(
            self.learned_count, dtype=torch.long, device=self.learned.device
        )

    def snapshot_usage_token_ids(self) -> tuple[int, ...]:
        if self.usage_counts is None:
            raise RuntimeError("recursive usage collection is not configured")
        rows = torch.nonzero(self.usage_counts > 0, as_tuple=False).squeeze(1)
        return tuple(
            self.base_size + int(row) for row in rows.detach().cpu().tolist()
        )

    def active_pair_keys(self) -> tuple[int, ...]:
        left = self.rule_left.detach().cpu().tolist()
        right = self.rule_right.detach().cpu().tolist()
        active = self.rule_active.detach().cpu().tolist()
        return tuple(
            recursive_pair_key(int(left[row]), int(right[row]))
            for row, enabled in enumerate(active)
            if enabled
        )

    def token_span(self, token_id: int) -> tuple[int, ...]:
        token_id = int(token_id)
        if 0 <= token_id < self.base_size:
            return (token_id,)
        row = token_id - self.base_size
        if row < 0 or row >= self.learned_count:
            raise ValueError("recursive token id is outside the vocabulary")
        start, stop = int(self.span_offsets[row]), int(self.span_offsets[row + 1])
        return tuple(int(value) for value in self.span_values[start:stop].tolist())

    def _validate_metadata(self) -> None:
        seen_pairs: set[tuple[int, int]] = set()
        seen_spans: set[tuple[int, ...]] = set()
        left_values = self.rule_left.detach().cpu().tolist()
        right_values = self.rule_right.detach().cpu().tolist()
        generation_values = self.rule_generation.detach().cpu().tolist()
        active_values = self.rule_active.detach().cpu().tolist()
        offsets = self.span_offsets.detach().cpu().tolist()
        span_values = self.span_values.detach().cpu().tolist()
        row_spans = [
            tuple(int(value) for value in span_values[offsets[row] : offsets[row + 1]])
            for row in range(self.learned_count)
        ]
        generations = set(int(value) for value in generation_values)
        if generations and generations != set(range(1, max(generations) + 1)):
            raise ValueError("recursive structural generations are not contiguous")
        for row in range(self.learned_count):
            token_id = self.base_size + row
            left = int(left_values[row])
            right = int(right_values[row])
            generation = int(generation_values[row])
            if left >= token_id or right >= token_id:
                raise ValueError("recursive rule references a non-earlier token")
            left_generation = (
                0 if left < self.base_size else int(generation_values[left - self.base_size])
            )
            right_generation = (
                0 if right < self.base_size else int(generation_values[right - self.base_size])
            )
            if left_generation >= generation or right_generation >= generation:
                raise ValueError("recursive rule parent is not from an earlier generation")
            pair = (left, right)
            span = row_spans[row]
            expected = (
                ((left,) if left < self.base_size else row_spans[left - self.base_size])
                + ((right,) if right < self.base_size else row_spans[right - self.base_size])
            )
            if pair in seen_pairs or span in seen_spans:
                raise ValueError("recursive rule or canonical span is duplicated")
            if span != expected or not span or len(span) > self.max_span_length:
                raise ValueError("recursive canonical span metadata is invalid")
            if min(span) < 0 or max(span) >= self.base_size:
                raise ValueError("recursive canonical span contains a non-base id")
            seen_pairs.add(pair)
            seen_spans.add(span)
        for row in range(self.learned_count):
            if not bool(active_values[row]):
                continue
            for parent in (int(left_values[row]), int(right_values[row])):
                if parent >= self.base_size and not bool(
                    active_values[parent - self.base_size]
                ):
                    raise ValueError("active recursive rule has an inactive parent")

    def metadata_sha256(self) -> str:
        payload = {
            name: getattr(self, name).detach().cpu().tolist()
            for name in (
                "rule_left", "rule_right", "rule_generation", "span_offsets",
                "span_values", "rule_active",
            )
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _lookup(
        self, token_ids: torch.Tensor, base_embedding: torch.Tensor
    ) -> torch.Tensor:
        if base_embedding.ndim != 2 or tuple(base_embedding.shape) != (
            self.base_size, self.dim
        ):
            raise ValueError("base embedding does not match recursive cascade")
        if token_ids.numel() and (
            int(token_ids.min()) < 0
            or int(token_ids.max()) >= self.base_size + self.learned_count
        ):
            raise ValueError("recursive token id is outside the vocabulary")
        result = torch.empty(
            (token_ids.numel(), self.dim),
            dtype=base_embedding.dtype,
            device=base_embedding.device,
        )
        base = token_ids < self.base_size
        if bool(base.any()):
            result[base] = base_embedding[token_ids[base]]
        if bool((~base).any()):
            result[~base] = self.learned[token_ids[~base] - self.base_size].to(
                dtype=base_embedding.dtype
            )
        return result

    @staticmethod
    def _nonoverlapping_leftmost(candidate: torch.Tensor) -> torch.Tensor:
        if candidate.dtype != torch.bool or candidate.ndim != 1:
            raise ValueError("recursive merge candidates must be a boolean vector")
        if not candidate.numel():
            return candidate
        positions = torch.arange(candidate.numel(), device=candidate.device)
        previous_false = torch.cat(
            (torch.ones(1, dtype=torch.bool, device=candidate.device), ~candidate[:-1])
        )
        starts = candidate & previous_false
        start_positions = torch.where(starts, positions, torch.full_like(positions, -1))
        last_start = torch.cummax(start_positions, dim=0).values
        return candidate & ((positions - last_start).remainder(2) == 0)

    def retokenize(
        self, content: torch.Tensor, segment: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the complete rule DAG to one packed batch on its current device."""

        if content.dtype != torch.long or segment.dtype != torch.long:
            raise ValueError("recursive packed tokens and segments must use torch.long")
        if content.ndim != 1 or segment.shape != content.shape:
            raise ValueError("recursive packed tokens and segments must align")
        if content.numel() and (
            int(content.min()) < 0 or int(content.max()) >= self.base_size
        ):
            raise ValueError("recursive tokenizer input must contain base ids only")
        emitted, emitted_segment = content, segment
        for generation in range(1, self.generation + 1):
            if emitted.numel() < 2:
                break
            rows = torch.nonzero(
                (self.rule_generation == generation) & self.rule_active,
                as_tuple=False,
            ).squeeze(1)
            if not rows.numel():
                continue
            rule_keys = _recursive_pair_keys(
                self.rule_left[rows], self.rule_right[rows]
            )
            order = torch.argsort(rule_keys, stable=True)
            rule_keys = rule_keys[order]
            rule_ids = rows[order] + self.base_size
            inside = emitted_segment[:-1] == emitted_segment[1:]
            adjacent = _recursive_pair_keys(emitted[:-1], emitted[1:])
            positions = torch.searchsorted(rule_keys, adjacent)
            safe = positions.clamp(max=rule_keys.numel() - 1)
            found = inside & (positions < rule_keys.numel()) & (
                rule_keys[safe] == adjacent
            )
            chosen = self._nonoverlapping_leftmost(found)
            if not bool(chosen.any()):
                continue
            chosen_positions = torch.nonzero(chosen, as_tuple=False).squeeze(1)
            replacement = emitted.clone()
            replacement[chosen_positions] = rule_ids[safe[chosen_positions]]
            keep = torch.ones_like(emitted, dtype=torch.bool)
            keep[chosen_positions + 1] = False
            emitted = replacement[keep]
            emitted_segment = emitted_segment[keep]
        return emitted, emitted_segment

    def pool(
        self,
        base_embedding: torch.Tensor,
        content: torch.Tensor,
        segment: torch.Tensor,
        language: torch.Tensor,
        n_sentences: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retokenize, train emitted rows, and observe next-generation pairs."""

        emitted, emitted_segment = self.retokenize(content, segment)
        if self.training and self.usage_counts is not None and emitted.numel():
            learned_rows = emitted[emitted >= self.base_size] - self.base_size
            if learned_rows.numel():
                self.usage_counts.index_add_(
                    0,
                    learned_rows,
                    torch.ones_like(learned_rows, dtype=self.usage_counts.dtype),
                )
        vectors = self._lookup(emitted, base_embedding)
        weight = torch.ones(emitted.numel(), dtype=vectors.dtype, device=vectors.device)
        numerator = torch.zeros(
            n_sentences, self.dim, dtype=vectors.dtype, device=vectors.device
        )
        denominator = torch.zeros(n_sentences, dtype=vectors.dtype, device=vectors.device)
        denominator.index_add_(0, emitted_segment, weight)
        if emitted.numel() > 1:
            inside = emitted_segment[:-1] == emitted_segment[1:]
            if bool(inside.any()):
                left, right = emitted[:-1][inside], emitted[1:][inside]
                pair_segment = emitted_segment[:-1][inside]
                pair_language = language[pair_segment]
                pair_key = _recursive_pair_keys(left, right)
                bucket = _proposal_hash(left, right, self.residual_buckets)
                probability = torch.sigmoid(
                    self.proposal_score[bucket] + self.language_bias[pair_language]
                )
                if (
                    self.discovery is not None
                    and self.training
                    and probability.requires_grad
                ):
                    saved_key = pair_key.detach()
                    saved_language = pair_language.detach()
                    saved_probability = probability.detach()

                    def observe(gradient: torch.Tensor) -> None:
                        if self.discovery is not None:
                            self.discovery.observe(
                                saved_key, saved_language, saved_probability, gradient
                            )

                    probability.register_hook(observe)
                positions = torch.nonzero(inside, as_tuple=False).squeeze(1)
                half = 0.5 * probability
                weight = weight.index_add(0, positions, -half)
                weight = weight.index_add(0, positions + 1, -half)
                _weighted_index_add_(
                    numerator,
                    pair_segment,
                    self.proposal_merged[bucket],
                    probability,
                )
                denominator.index_add_(0, pair_segment, -probability)
        _weighted_index_add_(numerator, emitted_segment, vectors, weight)
        return (
            numerator / denominator.clamp_min(1e-3).unsqueeze(1),
            emitted,
            emitted_segment,
        )

    @staticmethod
    def _optimizer_row(
        optimizer: torch.optim.Optimizer,
        parameter: nn.Parameter,
        row: int,
    ) -> dict[Any, Any]:
        state = optimizer.state.get(parameter, {})
        if not state:
            raise ValueError("recursive promotion requires initialized parent Adam state")
        result: dict[Any, Any] = {}
        for name, value in state.items():
            if not torch.is_tensor(value):
                result[name] = value
            elif value.ndim == 0:
                result[name] = value.detach().clone()
            elif tuple(value.shape) == tuple(parameter.shape):
                result[name] = value[row].detach().clone()
            else:
                raise ValueError(f"unknown recursive optimizer state shape: {name}")
        return result

    @staticmethod
    def _average_states(left: dict[Any, Any], right: dict[Any, Any]) -> dict[Any, Any]:
        if set(left) != set(right):
            raise ValueError("recursive parent Adam states differ")
        result: dict[Any, Any] = {}
        for name in left:
            a, b = left[name], right[name]
            if torch.is_tensor(a) != torch.is_tensor(b):
                raise ValueError(f"recursive parent Adam state type differs: {name}")
            if not torch.is_tensor(a):
                if a != b:
                    raise ValueError(f"recursive parent Adam scalar differs: {name}")
                result[name] = a
            elif a.ndim == 0:
                if not torch.equal(a, b):
                    raise ValueError(f"recursive parent Adam step differs: {name}")
                result[name] = a.detach().clone()
            else:
                if a.shape != b.shape:
                    raise ValueError(f"recursive parent Adam row differs: {name}")
                result[name] = (0.5 * (a + b)).detach().clone()
        return result

    def _parent_state(
        self,
        optimizer: torch.optim.Optimizer,
        base_embedding: nn.Parameter,
        token_id: int,
    ) -> dict[Any, Any]:
        if token_id < self.base_size:
            return self._optimizer_row(optimizer, base_embedding, token_id)
        return self._optimizer_row(
            optimizer, self.learned, token_id - self.base_size
        )

    def promotable_pair_keys(
        self, pair_keys: Iterable[int]
    ) -> tuple[tuple[int, ...], dict[str, int]]:
        existing_pairs = {
            recursive_pair_key(int(left), int(right))
            for left, right in zip(
                self.rule_left.detach().cpu().tolist(),
                self.rule_right.detach().cpu().tolist(),
                strict=True,
            )
        }
        existing_spans = {
            self.token_span(token_id)
            for token_id in range(self.base_size, self.base_size + self.learned_count)
        }
        accepted: list[int] = []
        rejections = {
            "already_rule": 0,
            "unknown_parent": 0,
            "duplicate_span": 0,
            "span_too_long": 0,
        }
        total = self.base_size + self.learned_count
        for key in sorted({int(value) for value in pair_keys}):
            left, right = decode_recursive_pair_key(key)
            if key in existing_pairs:
                rejections["already_rule"] += 1
                continue
            if left >= total or right >= total:
                rejections["unknown_parent"] += 1
                continue
            span = self.token_span(left) + self.token_span(right)
            if len(span) > self.max_span_length:
                rejections["span_too_long"] += 1
                continue
            if span in existing_spans:
                rejections["duplicate_span"] += 1
                continue
            accepted.append(key)
            existing_spans.add(span)
        return tuple(accepted), rejections

    def deactivate_unobserved(
        self, observed_token_ids: Iterable[int]
    ) -> dict[str, Any]:
        """Deactivate the exact active sub-DAG absent from both audit windows.

        Every active ancestor of an observed learned token is retained.  All
        remaining active rules can be disabled without changing either observed
        token stream: they were neither emitted nor needed to emit a descendant.
        Physical rows and Adam state remain stable for cheap future reactivation.
        """

        observed = sorted({int(value) for value in observed_token_ids})
        if any(
            token_id < self.base_size
            or token_id >= self.base_size + self.learned_count
            for token_id in observed
        ):
            raise ValueError("recursive usage contains an unknown learned token")
        active = self.rule_active.detach().cpu().tolist()
        left = self.rule_left.detach().cpu().tolist()
        right = self.rule_right.detach().cpu().tolist()
        keep = [False] * self.learned_count
        if observed:
            rows = [token_id - self.base_size for token_id in observed]
            if not all(active[row] for row in rows):
                raise ValueError("recursive usage contains an inactive token")
            for row in rows:
                keep[row] = True
        # Child ids are always larger than learned parent ids, so one reverse
        # pass closes the retained set over every active learned ancestor.
        for row in range(self.learned_count - 1, -1, -1):
            if not keep[row]:
                continue
            for parent in (int(left[row]), int(right[row])):
                if parent >= self.base_size:
                    keep[parent - self.base_size] = True
        rows = [
            row for row, enabled in enumerate(active) if enabled and not keep[row]
        ]
        if rows:
            device_rows = torch.tensor(
                rows, dtype=torch.long, device=self.rule_active.device
            )
            self.rule_active[device_rows] = False
        token_ids = [self.base_size + row for row in rows]
        return {
            "deactivated_rows": len(token_ids),
            "active_rows": self.active_count,
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }

    def reactivate_pair_keys(self, pair_keys: Iterable[int]) -> dict[str, Any]:
        """Reactivate inactive exact rules whose complete parent path is active."""

        left_values = self.rule_left.detach().cpu().tolist()
        right_values = self.rule_right.detach().cpu().tolist()
        by_key = {
            recursive_pair_key(int(left), int(right)): row
            for row, (left, right) in enumerate(
                zip(left_values, right_values, strict=True)
            )
        }
        reactivated_keys: list[int] = []
        reactivated_ids: list[int] = []
        rejected_inactive_parent = 0
        active = self.rule_active.detach().cpu().tolist()
        for key in sorted({int(value) for value in pair_keys}):
            row = by_key.get(key)
            if row is None or bool(active[row]):
                continue
            parents = (int(left_values[row]), int(right_values[row]))
            if any(
                parent >= self.base_size
                and not bool(active[parent - self.base_size])
                for parent in parents
            ):
                rejected_inactive_parent += 1
                continue
            active[row] = True
            reactivated_keys.append(key)
            reactivated_ids.append(self.base_size + row)
        if reactivated_ids:
            rows = torch.tensor(
                [token_id - self.base_size for token_id in reactivated_ids],
                dtype=torch.long,
                device=self.rule_active.device,
            )
            self.rule_active[rows] = True
        return {
            "reactivated_rows": len(reactivated_ids),
            "active_rows": self.active_count,
            "pair_keys": reactivated_keys,
            "token_ids": reactivated_ids,
            "rejected_inactive_parent": rejected_inactive_parent,
        }

    def grow(
        self,
        pair_keys: Iterable[int],
        *,
        base_embedding: nn.Parameter,
        optimizer: torch.optim.Optimizer,
        initialization: str = "parent_mean",
    ) -> dict[str, Any]:
        """Atomically append one dense generation and migrate live Adam state."""

        if initialization not in {"parent_mean", "parent_sum"}:
            raise ValueError("unknown recursive row initialization")

        accepted, rejections = self.promotable_pair_keys(pair_keys)
        if not accepted:
            return {
                "generation": self.generation,
                "added_rows": 0,
                "learned_rows": self.learned_count,
                "active_rows": self.active_count,
                "rejections": rejections,
            }
        pairs = tuple(decode_recursive_pair_key(key) for key in accepted)
        old_parameter = self.learned
        old_count = self.learned_count
        new_values = []
        new_states = []
        spans = []
        for left, right in pairs:
            parent_ids = torch.tensor(
                [left, right], dtype=torch.long, device=base_embedding.device
            )
            parent_values = self._lookup(parent_ids, base_embedding)
            new_values.append(
                parent_values.sum(dim=0)
                if initialization == "parent_sum"
                else parent_values.mean(dim=0)
            )
            new_states.append(
                self._average_states(
                    self._parent_state(optimizer, base_embedding, left),
                    self._parent_state(optimizer, base_embedding, right),
                )
            )
            spans.append(self.token_span(left) + self.token_span(right))
        new_parameter = nn.Parameter(
            torch.cat((old_parameter.detach(), torch.stack(new_values)), dim=0),
            requires_grad=True if not old_count else old_parameter.requires_grad,
        )

        occurrences = sum(
            candidate is old_parameter
            for group in optimizer.param_groups
            for candidate in group["params"]
        )
        if occurrences not in ({0, 1} if not old_count else {1}):
            raise ValueError("recursive learned table must occur once in the optimizer")
        old_state = optimizer.state.get(old_parameter, {})
        migrated: dict[Any, Any] = {}
        for name in new_states[0]:
            additions = [state[name] for state in new_states]
            first = additions[0]
            if not torch.is_tensor(first):
                if any(value != first for value in additions[1:]):
                    raise ValueError(f"recursive added Adam scalar differs: {name}")
                migrated[name] = first
            elif first.ndim == 0:
                if any(not torch.equal(value, first) for value in additions[1:]):
                    raise ValueError(f"recursive added Adam step differs: {name}")
                if old_state and not torch.equal(old_state[name], first):
                    raise ValueError(f"recursive existing/added Adam step differs: {name}")
                migrated[name] = first.detach().clone()
            else:
                added = torch.stack(additions)
                if old_count:
                    old_value = old_state.get(name)
                    if (
                        not torch.is_tensor(old_value)
                        or tuple(old_value.shape) != tuple(old_parameter.shape)
                    ):
                        raise ValueError(f"recursive existing Adam rows differ: {name}")
                    added = added.to(device=old_value.device, dtype=old_value.dtype)
                    migrated[name] = torch.cat((old_value.detach(), added), dim=0)
                else:
                    migrated[name] = added
        for group in optimizer.param_groups:
            replaced: list[nn.Parameter] = []
            for parameter in group["params"]:
                replaced.append(
                    new_parameter if parameter is old_parameter else parameter
                )
                if occurrences == 0 and parameter is base_embedding:
                    replaced.append(new_parameter)
            group["params"] = replaced
        optimizer.state.pop(old_parameter, None)
        optimizer.state[new_parameter] = migrated
        self.learned = new_parameter

        device = self.rule_left.device
        self.rule_left = torch.cat(
            (self.rule_left, torch.tensor([left for left, _ in pairs], device=device))
        )
        self.rule_right = torch.cat(
            (self.rule_right, torch.tensor([right for _, right in pairs], device=device))
        )
        new_generation = self.generation + 1
        self.rule_generation = torch.cat(
            (
                self.rule_generation,
                torch.full((len(pairs),), new_generation, dtype=torch.long, device=device),
            )
        )
        self.rule_active = torch.cat(
            (
                self.rule_active,
                torch.ones(len(pairs), dtype=torch.bool, device=device),
            )
        )
        span_values = [value for span in spans for value in span]
        lengths = torch.tensor([len(span) for span in spans], dtype=torch.long, device=device)
        appended_offsets = self.span_offsets[-1] + lengths.cumsum(0)
        self.span_offsets = torch.cat((self.span_offsets, appended_offsets))
        self.span_values = torch.cat(
            (self.span_values, torch.tensor(span_values, dtype=torch.long, device=device))
        )
        self.discovery = None
        return {
            "generation": new_generation,
            "added_rows": len(pairs),
            "learned_rows": self.learned_count,
            "active_rows": self.active_count,
            "token_ids": list(range(self.base_size + old_count, self.base_size + self.learned_count)),
            "pairs": [[left, right] for left, right in pairs],
            "rejections": rejections,
            "initialization": initialization,
            "metadata_sha256": self.metadata_sha256(),
        }
