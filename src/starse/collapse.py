"""Learned collapsing of adjacent tokens into single units.

The earlier n-gram experiment *added* bigram rows next to the unigrams, so a
Georgian word cut into six pieces became six unigrams plus five bigrams — eleven
items instead of six. That raised fragmentation instead of lowering it, which is
why well-tokenised languages lost ground: an English sentence of four tokens
gained three bigram rows and half its pooled mass went to them.

Collapsing is a soft *replacement*: a learned merged row enters the numerator
while the neighbouring unigram rows lose mass. That changes the direction toward
units the tokenizer should have produced and targets tokens-per-word directly —
the quantity whose rank correlation with held-out Recall@1 is -0.607.

For adjacent pair i, a merge probability p_i gates a learned merged row M_i:

    numerator   = sum_i (1 - (p_{i-1} + p_i)/2) * E[t_i]  +  sum_i p_i * M_i
    denominator = n - sum_i p_i

At p = 0 this is exactly the plain mean, so the control is recovered by
construction. Increasing p rotates the pooled direction from neighbouring
unigrams toward the learned merged row; overlapping candidates share the
reduction instead of needing a discrete matching. Under cosine scoring the
positive scalar denominator cancels, so the mechanism is the relative numerator
composition, not the smaller denominator.

The merge score carries a per-language bias, so a language whose words are
already single tokens can shut merging off globally rather than paying for a
mechanism it does not need — the failure mode of the ungated n-gram arm.
"""

from __future__ import annotations

import heapq
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

_MIX_A = 2654435761
_MIX_B = 40503
_SKETCH_A = (6364136223846793005, 3202034522624059733, 3935559000370003845)
_SKETCH_B = (1442695040888963407, 2691343689449507681, 4768777513237032717)
_SKETCH_SIGN_A = (2862933555777941757, 7046029254386353131, 3935559000370003845)
_MAX_WEIGHTED_SOURCE_BYTES = 256 * 1024 * 1024


def _weighted_index_add_(
    target: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_chunk_bytes: int = _MAX_WEIGHTED_SOURCE_BYTES,
) -> torch.Tensor:
    """Index-add weighted rows without materialising an unbounded N x dim product."""
    if source.ndim != 2 or weight.ndim != 1 or source.shape[0] != weight.shape[0]:
        raise ValueError("weighted index-add needs aligned matrix rows and scalar weights")
    if index.ndim != 1 or index.shape[0] != source.shape[0]:
        raise ValueError("weighted index-add needs one destination per source row")
    if max_chunk_bytes <= 0:
        raise ValueError("weighted index-add chunk budget must be positive")
    bytes_per_row = max(1, source.shape[1] * source.element_size())
    rows_per_chunk = max(1, max_chunk_bytes // bytes_per_row)
    for start in range(0, source.shape[0], rows_per_chunk):
        stop = min(source.shape[0], start + rows_per_chunk)
        target.index_add_(
            0,
            index[start:stop],
            source[start:stop] * weight[start:stop].unsqueeze(1),
        )
    return target


@dataclass(frozen=True)
class PromotedVocabulary:
    """Deterministic result of bounded exact-pair promotion."""

    pair_keys: tuple[int, ...]
    records: tuple[dict[str, Any], ...]
    candidate_count: int
    reservation_memberships: tuple[tuple[int, int], ...]


class ContrastiveVocabularyDiscovery:
    """Bounded candidate table with signed CountSketch admission.

    The sketch receives ``-p * dL/dp`` after each backward pass. It transfers
    only the largest absolute pair/language aggregates from a forward. A fixed
    signed CountSketch retains cumulative admission evidence even for keys not
    currently in the exact table; the bounded exact table stores utility,
    probability and support only from admission onward.
    """

    admission_mode = "utility"

    def __init__(
        self,
        *,
        n_languages: int,
        candidate_capacity: int,
        top_per_forward: int,
    ) -> None:
        if n_languages <= 0:
            raise ValueError("n_languages must be positive")
        if candidate_capacity <= 0:
            raise ValueError("candidate_capacity must be positive")
        if top_per_forward <= 0:
            raise ValueError("top_per_forward must be positive")
        self.n_languages = int(n_languages)
        self.candidate_capacity = int(candidate_capacity)
        self.top_per_forward = int(top_per_forward)
        # composite pair/language key -> [signed utility, p sum, captured count]
        self.records: dict[int, list[float | int]] = {}
        self._sketch_width = max(64, self.candidate_capacity)
        self._utility_sketch = torch.zeros(
            (len(_SKETCH_A), self._sketch_width), dtype=torch.float64
        )
        # Lazy min-heap over retained rows.  Updates append a new rank and leave
        # the previous entry stale; periodic rebuilding gives the metadata a
        # fixed bound without sorting the entire sketch on every forward.
        self._heap: list[tuple[float, int, int, int]] = []
        self.observed_occurrences = 0
        self.transferred_records = 0
        self.unique_records_before_transfer = 0
        self.dropped_by_top_per_forward = 0
        self.retained_updates = 0
        self.admitted_into_free_slots = 0
        self.capacity_replacements = 0
        self.capacity_rejections = 0

    @staticmethod
    def _rank(key: int, row: list[float | int]) -> tuple[float, int, int]:
        return float(row[0]), int(row[2]), -int(key)

    def _push_heap(self, key: int) -> None:
        heapq.heappush(self._heap, (*self._rank(key, self.records[key]), int(key)))

    def _discard_stale_heap_entries(self) -> None:
        while self._heap:
            utility, count, negative_key, key = self._heap[0]
            row = self.records.get(key)
            if row is not None and (utility, count, negative_key) == self._rank(key, row):
                return
            heapq.heappop(self._heap)

    def _rebuild_heap(self) -> None:
        self._heap = [(*self._rank(key, row), key) for key, row in self.records.items()]
        heapq.heapify(self._heap)

    def _update_utility_sketch(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Update fixed memory and return the median signed estimate per key."""

        keys = keys.to(device="cpu", dtype=torch.long)
        values = values.to(device="cpu", dtype=torch.float64)
        estimates = []
        for depth, (multiplier, offset, sign_multiplier) in enumerate(zip(
            _SKETCH_A, _SKETCH_B, _SKETCH_SIGN_A, strict=True
        )):
            bucket = torch.remainder(keys * multiplier + offset, self._sketch_width)
            sign = torch.where(
                torch.bitwise_and(keys * sign_multiplier + offset, 1) == 0,
                1.0,
                -1.0,
            ).to(torch.float64)
            self._utility_sketch[depth].index_add_(0, bucket, values * sign)
            estimates.append(self._utility_sketch[depth, bucket] * sign)
        return torch.stack(estimates).median(dim=0).values

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
        finite = torch.isfinite(utility)
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
        count = torch.zeros(unique.numel(), dtype=torch.long, device=utility.device)
        utility_sum.index_add_(0, inverse, utility)
        probability_sum.index_add_(0, inverse, probability)
        count.index_add_(0, inverse, torch.ones_like(inverse))

        self.unique_records_before_transfer += int(unique.numel())
        keep = min(self.top_per_forward, unique.numel())
        if keep < unique.numel():
            self.dropped_by_top_per_forward += int(unique.numel() - keep)
            chosen = torch.topk(utility_sum.abs(), keep, sorted=False).indices
            unique = unique[chosen]
            utility_sum = utility_sum[chosen]
            probability_sum = probability_sum[chosen]
            count = count[chosen]

        unique_cpu = unique.cpu()
        utility_cpu = utility_sum.cpu()
        admission_estimates = self._update_utility_sketch(unique_cpu, utility_cpu)
        keys = unique_cpu.tolist()
        utilities = utility_cpu.tolist()
        probabilities = probability_sum.cpu().tolist()
        counts = count.cpu().tolist()
        self.transferred_records += len(keys)
        incoming = {
            int(key): [float(value), float(p_sum), int(n)]
            for key, value, p_sum, n in zip(
                keys, utilities, probabilities, counts, strict=True
            )
        }
        admission = {
            int(key): float(estimate)
            for key, estimate in zip(keys, admission_estimates.tolist(), strict=True)
        }

        # First update retained rows, then consider unseen rows in descending
        # batch rank.  This makes replacement deterministic and independent of
        # the device order returned by top-k.
        retained_keys = sorted(key for key in incoming if key in self.records)
        self.retained_updates += len(retained_keys)
        for key in retained_keys:
            value, p_sum, n = incoming.pop(key)
            row = self.records[key]
            row[0] = float(row[0]) + float(value)
            row[1] = float(row[1]) + float(p_sum)
            row[2] = int(row[2]) + int(n)
            self._push_heap(key)

        unseen = sorted(
            incoming.items(),
            key=lambda item: (admission[item[0]], int(item[1][2]), -item[0]),
            reverse=True,
        )
        for key, row in unseen:
            if len(self.records) < self.candidate_capacity:
                self.records[key] = row
                self._push_heap(key)
                self.admitted_into_free_slots += 1
                continue
            self._discard_stale_heap_entries()
            if not self._heap:
                self._rebuild_heap()
            admission_rank = (admission[key], int(row[2]), -key)
            if admission_rank <= self._heap[0][:3]:
                self.capacity_rejections += 1
                continue
            victim = heapq.heappop(self._heap)[3]
            del self.records[victim]
            self.records[key] = row
            self._push_heap(key)
            self.capacity_replacements += 1

        # The retained dictionary never exceeds candidate_capacity.  Lazy heap
        # metadata is also bounded: after a call it has at most 2x as many
        # entries as the registered candidate table.
        if len(self._heap) > 2 * self.candidate_capacity:
            self._rebuild_heap()

    def _prune(self) -> None:
        ranked = sorted(
            self.records.items(),
            key=lambda item: self._rank(item[0], item[1]),
            reverse=True,
        )[: self.candidate_capacity]
        self.records = dict(ranked)
        self._rebuild_heap()

    def snapshot_records(self) -> tuple[dict[str, float | int], ...]:
        """Return the complete bounded pair/language table deterministically."""

        self._prune()
        rows = []
        for composite, row in sorted(self.records.items()):
            pair_key, language = divmod(int(composite), self.n_languages)
            utility, probability_sum, count = float(row[0]), float(row[1]), int(row[2])
            rows.append({
                "composite_key": int(composite),
                "pair_key": pair_key,
                "language_index": language,
                "utility": utility,
                "probability_sum": probability_sum,
                "captured_support": count,
                "mean_probability": probability_sum / count if count else 0.0,
            })
        return tuple(rows)

    def state_dict(self) -> dict[str, Any]:
        """Return the complete bounded discovery state for an exact resume."""

        self._prune()
        return {
            "version": 2,
            "n_languages": self.n_languages,
            "candidate_capacity": self.candidate_capacity,
            "top_per_forward": self.top_per_forward,
            "sketch_width": self._sketch_width,
            "records": {
                int(key): [float(row[0]), float(row[1]), int(row[2])]
                for key, row in sorted(self.records.items())
            },
            "utility_sketch": self._utility_sketch.clone(),
            "observed_occurrences": self.observed_occurrences,
            "transferred_records": self.transferred_records,
            "admission_pressure": self.admission_pressure(),
        }

    def load_state_dict(self, raw: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict` after validation."""

        version = int(raw.get("version", -1))
        if version not in (1, 2):
            raise ValueError("unsupported discovery state version")
        if int(raw.get("n_languages", -1)) != self.n_languages:
            raise ValueError("discovery state language count does not match")
        if int(raw.get("candidate_capacity", -1)) != self.candidate_capacity:
            raise ValueError("discovery state candidate capacity does not match")
        if int(raw.get("top_per_forward", -1)) != self.top_per_forward:
            raise ValueError("discovery state top-per-forward does not match")
        if int(raw.get("sketch_width", -1)) != self._sketch_width:
            raise ValueError("discovery state sketch width does not match")
        sketch = raw.get("utility_sketch")
        if (
            not torch.is_tensor(sketch)
            or sketch.dtype != torch.float64
            or tuple(sketch.shape) != tuple(self._utility_sketch.shape)
            or not bool(torch.isfinite(sketch).all())
        ):
            raise ValueError("discovery state utility sketch is invalid")
        raw_records = raw.get("records")
        if not isinstance(raw_records, dict) or len(raw_records) > self.candidate_capacity:
            raise ValueError("discovery state records are invalid")
        records: dict[int, list[float | int]] = {}
        for raw_key, raw_row in raw_records.items():
            key = int(raw_key)
            if key < 0 or not isinstance(raw_row, (list, tuple)) or len(raw_row) != 3:
                raise ValueError("discovery state record is invalid")
            utility, probability, count = float(raw_row[0]), float(raw_row[1]), int(raw_row[2])
            if not (math.isfinite(utility) and math.isfinite(probability)) or count < 0:
                raise ValueError("discovery state record values are invalid")
            records[key] = [utility, probability, count]
        observed = int(raw.get("observed_occurrences", -1))
        transferred = int(raw.get("transferred_records", -1))
        if observed < 0 or transferred < 0:
            raise ValueError("discovery state counters are invalid")
        counter_names = tuple(self.admission_pressure())
        raw_pressure = raw.get("admission_pressure", {})
        if version == 2 and (
            not isinstance(raw_pressure, dict)
            or set(raw_pressure) != set(counter_names)
        ):
            raise ValueError("discovery state admission counters are invalid")
        pressure = {
            name: int(raw_pressure.get(name, 0)) for name in counter_names
        }
        if any(value < 0 for value in pressure.values()):
            raise ValueError("discovery state admission counters are invalid")

        self.records = records
        self._utility_sketch.copy_(sketch)
        self.observed_occurrences = observed
        self.transferred_records = transferred
        for name, value in pressure.items():
            setattr(self, name, value)
        self._rebuild_heap()

    def admission_pressure(self) -> dict[str, int]:
        return {
            "unique_records_before_transfer": self.unique_records_before_transfer,
            "dropped_by_top_per_forward": self.dropped_by_top_per_forward,
            "retained_updates": self.retained_updates,
            "admitted_into_free_slots": self.admitted_into_free_slots,
            "capacity_replacements": self.capacity_replacements,
            "capacity_rejections": self.capacity_rejections,
        }


    def promote(
        self,
        *,
        budget: int,
        min_support: int,
        per_language_quota: int,
        per_language_min_support: int,
        min_utility: float = 0.0,
    ) -> PromotedVocabulary:
        """Select utility-thresholded exact rows under a safety L0 ceiling."""

        if budget <= 0:
            raise ValueError("budget must be positive")
        if min_support <= 0 or per_language_min_support <= 0:
            raise ValueError("support thresholds must be positive")
        if per_language_quota < 0:
            raise ValueError("per_language_quota must be non-negative")
        if not math.isfinite(min_utility) or min_utility < 0:
            raise ValueError("min_utility must be finite and non-negative")
        self._prune()

        by_pair: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"utility": 0.0, "p_sum": 0.0, "count": 0, "languages": {}}
        )
        for composite, row in self.records.items():
            pair_key, language = divmod(composite, self.n_languages)
            utility, p_sum, count = float(row[0]), float(row[1]), int(row[2])
            target = by_pair[pair_key]
            target["utility"] += utility
            target["p_sum"] += p_sum
            target["count"] += count
            target["languages"][language] = {
                "utility": utility,
                "p_sum": p_sum,
                "count": count,
            }

        selected: set[int] = set()
        reservation_languages: dict[int, set[int]] = defaultdict(set)
        if per_language_quota:
            candidates_by_language: list[list[tuple[float, int, int, int]]] = [
                [] for _ in range(self.n_languages)
            ]
            for pair_key, row in by_pair.items():
                # A language reservation protects multilingual coverage, but
                # it cannot override the registered global U(q) > 0
                # eligibility constraint.  Build the per-language lists in
                # one pass over observed records; scanning every pair once per
                # marker is prohibitive for the 1,914-language run.
                if row["utility"] <= 0 or row["utility"] < min_utility:
                    continue
                for language, language_row in row["languages"].items():
                    if (language_row["utility"] > 0
                            and language_row["count"] >= per_language_min_support):
                        candidates_by_language[language].append((
                            float(language_row["utility"]),
                            int(language_row["count"]),
                            -int(pair_key),
                            int(pair_key),
                        ))
            for language, candidates in enumerate(candidates_by_language):
                candidates.sort(reverse=True)
                for candidate in candidates[:per_language_quota]:
                    pair_key = candidate[3]
                    selected.add(pair_key)
                    reservation_languages[pair_key].add(language)

        if len(selected) > budget:
            selected = set(sorted(
                selected,
                key=lambda key: (
                    float(by_pair[key]["utility"]),
                    int(by_pair[key]["count"]),
                    -key,
                ),
                reverse=True,
            )[:budget])
            reservation_languages = {
                key: languages for key, languages in reservation_languages.items()
                if key in selected
            }

        global_candidates = [
            key for key, row in by_pair.items()
            if (
                row["utility"] > 0
                and row["utility"] >= min_utility
                and row["count"] >= min_support
                and key not in selected
            )
        ]
        global_candidates.sort(
            key=lambda key: (
                float(by_pair[key]["utility"]),
                int(by_pair[key]["count"]),
                -key,
            ),
            reverse=True,
        )
        selected.update(global_candidates[: max(0, budget - len(selected))])

        pair_keys = tuple(sorted(selected))
        rows = []
        for pair_key in pair_keys:
            row = by_pair[pair_key]
            reserved = sorted(reservation_languages.get(pair_key, ()))
            rows.append({
                "pair_key": pair_key,
                "utility": float(row["utility"]),
                "captured_support": int(row["count"]),
                "mean_probability": (
                    float(row["p_sum"]) / int(row["count"]) if row["count"] else 0.0
                ),
                "language_count": len(row["languages"]),
                "reservation_language_count": len(reserved),
                "max_reservation_support": max(
                    (int(row["languages"][language]["count"]) for language in reserved),
                    default=0,
                ),
                "global_support_eligible": int(row["count"]) >= min_support,
            })
        return PromotedVocabulary(
            pair_keys=pair_keys,
            records=tuple(rows),
            candidate_count=len(by_pair),
            reservation_memberships=tuple(sorted(
                (pair_key, language)
                for pair_key, languages in reservation_languages.items()
                for language in languages
            )),
        )

    def audit_promotion(
        self,
        promoted: PromotedVocabulary,
        *,
        budget: int,
        min_support: int,
        per_language_quota: int,
        per_language_min_support: int,
        min_utility: float = 0.0,
        reservation_audit_quota: int | None = None,
        trainable_language_indices: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Compare a quota selection with the same discovery under quota zero."""

        if reservation_audit_quota is None:
            reservation_audit_quota = per_language_quota
        if reservation_audit_quota < 0:
            raise ValueError("reservation_audit_quota must be non-negative")
        if trainable_language_indices is None:
            trainable_language_indices = tuple(range(self.n_languages))
        else:
            trainable_language_indices = tuple(
                int(index) for index in trainable_language_indices
            )
        if (
            not trainable_language_indices
            or len(set(trainable_language_indices)) != len(trainable_language_indices)
            or any(
                index < 0 or index >= self.n_languages
                for index in trainable_language_indices
            )
        ):
            raise ValueError("trainable language indices are invalid")
        trainable_set = set(trainable_language_indices)
        global_only = (
            promoted
            if per_language_quota == 0
            else self.promote(
                budget=budget,
                min_support=min_support,
                per_language_quota=0,
                per_language_min_support=per_language_min_support,
                min_utility=min_utility,
            )
        )
        reservation_selection = (
            promoted
            if per_language_quota == reservation_audit_quota
            else self.promote(
                budget=budget,
                min_support=min_support,
                per_language_quota=reservation_audit_quota,
                per_language_min_support=per_language_min_support,
                min_utility=min_utility,
            )
        )
        all_candidate_counts = [0] * self.n_languages
        aggregate_utility: dict[int, float] = defaultdict(float)
        for composite, row in self.records.items():
            pair_key, language = divmod(int(composite), self.n_languages)
            if language not in trainable_set:
                raise ValueError(
                    "discovery contains a nontrainable language candidate"
                )
            all_candidate_counts[language] += 1
            aggregate_utility[pair_key] += float(row[0])
        all_eligible_counts = [0] * self.n_languages
        for composite, row in self.records.items():
            pair_key, language = divmod(int(composite), self.n_languages)
            if (
                aggregate_utility[pair_key] > 0
                and aggregate_utility[pair_key] >= min_utility
                and float(row[0]) > 0
                and int(row[2]) >= per_language_min_support
            ):
                all_eligible_counts[language] += 1
        candidate_counts = [
            all_candidate_counts[index] for index in trainable_language_indices
        ]
        eligible_counts = [
            all_eligible_counts[index] for index in trainable_language_indices
        ]
        all_reservation_counts = [0] * self.n_languages
        for _, language in reservation_selection.reservation_memberships:
            if language not in trainable_set:
                raise ValueError("reservation selected a nontrainable language")
            all_reservation_counts[language] += 1
        reservation_counts = [
            all_reservation_counts[index] for index in trainable_language_indices
        ]

        reservation_utility = math.fsum(
            float(record["utility"]) for record in reservation_selection.records
        )
        global_utility = math.fsum(
            float(record["utility"]) for record in global_only.records
        )
        utility_displaced = global_utility - reservation_utility
        membership_capacity = reservation_audit_quota * len(trainable_language_indices)
        reservation_keys = set(reservation_selection.pair_keys)
        global_keys = set(global_only.pair_keys)
        return {
            "deployed_selector": {
                "per_language_quota": per_language_quota,
                "matches_global_only": promoted.pair_keys == global_only.pair_keys,
                "reservation_audit_quota": reservation_audit_quota,
            },
            "candidate_table": {
                "capacity": self.candidate_capacity,
                "pair_language_records": len(self.records),
                "saturation": len(self.records) / self.candidate_capacity,
                "candidate_pairs": promoted.candidate_count,
                "initializer_languages_total": self.n_languages,
                "languages_total": len(trainable_language_indices),
                "trainable_language_indices": list(trainable_language_indices),
                "languages_with_candidates": sum(
                    count > 0 for count in candidate_counts
                ),
                "records_per_language": candidate_counts,
                "reservation_eligible_records_per_language": eligible_counts,
                "min_records_per_language": min(candidate_counts),
                "median_records_per_language": statistics.median(candidate_counts),
                "max_records_per_language": max(candidate_counts),
                "admission_pressure": {
                    
                        "unique_records_before_transfer": (
                            self.unique_records_before_transfer
                        ),
                        "dropped_by_top_per_forward": (
                            self.dropped_by_top_per_forward
                        ),
                        "transferred_records": self.transferred_records
                    ,
                    "retained_updates": self.retained_updates,
                    "admitted_into_free_slots": self.admitted_into_free_slots,
                    "capacity_replacements": self.capacity_replacements,
                    "capacity_rejections": self.capacity_rejections,
                },
            },
            "reservation": {
                "quota_per_language": reservation_audit_quota,
                "membership_capacity": membership_capacity,
                "memberships": len(reservation_selection.reservation_memberships),
                "membership_occupancy": (
                    len(reservation_selection.reservation_memberships)
                    / membership_capacity
                    if membership_capacity
                    else 0.0
                ),
                "unique_reserved_rows": len({
                    pair_key
                    for pair_key, _ in reservation_selection.reservation_memberships
                }),
                "languages_with_reservations": sum(
                    count > 0 for count in reservation_counts
                ),
                "memberships_per_language": reservation_counts,
            },
            "global_only_counterfactual": {
                "global_selected_rows": len(global_only.pair_keys),
                "global_selected_utility": global_utility,
                "reservation_selected_rows": len(reservation_selection.pair_keys),
                "reservation_selected_utility": reservation_utility,
                "utility_displaced_by_reservations": utility_displaced,
                "relative_utility_displaced_by_reservations": (
                    utility_displaced / global_utility if global_utility else None
                ),
                "reservation_only_rows": len(reservation_keys - global_keys),
                "global_only_rows": len(global_keys - reservation_keys),
            },
        }

    def select_external(
        self,
        pair_keys: tuple[int, ...] | list[int],
        *,
        minimum_support: int,
    ) -> PromotedVocabulary:
        """Validate and materialize a hash-pinned external promotion decision."""

        if minimum_support <= 0:
            raise ValueError("external-selection minimum support must be positive")
        keys = tuple(int(key) for key in pair_keys)
        if not keys or tuple(sorted(set(keys))) != keys:
            raise ValueError("external pair keys must be non-empty, sorted and unique")
        self._prune()
        by_pair: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"utility": 0.0, "p_sum": 0.0, "count": 0, "languages": {}}
        )
        for composite, raw in self.records.items():
            pair_key, language = divmod(composite, self.n_languages)
            utility, p_sum, count = float(raw[0]), float(raw[1]), int(raw[2])
            row = by_pair[pair_key]
            row["utility"] += utility
            row["p_sum"] += p_sum
            row["count"] += count
            row["languages"][language] = {
                "utility": utility, "p_sum": p_sum, "count": count,
            }
        missing = [key for key in keys if key not in by_pair]
        if missing:
            raise ValueError(f"external pair keys absent from discovery: {missing[:8]}")
        ineligible = [
            key for key in keys
            if by_pair[key]["utility"] <= 0 or by_pair[key]["count"] < minimum_support
        ]
        if ineligible:
            raise ValueError(
                f"external pair keys fail utility/support eligibility: {ineligible[:8]}"
            )
        rows = []
        for pair_key in keys:
            row = by_pair[pair_key]
            rows.append({
                "pair_key": pair_key,
                "utility": float(row["utility"]),
                "captured_support": int(row["count"]),
                "mean_probability": (
                    float(row["p_sum"]) / int(row["count"]) if row["count"] else 0.0
                ),
                "language_count": len(row["languages"]),
                "reservation_language_count": 0,
                "max_reservation_support": 0,
                "global_support_eligible": int(row["count"]) >= minimum_support,
            })
        return PromotedVocabulary(
            pair_keys=keys,
            records=tuple(rows),
            candidate_count=len(by_pair),
            reservation_memberships=(),
        )


class FrequencyVocabularyDiscovery:
    """Independent fixed-memory heavy hitters ranked only by occurrence count.

    Unlike :class:`ContrastiveVocabularyDiscovery`, this observer never reads
    the value or sign of ``-p * dL/dp``.  A Count-Min sketch sees every pair
    occurrence, while a bounded exact key table retains the largest sketch
    estimates.  The table is pair-level rather than pair/language-level so a
    globally frequent pair does not spend multiple candidate slots merely
    because it occurs in several languages.

    The sketch has the same depth and width rule as the utility collector.  Its
    estimates can over-count because of sketch collisions; they cannot
    under-count.  This is an independent fixed-memory frequency selector, not
    an exact unbounded corpus histogram.
    """

    admission_mode = "frequency"

    def __init__(
        self,
        *,
        n_languages: int,
        candidate_capacity: int,
        top_per_forward: int,
    ) -> None:
        if n_languages <= 0:
            raise ValueError("n_languages must be positive")
        if candidate_capacity <= 0:
            raise ValueError("candidate_capacity must be positive")
        if top_per_forward <= 0:
            raise ValueError("top_per_forward must be positive")
        self.n_languages = int(n_languages)
        self.candidate_capacity = int(candidate_capacity)
        self.top_per_forward = int(top_per_forward)
        # pair key -> [unused utility, unused probability sum, support estimate]
        self.records: dict[int, list[float | int]] = {}
        self._sketch_width = max(64, self.candidate_capacity)
        self._frequency_sketch = torch.zeros(
            (len(_SKETCH_A), self._sketch_width), dtype=torch.long
        )
        self._heap: list[tuple[int, int, int]] = []
        self.observed_occurrences = 0
        self.transferred_records = 0
        self.unique_records_before_transfer = 0
        self.dropped_by_top_per_forward = 0
        self.retained_updates = 0
        self.admitted_into_free_slots = 0
        self.capacity_replacements = 0
        self.capacity_rejections = 0

    @staticmethod
    def _rank(key: int, row: list[float | int]) -> tuple[int, int]:
        return int(row[2]), -int(key)

    def _push_heap(self, key: int) -> None:
        heapq.heappush(self._heap, (*self._rank(key, self.records[key]), int(key)))

    def _discard_stale_heap_entries(self) -> None:
        while self._heap:
            support, negative_key, key = self._heap[0]
            row = self.records.get(key)
            if row is not None and (support, negative_key) == self._rank(key, row):
                return
            heapq.heappop(self._heap)

    def _rebuild_heap(self) -> None:
        self._heap = [
            (*self._rank(key, row), key) for key, row in self.records.items()
        ]
        heapq.heapify(self._heap)

    def _update_frequency_sketch(
        self, keys: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Update Count-Min state for every key and return its current estimate."""

        keys = keys.to(dtype=torch.long)
        counts = counts.to(device=keys.device, dtype=torch.long)
        if self._frequency_sketch.device != keys.device:
            self._frequency_sketch = self._frequency_sketch.to(keys.device)
        estimates = []
        for depth, (multiplier, offset) in enumerate(
            zip(_SKETCH_A, _SKETCH_B, strict=True)
        ):
            bucket = torch.remainder(
                keys * multiplier + offset, self._sketch_width
            )
            self._frequency_sketch[depth].index_add_(0, bucket, counts)
            estimates.append(self._frequency_sketch[depth, bucket])
        return torch.stack(estimates).amin(dim=0)

    @staticmethod
    def _largest_with_deterministic_boundary(
        keys: torch.Tensor, estimates: torch.Tensor, keep: int
    ) -> torch.Tensor:
        """Keep the largest estimates, resolving the cutoff by smaller key."""

        if keep >= keys.numel():
            return torch.arange(keys.numel(), device=keys.device)
        boundary = torch.topk(estimates, keep, sorted=False).values.min()
        above = torch.nonzero(estimates > boundary, as_tuple=False).squeeze(1)
        ties = torch.nonzero(estimates == boundary, as_tuple=False).squeeze(1)
        remaining = keep - int(above.numel())
        if remaining <= 0:
            return above[:keep]
        tie_choice = torch.topk(
            -keys.index_select(0, ties), remaining, sorted=False
        ).indices
        return torch.cat((above, ties.index_select(0, tie_choice)))

    @torch.no_grad()
    def observe(
        self,
        pair_key: torch.Tensor,
        language: torch.Tensor,
        probability: torch.Tensor,
        gradient: torch.Tensor,
    ) -> None:
        del language, probability, gradient
        if not pair_key.numel():
            return
        pair_key = pair_key.detach().to(torch.long)
        self.observed_occurrences += int(pair_key.numel())
        unique, counts = torch.unique(pair_key, return_counts=True)
        estimates = self._update_frequency_sketch(unique, counts)

        self.unique_records_before_transfer += int(unique.numel())
        keep = min(self.top_per_forward, unique.numel())
        if keep < unique.numel():
            self.dropped_by_top_per_forward += int(unique.numel() - keep)
            chosen = self._largest_with_deterministic_boundary(
                unique, estimates, keep
            )
            unique = unique.index_select(0, chosen)
            estimates = estimates.index_select(0, chosen)

        keys = unique.cpu().tolist()
        supports = estimates.cpu().tolist()
        self.transferred_records += len(keys)
        incoming = {int(key): int(support) for key, support in zip(keys, supports)}

        retained_keys = sorted(key for key in incoming if key in self.records)
        self.retained_updates += len(retained_keys)
        for key in retained_keys:
            support = incoming.pop(key)
            self.records[key][2] = max(int(self.records[key][2]), support)
            self._push_heap(key)

        unseen = sorted(
            incoming.items(), key=lambda item: (item[1], -item[0]), reverse=True
        )
        for key, support in unseen:
            row: list[float | int] = [0.0, 0.0, int(support)]
            if len(self.records) < self.candidate_capacity:
                self.records[key] = row
                self._push_heap(key)
                self.admitted_into_free_slots += 1
                continue
            self._discard_stale_heap_entries()
            if not self._heap:
                self._rebuild_heap()
            admission_rank = (int(support), -int(key))
            if admission_rank <= self._heap[0][:2]:
                self.capacity_rejections += 1
                continue
            victim = heapq.heappop(self._heap)[2]
            del self.records[victim]
            self.records[key] = row
            self._push_heap(key)
            self.capacity_replacements += 1

        if len(self._heap) > 2 * self.candidate_capacity:
            self._rebuild_heap()

    def _prune(self) -> None:
        ranked = sorted(
            self.records.items(),
            key=lambda item: self._rank(item[0], item[1]),
            reverse=True,
        )[: self.candidate_capacity]
        self.records = dict(ranked)
        self._rebuild_heap()

    def snapshot_records(self) -> tuple[dict[str, float | int], ...]:
        """Return one deterministic row per retained global pair key."""

        self._prune()
        return tuple(
            {
                "composite_key": int(pair_key),
                "pair_key": int(pair_key),
                "language_index": -1,
                "utility": 0.0,
                "probability_sum": 0.0,
                "captured_support": int(row[2]),
                "mean_probability": 0.0,
            }
            for pair_key, row in sorted(self.records.items())
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the complete fixed-memory frequency state for exact resume."""

        self._prune()
        return {
            "version": 1,
            "protocol": "independent-frequency-count-min-v1",
            "n_languages": self.n_languages,
            "candidate_capacity": self.candidate_capacity,
            "top_per_forward": self.top_per_forward,
            "sketch_width": self._sketch_width,
            "records": {
                int(key): [0.0, 0.0, int(row[2])]
                for key, row in sorted(self.records.items())
            },
            "frequency_sketch": self._frequency_sketch.detach().cpu().clone(),
            "observed_occurrences": self.observed_occurrences,
            "transferred_records": self.transferred_records,
            "admission_pressure": self.admission_pressure(),
        }

    def load_state_dict(self, raw: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict` after validation."""

        if (
            int(raw.get("version", -1)) != 1
            or raw.get("protocol") != "independent-frequency-count-min-v1"
        ):
            raise ValueError("unsupported frequency discovery state")
        if int(raw.get("n_languages", -1)) != self.n_languages:
            raise ValueError("frequency discovery language count does not match")
        if int(raw.get("candidate_capacity", -1)) != self.candidate_capacity:
            raise ValueError("frequency discovery candidate capacity does not match")
        if int(raw.get("top_per_forward", -1)) != self.top_per_forward:
            raise ValueError("frequency discovery top-per-forward does not match")
        if int(raw.get("sketch_width", -1)) != self._sketch_width:
            raise ValueError("frequency discovery sketch width does not match")
        sketch = raw.get("frequency_sketch")
        if (
            not torch.is_tensor(sketch)
            or sketch.dtype != torch.long
            or tuple(sketch.shape) != tuple(self._frequency_sketch.shape)
            or bool((sketch < 0).any())
        ):
            raise ValueError("frequency discovery sketch is invalid")
        raw_records = raw.get("records")
        if not isinstance(raw_records, dict) or len(raw_records) > self.candidate_capacity:
            raise ValueError("frequency discovery records are invalid")
        records: dict[int, list[float | int]] = {}
        for raw_key, raw_row in raw_records.items():
            key = int(raw_key)
            if key < 0 or not isinstance(raw_row, (list, tuple)) or len(raw_row) != 3:
                raise ValueError("frequency discovery record is invalid")
            support = int(raw_row[2])
            if support < 0:
                raise ValueError("frequency discovery support is invalid")
            records[key] = [0.0, 0.0, support]
        observed = int(raw.get("observed_occurrences", -1))
        transferred = int(raw.get("transferred_records", -1))
        if observed < 0 or transferred < 0:
            raise ValueError("frequency discovery counters are invalid")
        counter_names = tuple(self.admission_pressure())
        pressure = raw.get("admission_pressure")
        if not isinstance(pressure, dict) or set(pressure) != set(counter_names):
            raise ValueError("frequency discovery admission counters are invalid")
        pressure = {name: int(pressure[name]) for name in counter_names}
        if any(value < 0 for value in pressure.values()):
            raise ValueError("frequency discovery admission counters are invalid")

        self.records = records
        self._frequency_sketch = sketch.detach().cpu().clone()
        self.observed_occurrences = observed
        self.transferred_records = transferred
        for name, value in pressure.items():
            setattr(self, name, value)
        self._rebuild_heap()

    def admission_pressure(self) -> dict[str, int]:
        return {
            "unique_records_before_transfer": self.unique_records_before_transfer,
            "dropped_by_top_per_forward": self.dropped_by_top_per_forward,
            "retained_updates": self.retained_updates,
            "admitted_into_free_slots": self.admitted_into_free_slots,
            "capacity_replacements": self.capacity_replacements,
            "capacity_rejections": self.capacity_rejections,
        }



class CollapseTable(nn.Module):
    def __init__(
        self,
        buckets: int,
        dim: int,
        n_languages: int,
        bias_init: float = -2.0,
        *,
        pair_key_base: int | None = None,
        pair_keys: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        pair_slots: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        reclaimed_token_ids: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        residual_buckets: int = 0,
    ) -> None:
        super().__init__()
        self.buckets = int(buckets)
        self.residual_buckets = int(residual_buckets)
        if self.residual_buckets < 0:
            raise ValueError("residual_buckets must be non-negative")
        self.pair_key_base = int(pair_key_base or 0)
        reclaimed = torch.as_tensor(
            reclaimed_token_ids if reclaimed_token_ids is not None else [], dtype=torch.long
        )
        self.merged = nn.Parameter(torch.zeros(
            0 if reclaimed.numel() else self.buckets, dim
        ))
        self.score = nn.Parameter(torch.zeros(self.buckets))
        self.residual_merged = nn.Parameter(torch.zeros(self.residual_buckets, dim))
        self.residual_score = nn.Parameter(torch.zeros(self.residual_buckets))
        self.language_bias = nn.Parameter(torch.full((n_languages,), float(bias_init)))
        keys = torch.as_tensor(pair_keys if pair_keys is not None else [], dtype=torch.long)
        slots = torch.as_tensor(pair_slots if pair_slots is not None else [], dtype=torch.long)
        if keys.numel() != slots.numel():
            raise ValueError("pair_keys and pair_slots must have the same length")
        if keys.numel() and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("pair_keys must be strictly increasing")
        if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= self.buckets):
            raise ValueError("pair_slots must index the collapse table")
        if reclaimed.numel() and reclaimed.numel() != keys.numel():
            raise ValueError("reclaimed_token_ids must align one-to-one with pair_keys")
        if reclaimed.unique().numel() != reclaimed.numel():
            raise ValueError("reclaimed_token_ids must be unique")
        self.register_buffer("pair_keys", keys, persistent=True)
        self.register_buffer("pair_slots", slots, persistent=True)
        self.register_buffer("reclaimed_token_ids", reclaimed, persistent=True)
        self.discovery: (
            ContrastiveVocabularyDiscovery | FrequencyVocabularyDiscovery | None
        ) = None
        self.diagnostic_discovery: (
            ContrastiveVocabularyDiscovery | FrequencyVocabularyDiscovery | None
        ) = None
        self._vocabulary_telemetry: Any | None = None
        self._loss_path_scale: float | None = None
        self._loss_path_observer: Any | None = None

    def extra_repr(self) -> str:
        return (f"buckets={self.buckets}, explicit_pairs={self.pair_keys.numel()}, "
                f"residual_buckets={self.residual_buckets}, "
                f"reclaimed_rows={self.reclaimed_token_ids.numel()}")

    def set_vocabulary_telemetry(self, window: Any | None) -> None:
        """Attach detached run telemetry without making it module state."""

        self._vocabulary_telemetry = window

    def begin_loss_path_pass(self, scale: float, observer: Any) -> None:
        """Scale every soft pair gate and capture occurrence gradients.

        This is intentionally transient (not module state): an audit pass may
        read the graph but must never alter a checkpoint or discovery table.
        """

        if self._loss_path_scale is not None or self._loss_path_observer is not None:
            raise RuntimeError("a loss-path pass is already active")
        if float(scale) not in (0.0, 0.5, 1.0):
            raise ValueError("registered loss-path scale must be 0, 0.5, or 1")
        if observer is None or not callable(observer):
            raise TypeError("loss-path observer must be callable")
        self._loss_path_scale = float(scale)
        self._loss_path_observer = observer

    def end_loss_path_pass(self) -> None:
        if self._loss_path_scale is None or self._loss_path_observer is None:
            raise RuntimeError("no loss-path pass is active")
        self._loss_path_scale = None
        self._loss_path_observer = None

    def abort_loss_path_pass(self) -> None:
        self._loss_path_scale = None
        self._loss_path_observer = None

    def _hash(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a * _MIX_A + b * _MIX_B).abs()) % self.buckets

    def _hash_residual(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.residual_buckets <= 0:
            raise ValueError("residual hash requires a positive residual capacity")
        return ((a * _MIX_A + b * _MIX_B).abs()) % self.residual_buckets

    def _exact_key(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.pair_key_base <= 0:
            raise ValueError("pair_key_base is required for exact-pair vocabulary")
        return a.to(torch.long) * self.pair_key_base + b.to(torch.long)

    def configure_discovery(
        self,
        *,
        candidate_capacity: int,
        top_per_forward: int,
        admission_mode: str = "utility",
    ) -> None:
        if self.pair_keys.numel() and not self.residual_buckets:
            raise ValueError("cannot discover after explicit vocabulary promotion")
        discovery_class = {
            "utility": ContrastiveVocabularyDiscovery,
            "frequency": FrequencyVocabularyDiscovery,
        }.get(str(admission_mode))
        if discovery_class is None:
            raise ValueError("discovery admission_mode must be utility or frequency")
        self.discovery = discovery_class(
            n_languages=self.language_bias.numel(),
            candidate_capacity=candidate_capacity,
            top_per_forward=top_per_forward,
        )

    def configure_diagnostic_discovery(
        self,
        *,
        candidate_capacity: int,
        top_per_forward: int,
        admission_mode: str = "utility",
    ) -> None:
        """Attach a resettable observer which cannot affect promotion state."""

        if self.pair_keys.numel() and not self.residual_buckets:
            raise ValueError("cannot diagnose discovery after explicit promotion")
        discovery_class = {
            "utility": ContrastiveVocabularyDiscovery,
            "frequency": FrequencyVocabularyDiscovery,
        }.get(str(admission_mode))
        if discovery_class is None:
            raise ValueError("discovery admission_mode must be utility or frequency")
        self.diagnostic_discovery = discovery_class(
            n_languages=self.language_bias.numel(),
            candidate_capacity=candidate_capacity,
            top_per_forward=top_per_forward,
        )

    def _explicit_lookup(self, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.pair_keys.numel():
            return torch.zeros_like(key, dtype=torch.bool), torch.zeros_like(key)
        positions = torch.searchsorted(self.pair_keys, key)
        safe = positions.clamp(max=self.pair_keys.numel() - 1)
        found = (positions < self.pair_keys.numel()) & (self.pair_keys[safe] == key)
        return found, self.pair_slots[safe]

    @staticmethod
    def _fold_rows(value: torch.Tensor, rows: int) -> torch.Tensor:
        if value.ndim == 0:
            return value.detach().clone()
        if rows <= 0 or value.shape[0] < rows:
            raise ValueError("residual folding requires no more rows than its source")
        destination = torch.arange(value.shape[0], device=value.device) % rows
        folded = torch.zeros(
            (rows, *value.shape[1:]), dtype=value.dtype, device=value.device
        )
        folded.index_add_(0, destination, value)
        counts = torch.bincount(destination, minlength=rows).to(value.dtype)
        return folded / counts.reshape(rows, *([1] * (value.ndim - 1)))

    @staticmethod
    def _row_state(
        state: dict[Any, Any],
        source: nn.Parameter,
        rows: torch.Tensor,
    ) -> dict[Any, Any]:
        if not state:
            raise ValueError("growth requires initialized optimizer state")
        result: dict[Any, Any] = {}
        for name, value in state.items():
            if not torch.is_tensor(value):
                result[name] = value
            elif value.ndim == 0:
                result[name] = value.detach().clone()
            elif tuple(value.shape) == tuple(source.shape):
                result[name] = value.index_select(0, rows.to(value.device)).detach().clone()
            else:
                raise ValueError(f"unknown optimizer row-state shape: {name}")
        return result

    @staticmethod
    def _concat_row_states(
        left: dict[Any, Any] | None,
        right: dict[Any, Any],
        order: torch.Tensor,
    ) -> dict[Any, Any]:
        if left is None:
            return {
                name: (
                    value.index_select(0, order.to(value.device))
                    if torch.is_tensor(value) and value.ndim
                    else value.detach().clone()
                    if torch.is_tensor(value)
                    else value
                )
                for name, value in right.items()
            }
        if set(left) != set(right):
            raise ValueError("optimizer state keys differ across growth tiers")
        result: dict[Any, Any] = {}
        for name in left:
            old_value, new_value = left[name], right[name]
            if torch.is_tensor(old_value) != torch.is_tensor(new_value):
                raise ValueError(f"optimizer state type differs: {name}")
            if not torch.is_tensor(old_value):
                if old_value != new_value:
                    raise ValueError(f"optimizer scalar state differs: {name}")
                result[name] = old_value
            elif old_value.ndim == 0:
                if not torch.equal(old_value, new_value):
                    raise ValueError(f"optimizer step differs: {name}")
                result[name] = old_value.detach().clone()
            else:
                combined = torch.cat((old_value, new_value))
                result[name] = combined.index_select(
                    0, order.to(combined.device)
                ).detach().clone()
        return result

    def grow_exact_vocabulary(
        self,
        pair_keys: tuple[int, ...] | list[int],
        *,
        optimizer: torch.optim.Optimizer,
        residual_buckets: int,
    ) -> dict[str, int]:
        """Append exact rows while retaining a residual proposal channel."""

        additions = torch.as_tensor(
            pair_keys, dtype=torch.long, device=self.merged.device
        )
        if not additions.numel():
            raise ValueError("growth requires at least one exact pair")
        if additions.numel() > 1 and not bool((additions[1:] > additions[:-1]).all()):
            raise ValueError("growth pair keys must be strictly increasing")
        if self.reclaimed_token_ids.numel():
            raise ValueError("reclaimed vocabulary cannot attach a residual tier")
        if int(residual_buckets) <= 0:
            raise ValueError("growth residual capacity must be positive")
        if self.residual_buckets and int(residual_buckets) != self.residual_buckets:
            raise ValueError("growth cannot resize an existing residual tier")
        if self.pair_keys.numel():
            positions = torch.searchsorted(self.pair_keys, additions)
            safe = positions.clamp(max=self.pair_keys.numel() - 1)
            if bool(
                ((positions < self.pair_keys.numel()) & (self.pair_keys[safe] == additions)).any()
            ):
                raise ValueError("growth pair already exists in the exact vocabulary")

        old_merged, old_score = self.merged, self.score
        old_residual_merged, old_residual_score = (
            self.residual_merged,
            self.residual_score,
        )
        first = not self.residual_buckets
        tracked = (old_merged, old_score, old_residual_merged, old_residual_score)
        occurrences = {
            parameter: sum(
                candidate is parameter
                for group in optimizer.param_groups
                for candidate in group["params"]
            )
            for parameter in tracked
        }
        residual_occurrences = {0, 1} if first else {1}
        if (
            occurrences[old_merged] != 1
            or occurrences[old_score] != 1
            or occurrences[old_residual_merged] not in residual_occurrences
            or occurrences[old_residual_score] not in residual_occurrences
        ):
            raise ValueError("growth parameters must occur once in the optimizer")

        source_merged = old_merged if first else old_residual_merged
        source_score = old_score if first else old_residual_score
        source_index = (
            self._hash(
                torch.div(additions, self.pair_key_base, rounding_mode="floor"),
                additions.remainder(self.pair_key_base),
            )
            if first
            else self._hash_residual(
                torch.div(additions, self.pair_key_base, rounding_mode="floor"),
                additions.remainder(self.pair_key_base),
            )
        )
        source_merged_state = optimizer.state.get(source_merged, {})
        source_score_state = optimizer.state.get(source_score, {})
        added_merged_state = self._row_state(
            source_merged_state, source_merged, source_index
        )
        added_score_state = self._row_state(
            source_score_state, source_score, source_index
        )

        old_keys = self.pair_keys
        old_slots = self.pair_slots
        combined_keys = torch.cat((old_keys, additions))
        order = torch.argsort(combined_keys, stable=True)
        combined_merged = torch.cat(
            (
                old_merged.index_select(0, old_slots),
                source_merged.index_select(0, source_index),
            )
        ).index_select(0, order)
        combined_score = torch.cat(
            (
                old_score.index_select(0, old_slots),
                source_score.index_select(0, source_index),
            )
        ).index_select(0, order)
        old_merged_state = (
            None
            if not old_keys.numel()
            else self._row_state(
                optimizer.state.get(old_merged, {}), old_merged, old_slots
            )
        )
        old_score_state = (
            None
            if not old_keys.numel()
            else self._row_state(
                optimizer.state.get(old_score, {}), old_score, old_slots
            )
        )
        new_merged_state = self._concat_row_states(
            old_merged_state, added_merged_state, order
        )
        new_score_state = self._concat_row_states(
            old_score_state, added_score_state, order
        )

        if first:
            if source_merged.shape[0] < int(residual_buckets):
                raise ValueError("residual capacity exceeds the proposal table")
            residual_merged_value = self._fold_rows(
                source_merged.detach(), int(residual_buckets)
            )
            residual_score_value = self._fold_rows(
                source_score.detach(), int(residual_buckets)
            )
            new_residual_merged_state = {
                name: (
                    self._fold_rows(value, int(residual_buckets))
                    if torch.is_tensor(value) and value.ndim
                    else value.detach().clone()
                    if torch.is_tensor(value)
                    else value
                )
                for name, value in source_merged_state.items()
            }
            new_residual_score_state = {
                name: (
                    self._fold_rows(value, int(residual_buckets))
                    if torch.is_tensor(value) and value.ndim
                    else value.detach().clone()
                    if torch.is_tensor(value)
                    else value
                )
                for name, value in source_score_state.items()
            }
        else:
            residual_merged_value = old_residual_merged.detach().clone()
            residual_score_value = old_residual_score.detach().clone()
            new_residual_merged_state = optimizer.state[old_residual_merged]
            new_residual_score_state = optimizer.state[old_residual_score]

        new_merged = nn.Parameter(
            combined_merged.detach().clone(), requires_grad=old_merged.requires_grad
        )
        new_score = nn.Parameter(
            combined_score.detach().clone(), requires_grad=old_score.requires_grad
        )
        new_residual_merged = nn.Parameter(
            residual_merged_value,
            requires_grad=source_merged.requires_grad,
        )
        new_residual_score = nn.Parameter(
            residual_score_value,
            requires_grad=source_score.requires_grad,
        )
        replacements = {
            old_merged: new_merged,
            old_score: new_score,
            old_residual_merged: new_residual_merged,
            old_residual_score: new_residual_score,
        }
        for group in optimizer.param_groups:
            replaced = []
            for parameter in group["params"]:
                replaced.append(replacements.get(parameter, parameter))
                # Resume compatibility deliberately removes zero-sized tensors
                # from AdamW.  The first growth transition creates the real
                # residual tier, so attach each new parameter beside the source
                # parameter whose folded value and moments initialize it.
                if first and parameter is old_score:
                    # Match ``CollapseTable`` registration order exactly.
                    # Optimizer checkpoint loading binds state positionally, so
                    # putting residual_merged beside merged would swap the
                    # score vector and residual matrix after resume.
                    if occurrences[old_residual_merged] == 0:
                        replaced.append(new_residual_merged)
                    if occurrences[old_residual_score] == 0:
                        replaced.append(new_residual_score)
            group["params"] = replaced
        for parameter in tracked:
            optimizer.state.pop(parameter, None)
        optimizer.state[new_merged] = new_merged_state
        optimizer.state[new_score] = new_score_state
        optimizer.state[new_residual_merged] = new_residual_merged_state
        optimizer.state[new_residual_score] = new_residual_score_state

        self.merged = new_merged
        self.score = new_score
        self.residual_merged = new_residual_merged
        self.residual_score = new_residual_score
        self.pair_keys = combined_keys.index_select(0, order)
        self.pair_slots = torch.arange(
            self.pair_keys.numel(), dtype=torch.long, device=self.pair_keys.device
        )
        old_exact_rows = int(old_keys.numel())
        self.buckets = int(self.pair_keys.numel())
        self.residual_buckets = int(residual_buckets)
        self.discovery = None
        self.diagnostic_discovery = None
        return {
            "old_exact_rows": old_exact_rows,
            "added_exact_rows": int(additions.numel()),
            "active_exact_rows": int(self.pair_keys.numel()),
            "residual_rows": self.residual_buckets,
        }

    def promote_exact_vocabulary(
        self,
        pair_keys: tuple[int, ...] | list[int],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, int]:
        """Map selected exact pairs to collision-free live parameter rows."""

        keys = torch.as_tensor(pair_keys, dtype=torch.long, device=self.merged.device)
        if keys.numel() and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("promoted pair keys must be strictly increasing")
        left = torch.div(keys, self.pair_key_base, rounding_mode="floor")
        right = keys.remainder(self.pair_key_base)
        source = self._hash(left, right)

        # Preserve each original hash row for one pair. Colliding extra pairs
        # receive unused rows initialized, including Adam moments, from the same
        # source bucket.
        reserved = set(int(value) for value in source.cpu().tolist())
        free = iter(index for index in range(self.buckets) if index not in reserved)
        seen: set[int] = set()
        slots = []
        copied = []
        for src in source.cpu().tolist():
            src = int(src)
            if src not in seen:
                slot = src
                seen.add(src)
            else:
                try:
                    slot = next(free)
                except StopIteration as error:
                    raise ValueError("not enough collapse rows for collision-free promotion") from error
                copied.append((src, slot))
            slots.append(slot)

        with torch.no_grad():
            for src, dst in copied:
                self.merged[dst].copy_(self.merged[src])
                self.score[dst].copy_(self.score[src])
            if optimizer is not None:
                for parameter in (self.merged, self.score):
                    state = optimizer.state.get(parameter, {})
                    for value in state.values():
                        if (torch.is_tensor(value) and value.ndim
                                and value.shape[0] == self.buckets):
                            for src, dst in copied:
                                value[dst].copy_(value[src])

        self.pair_keys = keys
        self.pair_slots = torch.tensor(slots, dtype=torch.long, device=self.merged.device)
        self.discovery = None
        self.diagnostic_discovery = None
        return {
            "active_pairs": int(keys.numel()),
            "unique_source_buckets": len(reserved),
            "promoted_hash_collisions": len(copied),
        }

    def promote_exact_vocabulary_to_reclaimed(
        self,
        pair_keys: tuple[int, ...] | list[int],
        reclaimed_token_ids: tuple[int, ...] | list[int] | torch.Tensor,
        embedding_weight: nn.Parameter,
        *,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, int]:
        """Move a promoted exact vocabulary into existing base-table rows.

        The transition is elementwise-equivalent to ``promote_exact_vocabulary``
        at the promotion boundary. AdamW moments follow the copied vectors, and
        the obsolete full merged parameter/state is removed before training
        continues.
        """

        keys = torch.as_tensor(pair_keys, dtype=torch.long, device=self.merged.device)
        reclaimed = torch.as_tensor(
            reclaimed_token_ids, dtype=torch.long, device=self.merged.device
        )
        if not keys.numel() or keys.numel() != reclaimed.numel():
            raise ValueError("direct reclaimed promotion needs one base row per exact pair")
        if not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("promoted pair keys must be strictly increasing")
        if (reclaimed.unique().numel() != reclaimed.numel()
                or int(reclaimed.min()) < 0
                or int(reclaimed.max()) >= embedding_weight.shape[0]):
            raise ValueError("invalid direct-promotion reclaimed token ids")
        if self.pair_keys.numel() or self.reclaimed_token_ids.numel():
            raise ValueError("direct reclaimed promotion can run only once")
        if optimizer is None:
            raise ValueError("direct reclaimed promotion requires the live optimizer")

        left = torch.div(keys, self.pair_key_base, rounding_mode="floor")
        right = keys.remainder(self.pair_key_base)
        source = self._hash(left, right)
        unique_sources = int(source.unique().numel())
        old_merged, old_score = self.merged, self.score
        if tuple(old_merged.shape) != (self.buckets, embedding_weight.shape[1]):
            raise ValueError("direct promotion source table shape mismatch")

        occurrences = {
            "merged": sum(parameter is old_merged for group in optimizer.param_groups
                          for parameter in group["params"]),
            "score": sum(parameter is old_score for group in optimizer.param_groups
                         for parameter in group["params"]),
            "embedding": sum(parameter is embedding_weight for group in optimizer.param_groups
                             for parameter in group["params"]),
        }
        if occurrences != {"merged": 1, "score": 1, "embedding": 1}:
            raise ValueError(f"unexpected optimizer parameter membership: {occurrences}")
        merged_state = optimizer.state.get(old_merged)
        score_state = optimizer.state.get(old_score)
        embedding_state = optimizer.state.get(embedding_weight)
        if not merged_state or not score_state or not embedding_state:
            raise ValueError("AdamW state must exist before direct promotion")
        if set(merged_state) != set(score_state) or set(merged_state) != set(embedding_state):
            raise ValueError("optimizer state keys differ across promoted parameters")

        new_score_state: dict[object, object] = {}
        with torch.no_grad():
            embedding_weight.index_copy_(
                0, reclaimed.to(embedding_weight.device),
                old_merged.index_select(0, source).to(embedding_weight.device),
            )
            for state_key in merged_state:
                merged_value = merged_state[state_key]
                score_value = score_state[state_key]
                embedding_value = embedding_state[state_key]
                if not torch.is_tensor(merged_value):
                    if not (merged_value == score_value == embedding_value):
                        raise ValueError(f"optimizer scalar state mismatch: {state_key}")
                    new_score_state[state_key] = score_value
                    continue
                if merged_value.ndim == 0:
                    if not (torch.equal(merged_value, score_value)
                            and torch.equal(merged_value, embedding_value)):
                        raise ValueError(f"optimizer step mismatch: {state_key}")
                    new_score_state[state_key] = score_value.detach().clone()
                    continue
                if (tuple(merged_value.shape) != tuple(old_merged.shape)
                        or tuple(embedding_value.shape) != tuple(embedding_weight.shape)
                        or tuple(score_value.shape) != tuple(old_score.shape)):
                    raise ValueError(f"unknown optimizer row-state shape: {state_key}")
                embedding_value.index_copy_(
                    0, reclaimed.to(embedding_value.device),
                    merged_value.index_select(0, source).to(embedding_value.device),
                )
                new_score_state[state_key] = score_value.index_select(
                    0, source.to(score_value.device)
                ).detach().clone()

        new_score = nn.Parameter(
            old_score.index_select(0, source).detach().clone(),
            requires_grad=old_score.requires_grad,
        )
        for group in optimizer.param_groups:
            replacement = []
            for parameter in group["params"]:
                if parameter is old_merged:
                    continue
                replacement.append(new_score if parameter is old_score else parameter)
            group["params"] = replacement
        optimizer.state.pop(old_merged)
        optimizer.state.pop(old_score)
        optimizer.state[new_score] = new_score_state

        self.merged = nn.Parameter(
            torch.empty((0, embedding_weight.shape[1]), dtype=old_merged.dtype,
                        device=old_merged.device),
            requires_grad=False,
        )
        self.score = new_score
        self.pair_keys = keys
        self.pair_slots = torch.arange(keys.numel(), device=keys.device)
        self.reclaimed_token_ids = reclaimed
        self.buckets = int(keys.numel())
        self.discovery = None
        self.diagnostic_discovery = None
        return {
            "active_pairs": int(keys.numel()),
            "unique_source_buckets": unique_sources,
            "promoted_hash_collisions": int(keys.numel()) - unique_sources,
            "reclaimed_rows": int(reclaimed.numel()),
            "released_merged_rows": int(old_merged.shape[0]),
        }

    def compact_exact_vocabulary(self) -> dict[str, int]:
        """Drop inactive discovery rows after training, before final save."""

        active = int(self.pair_keys.numel())
        if not active:
            raise ValueError("cannot compact an empty exact vocabulary")
        old_buckets = self.buckets
        with torch.no_grad():
            merged = self.merged[self.pair_slots].detach().clone()
            score = self.score[self.pair_slots].detach().clone()
        self.merged = nn.Parameter(merged, requires_grad=self.merged.requires_grad)
        self.score = nn.Parameter(score, requires_grad=self.score.requires_grad)
        self.pair_slots = torch.arange(active, device=self.pair_keys.device)
        self.buckets = active
        return {"old_rows": old_buckets, "active_rows": active, "dropped_rows": old_buckets - active}

    def pool(
        self,
        vectors: torch.Tensor,     # (N, dim) token vectors, sentence-ordered
        content: torch.Tensor,     # (N,) token ids
        segment: torch.Tensor,     # (N,) sentence index
        language: torch.Tensor,    # (B,) language per sentence
        n_sentences: int,
        embedding_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = content.numel()
        dim = vectors.shape[1]
        weight = torch.ones(n, dtype=vectors.dtype, device=vectors.device)
        numerator = torch.zeros(n_sentences, dim, dtype=vectors.dtype, device=vectors.device)
        denominator = torch.zeros(n_sentences, dtype=vectors.dtype, device=vectors.device)
        denominator.index_add_(0, segment, weight)
        telemetry_observed = False

        if n > 1:
            left, right = content[:-1], content[1:]
            inside = segment[:-1] == segment[1:]
            if bool(inside.any()):
                pair_left = left[inside]
                pair_right = right[inside]
                pair_segment = segment[:-1][inside]
                pair_language = language[pair_segment]
                exact_key = self._exact_key(pair_left, pair_right)
                all_positions = torch.nonzero(inside, as_tuple=False).squeeze(1)
                discovery_probability = None
                discovery_key = None
                discovery_language = None
                if self.pair_keys.numel() and self.residual_buckets:
                    found, exact_index = self._explicit_lookup(exact_key)
                    missed = ~found
                    exact_index = exact_index[found]
                    residual_index = self._hash_residual(
                        pair_left[missed], pair_right[missed]
                    )
                    exact_probability = torch.sigmoid(
                        self.score[exact_index]
                        + self.language_bias[language[pair_segment[found]]]
                    )
                    residual_probability = torch.sigmoid(
                        self.residual_score[residual_index]
                        + self.language_bias[language[pair_segment[missed]]]
                    )
                    p = torch.cat((exact_probability, residual_probability))
                    pair_segment = torch.cat((pair_segment[found], pair_segment[missed]))
                    pair_positions = torch.cat((all_positions[found], all_positions[missed]))
                    merged = torch.cat(
                        (
                            self.merged[exact_index],
                            self.residual_merged[residual_index],
                        )
                    )
                    telemetry_index = torch.cat(
                        (exact_index, residual_index + self.buckets)
                    )
                    telemetry_exact_hits = torch.cat(
                        (
                            torch.ones_like(exact_index, dtype=torch.bool),
                            torch.zeros_like(residual_index, dtype=torch.bool),
                        )
                    )
                    discovery_probability = residual_probability
                    discovery_key = exact_key[missed]
                    discovery_language = pair_language[missed]
                    audit_key = torch.cat((exact_key[found], exact_key[missed]))
                elif self.pair_keys.numel():
                    found, index = self._explicit_lookup(exact_key)
                    pair_segment = pair_segment[found]
                    pair_language = pair_language[found]
                    exact_key = exact_key[found]
                    index = index[found]
                    pair_positions = all_positions[found]
                    p = torch.sigmoid(
                        self.score[index]
                        + self.language_bias[language[pair_segment]]
                    )
                    if self.reclaimed_token_ids.numel():
                        if embedding_weight is None:
                            raise ValueError(
                                "reclaimed collapse requires the base embedding table"
                            )
                        merged = embedding_weight[
                            self.reclaimed_token_ids.index_select(0, index)
                        ]
                    else:
                        merged = self.merged[index]
                    telemetry_index = index
                    telemetry_exact_hits = torch.ones_like(index, dtype=torch.bool)
                    discovery_probability = p
                    discovery_key = exact_key
                    discovery_language = pair_language
                    audit_key = exact_key
                else:
                    index = self._hash(pair_left, pair_right)
                    pair_positions = all_positions
                    p = torch.sigmoid(
                        self.score[index]
                        + self.language_bias[language[pair_segment]]
                    )
                    merged = self.merged[index]
                    telemetry_index = index
                    telemetry_exact_hits = torch.zeros_like(index, dtype=torch.bool)
                    discovery_probability = p
                    discovery_key = exact_key
                    discovery_language = pair_language
                    audit_key = exact_key
                if self._loss_path_scale is not None:
                    if self._loss_path_observer is None:
                        raise RuntimeError("loss-path scale has no observer")
                    original_probability = p.detach()
                    p = p * self._loss_path_scale
                    saved_audit_key = audit_key.detach()
                    saved_audit_probability = original_probability
                    audit_observer = self._loss_path_observer

                    def observe_loss_path(gradient):
                        audit_observer(
                            saved_audit_key, saved_audit_probability, gradient
                        )

                    if p.requires_grad:
                        p.register_hook(observe_loss_path)
                if not p.numel():
                    _weighted_index_add_(numerator, segment, vectors, weight)
                    if self._vocabulary_telemetry is not None:
                        self._vocabulary_telemetry.observe_q15(
                            base_tokens=n,
                            probabilities=torch.empty(
                                0, dtype=torch.float32, device=content.device
                            ),
                            hash_indices=telemetry_index,
                            effective_tokens=denominator.sum(),
                            exact_hits=telemetry_exact_hits,
                        )
                        telemetry_observed = True
                    return numerator / denominator.clamp(min=1e-3).unsqueeze(1)
                discoveries = tuple(
                    observer for observer in (
                        self.discovery, self.diagnostic_discovery
                    ) if observer is not None
                )
                if (
                    discoveries
                    and self.training
                    and discovery_probability is not None
                    and discovery_probability.requires_grad
                    and discovery_probability.numel()
                ):
                    saved_key = discovery_key.detach()
                    saved_language = discovery_language.detach()
                    saved_probability = discovery_probability.detach()
                    def observe_all(gradient):
                        for observer in discoveries:
                            observer.observe(
                                saved_key, saved_language, saved_probability, gradient
                            )
                    discovery_probability.register_hook(observe_all)
                # each participant of a merge gives up half a slot
                half = 0.5 * p
                weight = weight.index_add(0, pair_positions, -half)
                weight = weight.index_add(0, pair_positions + 1, -half)
                if self.reclaimed_token_ids.numel():
                    if embedding_weight is None:
                        raise ValueError("reclaimed collapse requires the base embedding table")
                    merged = embedding_weight[
                        self.reclaimed_token_ids.index_select(0, index)
                    ]
                _weighted_index_add_(numerator, pair_segment, merged, p)
                denominator.index_add_(0, pair_segment, -p)

                if self._vocabulary_telemetry is not None:
                    self._vocabulary_telemetry.observe_q15(
                        base_tokens=n,
                        probabilities=p,
                        hash_indices=telemetry_index,
                        effective_tokens=denominator.sum(),
                        exact_hits=telemetry_exact_hits,
                    )
                    telemetry_observed = True

        _weighted_index_add_(numerator, segment, vectors, weight)
        if not telemetry_observed and self._vocabulary_telemetry is not None:
            self._vocabulary_telemetry.observe_q15(
                base_tokens=n,
                probabilities=torch.empty(
                    0, dtype=torch.float32, device=content.device
                ),
                hash_indices=torch.empty(
                    0, dtype=torch.long, device=content.device
                ),
                effective_tokens=denominator.sum(),
                exact_hits=torch.empty(
                    0, dtype=torch.bool, device=content.device
                ),
            )
        return numerator / denominator.clamp(min=1e-3).unsqueeze(1)

    def mean_merge_probability(self, content, segment, language) -> torch.Tensor:
        if content.numel() < 2:
            return torch.zeros((), device=content.device)
        inside = segment[:-1] == segment[1:]
        if not bool(inside.any()):
            return torch.zeros((), device=content.device)
        left, right = content[:-1][inside], content[1:][inside]
        pair_segment = segment[:-1][inside]
        if self.pair_keys.numel():
            found, index = self._explicit_lookup(self._exact_key(left, right))
            index = index[found]
            pair_segment = pair_segment[found]
        else:
            index = self._hash(left, right)
        if not index.numel():
            return torch.zeros((), device=content.device)
        p = torch.sigmoid(self.score[index] + self.language_bias[language[pair_segment]])
        return p.mean()


def ablate_language_bias(module: nn.Module, mode: str = "none") -> None:
    """Apply an evaluation-only necessity ablation to a loaded collapse model.

    ``mean`` replaces every language bias with their learned mean. It preserves
    the overall merge tendency while removing language-specific routing; every
    other learned parameter remains unchanged. Because the full model was
    trained with language routing active, this does not replace a separately
    trained shared-bias control for a training-time causal claim.
    """

    if mode == "none":
        return
    if mode != "mean":
        raise ValueError(f"Unknown collapse language-bias ablation: {mode!r}")
    tables = [child for child in module.modules() if isinstance(child, CollapseTable)]
    if len(tables) != 1:
        raise ValueError(f"Expected exactly one CollapseTable, found {len(tables)}")
    with torch.no_grad():
        bias = tables[0].language_bias
        bias.fill_(bias.float().mean().to(dtype=bias.dtype, device=bias.device))
