"""Training-only selectors for monotone exact-pair vocabulary growth."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LanguageGrowthRecord:
    language_index: int
    utility: float
    support: int
    probability_sum: float


@dataclass(frozen=True)
class PairGrowthRecord:
    pair_key: int
    utility: float
    support: int
    probability_sum: float
    languages: tuple[LanguageGrowthRecord, ...]

    @property
    def mean_probability(self) -> float:
        return self.probability_sum / self.support if self.support else 0.0

    def has_supported_positive_language(self, minimum_support: int) -> bool:
        return any(
            row.utility > 0 and row.support >= minimum_support
            for row in self.languages
        )


@dataclass(frozen=True)
class GrowthWindow:
    n_languages: int
    pairs: tuple[PairGrowthRecord, ...]

    @classmethod
    def from_discovery(cls, discovery: Any) -> GrowthWindow:
        n_languages = int(getattr(discovery, "n_languages", 0))
        if n_languages <= 0:
            raise ValueError("growth discovery has no valid language universe")
        rows = tuple(discovery.snapshot_records())
        seen_composites: set[int] = set()
        by_pair: dict[int, list[LanguageGrowthRecord]] = {}
        for raw in rows:
            composite = int(raw["composite_key"])
            pair_key = int(raw["pair_key"])
            language = int(raw["language_index"])
            utility = float(raw["utility"])
            probability_sum = float(raw["probability_sum"])
            support = int(raw["captured_support"])
            if composite in seen_composites:
                raise ValueError("duplicate composite growth identity")
            seen_composites.add(composite)
            if composite != pair_key * n_languages + language:
                raise ValueError("growth composite identity is inconsistent")
            if (
                pair_key < 0
                or language < 0
                or language >= n_languages
                or support <= 0
                or not math.isfinite(utility)
                or not math.isfinite(probability_sum)
                or probability_sum < 0
                or probability_sum > support
            ):
                raise ValueError("growth discovery record is invalid")
            by_pair.setdefault(pair_key, []).append(
                LanguageGrowthRecord(
                    language_index=language,
                    utility=utility,
                    support=support,
                    probability_sum=probability_sum,
                )
            )
        pairs = []
        for pair_key, language_rows in sorted(by_pair.items()):
            language_rows.sort(key=lambda row: row.language_index)
            if len({row.language_index for row in language_rows}) != len(language_rows):
                raise ValueError("duplicate pair/language growth identity")
            pairs.append(
                PairGrowthRecord(
                    pair_key=pair_key,
                    utility=math.fsum(row.utility for row in language_rows),
                    support=sum(row.support for row in language_rows),
                    probability_sum=math.fsum(
                        row.probability_sum for row in language_rows
                    ),
                    languages=tuple(language_rows),
                )
            )
        return cls(n_languages=n_languages, pairs=tuple(pairs))

    def by_key(self) -> dict[int, PairGrowthRecord]:
        return {row.pair_key: row for row in self.pairs}

    def state_dict(self) -> dict[str, Any]:
        """Return a weights-only-safe, schema-versioned checkpoint payload."""

        return {
            "version": 1,
            "n_languages": self.n_languages,
            "pairs": [
                {
                    "pair_key": row.pair_key,
                    "utility": row.utility,
                    "support": row.support,
                    "probability_sum": row.probability_sum,
                    "languages": [
                        {
                            "language_index": language.language_index,
                            "utility": language.utility,
                            "support": language.support,
                            "probability_sum": language.probability_sum,
                        }
                        for language in row.languages
                    ],
                }
                for row in self.pairs
            ],
        }

    @classmethod
    def load_state_dict(cls, state: dict[str, Any]) -> GrowthWindow:
        if set(state) != {"version", "n_languages", "pairs"} or state["version"] != 1:
            raise ValueError("unknown growth-window checkpoint schema")
        n_languages = int(state["n_languages"])
        if n_languages <= 0 or not isinstance(state["pairs"], list):
            raise ValueError("invalid growth-window checkpoint")
        pairs = []
        previous_key = -1
        for raw in state["pairs"]:
            if not isinstance(raw, dict) or set(raw) != {
                "pair_key", "utility", "support", "probability_sum", "languages"
            }:
                raise ValueError("invalid growth-window pair checkpoint")
            key = int(raw["pair_key"])
            if key <= previous_key or not isinstance(raw["languages"], list):
                raise ValueError("growth-window keys must be unique and sorted")
            previous_key = key
            languages = tuple(
                LanguageGrowthRecord(
                    language_index=int(language["language_index"]),
                    utility=float(language["utility"]),
                    support=int(language["support"]),
                    probability_sum=float(language["probability_sum"]),
                )
                for language in raw["languages"]
            )
            row = PairGrowthRecord(
                pair_key=key,
                utility=float(raw["utility"]),
                support=int(raw["support"]),
                probability_sum=float(raw["probability_sum"]),
                languages=languages,
            )
            if (
                row.support <= 0
                or not math.isfinite(row.utility)
                or not math.isfinite(row.probability_sum)
                or tuple(sorted({item.language_index for item in languages}))
                != tuple(item.language_index for item in languages)
                or any(
                    item.language_index < 0
                    or item.language_index >= n_languages
                    or item.support <= 0
                    or not math.isfinite(item.utility)
                    or not math.isfinite(item.probability_sum)
                    for item in languages
                )
                or row.support != sum(item.support for item in languages)
                or not math.isclose(
                    row.utility,
                    math.fsum(item.utility for item in languages),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    row.probability_sum,
                    math.fsum(item.probability_sum for item in languages),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("inconsistent growth-window checkpoint")
            pairs.append(row)
        return cls(n_languages=n_languages, pairs=tuple(pairs))


@dataclass(frozen=True)
class PairSignEvidence:
    pair_key: int
    utility: float
    support: int
    positive_support: int

    def lower_positive_rate(self, *, hypotheses: int, alpha: float) -> float:
        radius = math.sqrt(
            math.log((2.0 * max(1, hypotheses)) / alpha) / (2.0 * self.support)
        )
        return (self.positive_support / self.support) - radius


@dataclass(frozen=True)
class SignEvidenceWindow:
    n_languages: int
    pairs: tuple[PairSignEvidence, ...]

    @classmethod
    def from_discovery(cls, discovery: Any) -> SignEvidenceWindow:
        n_languages = int(getattr(discovery, "n_languages", 0))
        if n_languages <= 0:
            raise ValueError("sign discovery has no valid language universe")
        by_pair: dict[int, list[float | int]] = {}
        seen_composites: set[int] = set()
        for raw in discovery.snapshot_records():
            composite = int(raw["composite_key"])
            pair_key = int(raw["pair_key"])
            language = int(raw["language_index"])
            support = int(raw["captured_support"])
            positive = int(raw.get("positive_utility_support", -1))
            utility = float(raw["utility"])
            if composite in seen_composites:
                raise ValueError("duplicate composite sign identity")
            seen_composites.add(composite)
            if (
                composite != pair_key * n_languages + language
                or pair_key < 0
                or language < 0
                or language >= n_languages
                or support <= 0
                or positive < 0
                or positive > support
                or not math.isfinite(utility)
            ):
                raise ValueError("sign discovery record is invalid")
            row = by_pair.setdefault(pair_key, [0.0, 0, 0])
            row[0] = float(row[0]) + utility
            row[1] = int(row[1]) + support
            row[2] = int(row[2]) + positive
        pairs = tuple(
            PairSignEvidence(
                pair_key=pair_key,
                utility=float(values[0]),
                support=int(values[1]),
                positive_support=int(values[2]),
            )
            for pair_key, values in sorted(by_pair.items())
        )
        return cls(n_languages=n_languages, pairs=pairs)

    def by_key(self) -> dict[int, PairSignEvidence]:
        return {row.pair_key: row for row in self.pairs}


@dataclass(frozen=True)
class GrowthSelection:
    pair_keys: tuple[int, ...]
    candidate_count: int
    rejections: dict[str, int]
    language_allocations: tuple[tuple[int, int], ...] = ()
    global_backfill: int = 0


def _validate_support(min_support: int, per_language_min_support: int) -> None:
    if min_support <= 0 or per_language_min_support <= 0:
        raise ValueError("growth support thresholds must be positive")


def select_stable_growth(
    previous: GrowthWindow,
    current: GrowthWindow,
    existing_keys: Iterable[int],
    *,
    min_support: int = 32,
    per_language_min_support: int = 8,
) -> GrowthSelection:
    _validate_support(min_support, per_language_min_support)
    if previous.n_languages != current.n_languages:
        raise ValueError("growth windows use different language universes")
    old = previous.by_key()
    existing = {int(key) for key in existing_keys}
    rejected = {
        "not_consecutive": 0,
        "nonpositive_utility": 0,
        "insufficient_support": 0,
        "no_supported_positive_language": 0,
        "already_exact": 0,
    }
    selected = []
    for row in current.pairs:
        prior = old.get(row.pair_key)
        if row.pair_key in existing:
            rejected["already_exact"] += 1
        elif prior is None:
            rejected["not_consecutive"] += 1
        elif row.utility <= 0 or prior.utility <= 0:
            rejected["nonpositive_utility"] += 1
        elif row.support < min_support or prior.support < min_support:
            rejected["insufficient_support"] += 1
        elif not (
            row.has_supported_positive_language(per_language_min_support)
            and prior.has_supported_positive_language(per_language_min_support)
        ):
            rejected["no_supported_positive_language"] += 1
        else:
            selected.append(row.pair_key)
    return GrowthSelection(
        pair_keys=tuple(selected),
        candidate_count=len(current.pairs),
        rejections=rejected,
    )


def select_language_balanced_growth(
    previous: GrowthWindow,
    current: GrowthWindow,
    existing_keys: Iterable[int],
    *,
    limit: int,
    min_support: int = 1,
    per_language_min_support: int = 1,
) -> GrowthSelection:
    """Fill an active-row target with stable, language-balanced candidates.

    Utilities are normalized inside each language and half-window before equal
    language reservations are applied.  The token rows themselves remain
    shared: language identity affects admission only, not representation.
    """

    _validate_support(min_support, per_language_min_support)
    if isinstance(limit, bool) or int(limit) < 0:
        raise ValueError("growth selection limit must be nonnegative")
    limit = int(limit)
    if previous.n_languages != current.n_languages:
        raise ValueError("growth windows use different language universes")
    old = previous.by_key()
    existing = {int(key) for key in existing_keys}
    rejected = {
        "not_consecutive": 0,
        "nonpositive_utility": 0,
        "insufficient_support": 0,
        "no_stable_positive_language": 0,
        "already_exact": 0,
        "outside_active_target": 0,
    }
    eligible: list[tuple[PairGrowthRecord, PairGrowthRecord, dict[int, tuple[float, float]]]] = []
    for row in current.pairs:
        prior = old.get(row.pair_key)
        if row.pair_key in existing:
            rejected["already_exact"] += 1
            continue
        if prior is None:
            rejected["not_consecutive"] += 1
            continue
        if row.utility <= 0 or prior.utility <= 0:
            rejected["nonpositive_utility"] += 1
            continue
        if row.support < min_support or prior.support < min_support:
            rejected["insufficient_support"] += 1
            continue
        prior_languages = {item.language_index: item for item in prior.languages}
        stable_languages = {}
        for item in row.languages:
            old_item = prior_languages.get(item.language_index)
            if (
                old_item is not None
                and item.utility > 0
                and old_item.utility > 0
                and item.support >= per_language_min_support
                and old_item.support >= per_language_min_support
            ):
                stable_languages[item.language_index] = (
                    old_item.utility,
                    item.utility,
                )
        if not stable_languages:
            rejected["no_stable_positive_language"] += 1
            continue
        eligible.append((prior, row, stable_languages))

    previous_totals = [0.0] * previous.n_languages
    current_totals = [0.0] * current.n_languages
    for _, _, languages in eligible:
        for language, (old_utility, new_utility) in languages.items():
            previous_totals[language] += old_utility
            current_totals[language] += new_utility

    by_language: dict[int, list[tuple[float, int]]] = {}
    global_rows: list[tuple[float, int]] = []
    for prior, row, languages in eligible:
        language_scores = []
        for language, (old_utility, new_utility) in languages.items():
            score = min(
                old_utility / previous_totals[language],
                new_utility / current_totals[language],
            )
            by_language.setdefault(language, []).append((score, row.pair_key))
            language_scores.append(score)
        global_rows.append((max(language_scores), row.pair_key))
    for rows in by_language.values():
        rows.sort(key=lambda value: (-value[0], value[1]))
    global_rows.sort(key=lambda value: (-value[0], value[1]))

    languages = sorted(by_language)
    selected: list[int] = []
    selected_set: set[int] = set()
    allocations: list[tuple[int, int]] = []
    if languages and limit:
        quotient, remainder = divmod(limit, len(languages))
        for position, language in enumerate(languages):
            quota = quotient + int(position < remainder)
            admitted = 0
            for _, pair_key in by_language[language]:
                if admitted >= quota or len(selected) >= limit:
                    break
                if pair_key in selected_set:
                    continue
                selected.append(pair_key)
                selected_set.add(pair_key)
                admitted += 1
            allocations.append((language, admitted))
    backfill = 0
    for _, pair_key in global_rows:
        if len(selected) >= limit:
            break
        if pair_key in selected_set:
            continue
        selected.append(pair_key)
        selected_set.add(pair_key)
        backfill += 1
    rejected["outside_active_target"] = len(eligible) - len(selected)
    return GrowthSelection(
        pair_keys=tuple(selected),
        candidate_count=len(current.pairs),
        rejections=rejected,
        language_allocations=tuple(allocations),
        global_backfill=backfill,
    )


def select_familywise_sign_growth(
    previous: SignEvidenceWindow,
    current: SignEvidenceWindow,
    existing_keys: Iterable[int],
    *,
    alpha: float = 0.05,
) -> GrowthSelection:
    """Select pairs whose positive-utility rate clears a two-window FWER bound."""

    if (
        not math.isfinite(alpha)
        or alpha <= 0
        or alpha > 1
        or previous.n_languages != current.n_languages
    ):
        raise ValueError("invalid familywise sign-growth settings")
    old = previous.by_key()
    existing = {int(key) for key in existing_keys}
    previous_hypotheses = len(previous.pairs)
    current_hypotheses = len(current.pairs)
    rejected = {
        "already_exact": 0,
        "not_consecutive": 0,
        "nonpositive_utility": 0,
        "insufficient_familywise_sign_evidence": 0,
    }
    selected = []
    for row in current.pairs:
        prior = old.get(row.pair_key)
        if row.pair_key in existing:
            rejected["already_exact"] += 1
        elif prior is None:
            rejected["not_consecutive"] += 1
        elif row.utility <= 0 or prior.utility <= 0:
            rejected["nonpositive_utility"] += 1
        elif (
            row.lower_positive_rate(
                hypotheses=current_hypotheses, alpha=alpha
            )
            <= 0.5
            or prior.lower_positive_rate(
                hypotheses=previous_hypotheses, alpha=alpha
            )
            <= 0.5
        ):
            rejected["insufficient_familywise_sign_evidence"] += 1
        else:
            selected.append(row.pair_key)
    return GrowthSelection(
        pair_keys=tuple(selected),
        candidate_count=current_hypotheses,
        rejections=rejected,
    )


def hard_concrete_open_probability(
    probability: float,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> float:
    probability = float(probability)
    if (
        not math.isfinite(probability)
        or probability < 0
        or probability > 1
        or not math.isfinite(beta)
        or beta <= 0
        or not math.isfinite(gamma)
        or not math.isfinite(zeta)
        or not gamma < 0 < 1 < zeta
    ):
        raise ValueError("invalid hard-concrete probability or constants")
    if probability == 0:
        return 0.0
    if probability == 1:
        return 1.0
    logit = math.log(probability / (1.0 - probability))
    shifted = logit - beta * math.log(-gamma / zeta)
    return 1.0 / (1.0 + math.exp(-shifted))


def select_l0_growth(
    current: GrowthWindow,
    existing_keys: Iterable[int],
    *,
    min_support: int = 32,
    per_language_min_support: int = 8,
    open_threshold: float = 0.5,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> GrowthSelection:
    _validate_support(min_support, per_language_min_support)
    if not math.isfinite(open_threshold) or not 0 < open_threshold < 1:
        raise ValueError("growth open threshold must be between zero and one")
    existing = {int(key) for key in existing_keys}
    rejected = {
        "below_open_threshold": 0,
        "nonpositive_utility": 0,
        "insufficient_support": 0,
        "no_supported_positive_language": 0,
        "already_exact": 0,
    }
    selected = []
    for row in current.pairs:
        if row.pair_key in existing:
            rejected["already_exact"] += 1
        elif row.utility <= 0:
            rejected["nonpositive_utility"] += 1
        elif row.support < min_support:
            rejected["insufficient_support"] += 1
        elif not row.has_supported_positive_language(per_language_min_support):
            rejected["no_supported_positive_language"] += 1
        elif hard_concrete_open_probability(
            row.mean_probability, beta=beta, gamma=gamma, zeta=zeta
        ) < open_threshold:
            rejected["below_open_threshold"] += 1
        else:
            selected.append(row.pair_key)
    return GrowthSelection(
        pair_keys=tuple(selected),
        candidate_count=len(current.pairs),
        rejections=rejected,
    )


def expected_hard_concrete_l0(
    collapse: Any,
    *,
    beta: float = 2.0 / 3.0,
    gamma: float = -0.1,
    zeta: float = 1.1,
) -> torch.Tensor:
    if not beta > 0 or not gamma < 0 < 1 < zeta:
        raise ValueError("invalid hard-concrete constants")
    if int(getattr(collapse, "residual_buckets", 0)):
        logits = collapse.residual_score
    elif not int(collapse.pair_keys.numel()):
        logits = collapse.score
    else:
        raise ValueError("exact-only collapse has no proposal channel")
    if not logits.numel() or not bool(torch.isfinite(logits).all()):
        raise ValueError("proposal logits are empty or nonfinite")
    if not bool(torch.isfinite(collapse.language_bias).all()):
        raise ValueError("language bias is nonfinite")
    shift = float(beta) * math.log(-float(gamma) / float(zeta))
    combined = logits.float() + collapse.language_bias.float().mean()
    return torch.sigmoid(combined - shift).mean()
