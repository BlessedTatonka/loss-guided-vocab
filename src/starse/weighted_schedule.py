"""Deterministic low-discrepancy scheduling for weighted data streams."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


class SmoothWeightedRoundRobin:
    """Emit items at configured frequencies without multinomial burstiness."""

    _STATE_VERSION = 1

    def __init__(
        self,
        items: Sequence[str],
        weights: Sequence[float],
        *,
        seed: int,
    ) -> None:
        self.items = tuple(str(item) for item in items)
        raw_weights = tuple(float(weight) for weight in weights)
        if not self.items:
            raise ValueError("a schedule needs at least one item")
        if len(self.items) != len(raw_weights):
            raise ValueError("items and weights must have the same length")
        if len(set(self.items)) != len(self.items):
            raise ValueError("schedule items must be unique")
        if not all(math.isfinite(weight) for weight in raw_weights):
            raise ValueError("schedule weights must be finite")
        if any(weight < 0 for weight in raw_weights):
            raise ValueError("schedule weights must be non-negative")
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("at least one schedule weight must be positive")

        self.weights = tuple(weight / total for weight in raw_weights)
        tie_order = list(range(len(self.items)))
        random.Random(int(seed)).shuffle(tie_order)
        self.tie_ranks = tuple(tie_order.index(index) for index in range(len(self.items)))
        self.credits = [0.0] * len(self.items)
        self.counts = [0] * len(self.items)
        self.steps = 0

    def next_item(self) -> str:
        for index, weight in enumerate(self.weights):
            self.credits[index] += weight
        selected = max(
            range(len(self.items)),
            key=lambda index: (self.credits[index], -self.tie_ranks[index]),
        )
        self.credits[selected] -= 1.0
        self.counts[selected] += 1
        self.steps += 1
        return self.items[selected]

    def advance(self, steps: int) -> None:
        if int(steps) < 0:
            raise ValueError("advance steps must be non-negative")
        for _ in range(int(steps)):
            self.next_item()

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self._STATE_VERSION,
            "items": list(self.items),
            "weights": list(self.weights),
            "tie_ranks": list(self.tie_ranks),
            "credits": list(self.credits),
            "counts": list(self.counts),
            "steps": self.steps,
        }

    def load_state_dict(self, raw: Mapping[str, Any]) -> None:
        if int(raw.get("version", -1)) != self._STATE_VERSION:
            raise ValueError("unsupported schedule state version")
        if tuple(raw.get("items", ())) != self.items:
            raise ValueError("schedule state items do not match")
        incoming_weights = tuple(float(value) for value in raw.get("weights", ()))
        if incoming_weights != self.weights:
            raise ValueError("schedule state weights do not match")
        incoming_ties = tuple(int(value) for value in raw.get("tie_ranks", ()))
        if incoming_ties != self.tie_ranks:
            raise ValueError("schedule state tie ranks do not match")

        credits = [float(value) for value in raw.get("credits", ())]
        counts = [int(value) for value in raw.get("counts", ())]
        steps = int(raw.get("steps", -1))
        if len(credits) != len(self.items) or not all(map(math.isfinite, credits)):
            raise ValueError("schedule state credits are invalid")
        if len(counts) != len(self.items) or any(value < 0 for value in counts):
            raise ValueError("schedule state counts are invalid")
        if steps < 0 or sum(counts) != steps:
            raise ValueError("schedule state step count is invalid")
        self.credits = credits
        self.counts = counts
        self.steps = steps
