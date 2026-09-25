"""Overflow-safe gradient validation, measurement and clipping."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import TrainerCallback


@dataclass(frozen=True)
class GradientReport:
    """Detached scalar evidence from one pre-optimizer gradient pass."""

    total_norm: float
    clip_coefficient: float
    max_abs: float
    group_norms: dict[str, float]
    parameter_count: int


def stable_gradient_target(config: dict[str, Any]) -> float | None:
    """Validate the opt-in training contract and return its fixed norm target."""

    raw = config.get("vocabulary_telemetry")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("vocabulary telemetry settings must be an object")
    if raw.get("protocol") != "vocabulary-telemetry-v1":
        raise ValueError("vocabulary telemetry has an unknown protocol")
    target = raw.get("stable_max_grad_norm")
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise ValueError("vocabulary telemetry requires stable_max_grad_norm=1.0")
    if not math.isfinite(float(target)) or float(target) != 1.0:
        raise ValueError("vocabulary telemetry requires stable_max_grad_norm=1.0")
    return 1.0


def parameter_group_name(name: str) -> str:
    """Map SentenceTransformer parameter names to stable telemetry groups."""

    if "embedding.weight" in name:
        return "base_embedding"
    if name.endswith("collapse.merged") or name.endswith("online_collapse.merged"):
        return "merge_vectors"
    if name.endswith("collapse.score") or name.endswith("online_collapse.score_hash"):
        return "global_scores"
    if name.endswith("collapse.language_bias") or name.endswith(
        "online_collapse.language_bias"
    ):
        return "language_bias"
    if "online_collapse." in name and name.rsplit(".", 1)[-1] in {
        "left_projection",
        "right_projection",
        "interaction_weight",
    }:
        return "amortized_scorer"
    return "other"


def _scaled_squared_norm(gradient: torch.Tensor, *, name: str) -> tuple[float, float]:
    if gradient.is_sparse:
        raise TypeError(f"sparse gradient is unsupported by AdamW: {name}")
    detached = gradient.detach()
    maximum = detached.abs().max()
    if not bool(torch.isfinite(maximum)):
        raise FloatingPointError(f"nonfinite gradient in {name}")
    max_abs = float(maximum.item())
    if max_abs == 0.0:
        return 0.0, 0.0
    scaled_squared = (
        (detached.float() / maximum.float()).square().sum(dtype=torch.float64)
    )
    squared_norm = maximum.double().square() * scaled_squared
    if not bool(torch.isfinite(squared_norm)):
        raise FloatingPointError(f"nonfinite stable gradient norm in {name}")
    return float(squared_norm.item()), max_abs


def stable_clip_gradients(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    max_norm: float,
    group_name: Callable[[str], str],
) -> GradientReport:
    """Validate and clip dense gradients without float32 norm overflow."""

    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be positive and finite")

    gradients: list[tuple[str, torch.Tensor]] = []
    squared_norms: list[float] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    maximum = 0.0
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        squared_norm, max_abs = _scaled_squared_norm(gradient, name=name)
        gradients.append((name, gradient))
        squared_norms.append(squared_norm)
        grouped[group_name(name)].append(squared_norm)
        maximum = max(maximum, max_abs)
    if not gradients:
        raise RuntimeError("optimizer step has no gradients")

    total_norm = math.sqrt(math.fsum(squared_norms))
    if not math.isfinite(total_norm):
        raise FloatingPointError("nonfinite stable total gradient norm")
    coefficient = 1.0 if total_norm == 0.0 else min(1.0, max_norm / total_norm)
    with torch.no_grad():
        if coefficient < 1.0:
            for _name, gradient in gradients:
                gradient.mul_(coefficient)
        for name, gradient in gradients:
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f"nonfinite post-clip gradient in {name}")

    return GradientReport(
        total_norm=total_norm,
        clip_coefficient=coefficient,
        max_abs=maximum,
        group_norms={
            name: math.sqrt(math.fsum(values))
            for name, values in sorted(grouped.items())
        },
        parameter_count=len(gradients),
    )


class StableGradientCallback(TrainerCallback):
    """Close online cache auditing, then safely clip either training arm."""

    def __init__(self, module: torch.nn.Module, *, max_norm: float) -> None:
        if not math.isfinite(max_norm) or max_norm <= 0.0:
            raise ValueError("max_norm must be positive and finite")
        self.module = module
        self.max_norm = float(max_norm)
        self.completed_cache_audits = 0
        self.completed_steps = 0
        self.last_report: GradientReport | None = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        del args, state, control, kwargs
        if model is None:
            raise RuntimeError("stable gradient callback requires the trainer model")
        if getattr(self.module, "mode", "") == "collapse_online":
            self.module.online_collapse.assert_cache_audit_empty()
            self.completed_cache_audits += 1
        report = stable_clip_gradients(
            model.named_parameters(),
            max_norm=self.max_norm,
            group_name=parameter_group_name,
        )
        self.last_report = report
        self.completed_steps += 1
