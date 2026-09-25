"""Detached, bounded training-window vocabulary telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from starse.gradient_safety import GradientReport, stable_gradient_target


class VocabularyWindow:
    """Accumulate one logging window without retaining text or exact pair keys."""

    def __init__(self) -> None:
        self._phase: str | None = None
        self.reset()

    def begin_record(self) -> None:
        self._begin_phase("record")

    def begin_replay(self) -> None:
        self._begin_phase("replay")

    def _begin_phase(self, phase: str) -> None:
        if self._phase is not None:
            raise RuntimeError("vocabulary telemetry phase is already active")
        self._phase = phase

    def end_phase(self) -> None:
        if self._phase is None:
            raise RuntimeError("no telemetry phase is active")
        self._phase = None

    def abort_phase(self) -> None:
        self._phase = None

    def _should_record(self) -> bool:
        if self._phase is None:
            raise RuntimeError("no telemetry phase is active")
        return self._phase == "record"

    def _set_kind(self, kind: str) -> None:
        if self._kind is None:
            self._kind = kind
        elif self._kind != kind:
            raise RuntimeError("vocabulary telemetry window mixed model methods")

    def observe_online(
        self,
        *,
        base_tokens: int,
        edge_scores: torch.Tensor,
        marginals: torch.Tensor,
        map_edges: torch.Tensor,
        hash_indices: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> None:
        if not self._should_record():
            return
        self._set_kind("online")
        if base_tokens < 0:
            raise ValueError("base_tokens must be non-negative")
        if not (
            edge_scores.shape == marginals.shape == map_edges.shape == edge_mask.shape
            and edge_scores.ndim == 2
        ):
            raise ValueError("online edge telemetry tensors must have aligned rank-two shapes")
        if map_edges.dtype != torch.bool or edge_mask.dtype != torch.bool:
            raise TypeError("online MAP and mask telemetry tensors must be boolean")
        valid_count = int(edge_mask.sum().item())
        if hash_indices.ndim != 1 or hash_indices.numel() != valid_count:
            raise ValueError("hash_indices must align with valid online edges")
        valid_scores = edge_scores.detach()[edge_mask]
        valid_marginals = marginals.detach()[edge_mask]
        valid_map = map_edges.detach()[edge_mask]
        if not bool(torch.isfinite(valid_scores).all()):
            raise FloatingPointError("nonfinite online edge scores")
        if not bool(torch.isfinite(valid_marginals).all()):
            raise FloatingPointError("nonfinite online marginals")
        if valid_marginals.numel() and bool(
            ((valid_marginals < 0) | (valid_marginals > 1)).any()
        ):
            raise ValueError("online marginals must lie in [0, 1]")

        selected = int(valid_map.sum().item())
        hashes = hash_indices.detach().to(device="cpu", dtype=torch.long)
        self._base_tokens += int(base_tokens)
        self._effective_tokens += float(base_tokens - selected)
        self._eligible_edges += valid_count
        self._selected_merges += selected
        self._positive_edges += int((valid_scores > 0).sum().item())
        self._marginal_sum += float(valid_marginals.float().sum().item())
        self._touched_buckets.update(int(value) for value in hashes.tolist())
        if selected:
            self._selected_buckets.update(
                int(value) for value in hashes[valid_map.to(device="cpu")].tolist()
            )

    def observe_q15(
        self,
        *,
        base_tokens: int,
        probabilities: torch.Tensor,
        hash_indices: torch.Tensor,
        effective_tokens: torch.Tensor,
        exact_hits: torch.Tensor | None = None,
    ) -> None:
        if not self._should_record():
            return
        self._set_kind("q15")
        if base_tokens < 0:
            raise ValueError("base_tokens must be non-negative")
        if probabilities.ndim != 1 or hash_indices.ndim != 1:
            raise ValueError("Q15 probabilities and hash_indices must be one-dimensional")
        if probabilities.numel() != hash_indices.numel():
            raise ValueError("Q15 probabilities and hash_indices must align")
        if exact_hits is not None and (
            exact_hits.dtype != torch.bool
            or exact_hits.ndim != 1
            or exact_hits.numel() != probabilities.numel()
        ):
            raise ValueError("Q15 exact hits must be an aligned boolean vector")
        detached = probabilities.detach()
        if not bool(torch.isfinite(detached).all()):
            raise FloatingPointError("nonfinite Q15 probabilities")
        if detached.numel() and bool(((detached < 0) | (detached > 1)).any()):
            raise ValueError("Q15 probabilities must lie in [0, 1]")
        effective = float(effective_tokens.detach().float().item())
        if not math.isfinite(effective):
            raise FloatingPointError("nonfinite Q15 effective token count")
        if effective < 0.0 or effective > float(base_tokens):
            raise ValueError("Q15 effective token count is outside the base count")

        self._base_tokens += int(base_tokens)
        self._effective_tokens += effective
        self._eligible_edges += int(detached.numel())
        self._probability_sum += float(detached.float().sum().item())
        self._above_half += int((detached > 0.5).sum().item())
        self._exact_hits += (
            0 if exact_hits is None else int(exact_hits.detach().sum().item())
        )
        self._touched_buckets.update(
            int(value)
            for value in hash_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        )

    def snapshot(self) -> dict[str, float | int]:
        if self._phase is not None:
            raise RuntimeError("vocabulary telemetry phase is active")
        base = self._base_tokens
        edges = self._eligible_edges
        touched = len(self._touched_buckets)
        metrics: dict[str, float | int] = {
            "vocab/base_token_occurrences": base,
            "vocab/effective_token_count": self._effective_tokens,
            "vocab/token_compression_ratio": (
                self._effective_tokens / base if base else 0.0
            ),
            "vocab/eligible_edge_occurrences": edges,
            "vocab/distinct_touched_hash_buckets": touched,
            "vocab/touched_bucket_occurrence_pressure": (
                edges / touched if touched else 0.0
            ),
        }
        if self._kind == "online":
            metrics.update(
                {
                    "vocab/hard_selected_merges": self._selected_merges,
                    "vocab/hard_merge_rate": (
                        self._selected_merges / edges if edges else 0.0
                    ),
                    "vocab/positive_edge_occurrences": self._positive_edges,
                    "vocab/positive_edge_fraction": (
                        self._positive_edges / edges if edges else 0.0
                    ),
                    "vocab/exact_marginal_mean": (
                        self._marginal_sum / edges if edges else 0.0
                    ),
                    "vocab/map_selected_distinct_hash_buckets": len(
                        self._selected_buckets
                    ),
                }
            )
        elif self._kind == "q15":
            metrics.update(
                {
                    "vocab/mean_merge_probability": (
                        self._probability_sum / edges if edges else 0.0
                    ),
                    "vocab/expected_merge_mass": self._probability_sum,
                    "vocab/above_half_occurrences": self._above_half,
                    "vocab/above_half_fraction": (
                        self._above_half / edges if edges else 0.0
                    ),
                    "vocab/exact_hit_occurrences": self._exact_hits,
                    "vocab/exact_hit_rate": self._exact_hits / edges if edges else 0.0,
                }
            )
        return metrics

    def reset(self) -> None:
        if self._phase is not None:
            raise RuntimeError("vocabulary telemetry phase is active")
        self._kind: str | None = None
        self._base_tokens = 0
        self._effective_tokens = 0.0
        self._eligible_edges = 0
        self._selected_merges = 0
        self._positive_edges = 0
        self._marginal_sum = 0.0
        self._probability_sum = 0.0
        self._above_half = 0
        self._exact_hits = 0
        self._touched_buckets: set[int] = set()
        self._selected_buckets: set[int] = set()


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VocabularyTelemetrySettings:
    arm: str
    interval_steps: int
    stable_max_grad_norm: float
    merge_sample_rows: int
    merge_sample_seed: int
    corpus_manifest: Path
    corpus_manifest_sha256: str


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_vocabulary_telemetry_config(
    config: dict[str, Any], module: Any
) -> VocabularyTelemetrySettings | None:
    raw = config.get("vocabulary_telemetry")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("vocabulary telemetry settings must be an object")
    required = {
        "protocol",
        "arm",
        "interval_steps",
        "stable_max_grad_norm",
        "merge_sample_rows",
        "merge_sample_seed",
        "corpus_manifest",
        "corpus_manifest_sha256",
    }
    if set(raw) != required:
        raise ValueError("vocabulary telemetry settings have an invalid schema")
    target = stable_gradient_target(config)
    assert target is not None
    arm = str(raw["arm"])
    mode = str(getattr(module, "mode", ""))
    if arm == "q15":
        table = getattr(module, "collapse", None)
        if mode != "collapse_dynamic" or table is None:
            raise ValueError("Q15 telemetry requires collapse_dynamic")
        pair_keys = getattr(table, "pair_keys", torch.empty(0, dtype=torch.long))
        growth = config.get("vocabulary_growth")
        if pair_keys.numel() and not growth:
            raise ValueError("Q15 telemetry requires an unpromoted hash table")
        if (
            (getattr(table, "discovery", None) is not None or getattr(
                table, "diagnostic_discovery", None
            ) is not None)
            and not growth
        ):
            raise ValueError("Q15 telemetry forbids discovery observers")
        if config.get("dynamic_vocabulary") or config.get("reclaimed_vocabulary"):
            raise ValueError("Q15 telemetry forbids dynamic vocabulary settings")
        if config.get("online_tokenizer"):
            raise ValueError("Q15 telemetry forbids online tokenizer settings")
    elif arm == "q29":
        table = getattr(module, "collapse", None)
        if mode != "collapse_dynamic" or table is None:
            raise ValueError("Q29 telemetry requires collapse_dynamic")
        pair_keys = getattr(table, "pair_keys", torch.empty(0, dtype=torch.long))
        if not pair_keys.numel():
            raise ValueError("Q29 telemetry requires a promoted exact table")
        if (
            getattr(table, "discovery", None) is not None
            or getattr(table, "diagnostic_discovery", None) is not None
        ):
            raise ValueError("Q29 telemetry forbids discovery observers")
        if (
            config.get("dynamic_vocabulary")
            or config.get("reclaimed_vocabulary")
            or config.get("vocabulary_growth")
            or config.get("online_tokenizer")
        ):
            raise ValueError("Q29 telemetry requires a fixed exact vocabulary")
    elif arm == "online":
        if mode != "collapse_online" or getattr(module, "online_collapse", None) is None:
            raise ValueError("online telemetry requires collapse_online")
        if config.get("dynamic_vocabulary") or config.get("reclaimed_vocabulary"):
            raise ValueError("online telemetry forbids dynamic vocabulary settings")
    else:
        raise ValueError("vocabulary telemetry arm must be q15, q29 or online")

    interval_steps = int(raw["interval_steps"])
    merge_sample_rows = int(raw["merge_sample_rows"])
    merge_sample_seed = int(raw["merge_sample_seed"])
    smoke_identity = (
        str(config.get("run_name", ""))
        == f"multilingual-sota-expanded10k-{arm}-smoke-v1"
        and str(config.get("experiment_arm", "")) == f"{arm}-smoke-v1"
        and int(dict(config.get("training") or {}).get("max_steps", 0)) == 1
    )
    growth_smoke = bool(config.get("vocabulary_growth")) and int(
        dict(config.get("training") or {}).get("max_steps", 0)
    ) <= 3
    if interval_steps != 100 and not (
        interval_steps == 1 and (smoke_identity or growth_smoke)
    ):
        raise ValueError("vocabulary telemetry interval must be 100 steps")
    if merge_sample_rows != 4096:
        raise ValueError("vocabulary telemetry merge sample must contain 4096 rows")
    if merge_sample_seed != 20260805:
        raise ValueError("vocabulary telemetry merge sample seed must be 20260805")
    corpus_manifest = Path(str(raw["corpus_manifest"])).expanduser().resolve()
    if not corpus_manifest.is_file():
        raise ValueError("vocabulary telemetry corpus manifest is missing")
    expected_digest = str(raw["corpus_manifest_sha256"])
    if _SHA256.fullmatch(expected_digest) is None:
        raise ValueError("vocabulary telemetry manifest hash is malformed")
    if _sha256(corpus_manifest) != expected_digest:
        raise ValueError("vocabulary telemetry manifest hash mismatch")
    return VocabularyTelemetrySettings(
        arm=arm,
        interval_steps=interval_steps,
        stable_max_grad_norm=target,
        merge_sample_rows=merge_sample_rows,
        merge_sample_seed=merge_sample_seed,
        corpus_manifest=corpus_manifest,
        corpus_manifest_sha256=expected_digest,
    )


def _tensor_statistics(
    values: torch.Tensor, *, prefix: str
) -> dict[str, float]:
    detached = values.detach().float().reshape(-1)
    if not detached.numel():
        raise ValueError(f"{prefix} statistics require at least one value")
    if not bool(torch.isfinite(detached).all()):
        raise FloatingPointError(f"nonfinite {prefix} values")
    return {
        f"{prefix}_mean": float(detached.mean().item()),
        f"{prefix}_std": float(detached.std(unbiased=False).item()),
        f"{prefix}_min": float(detached.min().item()),
        f"{prefix}_median": float(torch.quantile(detached, 0.5).item()),
        f"{prefix}_max": float(detached.max().item()),
        f"{prefix}_positive_fraction": float((detached > 0).float().mean().item()),
    }


class VocabularyTelemetryCoordinator:
    """Combine window, parameter and gradient metrics into durable JSONL."""

    def __init__(
        self,
        *,
        module: Any,
        window: VocabularyWindow,
        gradient_callback: Any,
        output_path: Path,
        arm: str,
        config_sha256: str,
        corpus_manifest_sha256: str,
        merge_sample_rows: int,
        merge_sample_seed: int,
        growth_callback: Any | None = None,
    ) -> None:
        if arm not in {"q15", "q29", "online"}:
            raise ValueError("telemetry arm must be q15, q29 or online")
        if _SHA256.fullmatch(config_sha256) is None:
            raise ValueError("config_sha256 must be lowercase hexadecimal")
        if _SHA256.fullmatch(corpus_manifest_sha256) is None:
            raise ValueError("corpus_manifest_sha256 must be lowercase hexadecimal")
        if merge_sample_rows <= 0:
            raise ValueError("merge_sample_rows must be positive")
        mode = str(getattr(module, "mode", ""))
        if arm == "online":
            if mode != "collapse_online" or getattr(module, "online_collapse", None) is None:
                raise ValueError("online telemetry requires collapse_online")
            table = module.online_collapse
            scores = table.score_hash
        else:
            if mode not in {"collapse", "collapse_shared", "collapse_dynamic"}:
                raise ValueError("Q15 telemetry requires an uncompiled collapse table")
            table = getattr(module, "collapse", None)
            if table is None:
                raise ValueError("Q15 telemetry requires a collapse table")
            scores = table.score
        merged = table.merged
        if merged.ndim != 2 or not merged.shape[0]:
            raise ValueError("telemetry merge table must be a non-empty matrix")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(merge_sample_seed))
        sample_count = min(int(merge_sample_rows), int(merged.shape[0]))
        self.sampled_merge_rows = tuple(
            int(value)
            for value in torch.randperm(
                int(merged.shape[0]), generator=generator, device="cpu"
            )[:sample_count].tolist()
        )
        self.module = module
        self.table = table
        self.scores = scores
        self.merge_sample_rows = int(merge_sample_rows)
        self.merge_sample_seed = int(merge_sample_seed)
        self.growth_callback = growth_callback
        self.window = window
        self.gradient_callback = gradient_callback
        self.output_path = Path(output_path)
        self.arm = arm
        self.config_sha256 = config_sha256
        self.corpus_manifest_sha256 = corpus_manifest_sha256

    def _parameter_metrics(self) -> dict[str, float | int]:
        embedding = self.module.embedding.weight
        exact_rows = int(getattr(self.table, "pair_keys", torch.empty(0)).numel())
        residual_rows = int(getattr(self.table, "residual_buckets", 0))
        if residual_rows:
            merged = self.table.residual_merged
            scores = self.table.residual_score
        else:
            merged = self.table.merged
            scores = self.table.score if self.arm == "q15" else self.scores
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.merge_sample_seed)
        sample_index = torch.randperm(
            int(merged.shape[0]), generator=generator, device="cpu"
        )[: min(self.merge_sample_rows, int(merged.shape[0]))].to(merged.device)
        sampled_norms = merged.index_select(0, sample_index).detach().float().norm(dim=1)
        metrics: dict[str, float | int] = {
            "vocab/base_rows": int(embedding.shape[0]),
            "vocab/merge_capacity_rows": int(self.table.buckets),
            "vocab/exact_rows": exact_rows,
            "vocab/residual_rows": residual_rows,
            "vocab/total_learned_pair_rows": exact_rows + (
                residual_rows if residual_rows else int(self.table.buckets)
            ),
        }
        metrics.update(
            _tensor_statistics(scores, prefix="vocab/global_score")
        )
        metrics.update(
            _tensor_statistics(
                self.table.language_bias, prefix="vocab/language_bias"
            )
        )
        merge_statistics = _tensor_statistics(
            sampled_norms, prefix="vocab/merge_vector_norm"
        )
        merge_statistics.pop("vocab/merge_vector_norm_positive_fraction")
        metrics.update(merge_statistics)
        if self.growth_callback is not None:
            metrics.update(
                {
                    "vocab/last_growth_additions": int(
                        self.growth_callback.last_additions
                    ),
                    "vocab/cumulative_growth_additions": int(
                        self.growth_callback.cumulative_additions
                    ),
                    "vocab/selector_candidate_pairs": int(
                        self.growth_callback.selector_candidates
                    ),
                    "vocab/selector_eligible_pairs": int(
                        self.growth_callback.selector_eligible
                    ),
                }
            )
        return metrics

    def _gradient_metrics(self) -> dict[str, float | int]:
        report = getattr(self.gradient_callback, "last_report", None)
        if not isinstance(report, GradientReport):
            raise RuntimeError("vocabulary telemetry has no stable gradient report")
        metrics: dict[str, float | int] = {
            "gradient/stable_total_norm": report.total_norm,
            "gradient/clip_coefficient": report.clip_coefficient,
            "gradient/max_abs": report.max_abs,
            "gradient/parameter_count": report.parameter_count,
        }
        metrics.update(
            {
                f"gradient/group/{name}_norm": value
                for name, value in sorted(report.group_norms.items())
            }
        )
        return metrics

    def _existing_bytes(self, *, step: int) -> bytes:
        if not self.output_path.exists():
            return b""
        existing = self.output_path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            raise ValueError("vocabulary metrics JSONL has a partial final line")
        previous_step = 0
        for raw_line in existing.splitlines():
            record = json.loads(raw_line)
            if (
                record.get("protocol") != "vocabulary-training-metrics-v1"
                or record.get("arm") != self.arm
                or record.get("config_sha256") != self.config_sha256
                or record.get("corpus_manifest_sha256")
                != self.corpus_manifest_sha256
            ):
                raise ValueError("existing vocabulary metrics identity differs")
            current = int(record.get("optimizer_step", -1))
            if current <= previous_step:
                raise ValueError("existing vocabulary metric steps are not increasing")
            previous_step = current
        if step <= previous_step:
            raise ValueError("vocabulary metric step is not newer than existing records")
        return existing

    def _replace_jsonl(self, record: dict[str, object]) -> None:
        step = int(record["optimizer_step"])
        existing = self._existing_bytes(step=step)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"stale vocabulary metrics temporary file: {temporary}")
        line = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        with temporary.open("xb") as handle:
            handle.write(existing)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.output_path)
        directory = os.open(self.output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def flush(self, step: int) -> dict[str, float | int]:
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("optimizer step must be a positive integer")
        metrics = self.window.snapshot()
        metrics.update(self._parameter_metrics())
        metrics.update(self._gradient_metrics())
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in metrics.values()
        ):
            raise FloatingPointError("nonfinite vocabulary telemetry metric")
        record: dict[str, object] = {
            "protocol": "vocabulary-training-metrics-v1",
            "arm": self.arm,
            "optimizer_step": step,
            "config_sha256": self.config_sha256,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "metrics": metrics,
        }
        self._replace_jsonl(record)
        self.window.reset()
        return metrics


class VocabularyLoggingCallback(TrainerCallback):
    """Merge one durable vocabulary record into the Trainer log dictionary."""

    def __init__(
        self,
        coordinator: VocabularyTelemetryCoordinator,
        *,
        interval_steps: int,
    ) -> None:
        if interval_steps <= 0:
            raise ValueError("vocabulary logging interval must be positive")
        self.coordinator = coordinator
        self.interval_steps = int(interval_steps)
        self.last_flushed_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        step = int(state.global_step)
        if (
            logs is None
            or step <= 0
            or step % self.interval_steps
            or step == self.last_flushed_step
        ):
            return
        metrics = self.coordinator.flush(step)
        logs.update(metrics)
        self.last_flushed_step = step


def install_vocabulary_logging_callback(
    trainer: Any, callback: VocabularyLoggingCallback
) -> None:
    callbacks = trainer.callback_handler.callbacks
    if any(isinstance(current, VocabularyLoggingCallback) for current in callbacks):
        raise RuntimeError("vocabulary logging callback is already installed")
    callbacks.insert(0, callback)
