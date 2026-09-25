"""Periodic, artifact-backed MTEB evaluation during StaRSE training."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _task_name(task: Any, index: int = 0) -> str:
    direct = getattr(task, "task_name", None)
    if isinstance(direct, str) and direct:
        return direct
    metadata = getattr(task, "metadata", None)
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"task_{index}"


def _normalize_mteb_results(results: Any) -> dict[str, Any]:
    if isinstance(results, dict):
        return results
    if not isinstance(results, list):
        results = [results]
    normalized: dict[str, Any] = {}
    for index, item in enumerate(results):
        name = _task_name(item, index)
        if isinstance(item, dict):
            normalized[name] = item
        elif hasattr(item, "to_dict"):
            normalized[name] = item.to_dict()
        else:
            normalized[name] = {"result": str(item)}
    return normalized


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def task_main_score(result: Any) -> float | None:
    """Extract the canonical test-split main score from an MTEB task result."""

    if isinstance(result, dict) and len(result) == 1 and "scores" not in result:
        return task_main_score(next(iter(result.values())))
    if isinstance(result, dict):
        scores = result.get("scores")
        if isinstance(scores, dict):
            for split in ("test", "dev", "validation"):
                if split not in scores:
                    continue
                items = scores[split]
                if not isinstance(items, list):
                    items = [items]
                values = [
                    score
                    for item in items
                    if isinstance(item, dict)
                    and (score := _finite_score(item.get("main_score"))) is not None
                ]
                if values:
                    return sum(values) / len(values)
        direct = _finite_score(result.get("main_score"))
        if direct is not None:
            return direct
        for value in result.values():
            nested = task_main_score(value)
            if nested is not None:
                return nested
    elif isinstance(result, list):
        for value in result:
            nested = task_main_score(value)
            if nested is not None:
                return nested
    return None


def _unwrap_model(model: Any) -> Any:
    seen: set[int] = set()
    while hasattr(model, "module") and id(model) not in seen:
        seen.add(id(model))
        model = model.module
    return model


@dataclass(frozen=True)
class PeriodicMTEBSettings:
    task: str
    eval_steps: int
    evaluate_at_start: bool
    batch_size: int
    max_length: int
    fail_on_error: bool
    verbosity: int
    output_dir: Path

    @classmethod
    def from_config(
        cls,
        raw: dict[str, Any],
        *,
        training_settings: dict[str, Any],
        output_dir: Path,
    ) -> PeriodicMTEBSettings:
        task = str(raw.get("task", "RuBQRetrieval"))
        if task != "RuBQRetrieval":
            raise ValueError("periodic retrieval evaluation is pinned to RuBQRetrieval")
        eval_steps = int(raw.get("eval_steps", 0))
        if eval_steps <= 0:
            raise ValueError("evaluation.rubq_retrieval.eval_steps must be positive")
        if str(training_settings.get("save_strategy", "steps")) != "steps":
            raise ValueError("periodic RuBQRetrieval evaluation requires step checkpoints")
        save_steps = int(training_settings.get("save_steps", 0))
        if save_steps <= 0 or eval_steps % save_steps:
            raise ValueError(
                "RuBQRetrieval eval_steps must be a multiple of save_steps so every "
                "metric is bound to a reloadable checkpoint"
            )
        batch_size = int(raw.get("batch_size", 1024))
        max_length = int(raw.get("max_length", 512))
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("RuBQRetrieval batch_size and max_length must be positive")
        return cls(
            task=task,
            eval_steps=eval_steps,
            evaluate_at_start=bool(raw.get("evaluate_at_start", True)),
            batch_size=batch_size,
            max_length=max_length,
            fail_on_error=bool(raw.get("fail_on_error", True)),
            verbosity=int(raw.get("verbosity", 0)),
            output_dir=output_dir / "evaluation" / "rubq_retrieval",
        )


class PeriodicMTEBEvaluationCallback(TrainerCallback):
    """Run RuBQRetrieval at initialization and after selected checkpoint saves."""

    def __init__(self, settings: PeriodicMTEBSettings) -> None:
        self.settings = settings
        self._evaluated_steps: set[int] = set()
        self._task: Any | None = None

    def _resolve_task(self) -> Any:
        if self._task is None:
            import mteb

            tasks = list(mteb.get_tasks(tasks=[self.settings.task]))
            by_name = {_task_name(task, index): task for index, task in enumerate(tasks)}
            if self.settings.task not in by_name:
                raise ValueError(
                    f"MTEB task {self.settings.task!r} was not found; "
                    f"resolved={sorted(by_name)}"
                )
            self._task = by_name[self.settings.task]
        return self._task

    def _log_comet(self, summary: dict[str, Any]) -> None:
        try:
            import comet_ml

            experiment = comet_ml.get_global_experiment()
            if experiment is None:
                return
            step = int(summary["step"])
            experiment.log_metric(
                "eval_rubq_retrieval_ok",
                1.0 if summary.get("status") == "ok" else 0.0,
                step=step,
            )
            score = _finite_score(summary.get("main_score"))
            if score is not None:
                experiment.log_metric(
                    "eval_rubq_retrieval_main_score", score, step=step
                )
        except Exception:
            # The JSON artifact remains authoritative even if the tracker is down.
            pass

    def evaluate(self, model: Any, *, step: int, reason: str) -> dict[str, Any]:
        if step in self._evaluated_steps:
            summary_path = self.settings.output_dir / f"step-{step:06d}" / "summary.json"
            return json.loads(summary_path.read_text(encoding="utf-8"))

        step_dir = self.settings.output_dir / f"step-{step:06d}"
        summary_path = step_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._evaluated_steps.add(step)
            self._log_comet(summary)
            return summary

        started = time.perf_counter()
        if model is None:
            raise TypeError("Trainer callback did not provide a model for MTEB evaluation")
        eval_model = _unwrap_model(model)
        was_training = bool(eval_model.training)
        tokenizer = None
        original_truncation = None
        try:
            import mteb

            module = eval_model[0]
            tokenizer = getattr(module, "tokenizer", None)
            if tokenizer is None or not hasattr(tokenizer, "enable_truncation"):
                raise TypeError("RuBQRetrieval evaluation requires a truncatable tokenizer")
            original_truncation = tokenizer.truncation
            tokenizer.enable_truncation(max_length=self.settings.max_length)
            eval_model.eval()
            task = self._resolve_task()
            runner = mteb.MTEB(tasks=[task])
            task_results = runner.run(
                model=eval_model,
                output_folder=str(step_dir / "mteb_results"),
                overwrite_results=True,
                verbosity=self.settings.verbosity,
                encode_kwargs={
                    "batch_size": self.settings.batch_size,
                    "normalize_embeddings": True,
                },
                co2_tracker=False,
                raise_error=True,
            )
            normalized = _normalize_mteb_results(task_results)
            if self.settings.task not in normalized and len(normalized) == 1:
                normalized = {self.settings.task: next(iter(normalized.values()))}
            score = task_main_score(normalized.get(self.settings.task))
            if score is None:
                raise RuntimeError("RuBQRetrieval result has no finite main_score")
            _write_json(step_dir / "raw_result.json", normalized)
            summary = {
                "status": "ok",
                "step": int(step),
                "reason": reason,
                "task": self.settings.task,
                "metric": "main_score",
                "main_score": score,
                "batch_size": self.settings.batch_size,
                "max_length": self.settings.max_length,
                "normalize_embeddings": True,
                "wall_seconds": time.perf_counter() - started,
                "checkpoint": (
                    None
                    if step == 0
                    else str(self.settings.output_dir.parents[1] / f"checkpoint-{step}")
                ),
            }
        except Exception as exc:
            summary = {
                "status": "error",
                "step": int(step),
                "reason": reason,
                "task": self.settings.task,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
            }
            _write_json(summary_path, summary)
            self._log_comet(summary)
            print(json.dumps({"event": "rubq_evaluation", **summary}), flush=True)
            if self.settings.fail_on_error:
                raise
            return summary
        finally:
            if tokenizer is not None:
                if original_truncation is None:
                    tokenizer.no_truncation()
                else:
                    tokenizer.enable_truncation(**original_truncation)
            if was_training:
                eval_model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _write_json(summary_path, summary)
        self._evaluated_steps.add(step)
        self._log_comet(summary)
        print(json.dumps({"event": "rubq_evaluation", **summary}), flush=True)
        return summary

    def on_train_begin(self, args, state, control, **kwargs):
        if (
            self.settings.evaluate_at_start
            and int(getattr(state, "global_step", 0) or 0) == 0
            and bool(getattr(state, "is_world_process_zero", True))
        ):
            self.evaluate(kwargs.get("model"), step=0, reason="train_begin")
        return control

    def on_save(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if (
            step > 0
            and step % self.settings.eval_steps == 0
            and bool(getattr(state, "is_world_process_zero", True))
        ):
            self.evaluate(kwargs.get("model"), step=step, reason="checkpoint_save")
        return control
