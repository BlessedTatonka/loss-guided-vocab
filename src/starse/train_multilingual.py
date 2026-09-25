"""Train a multilingual StaRSE model from infinite randomized Parquet streams."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import TrainerCallback

from starse.checkpoint_precision import (
    BFloat16CheckpointCallback,
    rewrite_safetensors_bf16,
)
from starse.gradient_safety import StableGradientCallback, stable_gradient_target
from starse.streaming import build_streaming_dataset
from starse.vocabulary_growth import (
    GrowthWindow,
    SignEvidenceWindow,
    expected_hard_concrete_l0,
    select_familywise_sign_growth,
    select_l0_growth,
    select_language_balanced_growth,
    select_stable_growth,
)
from starse.vocabulary_telemetry import (
    VocabularyLoggingCallback,
    VocabularyTelemetryCoordinator,
    VocabularyWindow,
    canonical_config_sha256,
    install_vocabulary_logging_callback,
    validate_vocabulary_telemetry_config,
)


@dataclass(frozen=True)
class DynamicResumeMetadata:
    global_step: int
    discovery_sidecar: Path | None
    post_promotion: bool


def _disable_empty_trainable_parameters(model: torch.nn.Module) -> tuple[str, ...]:
    """Keep zero-sized compatibility tensors out of resumed optimizer groups."""

    disabled = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.numel() == 0:
            parameter.requires_grad_(False)
            disabled.append(name)
    return tuple(disabled)


def _resolve_dynamic_promotion_step(
    settings: dict[str, Any], *, max_steps: int
) -> int:
    """Resolve either an in-training promotion or a discovery-only horizon."""

    if bool(settings.get("promote_during_training", True)):
        promotion_step = int(settings.get("promotion_step", 1000))
        if not 0 < promotion_step <= max_steps:
            raise ValueError(
                "dynamic promotion_step must be between zero and max_steps inclusive"
            )
        return promotion_step
    if "promotion_step" in settings:
        raise ValueError(
            "continuous dynamic discovery may not set promotion_step"
        )
    return max_steps + 1


def _dynamic_resume_preflight(
    checkpoint: Path, *, promotion_step: int
) -> DynamicResumeMetadata:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"dynamic resume is missing trainer state: {state_path}")
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(raw.get("global_step", -1))
    if global_step < 0:
        raise ValueError("dynamic resume trainer state has an invalid global step")
    post_promotion = global_step >= int(promotion_step)
    sidecar = checkpoint / "dynamic_vocabulary_discovery.pt"
    if not post_promotion and not sidecar.is_file():
        raise ValueError(
            f"pre-promotion dynamic resume requires discovery sidecar: {sidecar}"
        )
    return DynamicResumeMetadata(
        global_step=global_step,
        discovery_sidecar=None if post_promotion else sidecar,
        post_promotion=post_promotion,
    )


def _checkpoint_global_step(checkpoint: Path, *, label: str) -> int:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"{label} resume is missing trainer state: {state_path}")
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    value = raw.get("global_step")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} resume trainer state has an invalid global step")
    global_step = int(value)
    if global_step < 0 or float(value) != global_step:
        raise ValueError(f"{label} resume trainer state has an invalid global step")
    return global_step


def _cast_bf16_resume_model_to_master_float(
    model: torch.nn.Module, checkpoint: Path
) -> None:
    """Restore quantized BF16 values as FP32 master weights for autocast training."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    precision_path = checkpoint / "checkpoint_precision.json"
    model_path = checkpoint / "model.safetensors"
    if not precision_path.is_file():
        raise FileNotFoundError(f"BF16 resume precision manifest is missing: {precision_path}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    manifest = json.loads(precision_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != "bf16-checkpoint-precision-v1"
        or manifest.get("stored_floating_dtype") != "bfloat16"
        or manifest.get("model_sha256") != _sha256(model_path)
    ):
        raise ValueError("BF16 resume precision identity differs")
    model.float()
    checked = 0
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        checked += 1
        if parameter.dtype != torch.float32 or not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"BF16 resume master parameter is invalid: {name}")
    if not checked:
        raise ValueError("BF16 resume model has no floating master parameters")


def _resolve_resume_scheduler_start_batch(
    streaming_settings: dict[str, Any],
    training_settings: dict[str, Any],
    *,
    global_step: int,
) -> int:
    if global_step < 0:
        raise ValueError("resume global step must be non-negative")
    accumulated_batches = global_step * int(
        training_settings.get("gradient_accumulation_steps", 1)
    )
    configured_start = int(streaming_settings.get("scheduler_start_batch", 0))
    if configured_start not in (0, accumulated_batches):
        raise ValueError(
            "configured scheduler_start_batch disagrees with resume global step"
        )
    return accumulated_batches


def _resolve_nondynamic_resume_scheduler_start_batch(
    config: dict[str, Any],
    streaming_settings: dict[str, Any],
    training_settings: dict[str, Any],
    checkpoint: Path,
) -> int | None:
    """Advance deterministic streams for online and telemetry-only resumes."""

    online = bool(config.get("online_tokenizer"))
    telemetry = dict(config.get("vocabulary_telemetry") or {})
    if not online and not telemetry:
        return None
    label = "online tokenizer" if online else f"{telemetry.get('arm', 'vocabulary')} telemetry"
    global_step = _checkpoint_global_step(Path(checkpoint), label=label)
    return _resolve_resume_scheduler_start_batch(
        streaming_settings,
        training_settings,
        global_step=global_step,
    )


def _validate_online_tokenizer_config(
    config: dict[str, Any], module: Any
) -> None:
    mode = str(getattr(module, "mode", ""))
    raw = config.get("online_tokenizer")
    if mode != "collapse_online":
        if raw:
            raise ValueError("online_tokenizer config requires collapse_online model")
        return
    if config.get("dynamic_vocabulary"):
        raise ValueError("collapse_online may not configure dynamic_vocabulary")
    if config.get("reclaimed_vocabulary"):
        raise ValueError("collapse_online may not configure reclaimed_vocabulary")
    expected = {
        "protocol": "structured-st-v1",
        "rank": int(module.online_collapse.rank),
        "matching": "exact_monomer_dimer",
        "tie_break": "keep_boundary",
    }
    if not isinstance(raw, dict) or raw != expected:
        raise ValueError(
            f"online_tokenizer must match the registered structured contract: {expected}"
        )
    loss = dict(config.get("loss") or {})
    if (
        loss.get("type") != "cached_multiple_negatives"
        or tuple(loss.get("directions") or ())
        != ("query_to_doc", "doc_to_query")
        or loss.get("partition_mode") != "per_direction"
        or int(loss.get("mini_batch_size", 0)) <= 0
    ):
        raise ValueError("collapse_online requires the strict cached symmetric loss")
    training = dict(config.get("training") or {})
    if int(training.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("collapse_online requires gradient accumulation of one")


def _resolve_vocabulary_growth_settings(
    config: dict[str, Any], module: Any
) -> dict[str, Any] | None:
    """Validate the frozen adaptive-growth contract before touching training state."""

    raw = config.get("vocabulary_growth")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("vocabulary_growth must be an object")
    required = {
        "protocol", "policy", "interval_steps", "residual_buckets",
        "candidate_capacity", "top_per_forward", "min_support",
        "per_language_min_support", "max_parameter_bytes", "l0",
    }
    optional = {"single_transition"}
    if not required.issubset(raw) or set(raw) - required - optional:
        raise ValueError("vocabulary_growth has an invalid schema")
    if raw["protocol"] != "adaptive-exact-growth-v1":
        raise ValueError("unknown vocabulary growth protocol")
    policy = str(raw["policy"])
    if policy not in {"stable", "l0"}:
        raise ValueError("vocabulary growth policy must be stable or l0")
    if str(getattr(module, "mode", "")) != "collapse_dynamic":
        raise ValueError("adaptive vocabulary growth requires collapse_dynamic")
    if config.get("dynamic_vocabulary"):
        raise ValueError("adaptive growth may not mix with legacy dynamic_vocabulary")
    if config.get("reclaimed_vocabulary") or config.get("online_tokenizer"):
        raise ValueError("adaptive growth is incompatible with reclaimed or online vocabulary")
    settings = dict(raw)
    if "single_transition" in settings and not isinstance(
        settings["single_transition"], bool
    ):
        raise ValueError("vocabulary growth single_transition must be boolean")
    for name in (
        "interval_steps", "residual_buckets", "candidate_capacity",
        "top_per_forward", "min_support", "per_language_min_support",
        "max_parameter_bytes",
    ):
        value = settings[name]
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"vocabulary growth {name} must be positive")
        settings[name] = int(value)
    training = dict(config.get("training") or {})
    save_steps = int(training.get("save_steps", 0))
    if save_steps <= 0 or save_steps % settings["interval_steps"]:
        raise ValueError("vocabulary growth interval must divide save_steps")
    if settings["residual_buckets"] != 65536:
        raise ValueError("vocabulary growth residual capacity must be 65536")
    l0 = settings["l0"]
    expected_l0 = {
        "weight": 1e-5,
        "beta": 2.0 / 3.0,
        "gamma": -0.1,
        "zeta": 1.1,
        "open_threshold": 0.5,
    }
    if not isinstance(l0, dict) or set(l0) != set(expected_l0):
        raise ValueError("vocabulary growth L0 settings have an invalid schema")
    for name, expected in expected_l0.items():
        if not math.isclose(float(l0[name]), expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"vocabulary growth L0 {name} differs from frozen contract")
    settings["l0"] = {name: float(value) for name, value in l0.items()}
    return settings


def _resolve_recursive_vocabulary_settings(
    config: dict[str, Any], module: Any, *, max_steps: int | None = None
) -> dict[str, Any] | None:
    """Validate the recursive-cascade lifecycle before constructing Trainer."""

    raw = config.get("recursive_vocabulary")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("recursive_vocabulary has an invalid schema")
    protocol = raw.get("protocol")
    if protocol not in {
        "recursive-cascade-v1",
        "recursive-cascade-bidirectional-v1",
        "recursive-cascade-familywise-sign-v1",
        "recursive-cascade-balanced-warmup-v1",
        "recursive-cascade-monotone-warmup-v1",
    }:
        raise ValueError("unknown recursive vocabulary protocol")
    required = {
        "protocol", "interval_steps", "residual_buckets",
        "max_parameter_bytes", "max_span_length",
    }
    if protocol == "recursive-cascade-familywise-sign-v1":
        required.add("evidence_alpha")
    else:
        required.update({"min_support", "per_language_min_support"})
    if protocol == "recursive-cascade-balanced-warmup-v1":
        required.update({"warmup_steps", "target_active_rows", "initialization"})
    if protocol == "recursive-cascade-monotone-warmup-v1":
        required.update({"warmup_steps", "target_learned_rows", "initialization"})
    if set(raw) != required:
        raise ValueError("recursive_vocabulary has an invalid schema")
    if str(getattr(module, "mode", "")) != "collapse_recursive":
        raise ValueError("recursive vocabulary requires collapse_recursive mode")
    if any(
        config.get(name)
        for name in (
            "dynamic_vocabulary", "reclaimed_vocabulary", "online_tokenizer",
            "vocabulary_growth",
        )
    ):
        raise ValueError("recursive vocabulary cannot mix with another vocabulary mode")
    settings = dict(raw)
    for name in required - {"protocol", "evidence_alpha", "initialization"}:
        value = settings[name]
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"recursive vocabulary {name} must be positive")
        settings[name] = int(value)
    if "evidence_alpha" in settings:
        alpha = settings["evidence_alpha"]
        if (
            isinstance(alpha, bool)
            or not math.isfinite(float(alpha))
            or not 0 < float(alpha) <= 1
        ):
            raise ValueError("recursive vocabulary evidence_alpha must be in (0, 1]")
        settings["evidence_alpha"] = float(alpha)
    if "initialization" in settings:
        if settings["initialization"] not in {"parent_mean", "parent_sum"}:
            raise ValueError("recursive vocabulary initialization is invalid")
    interval = settings["interval_steps"]
    if interval % 2:
        raise ValueError("recursive vocabulary interval must split into two halves")
    save_steps = int(dict(config.get("training") or {}).get("save_steps", 0))
    if save_steps <= 0 or save_steps % interval:
        raise ValueError("recursive vocabulary interval must divide save_steps")
    if max_steps is not None and int(max_steps) % interval:
        raise ValueError(
            "recursive vocabulary max_steps must end on a full decision boundary"
        )
    if protocol in {
        "recursive-cascade-balanced-warmup-v1",
        "recursive-cascade-monotone-warmup-v1",
    }:
        warmup_steps = settings["warmup_steps"]
        if warmup_steps % interval:
            raise ValueError("recursive vocabulary warmup must end on a full boundary")
        if max_steps is not None and warmup_steps > int(max_steps):
            raise ValueError("recursive vocabulary warmup exceeds max_steps")
    cascade = getattr(module, "recursive_cascade", None)
    if cascade is None:
        raise ValueError("recursive cascade table is missing")
    if settings["residual_buckets"] != int(cascade.residual_buckets):
        raise ValueError("recursive proposal capacity differs from initialized model")
    if settings["max_span_length"] != int(cascade.max_span_length):
        raise ValueError("recursive span limit differs from initialized model")
    return settings


def _parse_root_assignments(
    assignments: list[str] | None,
    *,
    fineweb_root: Path | None,
    parallel_root: Path | None,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}

    def add(name: str, path: Path) -> None:
        if not name or name in roots:
            raise ValueError(f"duplicate dataset root: {name!r}")
        roots[name] = path.expanduser().resolve()

    if fineweb_root is not None:
        add("fineweb", fineweb_root)
    if parallel_root is not None:
        add("parallel", parallel_root)
    for assignment in assignments or ():
        if "=" not in assignment:
            raise ValueError(f"dataset root must be NAME=PATH: {assignment!r}")
        name, raw_path = assignment.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError(f"dataset root must be NAME=PATH: {assignment!r}")
        add(name.strip(), Path(raw_path))
    if not roots:
        raise ValueError("at least one dataset root is required")
    return roots


def _resolve_trainable_language_indices(
    model_languages: list[str], configured_languages: list[str] | None
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not model_languages or len(set(model_languages)) != len(model_languages):
        raise ValueError("model languages must be non-empty and unique")
    selected = tuple(model_languages if configured_languages is None else configured_languages)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(not isinstance(language, str) or not language for language in selected)
    ):
        raise ValueError("trainable languages must be non-empty unique names")
    lookup = {language: index for index, language in enumerate(model_languages)}
    unknown = sorted(set(selected) - set(lookup))
    if unknown:
        raise ValueError(f"unknown trainable language markers: {unknown}")
    return selected, tuple(lookup[language] for language in selected)


class BoundedVocabularyCallback(TrainerCallback):
    """Promote a hard-budget exact vocabulary at one registered training step."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        output_dir: Path,
        settings: dict[str, Any],
        reclaimed_settings: dict[str, Any] | None = None,
    ) -> None:
        self.module = module
        self.output_dir = output_dir
        self.settings = settings
        self.reclaimed_settings = reclaimed_settings
        self.memory_report: dict[str, Any] | None = None
        self.external_selection_report: dict[str, Any] | None = None
        collapse = getattr(module, "collapse", None)
        self.promotion_done = bool(collapse is not None and collapse.pair_keys.numel())

    def _export_diagnostic_candidates(self, discovery, step: int) -> None:
        self._export_candidates(
            discovery,
            step,
            csv_name="dynamic_vocabulary_candidates.csv",
            manifest_name="dynamic_vocabulary_candidate_manifest.json",
            protocol="q23-proportional-fair-vocabulary-v1",
            stage="bounded-candidate-export",
        )
    def _export_candidates(
        self,
        discovery,
        step: int,
        *,
        csv_name: str,
        manifest_name: str,
        protocol: str,
        stage: str,
    ) -> None:
        rows = discovery.snapshot_records()
        if not rows:
            raise RuntimeError("diagnostic candidate export is empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / csv_name
        csv_tmp = csv_path.with_suffix(".csv.tmp")
        fieldnames = [
            "composite_key", "pair_key", "language_index", "utility",
            "probability_sum", "captured_support", "mean_probability",
        ]
        with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(csv_tmp, csv_path)
        manifest = {
            "protocol": protocol,
            "stage": stage,
            "export_step": int(step),
            "admission_mode": str(
                getattr(discovery, "admission_mode", "utility")
            ),
            "settings": self.settings,
            "observed_occurrences": int(discovery.observed_occurrences),
            "transferred_records": int(discovery.transferred_records),
            "bounded_pair_language_records": len(rows),
            "candidate_pairs": len({int(row["pair_key"]) for row in rows}),
            "candidate_csv": str(csv_path.resolve()),
            "candidate_csv_sha256": _sha256(csv_path),
        }
        manifest_path = self.output_dir / manifest_name
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest_path)
        print(
            f"exported {len(rows)} bounded pair/language records over "
            f"{manifest['candidate_pairs']} pairs at step {step}",
            flush=True,
        )

    def on_save(self, args, state, control, **kwargs):
        if self.promotion_done:
            return control
        discovery = self.module.collapse.discovery
        if discovery is None:
            raise RuntimeError("pre-promotion checkpoint has no discovery state")
        checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        sidecar = checkpoint / "dynamic_vocabulary_discovery.pt"
        temporary = checkpoint / "dynamic_vocabulary_discovery.pt.tmp"
        torch.save(discovery.state_dict(), temporary)
        os.replace(temporary, sidecar)
        return control

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        """Materialize a frozen selection at an exact pre-step resume boundary."""

        if (
            self.promotion_done
            or not bool(self.settings.get("resume_boundary_promotion", False))
            or int(state.global_step) + 1 != int(self.settings["promotion_step"])
        ):
            return control
        if optimizer is None:
            raise RuntimeError("resume-boundary promotion requires the loaded optimizer")
        original_step = int(state.global_step)
        state.global_step = int(self.settings["promotion_step"])
        try:
            result = self.on_step_end(
                args, state, control, optimizer=optimizer, **kwargs
            )
        finally:
            state.global_step = original_step
        if not self.promotion_done:
            raise RuntimeError("resume-boundary promotion did not materialize a vocabulary")
        print(
            f"external vocabulary materialized before optimizer step "
            f"{original_step + 1}",
            flush=True,
        )
        return result

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        window_steps = tuple(self.settings.get("diagnostic_window_export_steps", ()))
        current_step = int(state.global_step)
        if current_step in window_steps:
            collapse = self.module.collapse
            diagnostic = collapse.diagnostic_discovery
            if diagnostic is None:
                raise RuntimeError("window export has no diagnostic discovery statistics")
            window_index = window_steps.index(current_step) + 1
            self._export_candidates(
                diagnostic,
                current_step,
                csv_name=f"dynamic_vocabulary_window{window_index}.csv",
                manifest_name=f"dynamic_vocabulary_window{window_index}_manifest.json",
                protocol="q24-cross-window-stable-vocabulary-v1",
                stage=f"bounded-window-{window_index}-export",
            )
            if window_index < len(window_steps):
                collapse.configure_diagnostic_discovery(
                    candidate_capacity=int(self.settings["candidate_capacity"]),
                    top_per_forward=int(self.settings["top_per_forward"]),
                    admission_mode=str(
                        self.settings.get("admission_mode", "utility")
                    ),
                )
        export_step = int(self.settings.get("diagnostic_export_step", 0))
        if (not self.promotion_done and export_step
                and current_step == export_step):
            collapse = self.module.collapse
            discovery = collapse.discovery
            if discovery is None:
                raise RuntimeError("diagnostic export has no discovery statistics")
            self._export_diagnostic_candidates(discovery, int(state.global_step))
            if bool(self.settings.get("diagnostic_export_only", False)):
                control.should_training_stop = True
                return control
        if self.promotion_done or int(state.global_step) != int(self.settings["promotion_step"]):
            return control
        collapse = self.module.collapse
        discovery = collapse.discovery
        if discovery is None:
            raise RuntimeError("dynamic vocabulary promotion has no discovery statistics")
        external = dict(self.settings.get("external_selection") or {})
        selection_audit = None
        if external:
            selection_path = Path(external["path"])
            if _sha256(selection_path) != external["sha256"]:
                raise RuntimeError("external vocabulary selection hash mismatch")
            keys = []
            with selection_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"pair_key", "selector"}
                if not required.issubset(reader.fieldnames or ()):
                    raise RuntimeError("external vocabulary selection columns missing")
                for row in reader:
                    if row["selector"] != external["selector"]:
                        raise RuntimeError("external vocabulary selector label mismatch")
                    keys.append(int(row["pair_key"]))
            keys = sorted(keys)
            expected = int(external["expected_rows"])
            if len(keys) != expected or len(set(keys)) != expected:
                raise RuntimeError("external vocabulary selection cardinality mismatch")
            promoted = discovery.select_external(
                keys, minimum_support=int(self.settings["per_language_min_support"])
            )
            self.external_selection_report = {
                "protocol": external["protocol"],
                "path": str(selection_path.resolve()),
                "sha256": external["sha256"],
                "selector": external["selector"],
                "expected_rows": expected,
            }
        else:
            promoted = discovery.promote(
                budget=int(self.settings["budget"]),
                min_support=int(self.settings["min_support"]),
                per_language_quota=int(self.settings["per_language_quota"]),
                per_language_min_support=int(self.settings["per_language_min_support"]),
                min_utility=float(self.settings.get("min_utility", 0.0)),
            )
            selection_audit = discovery.audit_promotion(
                promoted,
                budget=int(self.settings["budget"]),
                min_support=int(self.settings["min_support"]),
                per_language_quota=int(self.settings["per_language_quota"]),
                per_language_min_support=int(self.settings["per_language_min_support"]),
                min_utility=float(self.settings.get("min_utility", 0.0)),
                reservation_audit_quota=int(
                    self.settings.get(
                        "reservation_audit_quota",
                        self.settings["per_language_quota"],
                    )
                ),
                trainable_language_indices=tuple(
                    int(index)
                    for index in self.settings.get(
                        "trainable_language_indices",
                        range(discovery.n_languages),
                    )
                ),
            )
            selection_audit["trainable_languages"] = {
                "names": list(self.settings.get("trainable_languages", ())),
                "indices": list(
                    self.settings.get(
                        "trainable_language_indices",
                        range(discovery.n_languages),
                    )
                ),
            }
        if not promoted.pair_keys:
            raise RuntimeError("dynamic vocabulary promotion selected no positive supported pairs")
        reclaim_report = None
        if self.reclaimed_settings is None:
            mapping = self.module.promote_dynamic_vocabulary(
                promoted.pair_keys,
                optimizer=optimizer,
            )
        else:
            inventory_root = Path(self.reclaimed_settings["inventory"])
            inventory_path = inventory_root / "manifest.json"
            selection_path = inventory_root / "reclaim_antichain.csv"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            if (inventory.get("protocol") != "q21a-zero-support-reclaimed-slots-v1"
                    or not inventory["inventory"]["structural_gate_pass"]):
                raise RuntimeError("direct-promotion reclaim inventory is not authorized")
            expected_outputs = {
                Path(row["path"]).name: row["sha256"] for row in inventory["outputs"]
            }
            if expected_outputs.get(selection_path.name) != _sha256(selection_path):
                raise RuntimeError("direct-promotion reclaim selection hash mismatch")
            candidates = []
            with selection_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if len(candidates) == len(promoted.pair_keys):
                        break
                    if int(row["training_support"]) != 0:
                        raise RuntimeError("direct-promotion candidate has training support")
                    candidates.append((
                        int(row["token_id"]), int(row["left_parent_id"]),
                        int(row["right_parent_id"]),
                    ))
            if len(candidates) != len(promoted.pair_keys):
                raise RuntimeError("direct-promotion reclaim inventory is too small")
            allocated_before = (
                int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None
            )
            peak_before = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            )
            mapping = self.module.promote_dynamic_vocabulary_to_reclaimed(
                promoted.pair_keys,
                [row[0] for row in candidates],
                [row[1] for row in candidates],
                [row[2] for row in candidates],
                optimizer=optimizer,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                allocated_after = int(torch.cuda.memory_allocated())
                reduction = 100.0 * (1.0 - allocated_after / allocated_before)
                torch.cuda.reset_peak_memory_stats()
            else:
                allocated_after, reduction = None, None
            stale_shapes = []
            released_shape = (int(mapping["released_merged_rows"]),
                              int(self.module.embedding.weight.shape[1]))
            for parameter, state_row in optimizer.state.items():
                for state_name, value in state_row.items():
                    if torch.is_tensor(value) and tuple(value.shape) == released_shape:
                        stale_shapes.append({
                            "parameter_shape": list(parameter.shape),
                            "state": str(state_name), "shape": list(value.shape),
                        })
            if stale_shapes:
                raise RuntimeError(f"stale full merged optimizer state: {stale_shapes}")
            self.memory_report = {
                "pre_release_allocated_bytes": allocated_before,
                "post_release_allocated_bytes": allocated_after,
                "immediate_allocated_reduction_percent": reduction,
                "prepromotion_peak_allocated_bytes": peak_before,
                "postpromotion_peak_reset": torch.cuda.is_available(),
            }
            reclaim_report = {
                "protocol": self.reclaimed_settings["protocol"],
                "inventory_manifest": str(inventory_path.resolve()),
                "inventory_manifest_sha256": _sha256(inventory_path),
                "selection_sha256": _sha256(selection_path),
                "selected_prefix_rows": len(candidates),
                "memory": self.memory_report,
                "stale_optimizer_state_shapes": stale_shapes,
            }
        key_base = int(collapse.pair_key_base)
        rows = []
        for record in promoted.records:
            pair_key = int(record["pair_key"])
            rows.append({
                **record,
                "left_id": pair_key // key_base,
                "right_id": pair_key % key_base,
            })
        manifest = {
            "protocol": (
                self.reclaimed_settings["protocol"] if self.reclaimed_settings
                else "bounded-contrastive-vocabulary-v1"
            ),
            "promotion_step": int(state.global_step),
            "settings": self.settings,
            "discovery": {
                "observed_occurrences": discovery.observed_occurrences,
                "transferred_records": discovery.transferred_records,
                "bounded_pair_language_records": len(discovery.records),
                "candidate_pairs": promoted.candidate_count,
            },
            "promotion": mapping,
            "selected": rows,
        }
        if selection_audit is not None:
            manifest["selection_audit"] = selection_audit
        if reclaim_report is not None:
            manifest["reclaimed_promotion"] = reclaim_report
        if self.external_selection_report is not None:
            manifest["external_selection"] = self.external_selection_report
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "dynamic_vocabulary_manifest.json"
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        with (self.output_dir / "dynamic_vocabulary_selected.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = [
                "pair_key", "left_id", "right_id", "utility",
                "captured_support", "mean_probability", "language_count",
                "reservation_language_count", "max_reservation_support",
                "global_support_eligible",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"dynamic vocabulary promoted {mapping['active_pairs']} exact pairs "
            f"from {promoted.candidate_count} candidates at step {state.global_step}; "
            f"hash collisions split: {mapping['promoted_hash_collisions']}",
            flush=True,
        )
        self.promotion_done = True
        return control


def _exact_vocabulary_sha256(pair_keys: torch.Tensor) -> str:
    keys = [int(value) for value in pair_keys.detach().to("cpu", torch.long).tolist()]
    payload = json.dumps(keys, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AdaptiveVocabularyGrowthCallback(TrainerCallback):
    """Make repeated, monotone exact-vocabulary transitions during one run."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        output_dir: Path,
        settings: dict[str, Any],
        resume_sidecar: Path | None = None,
        resume_step: int | None = None,
    ) -> None:
        self.module = module
        self.output_dir = Path(output_dir)
        self.settings = settings
        self.previous_window: GrowthWindow | None = None
        self.last_decision_step = 0
        self.last_additions = 0
        self.cumulative_additions = int(module.collapse.pair_keys.numel())
        self.selector_candidates = 0
        self.selector_eligible = 0
        self._settings_sha256 = hashlib.sha256(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if resume_sidecar is not None:
            self._restore(Path(resume_sidecar), expected_step=resume_step)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "protocol": "adaptive-exact-growth-state-v1",
            "settings_sha256": self._settings_sha256,
            "policy": self.settings["policy"],
            "last_decision_step": self.last_decision_step,
            "last_additions": self.last_additions,
            "cumulative_additions": self.cumulative_additions,
            "selector_candidates": self.selector_candidates,
            "selector_eligible": self.selector_eligible,
            "exact_keys_sha256": _exact_vocabulary_sha256(
                self.module.collapse.pair_keys
            ),
            "previous_window": (
                None if self.previous_window is None else self.previous_window.state_dict()
            ),
        }

    def _restore(self, sidecar: Path, *, expected_step: int | None) -> None:
        if not sidecar.is_file():
            raise ValueError(f"adaptive growth resume sidecar is missing: {sidecar}")
        state = torch.load(sidecar, map_location="cpu", weights_only=True)
        required = {
            "version", "protocol", "settings_sha256", "policy",
            "last_decision_step", "last_additions", "cumulative_additions",
            "selector_candidates", "selector_eligible", "exact_keys_sha256",
            "previous_window",
        }
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("adaptive growth resume state has an invalid schema")
        if (
            state["version"] != 1
            or state["protocol"] != "adaptive-exact-growth-state-v1"
            or state["settings_sha256"] != self._settings_sha256
            or state["policy"] != self.settings["policy"]
        ):
            raise ValueError("adaptive growth resume identity differs")
        if state["exact_keys_sha256"] != _exact_vocabulary_sha256(
            self.module.collapse.pair_keys
        ):
            raise ValueError("adaptive growth resume exact-key hash differs")
        decision_step = int(state["last_decision_step"])
        if expected_step is not None and decision_step != int(expected_step):
            raise ValueError("adaptive growth resume step differs from checkpoint")
        self.last_decision_step = decision_step
        self.last_additions = int(state["last_additions"])
        self.cumulative_additions = int(state["cumulative_additions"])
        self.selector_candidates = int(state["selector_candidates"])
        self.selector_eligible = int(state["selector_eligible"])
        previous = state["previous_window"]
        self.previous_window = (
            None if previous is None else GrowthWindow.load_state_dict(previous)
        )

    def _preflight_additions(self, additions: tuple[int, ...]) -> None:
        if not additions:
            return
        merged = self.module.collapse.merged
        dimensions = int(merged.shape[1])
        # Parameter + Adam first/second moments. This is an explicit addition
        # ceiling, not a selector budget: the full eligible set passes or fails.
        bytes_per_row = (dimensions + 1) * max(4, merged.element_size()) * 3
        required = len(additions) * bytes_per_row
        if required > int(self.settings["max_parameter_bytes"]):
            raise MemoryError(
                f"complete vocabulary growth requires {required} parameter bytes"
            )

    def _reset_discovery(self) -> None:
        self.module.configure_dynamic_vocabulary_discovery(
            candidate_capacity=int(self.settings["candidate_capacity"]),
            top_per_forward=int(self.settings["top_per_forward"]),
        )

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        del args, kwargs
        step = int(state.global_step)
        interval = int(self.settings["interval_steps"])
        if step <= 0 or step % interval or step == self.last_decision_step:
            return control
        discovery = self.module.collapse.discovery
        if discovery is None:
            raise RuntimeError("adaptive vocabulary growth has no discovery observer")
        current = GrowthWindow.from_discovery(discovery)
        existing = tuple(int(value) for value in self.module.collapse.pair_keys.tolist())
        selection = None
        growth_locked = bool(self.settings.get("single_transition", False) and existing)
        if growth_locked:
            self.previous_window = current
        elif self.settings["policy"] == "stable":
            if self.previous_window is not None:
                selection = select_stable_growth(
                    self.previous_window,
                    current,
                    existing,
                    min_support=int(self.settings["min_support"]),
                    per_language_min_support=int(
                        self.settings["per_language_min_support"]
                    ),
                )
            self.previous_window = current
        else:
            l0 = self.settings["l0"]
            selection = select_l0_growth(
                current,
                existing,
                min_support=int(self.settings["min_support"]),
                per_language_min_support=int(self.settings["per_language_min_support"]),
                open_threshold=float(l0["open_threshold"]),
                beta=float(l0["beta"]),
                gamma=float(l0["gamma"]),
                zeta=float(l0["zeta"]),
            )
        additions = () if selection is None else selection.pair_keys
        self._preflight_additions(additions)
        transition = None
        if additions:
            if optimizer is None:
                raise RuntimeError("adaptive vocabulary growth requires the live optimizer")
            transition = self.module.grow_dynamic_vocabulary(
                additions,
                optimizer=optimizer,
                residual_buckets=int(self.settings["residual_buckets"]),
            )
        self.last_decision_step = step
        self.last_additions = len(additions)
        self.cumulative_additions = int(self.module.collapse.pair_keys.numel())
        self.selector_candidates = 0 if selection is None else selection.candidate_count
        self.selector_eligible = len(additions)
        manifest = {
            "protocol": "adaptive-exact-growth-decision-v1",
            "policy": self.settings["policy"],
            "optimizer_step": step,
            "candidate_pairs": self.selector_candidates,
            "eligible_pairs": self.selector_eligible,
            "added_exact_rows": self.last_additions,
            "cumulative_exact_rows": self.cumulative_additions,
            "residual_rows": int(getattr(self.module.collapse, "residual_buckets", 0)),
            "rejections": {} if selection is None else selection.rejections,
            "transition": transition,
            "growth_locked": growth_locked,
            "exact_keys_sha256": _exact_vocabulary_sha256(
                self.module.collapse.pair_keys
            ),
            "settings_sha256": self._settings_sha256,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"vocabulary_growth_step-{step:08d}.json"
        temporary = path.with_suffix(".json.tmp")
        if path.exists() or temporary.exists():
            raise FileExistsError(path)
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        self._reset_discovery()
        return control

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        step = int(state.global_step)
        if step != self.last_decision_step:
            raise RuntimeError("adaptive growth save is not aligned to a decision step")
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        destination = checkpoint / "vocabulary_growth_state.pt"
        temporary = checkpoint / "vocabulary_growth_state.pt.tmp"
        if temporary.exists():
            raise FileExistsError(temporary)
        torch.save(self._state_dict(), temporary)
        os.replace(temporary, destination)
        return control


class RecursiveVocabularyGrowthCallback(TrainerCallback):
    """Collect two half-windows and make one atomic recursive transition."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        output_dir: Path,
        settings: dict[str, Any],
        resume_sidecar: Path | None = None,
        resume_step: int | None = None,
    ) -> None:
        self.module = module
        self.output_dir = Path(output_dir)
        self.settings = dict(settings)
        self.first_half: GrowthWindow | None = None
        self.first_half_sign: SignEvidenceWindow | None = None
        self.first_half_usage: tuple[int, ...] | None = None
        self.last_decision_step = 0
        self.last_additions = 0
        self.consecutive_empty_transitions = 0
        self._settings_sha256 = hashlib.sha256(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if resume_sidecar is not None:
            self._restore(Path(resume_sidecar), expected_step=resume_step)

    @property
    def balanced_warmup(self) -> bool:
        return self.settings["protocol"] == "recursive-cascade-balanced-warmup-v1"

    @property
    def monotone_warmup(self) -> bool:
        return self.settings["protocol"] == "recursive-cascade-monotone-warmup-v1"

    @property
    def warmup_limited(self) -> bool:
        return self.balanced_warmup or self.monotone_warmup

    def structure_frozen(self, step: int) -> bool:
        if not self.warmup_limited:
            return False
        if int(step) >= int(self.settings["warmup_steps"]):
            return True
        return self.monotone_warmup and self.cascade.learned_count >= int(
            self.settings["target_learned_rows"]
        )

    @property
    def cascade(self):
        cascade = getattr(self.module, "recursive_cascade", None)
        if cascade is None:
            raise RuntimeError("recursive vocabulary has no cascade table")
        return cascade

    def _state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "protocol": "recursive-cascade-state-v1",
            "settings_sha256": self._settings_sha256,
            "last_decision_step": self.last_decision_step,
            "last_additions": self.last_additions,
            "consecutive_empty_transitions": self.consecutive_empty_transitions,
            "generation": self.cascade.generation,
            "learned_rows": self.cascade.learned_count,
            "metadata_sha256": self.cascade.metadata_sha256(),
            # Checkpoints are required at full boundaries, never halfway through
            # a decision interval, so no partial discovery window is legal here.
            "first_half": None,
        }

    def _restore(self, sidecar: Path, *, expected_step: int | None) -> None:
        if not sidecar.is_file():
            raise ValueError(f"recursive vocabulary resume sidecar is missing: {sidecar}")
        state = torch.load(sidecar, map_location="cpu", weights_only=True)
        required = {
            "version", "protocol", "settings_sha256", "last_decision_step",
            "last_additions", "consecutive_empty_transitions", "generation",
            "learned_rows", "metadata_sha256", "first_half",
        }
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("recursive vocabulary resume state has an invalid schema")
        if (
            state["version"] != 1
            or state["protocol"] != "recursive-cascade-state-v1"
            or state["settings_sha256"] != self._settings_sha256
            or state["first_half"] is not None
        ):
            raise ValueError("recursive vocabulary resume identity differs")
        step = int(state["last_decision_step"])
        if expected_step is not None and step != int(expected_step):
            raise ValueError("recursive vocabulary resume step differs")
        if (
            int(state["generation"]) != self.cascade.generation
            or int(state["learned_rows"]) != self.cascade.learned_count
            or state["metadata_sha256"] != self.cascade.metadata_sha256()
        ):
            raise ValueError("recursive vocabulary resume structure differs")
        self.last_decision_step = step
        self.last_additions = int(state["last_additions"])
        self.consecutive_empty_transitions = int(
            state["consecutive_empty_transitions"]
        )
        if self.structure_frozen(step):
            self.cascade.discovery = None
            self.cascade.usage_counts = None

    def _reset_discovery(self) -> None:
        self.module.configure_recursive_vocabulary_discovery()

    def _preflight(self, pair_keys: tuple[int, ...]) -> None:
        if not pair_keys:
            return
        dim = int(self.cascade.dim)
        # Learned parameter and Adam first/second moments, plus five int64 DAG
        # fields per row.  This is fail-closed capacity validation, not ranking.
        required = len(pair_keys) * ((dim * 4 * 3) + 5 * 8)
        if required > int(self.settings["max_parameter_bytes"]):
            raise MemoryError(
                f"complete recursive generation requires {required} bytes"
            )

    def _write_active_mask(self, step: int) -> dict[str, Any]:
        mask = self.cascade.rule_active.detach().cpu().to(torch.uint8)
        padding = (-int(mask.numel())) % 8
        if padding:
            mask = torch.cat((mask, torch.zeros(padding, dtype=torch.uint8)))
        if mask.numel():
            weights = torch.tensor(
                [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int16
            )
            packed = (mask.reshape(-1, 8).to(torch.int16) * weights).sum(1).to(
                torch.uint8
            )
            payload = packed.numpy().tobytes()
        else:
            payload = b""
        path = self.output_dir / f"recursive_vocabulary_active_step-{step:08d}.bin"
        temporary = path.with_suffix(".bin.tmp")
        if path.exists() or temporary.exists():
            raise FileExistsError(path)
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return {
            "protocol": "recursive-active-mask-little-bitpack-v1",
            "path": path.name,
            "learned_rows": self.cascade.learned_count,
            "active_rows": self.cascade.active_count,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        del args, kwargs
        step = int(state.global_step)
        interval = int(self.settings["interval_steps"])
        half = interval // 2
        if self.structure_frozen(step - 1):
            return control
        if step <= 0 or step % half:
            return control
        discovery = self.cascade.discovery
        if discovery is None:
            raise RuntimeError("recursive vocabulary has no discovery observer")
        familywise_sign = (
            self.settings["protocol"] == "recursive-cascade-familywise-sign-v1"
        )
        bidirectional = self.settings["protocol"] in {
            "recursive-cascade-bidirectional-v1",
            "recursive-cascade-familywise-sign-v1",
            "recursive-cascade-balanced-warmup-v1",
        }
        # Each conversion scans every exact pair-language record.  The two
        # selectors are mutually exclusive, so constructing both needlessly
        # doubled boundary time for the familywise protocol.
        current = (
            None if familywise_sign else GrowthWindow.from_discovery(discovery)
        )
        current_sign = (
            SignEvidenceWindow.from_discovery(discovery) if familywise_sign else None
        )
        current_usage = (
            self.cascade.snapshot_usage_token_ids() if bidirectional else ()
        )
        if step % interval:
            self.first_half = current
            self.first_half_sign = current_sign
            self.first_half_usage = current_usage
            self._reset_discovery()
            return control
        if step == self.last_decision_step:
            return control
        if not familywise_sign and self.first_half is None:
            raise RuntimeError("recursive decision has no first half-window")
        if self.first_half_usage is None:
            raise RuntimeError("recursive decision has no first half usage window")
        if familywise_sign and self.first_half_sign is None:
            raise RuntimeError("recursive decision has no first half sign window")
        deactivation = {
            "deactivated_rows": 0,
            "active_rows": self.cascade.active_count,
            "token_ids_sha256": hashlib.sha256(b"[]").hexdigest(),
        }
        if bidirectional:
            observed = tuple(sorted(set(self.first_half_usage) | set(current_usage)))
            deactivation = self.module.deactivate_unobserved_recursive_vocabulary(
                observed
            )
        all_existing = tuple(
            (int(left), int(right))
            for left, right in zip(
                self.cascade.rule_left.detach().cpu().tolist(),
                self.cascade.rule_right.detach().cpu().tolist(),
                strict=True,
            )
        )
        # Existing rule identities use Cantor keys, not the flat Q30 base.
        from starse.recursive_cascade import recursive_pair_key

        all_existing_keys = tuple(
            recursive_pair_key(*pair) for pair in all_existing
        )
        existing_keys = (
            self.cascade.active_pair_keys() if bidirectional else all_existing_keys
        )
        if self.monotone_warmup:
            available = max(
                0,
                int(self.settings["target_learned_rows"])
                - self.cascade.learned_count,
            )
            selection = select_language_balanced_growth(
                self.first_half,
                current,
                existing_keys,
                limit=available,
                min_support=int(self.settings["min_support"]),
                per_language_min_support=int(
                    self.settings["per_language_min_support"]
                ),
            )
        elif self.balanced_warmup:
            available = max(
                0,
                int(self.settings["target_active_rows"]) - self.cascade.active_count,
            )
            selection = select_language_balanced_growth(
                self.first_half,
                current,
                existing_keys,
                limit=available,
                min_support=int(self.settings["min_support"]),
                per_language_min_support=int(
                    self.settings["per_language_min_support"]
                ),
            )
        elif familywise_sign:
            selection = select_familywise_sign_growth(
                self.first_half_sign,
                current_sign,
                existing_keys,
                alpha=float(self.settings["evidence_alpha"]),
            )
        else:
            selection = select_stable_growth(
                self.first_half,
                current,
                existing_keys,
                min_support=int(self.settings["min_support"]),
                per_language_min_support=int(
                    self.settings["per_language_min_support"]
                ),
            )
        reactivation = {
            "reactivated_rows": 0,
            "active_rows": self.cascade.active_count,
            "pair_keys": [],
            "token_ids": [],
            "rejected_inactive_parent": 0,
        }
        if bidirectional:
            reactivation = self.module.reactivate_recursive_vocabulary(
                selection.pair_keys
            )
        known = set(all_existing_keys)
        novel_pair_keys = tuple(
            key for key in selection.pair_keys if key not in known
        )
        promotable, structural_rejections = self.cascade.promotable_pair_keys(
            novel_pair_keys
        )
        self._preflight(promotable)
        if promotable and optimizer is None:
            raise RuntimeError("recursive growth requires the live optimizer")
        transition = (
            self.module.grow_recursive_vocabulary(
                promotable,
                optimizer=optimizer,
                initialization=self.settings.get("initialization", "parent_mean"),
            )
            if promotable
            else {
                "generation": self.cascade.generation,
                "added_rows": 0,
                "learned_rows": self.cascade.learned_count,
                "active_rows": self.cascade.active_count,
                "rejections": structural_rejections,
            }
        )
        self.last_decision_step = step
        self.last_additions = len(promotable)
        structural_changes = (
            len(promotable)
            + int(reactivation["reactivated_rows"])
            + int(deactivation["deactivated_rows"])
        )
        self.consecutive_empty_transitions = (
            self.consecutive_empty_transitions + 1 if not structural_changes else 0
        )
        if self.balanced_warmup and self.cascade.active_count > int(
            self.settings["target_active_rows"]
        ):
            raise RuntimeError("recursive active vocabulary exceeded its target")
        if self.monotone_warmup and self.cascade.learned_count > int(
            self.settings["target_learned_rows"]
        ):
            raise RuntimeError("recursive learned vocabulary exceeded its target")
        active_mask = self._write_active_mask(step) if bidirectional else None
        manifest = {
            "protocol": (
                "recursive-cascade-familywise-sign-decision-v1"
                if familywise_sign
                else (
                    "recursive-cascade-balanced-warmup-decision-v1"
                    if self.balanced_warmup
                    else (
                        "recursive-cascade-monotone-warmup-decision-v1"
                        if self.monotone_warmup
                        else (
                            "recursive-cascade-bidirectional-decision-v1"
                            if bidirectional else "recursive-cascade-decision-v1"
                        )
                    )
                )
            ),
            "optimizer_step": step,
            "generation": self.cascade.generation,
            "candidate_pairs": selection.candidate_count,
            "criterion_eligible_pairs": len(selection.pair_keys),
            "added_rows": len(promotable),
            "learned_rows": self.cascade.learned_count,
            "active_rows": self.cascade.active_count,
            "deactivated_rows": int(deactivation["deactivated_rows"]),
            "reactivated_rows": int(reactivation["reactivated_rows"]),
            "consecutive_empty_transitions": self.consecutive_empty_transitions,
            "selector_rejections": selection.rejections,
            "language_allocations": [
                {"language_index": language, "rows": rows}
                for language, rows in selection.language_allocations
            ],
            "global_backfill": selection.global_backfill,
            "target_active_rows": self.settings.get("target_active_rows"),
            "target_learned_rows": self.settings.get("target_learned_rows"),
            "structure_frozen": self.structure_frozen(step),
            "structural_rejections": structural_rejections,
            "deactivation": deactivation,
            "reactivation": reactivation,
            "active_mask": active_mask,
            "transition": transition,
            "metadata_sha256": self.cascade.metadata_sha256(),
            "settings_sha256": self._settings_sha256,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"recursive_vocabulary_step-{step:08d}.json"
        temporary = path.with_suffix(".json.tmp")
        if path.exists() or temporary.exists():
            raise FileExistsError(path)
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        self.first_half = None
        self.first_half_sign = None
        self.first_half_usage = None
        if self.structure_frozen(step):
            self.cascade.discovery = None
            self.cascade.usage_counts = None
        else:
            self._reset_discovery()
        return control

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        step = int(state.global_step)
        if self.structure_frozen(step):
            self.last_decision_step = step
        if (
            step != self.last_decision_step
            or self.first_half is not None
            or self.first_half_sign is not None
            or self.first_half_usage is not None
        ):
            raise RuntimeError("recursive save is not aligned to a full decision boundary")
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        destination = checkpoint / "recursive_vocabulary_state.pt"
        temporary = checkpoint / "recursive_vocabulary_state.pt.tmp"
        if temporary.exists():
            raise FileExistsError(temporary)
        torch.save(self._state_dict(), temporary)
        os.replace(temporary, destination)
        return control


class HardConcreteL0Callback(TrainerCallback):
    """Add the expected hard-concrete architecture price before clipping."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        weight: float,
        beta: float,
        gamma: float,
        zeta: float,
    ) -> None:
        self.module = module
        self.weight = float(weight)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.last_penalty = 0.0

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        expected = expected_hard_concrete_l0(
            self.module.collapse,
            beta=self.beta,
            gamma=self.gamma,
            zeta=self.zeta,
        )
        penalty = expected * self.weight
        if penalty.ndim or not bool(torch.isfinite(penalty)):
            raise FloatingPointError("hard-concrete L0 penalty is nonfinite")
        penalty.backward()
        proposal = (
            self.module.collapse.residual_score
            if int(getattr(self.module.collapse, "residual_buckets", 0))
            else self.module.collapse.score
        )
        for name, parameter in (
            ("proposal score", proposal),
            ("language bias", self.module.collapse.language_bias),
        ):
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"hard-concrete L0 has invalid {name} gradient")
        self.last_penalty = float(penalty.detach().float().item())


class OnlineTokenizerInvariantCallback(TrainerCallback):
    """Make cache completion and finite gradients fatal before every online step."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.completed_audits = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        del args, state, control, kwargs
        self.module.online_collapse.assert_cache_audit_empty()
        checked = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            checked += 1
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"nonfinite gradient in {name}")
        if checked == 0:
            raise RuntimeError("online tokenizer optimizer step has no gradients")
        self.completed_audits += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SplitPenaltyLoss(torch.nn.Module):
    """Add an L1 on the split-vocabulary gate so a private row must earn itself.

    Without it every gate drifts open and the split table is free extra capacity,
    which answers a different question than "which pairs actually need
    separating".
    """

    def __init__(self, loss: torch.nn.Module, module: torch.nn.Module, weight: float) -> None:
        super().__init__()
        self.loss = loss
        self.module = module
        self.weight = float(weight)

    def forward(self, sentence_features: Any, labels: Any = None) -> torch.Tensor:
        value = self.loss(sentence_features, labels)
        split = getattr(self.module, "split", None)
        if split is not None and self.weight:
            term = (split.fusion_penalty() if getattr(self.module, "mode", "") == "fuse"
                    else split.penalty())
            value = value + self.weight * term
        assign = getattr(self.module, "assign", None)
        if assign is not None and self.weight:
            # Penalise assignment entropy: each language should commit to one
            # vocabulary rather than hedge across all of them.
            value = value + self.weight * assign.entropy()
        return value


class ScaledLoss(torch.nn.Module):
    """Multiply a scalar training objective without changing its composition."""

    def __init__(self, loss: torch.nn.Module, multiplier: float) -> None:
        super().__init__()
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("loss multiplier must be finite and greater than zero")
        self.loss = loss
        self.multiplier = float(multiplier)

    def forward(self, sentence_features: Any, labels: Any = None) -> torch.Tensor:
        return self.loss(sentence_features, labels) * self.multiplier


class CheckpointRunConfigCallback(TrainerCallback):
    """Put the exact public run contract inside every reloadable checkpoint."""

    def __init__(self, run_config: Path) -> None:
        self.run_config = Path(run_config).expanduser().resolve()
        if not self.run_config.is_file():
            raise FileNotFoundError(self.run_config)
        self.completed_saves = 0

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        output_dir = Path(args.output_dir).expanduser().resolve()
        if self.run_config.parent != output_dir:
            raise ValueError("checkpoint run config is outside Trainer output_dir")
        checkpoint = output_dir / f"checkpoint-{int(state.global_step)}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Trainer checkpoint is absent during on_save: {checkpoint}"
            )
        source = self.run_config.read_bytes()
        destination = checkpoint / "run_config.json"
        if destination.exists():
            if destination.read_bytes() != source:
                raise ValueError("checkpoint run_config.json differs from the run contract")
        else:
            temporary = checkpoint / "run_config.json.tmp"
            with temporary.open("xb") as handle:
                handle.write(source)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        self.completed_saves += 1
        return control


_OPENLEX_REPLAY_KEYS = {
    "protocol",
    "sample_pairs",
    "sample_seed",
    "squared_boundary_budget",
    "generator_atom_dim",
    "generator_hidden_dim",
    "residual_rank",
    "initial_non_atomic_score",
    "existing_token_length_bonus",
    "boundary_misalignment_penalty",
    "compression_weight",
    "learning_rate",
    "weight_decay",
}
_OPENLEX_REUSE_MDL_KEYS = _OPENLEX_REPLAY_KEYS | {"structural_objective"}
_OPENLEX_RECURRENT_RESIDUAL_KEYS = _OPENLEX_REUSE_MDL_KEYS | {
    "recurrent_residuals"
}
_OPENLEX_BATCH_RESIDUAL_KEYS = _OPENLEX_RECURRENT_RESIDUAL_KEYS | {
    "reuse_support_scope"
}


def _validate_openlex_replay_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the frozen production replay contract before data iteration."""

    raw = config.get("openlex_replay")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("OpenLex replay settings must use the exact registered schema")
    protocol = raw.get("protocol")
    if (
        protocol == "openlex-cvr-v1"
        and set(raw) == _OPENLEX_REPLAY_KEYS
    ):
        structural_objective = "expected_tokens"
        recurrent_residuals = False
        reuse_support_scope = "sample"
    elif (
        protocol == "openlex-cvr-reuse-mdl-v1"
        and set(raw) == _OPENLEX_REUSE_MDL_KEYS
        and raw.get("structural_objective") == "reuse_mdl"
    ):
        structural_objective = "reuse_mdl"
        recurrent_residuals = False
        reuse_support_scope = "sample"
    elif (
        protocol == "openlex-cvr-reuse-residual-v1"
        and set(raw) == _OPENLEX_RECURRENT_RESIDUAL_KEYS
        and raw.get("structural_objective") == "reuse_mdl"
        and raw.get("recurrent_residuals") is True
    ):
        structural_objective = "reuse_mdl"
        recurrent_residuals = True
        reuse_support_scope = "sample"
    elif (
        protocol == "openlex-cvr-reuse-residual-batch-v1"
        and set(raw) == _OPENLEX_BATCH_RESIDUAL_KEYS
        and raw.get("structural_objective") == "reuse_mdl"
        and raw.get("recurrent_residuals") is True
        and raw.get("reuse_support_scope") == "batch"
    ):
        structural_objective = "reuse_mdl"
        recurrent_residuals = True
        reuse_support_scope = "batch"
    else:
        raise ValueError("OpenLex replay settings must use the exact registered schema")
    loss = dict(config.get("loss") or {})
    training = dict(config.get("training") or {})
    if (
        loss.get("type") != "cached_multiple_negatives"
        or tuple(loss.get("directions", ())) != ("query_to_doc", "doc_to_query")
        or loss.get("partition_mode") != "per_direction"
        or int(loss.get("mini_batch_size", 0)) <= 0
    ):
        raise ValueError("OpenLex replay requires the exact symmetric cached loss")
    if int(training.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("OpenLex replay requires gradient_accumulation_steps=1")
    if not bool(training.get("bf16", False)) or not bool(
        training.get("save_model_bf16", False)
    ):
        raise ValueError("OpenLex replay requires BF16 training and saved model tables")
    if (
        float(loss.get("multiplier", 1.0)) != 1.0
        or float(training.get("edit_weight", 0.0)) != 0.0
        or float(training.get("split_penalty", 0.0)) != 0.0
        or bool(training.get("freeze_table", False))
    ):
        raise ValueError("OpenLex replay rejects additional loss wrappers or a frozen table")
    if float(raw.get("boundary_misalignment_penalty", -1.0)) != 10.0:
        raise ValueError("OpenLex replay requires boundary_misalignment_penalty=10")
    if float(raw.get("compression_weight", -1.0)) != 1.0:
        raise ValueError("OpenLex replay requires compression_weight=1")

    integer_positive = (
        "sample_pairs",
        "squared_boundary_budget",
        "generator_atom_dim",
        "generator_hidden_dim",
        "residual_rank",
    )
    if any(
        isinstance(raw.get(name), bool) or int(raw.get(name, 0)) <= 0
        for name in integer_positive
    ):
        raise ValueError("OpenLex replay protocol or positive dimensions are invalid")
    if isinstance(raw.get("sample_seed"), bool) or not isinstance(
        raw.get("sample_seed"), int
    ):
        raise ValueError("OpenLex replay sample_seed must be an integer")
    finite_nonnegative = (
        "existing_token_length_bonus",
        "learning_rate",
        "weight_decay",
    )
    if any(
        not math.isfinite(float(raw.get(name, float("nan"))))
        or float(raw[name]) < 0
        for name in finite_nonnegative
    ) or float(raw["learning_rate"]) == 0:
        raise ValueError("OpenLex replay optimizer settings must be finite and valid")
    if not math.isfinite(float(raw.get("initial_non_atomic_score", float("nan")))):
        raise ValueError("OpenLex replay initial score must be finite")
    validated = dict(raw)
    validated["structural_objective"] = structural_objective
    validated["recurrent_residuals"] = recurrent_residuals
    validated["reuse_support_scope"] = reuse_support_scope
    return validated


def _build_openlex_replay(
    model: torch.nn.Module,
    config: dict[str, Any],
    replay_settings: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Build the loss-owned policy while aliasing, then detaching, the q15 table."""

    from starse.openlex_model import OpenLexEncoder
    from starse.openlex_replay import OpenLexControlVariateLoss, OpenLexReplaySettings
    from starse.openlex_tokenizer import CanonicalVocabulary

    module = model[0]
    if not hasattr(module, "embedding") or not hasattr(module, "tokenizer"):
        raise ValueError("OpenLex replay requires a static tokenizer and embedding table")
    tokenizer_text = module.tokenizer.to_str()
    tokenizer_payload = json.loads(tokenizer_text)
    tokenizer_model = tokenizer_payload.get("model")
    if not isinstance(tokenizer_model, dict):
        raise ValueError("OpenLex replay tokenizer has no serializable model")
    vocabulary_size, dimension = module.embedding.weight.shape
    vocabulary = CanonicalVocabulary.from_model_json(
        tokenizer_model,
        vocabulary_size=int(vocabulary_size),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(replay_settings["sample_seed"]))
        openlex = OpenLexEncoder(
            atom_vocab_size=int(vocabulary_size),
            dimension=int(dimension),
            generator_atom_dim=int(replay_settings["generator_atom_dim"]),
            generator_hidden_dim=int(replay_settings["generator_hidden_dim"]),
            residual_rank=int(replay_settings["residual_rank"]),
            initial_non_atomic_score=float(
                replay_settings["initial_non_atomic_score"]
            ),
            existing_token_length_bonus=float(
                replay_settings["existing_token_length_bonus"]
            ),
            detach_base_for_openlex=True,
            straight_through_map=False,
            disable_candidate_residuals=not bool(
                replay_settings["recurrent_residuals"]
            ),
            dictionary_composition=True,
            boundary_misalignment_penalty=float(
                replay_settings["boundary_misalignment_penalty"]
            ),
        )
    openlex.atom_embedding.weight = module.embedding.weight
    loss_settings = dict(config["loss"])
    replay = OpenLexControlVariateLoss(
        model=model,
        openlex=openlex,
        vocabulary=vocabulary,
        settings=OpenLexReplaySettings(
            sample_pairs=int(replay_settings["sample_pairs"]),
            sample_seed=int(replay_settings["sample_seed"]),
            compression_weight=float(replay_settings["compression_weight"]),
            squared_boundary_budget=int(
                replay_settings["squared_boundary_budget"]
            ),
            structural_objective=str(replay_settings["structural_objective"]),
            recurrent_residuals=bool(replay_settings["recurrent_residuals"]),
            reuse_support_scope=str(replay_settings["reuse_support_scope"]),
        ),
        scale=float(loss_settings.get("scale", 20.0)),
        mini_batch_size=int(loss_settings["mini_batch_size"]),
        directions=tuple(loss_settings["directions"]),
        partition_mode=str(loss_settings["partition_mode"]),
    )
    identity: dict[str, object] = {
        "protocol": "openlex-cvr-identity-v1",
        "tokenizer_sha256": hashlib.sha256(tokenizer_text.encode("utf-8")).hexdigest(),
        "config_sha256": canonical_config_sha256(config),
        "table_shape": [int(vocabulary_size), int(dimension)],
        "policy_seed": int(replay_settings["sample_seed"]),
    }
    return replay, identity


def _build_openlex_optimizer(
    *,
    model: torch.nn.Module,
    replay: Any,
    training_settings: dict[str, Any],
    replay_settings: dict[str, Any],
) -> torch.optim.AdamW:
    """Give every base/policy parameter exactly one registered AdamW owner."""

    ordinary = [parameter for parameter in model.parameters() if parameter.requires_grad]
    policy = list(replay.policy_parameters())
    if not ordinary or not policy:
        raise ValueError("OpenLex replay optimizer has an empty parameter group")
    ordinary_ids = {id(parameter) for parameter in ordinary}
    policy_ids = {id(parameter) for parameter in policy}
    if len(ordinary_ids) != len(ordinary) or len(policy_ids) != len(policy):
        raise RuntimeError("OpenLex replay optimizer contains duplicate parameters")
    if ordinary_ids & policy_ids:
        raise RuntimeError("OpenLex replay table leaked into the policy optimizer group")
    return torch.optim.AdamW(
        [
            {
                "params": ordinary,
                "lr": float(training_settings["learning_rate"]),
                "weight_decay": float(training_settings["weight_decay"]),
            },
            {
                "params": policy,
                "lr": float(replay_settings["learning_rate"]),
                "weight_decay": float(replay_settings["weight_decay"]),
            },
        ]
    )


def _build_contrastive_loss(
    model: torch.nn.Module, loss_settings: dict[str, Any]
) -> torch.nn.Module:
    """Build the exact configured symmetric contrastive objective."""

    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesSymmetricRankingLoss,
    )

    loss_type = str(loss_settings.get("type", "multiple_negatives_symmetric"))
    scale = float(loss_settings.get("scale", 20.0))
    if loss_type == "multiple_negatives_symmetric":
        return MultipleNegativesSymmetricRankingLoss(model=model, scale=scale)
    if loss_type != "cached_multiple_negatives":
        raise ValueError(f"unknown multilingual contrastive loss type: {loss_type}")

    directions = tuple(
        str(direction)
        for direction in loss_settings.get(
            "directions", ("query_to_doc", "doc_to_query")
        )
    )
    partition_mode = str(loss_settings.get("partition_mode", "per_direction"))
    if (
        directions != ("query_to_doc", "doc_to_query")
        or partition_mode != "per_direction"
    ):
        raise ValueError(
            "cached multilingual loss requires symmetric query_to_doc/doc_to_query "
            "directions with partition_mode=per_direction"
        )
    mini_batch_size = int(loss_settings.get("mini_batch_size", 0))
    if mini_batch_size <= 0:
        raise ValueError("cached multilingual loss requires a positive mini_batch_size")

    from starse.cached_loss import (
        StaticCompatibleCachedMultipleNegativesRankingLoss,
    )

    return StaticCompatibleCachedMultipleNegativesRankingLoss(
        model=model,
        scale=scale,
        mini_batch_size=mini_batch_size,
        directions=directions,
        partition_mode="per_direction",
    )


def _run_loss_path_vocabulary_audit(
    *,
    trainer: Any,
    loss: torch.nn.Module,
    module: torch.nn.Module,
    config: dict[str, Any],
    roots: dict[str, Path],
    streaming_settings: dict[str, Any],
    output_dir: Path,
    completed_steps: int,
) -> dict[str, Any]:
    """Run a read-only, fresh-stream loss-path audit after prefill."""

    from starse.loss_path_vocabulary import (
        HalfEvidence,
        PathPass,
        completeness_error,
        select_loss_path_pairs,
        select_support_control,
    )

    settings = dict(config.get("loss_path_vocabulary") or {})
    batches = int(settings.get("audit_batches", 100))
    if batches <= 0 or batches % 2:
        raise ValueError("loss_path_vocabulary.audit_batches must be positive and even")
    if getattr(module, "mode", "") != "collapse_dynamic":
        raise ValueError("loss-path vocabulary requires collapse_dynamic mode")
    if module.collapse.pair_keys.numel():
        raise ValueError("loss-path audit must precede exact vocabulary materialization")
    if module.collapse.discovery is not None or module.collapse.diagnostic_discovery is not None:
        raise ValueError("loss-path audit rejects censored discovery observers")

    audit_streaming = dict(streaming_settings)
    audit_streaming["scheduler_start_batch"] = int(completed_steps)
    audit_dataset, _ = build_streaming_dataset(
        config["streams"], roots, audit_streaming
    )
    original_dataset = trainer.train_dataset
    trainer.train_dataset = audit_dataset
    try:
        dataloader = trainer.get_train_dataloader()
    finally:
        trainer.train_dataset = original_dataset

    halves = (HalfEvidence(), HalfEvidence())
    batch_reports: list[dict[str, Any]] = []
    model = trainer.model
    was_training = model.training
    model.train()
    iterator = iter(dataloader)
    try:
        for batch_index in range(batches):
            raw_batch = next(iterator)
            prepared = trainer._prepare_inputs(raw_batch)
            path_passes: list[PathPass] = []
            losses: list[float] = []
            for alpha in (0.0, 0.5, 1.0):
                capture = PathPass()
                model.zero_grad(set_to_none=True)
                module.collapse.begin_loss_path_pass(alpha, capture.observe)
                try:
                    features, labels = trainer.collect_features(dict(prepared))
                    with trainer.compute_loss_context_manager():
                        value = loss(features, labels)
                    if value.ndim or not bool(torch.isfinite(value)):
                        raise FloatingPointError("loss-path audit loss is nonfinite")
                    losses.append(float(value.detach().float().item()))
                    value.backward()
                except BaseException:
                    module.collapse.abort_loss_path_pass()
                    raise
                else:
                    module.collapse.end_loss_path_pass()
                path_passes.append(capture)
            model.zero_grad(set_to_none=True)
            halves[0 if batch_index < batches // 2 else 1].add_simpson(
                tuple(path_passes)
            )
            attribution_sum = float(
                sum(path_passes[0].tensors()[1] + 4.0 * path_passes[1].tensors()[1]
                    + path_passes[2].tensors()[1]) / 6.0
            )
            absolute, relative = completeness_error(
                tuple(losses), attribution_sum
            )
            batch_reports.append({
                "batch": batch_index + 1,
                "loss_alpha_0": losses[0],
                "loss_alpha_0p5": losses[1],
                "loss_alpha_1": losses[2],
                "attribution_sum": attribution_sum,
                "completeness_absolute": absolute,
                "completeness_relative": relative,
            })
            print(
                f"loss-path audit {batch_index + 1}/{batches}: "
                f"completeness abs={absolute:.6g} rel={relative:.6g}",
                flush=True,
            )
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()

    absolute_tolerance = float(settings.get("completeness_absolute_tolerance", 0.05))
    relative_tolerance = float(settings.get("completeness_relative_tolerance", 0.10))
    failed = [
        row for row in batch_reports
        if row["completeness_absolute"] > absolute_tolerance
        and row["completeness_relative"] > relative_tolerance
    ]
    if failed:
        raise RuntimeError(
            f"loss-path completeness failed on {len(failed)}/{batches} audit batches"
        )

    familywise_alpha = float(settings.get("familywise_alpha", 0.05))
    learned, evidence_rows = select_loss_path_pairs(
        halves[0], halves[1], familywise_alpha=familywise_alpha
    )
    allow_empty_smoke = bool(settings.get("allow_empty_smoke", False))
    if not learned and not allow_empty_smoke:
        raise RuntimeError("loss-path evidence selected an empty vocabulary")
    support = select_support_control(halves[0], halves[1], count=len(learned))

    evidence_path = output_dir / "loss_path_evidence.csv"
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        evidence_fields = (
            "pair_key", "half1_occurrences", "half1_positives", "half1_lower",
            "half2_occurrences", "half2_positives", "half2_lower",
            "attribution_sum", "selected",
        )
        writer = csv.DictWriter(handle, fieldnames=evidence_fields)
        writer.writeheader()
        writer.writerows(evidence_rows)
    for name, keys in (("loss_path_selected.csv", learned), ("support_selected.csv", support)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("pair_key", "selector"))
            writer.writeheader()
            selector = "loss_path_evidence" if name.startswith("loss_path") else "matched_support"
            writer.writerows({"pair_key": key, "selector": selector} for key in keys)
    report = {
        "protocol": "loss-path-learned-cardinality-v1",
        "completed_prefill_steps": int(completed_steps),
        "audit_batches": batches,
        "audit_scheduler_start_batch": int(completed_steps),
        "alpha_grid": [0.0, 0.5, 1.0],
        "integration": "Simpson",
        "familywise_alpha": familywise_alpha,
        "candidate_pairs_in_both_halves": len(evidence_rows),
        "learned_rows": len(learned),
        "support_rows": len(support),
        "allow_empty_smoke": allow_empty_smoke,
        "completeness_absolute_tolerance": absolute_tolerance,
        "completeness_relative_tolerance": relative_tolerance,
        "batch_reports": batch_reports,
    }
    (output_dir / "loss_path_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Training config must be a YAML object")
    for key in ("run_name", "streams", "streaming", "training"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    if not isinstance(config["streams"], list) or not config["streams"]:
        raise ValueError("streams must be a non-empty list")
    return config


def configure_comet(config: dict[str, Any], enabled: bool) -> str | list[str]:
    if not enabled:
        return "none"
    if not os.environ.get("COMET_API_KEY"):
        config_path = Path.home() / ".comet.config"
        if config_path.is_file():
            for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = raw_line.partition("=")
                if separator and key.strip().casefold() == "api_key" and value.strip():
                    os.environ["COMET_API_KEY"] = value.strip().strip("\"'")
                    break
    if not os.environ.get("COMET_API_KEY"):
        raise RuntimeError("COMET_API_KEY is required unless --no-comet is used")
    comet = dict(config.get("comet") or {})
    os.environ.setdefault("COMET_PROJECT_NAME", str(comet.get("project_name", "starse-multilingual")))
    os.environ.setdefault("COMET_EXPERIMENT_NAME", str(config["run_name"]))
    if comet.get("workspace"):
        os.environ.setdefault("COMET_WORKSPACE", str(comet["workspace"]))
    os.environ.setdefault("COMET_START_ONLINE", "1")
    return ["comet_ml"]


def _configure_tokenizer_truncation(tokenizer: Any, *, max_tokens: int) -> None:
    if max_tokens <= 0:
        raise ValueError("training max_tokens must be positive")
    tokenizer.enable_truncation(max_length=int(max_tokens))


def train(
    *,
    config: dict[str, Any],
    roots: dict[str, Path],
    model_path: str,
    output_dir: Path,
    max_steps_override: int | None,
    batch_size_override: int | None,
    resume_from_checkpoint: str | None,
    comet_enabled: bool,
    disable_bf16: bool,
    save_final_model: bool = True,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    settings = dict(config["training"])
    openlex_replay_settings = _validate_openlex_replay_config(config)
    max_steps = int(max_steps_override or settings["max_steps"])
    batch_size = int(batch_size_override or settings["per_device_train_batch_size"])
    streaming_settings = dict(config["streaming"])
    streaming_settings.setdefault("seed", int(config.get("seed", 42)))
    streaming_settings["batch_size"] = batch_size
    dynamic_resume: DynamicResumeMetadata | None = None
    if resume_from_checkpoint and config.get("dynamic_vocabulary"):
        promotion_step = _resolve_dynamic_promotion_step(
            dict(config["dynamic_vocabulary"]), max_steps=max_steps
        )
        dynamic_resume = _dynamic_resume_preflight(
            Path(resume_from_checkpoint), promotion_step=promotion_step
        )
        streaming_settings["scheduler_start_batch"] = (
            _resolve_resume_scheduler_start_batch(
                streaming_settings,
                settings,
                global_step=dynamic_resume.global_step,
            )
        )
    elif resume_from_checkpoint:
        resume_start = _resolve_nondynamic_resume_scheduler_start_batch(
            config,
            streaming_settings,
            settings,
            Path(resume_from_checkpoint),
        )
        if resume_start is not None:
            streaming_settings["scheduler_start_batch"] = resume_start
    if bool(streaming_settings.get("smooth_weighted_round_robin", False)):
        workers = int(settings.get("dataloader_num_workers", 4))
        replicas = int(streaming_settings.get("worker_replicas", 0))
        if workers != 1 or replicas not in (0, 1):
            raise ValueError(
                "deterministic smooth scheduling requires one DataLoader worker "
                "and at most one worker replica"
            )
    dataset, shard_counts = build_streaming_dataset(config["streams"], roots, streaming_settings)
    load_path = Path(resume_from_checkpoint) if resume_from_checkpoint else model_path
    model = SentenceTransformer(str(load_path))
    if resume_from_checkpoint and bool(settings.get("save_model_bf16", False)):
        _cast_bf16_resume_model_to_master_float(model, Path(resume_from_checkpoint))
    module = model[0]
    if config.get("loss_path_vocabulary") and config.get("dynamic_vocabulary"):
        raise ValueError(
            "loss_path_vocabulary and legacy dynamic_vocabulary are mutually exclusive"
        )
    if bool(settings.get("disable_dynamic_pairs", False)):
        if (
            getattr(module, "mode", "") != "collapse_dynamic"
            or module.collapse.pair_keys.numel()
        ):
            raise ValueError(
                "disable_dynamic_pairs requires an unmaterialized collapse_dynamic model"
            )
        with torch.no_grad():
            module.collapse.score.fill_(-30.0)
            module.collapse.language_bias.zero_()
            module.collapse.merged.zero_()
        for parameter in (
            module.collapse.score,
            module.collapse.language_bias,
            module.collapse.merged,
        ):
            parameter.requires_grad_(False)
        print("dynamic pair path disabled for the fixed-tokenizer control", flush=True)
    if resume_from_checkpoint:
        disabled_empty = _disable_empty_trainable_parameters(model)
        if disabled_empty:
            print(
                "resume optimizer compatibility: excluded empty parameters "
                + ", ".join(disabled_empty),
                flush=True,
            )
    _validate_online_tokenizer_config(config, module)
    vocabulary_growth_settings = _resolve_vocabulary_growth_settings(config, module)
    recursive_vocabulary_settings = _resolve_recursive_vocabulary_settings(
        config, module, max_steps=max_steps
    )
    vocabulary_telemetry_settings = validate_vocabulary_telemetry_config(
        config, module
    )
    vocabulary_window: VocabularyWindow | None = None
    if vocabulary_telemetry_settings is not None:
        vocabulary_window = VocabularyWindow()
        telemetry_table = (
            module.online_collapse
            if vocabulary_telemetry_settings.arm == "online"
            else module.collapse
        )
        telemetry_table.set_vocabulary_telemetry(vocabulary_window)
    max_tokens = int(settings.get("max_tokens", 0))
    if max_tokens:
        _configure_tokenizer_truncation(model[0].tokenizer, max_tokens=max_tokens)
        print(f"training tokenizer truncation: {max_tokens} tokens", flush=True)
    loss_settings = dict(config.get("loss") or {})
    openlex_replay: Any | None = None
    openlex_replay_identity: dict[str, object] | None = None
    if openlex_replay_settings is None:
        loss: torch.nn.Module = _build_contrastive_loss(model, loss_settings)
    else:
        openlex_replay, openlex_replay_identity = _build_openlex_replay(
            model,
            config,
            openlex_replay_settings,
        )
        loss = openlex_replay
        if resume_from_checkpoint:
            from starse.openlex_replay_checkpoint import (
                load_openlex_replay_checkpoint,
            )

            checkpoint = Path(resume_from_checkpoint)
            load_openlex_replay_checkpoint(
                checkpoint,
                replay=openlex_replay,
                expected_identity=openlex_replay_identity,
                expected_step=_checkpoint_global_step(
                    checkpoint,
                    label="OpenLex replay",
                ),
            )
    loss_multiplier = float(loss_settings.get("multiplier", 1.0))
    if loss_multiplier != 1.0:
        loss = ScaledLoss(loss, loss_multiplier)

    bf16 = bool(settings.get("bf16", True)) and not disable_bf16
    workers = int(settings.get("dataloader_num_workers", 4))
    report_to = configure_comet(config, comet_enabled)
    stable_gradient_norm_target = stable_gradient_target(config)
    evaluation_settings = dict(config.get("evaluation") or {})
    rubq_settings_raw = dict(evaluation_settings.get("rubq_retrieval") or {})
    rubq_callback: TrainerCallback | None = None
    if bool(rubq_settings_raw.get("enabled", False)):
        from starse.mteb_evaluation import (
            PeriodicMTEBEvaluationCallback,
            PeriodicMTEBSettings,
        )

        rubq_callback = PeriodicMTEBEvaluationCallback(
            PeriodicMTEBSettings.from_config(
                rubq_settings_raw,
                training_settings=settings,
                output_dir=output_dir,
            )
        )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        run_name=str(config["run_name"]),
        max_steps=max_steps,
        num_train_epochs=1,
        learning_rate=float(settings["learning_rate"]),
        max_grad_norm=(
            0.0
            if stable_gradient_norm_target is not None
            else float(settings.get("max_grad_norm", 1.0))
        ),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=int(settings.get("gradient_accumulation_steps", 1)),
        warmup_steps=int(settings.get("warmup_steps", 0)),
        weight_decay=float(settings.get("weight_decay", 0.0)),
        lr_scheduler_type=str(settings.get("lr_scheduler_type", "constant_with_warmup")),
        eval_strategy="no",
        save_strategy=str(settings.get("save_strategy", "steps")),
        save_steps=int(settings.get("save_steps", 5000)),
        save_total_limit=int(settings.get("save_total_limit", 3)),
        logging_steps=int(settings.get("logging_steps", 10)),
        report_to=report_to,
        bf16=bf16,
        dataloader_drop_last=True,
        dataloader_num_workers=workers,
        dataloader_prefetch_factor=int(settings.get("dataloader_prefetch_factor", 2)) if workers > 0 else None,
        dataloader_persistent_workers=workers > 0,
        dataloader_pin_memory=True,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        remove_unused_columns=False,
        ignore_data_skip=True,
        accelerator_config={"dispatch_batches": False},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {
        "run_name": config["run_name"],
        "seed": int(config.get("seed", 42)),
        "streams": [
            {
                "name": stream["name"],
                "weight": stream.get("weight", 1.0),
                "anchor_column": stream["anchor_column"],
                "positive_column": stream["positive_column"],
                "parquet_files": shard_counts[str(stream["name"])],
                "language_sampling": stream.get("language_sampling"),
            }
            for stream in config["streams"]
        ],
        "streaming": streaming_settings,
        "loss": loss_settings,
        "dynamic_vocabulary": dict(config.get("dynamic_vocabulary") or {}),
        "vocabulary_growth": dict(config.get("vocabulary_growth") or {}),
        "recursive_vocabulary": dict(config.get("recursive_vocabulary") or {}),
        "online_tokenizer": dict(config.get("online_tokenizer") or {}),
        "openlex_replay": dict(config.get("openlex_replay") or {}),
        "vocabulary_telemetry": dict(config.get("vocabulary_telemetry") or {}),
        "evaluation": evaluation_settings,
        "training": {
            **settings,
            "max_steps": max_steps,
            "per_device_train_batch_size": batch_size,
            "bf16": bf16,
        },
    }
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(public_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Translation-Edit Invariance. A full-sentence bitext loss constrains only the
    # *sum* of the lexical differences in a pair, so individual errors can cancel;
    # an edit term holds the sentence fixed and swaps one aligned phrase, which
    # turns one parallel sentence into several independent local constraints.
    edit_weight = float(settings.get("edit_weight", 0.0))
    edits_path = settings.get("edits_path")
    if edit_weight and edits_path:
        from starse.translation_edit import TranslationEditLoss

        loss = TranslationEditLoss(
            loss, model[0], edits_path,
            weight=edit_weight,
            edits_per_step=int(settings.get("edits_per_step", 2048)),
            tangent=bool(settings.get("edit_tangent", False)),
            seed=int(config.get("seed", 42)),
        )
        print(f"translation edits: {loss.n_edits} in buffer, "
              f"{loss.edits_per_step} per step, weight {edit_weight}", flush=True)

    split_penalty = float(settings.get("split_penalty", 0.0))
    if split_penalty and (getattr(model[0], "split", None) is not None
                          or getattr(model[0], "assign", None) is not None):
        loss = SplitPenaltyLoss(loss, model[0], split_penalty)
        print(f"split gate L1 weight {split_penalty}", flush=True)

    # The shared table rewrites roughly 85% of a row's magnitude in 3,000 steps,
    # which lets it absorb any per-language transform a conditioning head could
    # learn and leaves the head redundant. These two knobs exist to break that
    # redundancy: freeze the table so the head is the only adaptive path, or give
    # the head its own learning rate and weight decay so it can move first.
    freeze_table = bool(settings.get("freeze_table", False))
    conditioner_lr_scale = float(settings.get("conditioner_lr_scale", 1.0))
    conditioner_weight_decay = settings.get("conditioner_weight_decay")

    checkpoint_run_config_callback = CheckpointRunConfigCallback(run_config_path)
    callbacks: list[TrainerCallback] = [checkpoint_run_config_callback]
    if rubq_callback is not None:
        callbacks.append(rubq_callback)
        print(
            "periodic evaluation: RuBQRetrieval at initialization and every "
            f"{rubq_settings_raw['eval_steps']} checkpoint steps",
            flush=True,
        )
    openlex_checkpoint_callback: TrainerCallback | None = None
    openlex_logging_callback: TrainerCallback | None = None
    if openlex_replay is not None:
        if openlex_replay_identity is None:
            raise RuntimeError("OpenLex replay identity was not constructed")
        from starse.openlex_replay_checkpoint import (
            OpenLexReplayCheckpointCallback,
            OpenLexReplayLoggingCallback,
        )

        openlex_checkpoint_callback = OpenLexReplayCheckpointCallback(
            replay=openlex_replay,
            output_dir=output_dir,
            identity=openlex_replay_identity,
        )
        openlex_logging_callback = OpenLexReplayLoggingCallback(
            openlex_replay,
            max_grad_norm=1.0,
        )
        callbacks.extend((openlex_checkpoint_callback, openlex_logging_callback))
    bounded_callback: BoundedVocabularyCallback | None = None
    growth_callback: AdaptiveVocabularyGrowthCallback | None = None
    recursive_growth_callback: RecursiveVocabularyGrowthCallback | None = None
    l0_callback: HardConcreteL0Callback | None = None
    online_invariant_callback: OnlineTokenizerInvariantCallback | None = None
    stable_gradient_callback: StableGradientCallback | None = None
    bf16_checkpoint_callback: BFloat16CheckpointCallback | None = None
    if (
        getattr(module, "mode", "") == "collapse_dynamic"
        and bool(config.get("dynamic_vocabulary"))
        and vocabulary_telemetry_settings is None
        and vocabulary_growth_settings is None
    ):
        raw_dynamic = dict(config.get("dynamic_vocabulary") or {})
        configured_trainable_languages = raw_dynamic.get("trainable_languages")
        if (
            configured_trainable_languages is not None
            and not isinstance(configured_trainable_languages, list)
        ):
            raise ValueError("dynamic trainable_languages must be a list")
        trainable_languages, trainable_language_indices = (
            _resolve_trainable_language_indices(
                list(getattr(module, "languages", ())),
                configured_trainable_languages,
            )
        )
        dynamic_settings = {
            "promotion_step": _resolve_dynamic_promotion_step(
                raw_dynamic, max_steps=max_steps
            ),
            "budget": int(raw_dynamic.get("budget", 32768)),
            "candidate_capacity": int(raw_dynamic.get("candidate_capacity", 131072)),
            "top_per_forward": int(raw_dynamic.get("top_per_forward", 4096)),
            "admission_mode": str(raw_dynamic.get("admission_mode", "utility")),
            "min_support": int(raw_dynamic.get("min_support", 32)),
            "min_utility": float(raw_dynamic.get("min_utility", 0.0)),
            "per_language_quota": int(raw_dynamic.get("per_language_quota", 1024)),
            "reservation_audit_quota": int(
                raw_dynamic.get(
                    "reservation_audit_quota",
                    raw_dynamic.get("per_language_quota", 1024),
                )
            ),
            "per_language_min_support": int(
                raw_dynamic.get("per_language_min_support", 8)
            ),
            "trainable_languages": trainable_languages,
            "trainable_language_indices": trainable_language_indices,
            "diagnostic_export_step": int(raw_dynamic.get("diagnostic_export_step", 0)),
            "diagnostic_export_only": bool(raw_dynamic.get("diagnostic_export_only", False)),
            "diagnostic_window_export_steps": tuple(
                int(value)
                for value in raw_dynamic.get("diagnostic_window_export_steps", ())
            ),
            "external_selection": dict(raw_dynamic.get("external_selection") or {}),
            "resume_boundary_promotion": bool(
                raw_dynamic.get("resume_boundary_promotion", False)
            ),
        }
        if not 0 < dynamic_settings["budget"] <= module.ngram_buckets:
            raise ValueError("dynamic budget must fit the initialized collapse table")
        if dynamic_settings["candidate_capacity"] < dynamic_settings["budget"]:
            raise ValueError("dynamic candidate_capacity must be at least the budget")
        if dynamic_settings["admission_mode"] not in {"utility", "frequency"}:
            raise ValueError("dynamic admission_mode must be utility or frequency")
        if (
            not math.isfinite(dynamic_settings["min_utility"])
            or dynamic_settings["min_utility"] < 0
        ):
            raise ValueError("dynamic min_utility must be finite and non-negative")
        if dynamic_settings["reservation_audit_quota"] < 0:
            raise ValueError("dynamic reservation_audit_quota must be non-negative")
        external_selection = dynamic_settings["external_selection"]
        if external_selection:
            required = {"protocol", "path", "sha256", "selector", "expected_rows"}
            if set(external_selection) != required:
                raise ValueError("external-selection settings must contain exact schema")
            if external_selection["protocol"] != "frozen-external-vocabulary-v1":
                raise ValueError("unknown external-selection protocol")
            if int(external_selection["expected_rows"]) != dynamic_settings["budget"]:
                raise ValueError("external-selection rows must equal dynamic budget")
            digest = str(external_selection["sha256"])
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("external-selection sha256 must be a full digest")
            selection_path = Path(external_selection["path"])
            if not selection_path.is_file():
                raise ValueError("external-selection CSV is missing")
            if _sha256(selection_path) != digest:
                raise ValueError("external-selection preflight hash mismatch")
            if not str(external_selection["selector"]).strip():
                raise ValueError("external-selection selector must be non-empty")
            if dynamic_settings["diagnostic_export_only"]:
                raise ValueError("external selection cannot be diagnostic-export-only")
        if dynamic_settings["resume_boundary_promotion"]:
            if (
                dynamic_resume is None
                or not external_selection
                or dynamic_resume.post_promotion
                or dynamic_resume.global_step + 1
                != dynamic_settings["promotion_step"]
            ):
                raise ValueError(
                    "resume_boundary_promotion requires an external selection and a "
                    "checkpoint immediately before promotion"
                )
        export_step = dynamic_settings["diagnostic_export_step"]
        if export_step and not 0 < export_step < dynamic_settings["promotion_step"]:
            raise ValueError("diagnostic_export_step must precede dynamic promotion")
        if dynamic_settings["diagnostic_export_only"] and not export_step:
            raise ValueError("diagnostic_export_only requires diagnostic_export_step")
        if (
            dynamic_settings["admission_mode"] == "frequency"
            and not dynamic_settings["diagnostic_export_only"]
        ):
            raise ValueError(
                "frequency admission is an export-only selector collector"
            )
        window_steps = dynamic_settings["diagnostic_window_export_steps"]
        if window_steps:
            if (tuple(sorted(set(window_steps))) != window_steps
                    or not all(0 < step <= export_step for step in window_steps)
                    or window_steps[-1] != export_step):
                raise ValueError(
                    "diagnostic window steps must be unique, increasing, end at export_step"
                )
        if not module.collapse.pair_keys.numel():
            discovery_kwargs = {
                "candidate_capacity": dynamic_settings["candidate_capacity"],
                "top_per_forward": dynamic_settings["top_per_forward"],
            }
            if dynamic_settings["admission_mode"] != "utility":
                discovery_kwargs["admission_mode"] = dynamic_settings[
                    "admission_mode"
                ]
            module.configure_dynamic_vocabulary_discovery(
                **discovery_kwargs,
            )
            if window_steps:
                module.configure_dynamic_vocabulary_diagnostic_discovery(
                    candidate_capacity=dynamic_settings["candidate_capacity"],
                    top_per_forward=dynamic_settings["top_per_forward"],
                    admission_mode=dynamic_settings["admission_mode"],
                )
        if dynamic_resume is not None:
            promoted = bool(module.collapse.pair_keys.numel())
            if dynamic_resume.post_promotion != promoted:
                raise ValueError(
                    "dynamic resume checkpoint structure disagrees with promotion step"
                )
            if dynamic_resume.discovery_sidecar is not None:
                if module.collapse.discovery is None:
                    raise ValueError("pre-promotion resume did not configure discovery")
                discovery_state = torch.load(
                    dynamic_resume.discovery_sidecar,
                    map_location="cpu",
                    weights_only=True,
                )
                module.collapse.discovery.load_state_dict(discovery_state)
        raw_reclaimed = dict(config.get("reclaimed_vocabulary") or {})
        reclaimed_settings = None
        if raw_reclaimed:
            if raw_reclaimed.get("protocol") != "q21c-direct-reclaimed-promotion-v1":
                raise ValueError("unknown reclaimed-vocabulary training protocol")
            if not raw_reclaimed.get("direct_promotion"):
                raise ValueError("Q21c requires direct_promotion=true")
            inventory_root = Path(raw_reclaimed["inventory"])
            if not (inventory_root / "manifest.json").is_file():
                raise ValueError("Q21c reclaim inventory is missing")
            reclaimed_settings = raw_reclaimed
        bounded_callback = BoundedVocabularyCallback(
            module=module,
            output_dir=output_dir,
            settings=dynamic_settings,
            reclaimed_settings=reclaimed_settings,
        )
        callbacks.append(bounded_callback)
        print(
            "bounded dynamic vocabulary: "
            f"promote at step {dynamic_settings['promotion_step']}, "
            f"budget {dynamic_settings['budget']}, "
            f"minimum utility {dynamic_settings['min_utility']}, "
            f"admission {dynamic_settings['admission_mode']}, "
            f"candidate capacity {dynamic_settings['candidate_capacity']}",
            flush=True,
        )
    if vocabulary_growth_settings is not None:
        module.configure_dynamic_vocabulary_discovery(
            candidate_capacity=int(vocabulary_growth_settings["candidate_capacity"]),
            top_per_forward=int(vocabulary_growth_settings["top_per_forward"]),
        )
        resume_sidecar = None
        resume_step = None
        if resume_from_checkpoint:
            checkpoint = Path(resume_from_checkpoint)
            resume_step = _checkpoint_global_step(
                checkpoint, label="adaptive vocabulary growth"
            )
            resume_sidecar = checkpoint / "vocabulary_growth_state.pt"
        growth_callback = AdaptiveVocabularyGrowthCallback(
            module=module,
            output_dir=output_dir,
            settings=vocabulary_growth_settings,
            resume_sidecar=resume_sidecar,
            resume_step=resume_step,
        )
        if vocabulary_growth_settings["policy"] == "l0":
            l0 = vocabulary_growth_settings["l0"]
            l0_callback = HardConcreteL0Callback(
                module,
                weight=float(l0["weight"]),
                beta=float(l0["beta"]),
                gamma=float(l0["gamma"]),
                zeta=float(l0["zeta"]),
            )
            callbacks.append(l0_callback)
        callbacks.append(growth_callback)
        print(
            f"adaptive vocabulary growth: {vocabulary_growth_settings['policy']}, "
            f"every {vocabulary_growth_settings['interval_steps']} steps, "
            f"residual rows {vocabulary_growth_settings['residual_buckets']}",
            flush=True,
        )
    if recursive_vocabulary_settings is not None:
        module.configure_recursive_vocabulary_discovery()
        resume_sidecar = None
        resume_step = None
        if resume_from_checkpoint:
            checkpoint = Path(resume_from_checkpoint)
            resume_step = _checkpoint_global_step(
                checkpoint, label="recursive vocabulary"
            )
            resume_sidecar = checkpoint / "recursive_vocabulary_state.pt"
        recursive_growth_callback = RecursiveVocabularyGrowthCallback(
            module=module,
            output_dir=output_dir,
            settings=recursive_vocabulary_settings,
            resume_sidecar=resume_sidecar,
            resume_step=resume_step,
        )
        callbacks.append(recursive_growth_callback)
        print(
            "recursive cascade vocabulary: "
            f"every {recursive_vocabulary_settings['interval_steps']} steps, "
            "two stable half-windows, "
            f"protocol {recursive_vocabulary_settings['protocol']}, "
            f"max span {recursive_vocabulary_settings['max_span_length']}",
            flush=True,
        )
    if getattr(module, "mode", "") == "collapse_online":
        if stable_gradient_norm_target is None:
            online_invariant_callback = OnlineTokenizerInvariantCallback(module)
            callbacks.append(online_invariant_callback)
        print(
            "online tokenizer: structured-st-v1, "
            f"rank {module.online_collapse.rank}, exact MAP every optimizer step",
            flush=True,
        )
    if stable_gradient_norm_target is not None:
        stable_gradient_callback = StableGradientCallback(
            module, max_norm=stable_gradient_norm_target
        )
        callbacks.append(stable_gradient_callback)
        print(
            "stable gradient clipping: target global norm 1.0, "
            "Trainer clipping disabled",
            flush=True,
        )
    if bool(settings.get("save_model_bf16", False)):
        if not bf16:
            raise ValueError("BF16 checkpoint persistence requires BF16 training")
        bf16_checkpoint_callback = BFloat16CheckpointCallback(
            output_dir,
            keep_resume_checkpoints=int(
                settings.get("keep_resume_checkpoints", 1)
            ),
        )
        callbacks.append(bf16_checkpoint_callback)
        print(
            "checkpoint persistence: BF16 model snapshots, newest-only resume state",
            flush=True,
        )
    if freeze_table and hasattr(module, "embedding"):
        module.embedding.weight.requires_grad_(False)
        print("table frozen: only the conditioning head will train", flush=True)
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError(
                "freeze_table left nothing trainable — this mode has no conditioning "
                "head, so the run would just reproduce the initialization")

    optimizers = (None, None)
    if openlex_replay is not None:
        if conditioner_lr_scale != 1.0 or conditioner_weight_decay is not None:
            raise ValueError("OpenLex replay does not support conditioner optimizer overrides")
        optimizer = _build_openlex_optimizer(
            model=model,
            replay=openlex_replay,
            training_settings=settings,
            replay_settings=openlex_replay_settings,
        )
        optimizers = (optimizer, None)
        print(
            "OpenLex CVR: exact q15 table group plus detached policy group, "
            f"sample {openlex_replay_settings['sample_pairs']} aligned pairs",
            flush=True,
        )
    elif conditioner_lr_scale != 1.0 or conditioner_weight_decay is not None:
        head, rest = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (head if "conditioner." in name else rest).append(parameter)
        base_lr = float(settings["learning_rate"])
        groups = [{"params": rest, "lr": base_lr,
                   "weight_decay": float(settings["weight_decay"])}]
        if head:
            groups.append({
                "params": head,
                "lr": base_lr * conditioner_lr_scale,
                "weight_decay": float(settings["weight_decay"])
                if conditioner_weight_decay is None else float(conditioner_weight_decay),
            })
        optimizer = torch.optim.AdamW(groups)
        print(f"conditioner group: lr x{conditioner_lr_scale}, "
              f"wd {groups[-1]['weight_decay'] if head else 'n/a'}", flush=True)
        optimizers = (optimizer, None)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        optimizers=optimizers,
        callbacks=callbacks,
    )
    if openlex_logging_callback is not None:
        trainer.callback_handler.callbacks.remove(openlex_logging_callback)
        trainer.callback_handler.callbacks.insert(0, openlex_logging_callback)
    vocabulary_coordinator: VocabularyTelemetryCoordinator | None = None
    if vocabulary_telemetry_settings is not None:
        if vocabulary_window is None or stable_gradient_callback is None:
            raise RuntimeError("vocabulary telemetry trainer wiring is incomplete")
        vocabulary_coordinator = VocabularyTelemetryCoordinator(
            module=module,
            window=vocabulary_window,
            gradient_callback=stable_gradient_callback,
            output_path=output_dir / "vocabulary_metrics.jsonl",
            arm=vocabulary_telemetry_settings.arm,
            config_sha256=canonical_config_sha256(config),
            corpus_manifest_sha256=(
                vocabulary_telemetry_settings.corpus_manifest_sha256
            ),
            merge_sample_rows=vocabulary_telemetry_settings.merge_sample_rows,
            merge_sample_seed=vocabulary_telemetry_settings.merge_sample_seed,
            growth_callback=growth_callback,
        )
        install_vocabulary_logging_callback(
            trainer,
            VocabularyLoggingCallback(
                vocabulary_coordinator,
                interval_steps=vocabulary_telemetry_settings.interval_steps,
            ),
        )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    loss_path_report = None
    if config.get("loss_path_vocabulary"):
        loss_path_report = _run_loss_path_vocabulary_audit(
            trainer=trainer,
            loss=loss,
            module=module,
            config=config,
            roots=roots,
            streaming_settings=streaming_settings,
            output_dir=output_dir,
            completed_steps=max_steps,
        )
    compact_report = None
    if (
        vocabulary_growth_settings is None
        and getattr(module, "mode", "") == "collapse_dynamic"
        and module.collapse.pair_keys.numel()
    ):
        compact_report = module.compact_dynamic_vocabulary()
        (output_dir / "dynamic_vocabulary_compaction.json").write_text(
            json.dumps(compact_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"dynamic vocabulary compacted {compact_report['old_rows']} -> "
            f"{compact_report['active_rows']} rows before final save",
            flush=True,
        )
    if save_final_model:
        trainer.save_model()
        if bool(settings.get("save_model_bf16", False)):
            report = rewrite_safetensors_bf16(output_dir / "model.safetensors")
            (output_dir / "final_model_precision.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    metrics = dict(result.metrics)
    if torch.cuda.is_available():
        metrics["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if compact_report is not None:
        metrics["dynamic_vocabulary_active_rows"] = compact_report["active_rows"]
        metrics["dynamic_vocabulary_dropped_rows"] = compact_report["dropped_rows"]
    if loss_path_report is not None:
        metrics["loss_path_learned_rows"] = loss_path_report["learned_rows"]
        metrics["loss_path_support_rows"] = loss_path_report["support_rows"]
    if recursive_growth_callback is not None:
        metrics["recursive_vocabulary_generation"] = (
            module.recursive_cascade.generation
        )
        metrics["recursive_vocabulary_learned_rows"] = (
            module.recursive_cascade.learned_count
        )
        metrics["recursive_vocabulary_active_rows"] = (
            module.recursive_cascade.active_count
        )
        metrics["recursive_vocabulary_consecutive_empty_transitions"] = (
            recursive_growth_callback.consecutive_empty_transitions
        )
    if bounded_callback is not None and bounded_callback.memory_report is not None:
        metrics["reclaimed_vocabulary_memory"] = {
            **bounded_callback.memory_report,
            "postpromotion_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            ),
            "postpromotion_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
            ),
        }
    if online_invariant_callback is not None:
        module.online_collapse.assert_cache_audit_empty()
        metrics["online_tokenizer_completed_cache_audits"] = (
            online_invariant_callback.completed_audits
        )
    if stable_gradient_callback is not None:
        if getattr(module, "mode", "") == "collapse_online":
            module.online_collapse.assert_cache_audit_empty()
            metrics["online_tokenizer_completed_cache_audits"] = (
                stable_gradient_callback.completed_cache_audits
            )
        metrics["stable_gradient_completed_steps"] = (
            stable_gradient_callback.completed_steps
        )
    metrics["checkpoint_run_config_completed_saves"] = (
        checkpoint_run_config_callback.completed_saves
    )
    if rubq_callback is not None:
        metrics["rubq_retrieval_evaluated_steps"] = sorted(
            int(step) for step in rubq_callback._evaluated_steps
        )
    if openlex_checkpoint_callback is not None:
        metrics["openlex_replay_completed_saves"] = int(
            openlex_checkpoint_callback.completed_saves
        )
    (output_dir / "train_result.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fineweb-root", type=Path)
    parser.add_argument("--parallel-root", type=Path)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named dataset root; repeat for configs with more than two sources",
    )
    parser.add_argument("--model", required=True, help="Initialized StaticEmbedding model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--comet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable only for disposable profiling runs; normal experiments must save their final model",
    )
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    if args.print_config:
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        return 0
    roots = _parse_root_assignments(
        args.root,
        fineweb_root=args.fineweb_root,
        parallel_root=args.parallel_root,
    )
    metrics = train(
        config=config,
        roots=roots,
        model_path=args.model,
        output_dir=args.output_dir.resolve(),
        max_steps_override=args.max_steps,
        batch_size_override=args.batch_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
        comet_enabled=args.comet,
        disable_bf16=args.no_bf16,
        save_final_model=args.save_final_model,
    )
    print(json.dumps({"status": "ok", "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
