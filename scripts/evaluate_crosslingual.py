#!/usr/bin/env python3
"""Run the frozen multilingual long-run suite with revision preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from scripts import run_mteb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def model_file_hashes(model: Path) -> dict[str, str]:
    model = model.resolve()
    modules_path = model / "modules.json"
    if not modules_path.is_file():
        raise FileNotFoundError(modules_path)
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    if not isinstance(modules, list) or any(
        not isinstance(record, dict) or not isinstance(record.get("path"), str)
        for record in modules
    ):
        raise ValueError("model modules.json has an invalid module inventory")

    files = {
        path
        for name in (
            "modules.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "tokenizer.json",
            "language_conditioning.json",
            "dynamic_vocabulary_manifest.json",
        )
        if (path := model / name).is_file()
    }
    for record in modules:
        relative_module = Path(record["path"])
        if not relative_module.parts:
            continue
        module = (model / relative_module).resolve()
        try:
            module.relative_to(model)
        except ValueError as error:
            raise ValueError("model module path escapes the model root") from error
        if not module.is_dir():
            raise FileNotFoundError(module)
        files.update(path for path in module.rglob("*") if path.is_file())
    return {
        path.relative_to(model).as_posix(): _sha256(path)
        for path in sorted(files)
    }


def load_static_checkpoint_for_evaluation(
    path: Path, device: str
) -> tuple[Any, dict[str, object]]:
    """Load a BF16 snapshot without replacing its quantized on-disk values."""

    from sentence_transformers import SentenceTransformer

    resolved = Path(path).expanduser().resolve()
    model_path = resolved / "model.safetensors"
    tensors = load_file(str(model_path), device="cpu")
    floating = [tensor for tensor in tensors.values() if tensor.is_floating_point()]
    if not floating or any(tensor.dtype != torch.bfloat16 for tensor in floating):
        raise ValueError("evaluation checkpoint floating tensors are not all BF16")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in floating):
        raise FloatingPointError("evaluation checkpoint contains nonfinite tensors")
    model = SentenceTransformer(str(resolved), device=device)
    compute_dtype = "model-default"
    if str(device).startswith("cpu"):
        model = model.float()
        compute_dtype = "float32"
    return model, {
        "stored_floating_dtype": "bfloat16",
        "evaluation_compute_dtype": compute_dtype,
        "floating_tensor_count": len(floating),
    }


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite.get("protocol") != "multilingual-sota-validation-v1":
        raise ValueError("unknown multilingual validation protocol")
    if not isinstance(suite.get("benchmark"), str) or not suite["benchmark"]:
        raise ValueError("validation suite has no benchmark")
    tasks = suite.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("validation suite has no tasks")
    excluded = suite.get("reference_clean_excludes", [])
    if not isinstance(excluded, list) or any(
        not isinstance(name, str) for name in excluded
    ):
        raise ValueError("reference_clean_excludes must be a list of task names")
    unknown_exclusions = sorted(set(excluded) - set(tasks))
    if unknown_exclusions:
        raise ValueError(
            f"reference_clean_excludes contains an unknown task: {unknown_exclusions}"
        )
    if excluded and not (set(tasks) - set(excluded)):
        raise ValueError("reference_clean_excludes cannot exclude every task")
    for name, record in tasks.items():
        if not {"type", "dataset", "revision"} <= set(record) or set(record) - {
            "type", "dataset", "revision", "subsets"
        }:
            raise ValueError(f"validation task {name!r} has an invalid schema")
        revision = str(record["revision"])
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"validation task {name!r} needs a full commit revision")
        if "subsets" in record:
            policy = record["subsets"]
            if set(policy) != {"policy", "suffix", "expected_count", "sha256"}:
                raise ValueError(f"validation task {name!r} has an invalid subset policy")
            if policy["policy"] != "target-to-english":
                raise ValueError(f"validation task {name!r} has an unknown subset policy")
            if int(policy["expected_count"]) <= 0 or len(str(policy["sha256"])) != 64:
                raise ValueError(f"validation task {name!r} has an invalid subset freeze")
    return suite


def compute_reporting_macros(
    suite: dict[str, Any], scores: dict[str, float]
) -> dict[str, dict[str, Any]]:
    task_names = list(suite["tasks"])
    excluded = list(suite.get("reference_clean_excludes", ()))
    clean_task_names = [name for name in task_names if name not in excluded]
    return {
        "all_tasks": {
            "tasks": task_names,
            "score": math.fsum(float(scores[name]) for name in task_names)
            / len(task_names),
        },
        "reference_clean": {
            "tasks": clean_task_names,
            "excluded": excluded,
            "score": math.fsum(float(scores[name]) for name in clean_task_names)
            / len(clean_task_names),
        },
    }


def verify_task_revisions(
    suite: dict[str, Any], tasks: list[Any]
) -> dict[str, dict[str, str]]:
    expected = dict(suite["tasks"])
    observed: dict[str, dict[str, str]] = {}
    for index, task in enumerate(tasks):
        name = run_mteb.task_name(task, index)
        metadata = task.metadata
        dataset = dict(metadata.dataset)
        record = {
            "type": str(metadata.type),
            "dataset": str(dataset.get("path", "")),
            "revision": str(dataset.get("revision", "")),
        }
        if name not in expected:
            raise ValueError(f"resolved an unregistered validation task: {name}")
        for key, value in record.items():
            if value != str(expected[name][key]):
                raise ValueError(f"validation {key} mismatch for {name!r}")
        observed[name] = record
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"validation tasks did not resolve: {missing}")
    return observed


def model_language_codes(model: Path) -> list[str]:
    path = model / "language_conditioning.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    languages = raw.get("languages")
    if not isinstance(languages, list) or not languages:
        raise ValueError("model language-conditioning manifest has no languages")
    return sorted({str(language).split("_", 1)[0] for language in languages})


def resolve_model_languages(
    model: Path, reference_languages: Path | None
) -> tuple[tuple[str, ...], bool, Path]:
    condition = run_mteb.conditioning_config(str(model))
    if condition is not None:
        source = (model / "language_conditioning.json").resolve()
        return tuple(str(language) for language in condition["languages"]), True, source
    if reference_languages is None:
        raise ValueError(
            "an unconditioned model needs an explicit reference language universe"
        )
    source = reference_languages.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    languages = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("reference language universe is empty or contains duplicates")
    return languages, False, source


def resolve_task_subset_overrides(
    suite: dict[str, Any], tasks: list[Any], model_languages: tuple[str, ...]
) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for index, task in enumerate(tasks):
        name = run_mteb.task_name(task, index)
        task_type = str(getattr(task.metadata, "type", ""))
        if task_type == run_mteb.BITEXT_MINING_TASK_TYPE:
            supported = run_mteb.filter_conditioned_bitext_subsets(
                task, model_languages
            )
        elif task_type == run_mteb.STS_TASK_TYPE:
            supported = run_mteb.filter_conditioned_sts_subsets(
                task, model_languages
            )
        else:
            raise ValueError(f"unsupported validation task type for {name!r}")
        policy = dict(suite["tasks"][name].get("subsets") or {})
        if not policy:
            overrides[name] = list(supported)
            continue
        selected = sorted(
            subset for subset in supported if subset.endswith(str(policy["suffix"]))
        )
        encoded = json.dumps(
            selected, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if len(selected) != int(policy["expected_count"]):
            raise ValueError(f"validation subset count mismatch for {name!r}")
        if digest != str(policy["sha256"]):
            raise ValueError(f"validation subset hash mismatch for {name!r}")
        overrides[name] = selected
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/eval/multilingual-sota-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--collapse-bias-ablation",
        choices=("none", "mean"),
        default="none",
        help=(
            "Evaluation-only removal of language-specific collapse routing. "
            "The mean intervention preserves the learned global merge tendency."
        ),
    )
    parser.add_argument(
        "--reference-languages",
        type=Path,
        help=(
            "Frozen marker universe used only to match subsets for an "
            "unconditioned baseline model"
        ),
    )
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = args.model.expanduser().resolve()
    suite_path = args.suite.expanduser().resolve()
    output = args.output.expanduser().resolve()
    suite = load_suite(suite_path)
    model_languages, conditioned_model, language_source = resolve_model_languages(
        model, args.reference_languages
    )
    language_codes = sorted(
        {str(language).split("_", 1)[0] for language in model_languages}
    )
    names = list(suite["tasks"])
    # FineWeb2's marker universe includes glottocodes and retired codes which
    # are valid model routing labels but intentionally rejected by MTEB's ISO
    # language filter.  Resolve the complete registered tasks here; run_mteb's
    # conditioned subset preflight then keeps only exact marker-covered pairs.
    resolved = run_mteb.resolve_tasks(str(suite["benchmark"]), names, ())
    revision_preflight = verify_task_revisions(suite, resolved)
    task_subset_overrides = resolve_task_subset_overrides(
        suite, resolved, model_languages
    )
    runner_args = argparse.Namespace(
        model=str(model),
        revision=None,
        output_dir=output,
        benchmark=str(suite["benchmark"]),
        tasks=",".join(names),
        tasks_file=None,
        languages="",
        subsets="",
        task_subsets=task_subset_overrides,
        batch_size=args.batch_size,
        truncate_dim=None,
        device=args.device,
        normalize_embeddings=True,
        collapse_bias_ablation=args.collapse_bias_ablation,
        trust_remote_code=True,
        overwrite=args.overwrite,
        fail_on_error=True,
        verbosity=0,
    )
    summary = run_mteb.run(runner_args)
    scores = dict(summary.get("scores") or {})
    if summary.get("status") != "ok" or set(scores) != set(names):
        raise RuntimeError("frozen validation suite did not complete exactly")
    if any(
        not isinstance(score, (int, float)) or not math.isfinite(float(score))
        for score in scores.values()
    ):
        raise RuntimeError("frozen validation suite produced a missing/non-finite score")
    reporting_macros = compute_reporting_macros(suite, scores)
    summary_path = output / "mteb_eval_summary.json"
    _atomic_json(
        output / "validation_manifest.json",
        {
            "protocol": "multilingual-sota-validation-result-v1",
            "suite": str(suite_path),
            "suite_sha256": _sha256(suite_path),
            "revision_preflight": revision_preflight,
            "model_language_codes": language_codes,
            "conditioned_model": conditioned_model,
            "collapse_bias_ablation": args.collapse_bias_ablation,
            "reference_languages": str(language_source),
            "reference_languages_sha256": _sha256(language_source),
            "task_subset_overrides": task_subset_overrides,
            "model": str(model),
            "model_sha256": model_file_hashes(model),
            "summary": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "scores": scores,
            "unweighted_task_macro": float(summary["mean_main_score"]),
            "reporting_macros": reporting_macros,
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
