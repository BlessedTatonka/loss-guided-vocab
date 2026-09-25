#!/usr/bin/env python3
"""Reproducible end-to-end encoding cost for the q15 static encoders.

The benchmark includes tokenisation, language-marker handling, pooling,
normalisation, and transfer of the returned embeddings to NumPy. Every model is
run over exactly the same deterministically selected held-out texts. Quality is
not measured here; the held-out file is only a convenient immutable text source.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from starse.conditioned_module import marker_for


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("models must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("models must use non-empty NAME=PATH")
    path = Path(raw_path).resolve()
    if not (path / "model.safetensors").is_file():
        raise argparse.ArgumentTypeError(f"{path} has no model.safetensors")
    return name, path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_sample(path: Path, per_language: int) -> tuple[list[str], dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    texts: list[str] = []
    languages: list[str] = []
    for record in records:
        language = str(record["lang"])
        selected = list(record["s1"][:per_language])
        if len(selected) != per_language:
            raise ValueError(
                f"{language} has only {len(selected)} texts, requested {per_language}"
            )
        texts.extend(f"{marker_for(language)} {text}" for text in selected)
        languages.append(language)
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return texts, {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "selection": "first s1 texts in file order",
        "per_language": per_language,
        "languages": languages,
        "n_texts": len(texts),
        "selected_texts_sha256": hashlib.sha256(payload).hexdigest(),
    }


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "mean_seconds": statistics.fmean(values),
        "sample_sd_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_seconds": statistics.median(values),
        "p10_seconds": quantile(0.10),
        "p90_seconds": quantile(0.90),
    }


def model_inventory(model: SentenceTransformer, path: Path) -> dict[str, Any]:
    parameters = list(model.parameters())
    module = model[0]
    collapse = getattr(module, "collapse", None)
    state_path = path / "model.safetensors"
    root_files = [item for item in path.iterdir() if item.is_file()]
    result: dict[str, Any] = {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size()
                               for parameter in parameters),
        "model_safetensors_bytes": state_path.stat().st_size,
        "model_safetensors_sha256": sha256_file(state_path),
        "root_files_bytes": sum(item.stat().st_size for item in root_files),
        "root_file_names": sorted(item.name for item in root_files),
        "mode": getattr(module, "mode", None),
    }
    embedding = getattr(module, "embedding", None)
    if embedding is not None:
        result["base_embedding_shape"] = list(embedding.weight.shape)
    if collapse is not None:
        result.update({
            "auxiliary_rows": int(collapse.merged.shape[0]),
            "auxiliary_dimension": int(collapse.merged.shape[1]),
            "exact_pair_keys": int(collapse.pair_keys.numel()),
            "pair_scores": int(collapse.score.numel()),
            "reclaimed_base_rows": int(collapse.reclaimed_token_ids.numel()),
        })
    return result


def benchmark_model(
    name: str,
    path: Path,
    texts: list[str],
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
    device: str,
) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = SentenceTransformer(str(path), device=device)
    model.eval()
    inventory = model_inventory(model, path)
    rows = []
    for batch_size in batch_sizes:
        for _ in range(warmup):
            model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(repeats):
            started = time.perf_counter()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
        stats = distribution(elapsed)
        rows.append({
            "batch_size": batch_size,
            "warmup_repetitions": warmup,
            "measured_repetitions": repeats,
            "n_texts_per_repetition": len(texts),
            **stats,
            "median_texts_per_second": len(texts) / stats["median_seconds"],
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.startswith("cuda") else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.startswith("cuda") else None
            ),
        })
        del embeddings
    result = {"name": name, "path": str(path), "inventory": inventory, "latency": rows}
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for model in report["models"]:
        inventory = model["inventory"]
        for latency in model["latency"]:
            rows.append({
                "model": model["name"],
                "path": model["path"],
                "mode": inventory["mode"],
                "total_parameters": inventory["total_parameters"],
                "parameter_bytes": inventory["parameter_bytes"],
                "model_safetensors_bytes": inventory["model_safetensors_bytes"],
                "auxiliary_rows": inventory.get("auxiliary_rows", 0),
                "exact_pair_keys": inventory.get("exact_pair_keys", 0),
                **latency,
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="Repeat NAME=PATH for every model")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=64)
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    batch_sizes = args.batch_sizes or [1, 64, 512]
    if args.per_language <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("per-language and repeats must be positive; warmup must be non-negative")
    if any(value <= 0 for value in batch_sizes):
        parser.error("batch sizes must be positive")
    names = [name for name, _ in args.model]
    if len(names) != len(set(names)):
        parser.error("model names must be unique")

    texts, sample = frozen_sample(args.pairs, args.per_language)
    report = {
        "protocol": "q15-static-inference-cost-v1",
        "timer": "time.perf_counter with CUDA synchronization",
        "scope": "tokenization + marker routing + pooling + normalization + NumPy return",
        "sample": sample,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "device": args.device,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(args.device)
                    if args.device.startswith("cuda") else None),
        },
        "models": [],
    }
    for name, path in args.model:
        print(f"benchmarking {name}: {path}", flush=True)
        report["models"].append(benchmark_model(
            name, path, texts, batch_sizes, args.warmup, args.repeats, args.device
        ))
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "inference_cost.json"
    csv_path = args.output / "inference_cost.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    write_csv(csv_path, report)
    print(f"wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
