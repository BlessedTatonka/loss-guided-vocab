#!/usr/bin/env python3
"""Promote a given pair selection (e.g. the most frequent pairs) into exact rows of a collector model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-model", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selector", default="independent_frequency_count_min")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    discovery_model = args.discovery_model.resolve()
    selection = args.selection.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    with selection.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("selector") != args.selector for row in rows):
        raise ValueError("selection is empty or has the wrong selector")
    keys = sorted(int(row["pair_key"]) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("selection contains duplicate pair keys")
    if args.expected_rows is not None and len(keys) != args.expected_rows:
        raise ValueError(f"selection must contain {args.expected_rows} unique pair keys")

    model = SentenceTransformer(str(discovery_model))
    module = model[0]
    if getattr(module, "mode", "") != "collapse_dynamic":
        raise ValueError("source model is not a dynamic discovery prefill")
    if module.collapse.pair_keys.numel():
        raise ValueError("source model has already promoted exact pair rows")
    transition = module.promote_dynamic_vocabulary(keys)
    compact = module.compact_dynamic_vocabulary()
    output.mkdir(parents=True)
    model.save(str(output), safe_serialization=True)
    report = {
        "protocol": "pair-selector-control-initializer-v1",
        "discovery_model": str(discovery_model),
        "discovery_model_sha256": sha256(discovery_model / "model.safetensors"),
        "selection": str(selection),
        "selection_sha256": sha256(selection),
        "selector": args.selector,
        "initialization": "inherited proposal vectors and logits",
        "rows": len(keys),
        "transition": transition,
        "compaction": compact,
        "model_sha256": sha256(output / "model.safetensors"),
    }
    (output / "branch_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
