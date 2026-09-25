#!/usr/bin/env python3
"""Build the mmBERT static-table initializer used by the paper."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from starse.conditioned_module import build
from starse.init_model import initialize_model

BASE_MODELS = ("jhu-clsp/mmBERT-base", "jhu-clsp/mmBERT-small")
BASE_REVISIONS = (
    "c5955035435e2bf121cde7f3c8863ef52ff35d82",
    "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contextual-batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    languages = [
        line.strip()
        for line in (root / "configs/languages.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(languages) != 1_914 or len(languages) != len(set(languages)):
        raise ValueError("registered language-marker inventory must contain 1,914 unique entries")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    with tempfile.TemporaryDirectory(prefix="mrl-static-init-") as temporary:
        static_dir = Path(temporary) / "static"
        initialize_model(
            base_models=BASE_MODELS,
            base_revisions=BASE_REVISIONS,
            output_dir=static_dir,
            target_dim=1_024,
            sif_a=5e-5,
            zipf_exponent=1.0,
            trust_remote_code=False,
            representations=("input",),
            device=args.device,
            contextual_batch_size=args.contextual_batch_size,
        )
        module = build(
            static_dir,
            languages,
            "collapse_dynamic",
            ngram_buckets=262_144,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        module.save(str(output_dir))
        (output_dir / "initialization.json").write_bytes(
            (static_dir / "initialization.json").read_bytes()
        )


if __name__ == "__main__":
    main()

