#!/usr/bin/env python3
"""Run a training configuration with named data roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starse.train_multilingual import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-comet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = json.loads(args.roots.read_text(encoding="utf-8"))
    if not isinstance(roots, dict) or not roots:
        raise ValueError("--roots must contain a non-empty JSON object")

    argv = [
        "--config",
        str(args.config),
        "--model",
        args.model,
        "--output-dir",
        str(args.output_dir),
    ]
    for name, raw_path in sorted(roots.items()):
        if not isinstance(name, str) or not isinstance(raw_path, str):
            raise TypeError("every data-root name and path must be a string")
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"dataset root is missing: {name}={path}")
        argv.extend(("--root", f"{name}={path.resolve()}"))
    if args.resume_from_checkpoint:
        argv.extend(("--resume-from-checkpoint", args.resume_from_checkpoint))
    if args.max_steps is not None:
        argv.extend(("--max-steps", str(args.max_steps)))
    if args.no_comet:
        argv.append("--no-comet")
    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
