#!/usr/bin/env python3
"""Paired comparison of two cross-lingual evaluation outputs over language-pair subsets.

Both directories must come from scripts/evaluate_crosslingual.py (raw_task_results/*.json).
Reports the mean difference A - B in percentage points, a paired bootstrap 95% interval,
and the share of subsets in which A is ahead.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

TASKS = ("Tatoeba", "BUCC.v2", "FloresBitextMining")


def load_subsets(result_dir: Path, task: str) -> dict[str, float]:
    payload = json.loads((result_dir / "raw_task_results" / f"{task}.json").read_text(encoding="utf-8"))
    entry = payload.get(task, payload)
    scores = entry["scores"]
    split = scores.get("test") or next(iter(scores.values()))
    return {row["hf_subset"]: float(row.get("main_score", row.get("f1"))) for row in split}


def paired(a: dict[str, float], b: dict[str, float], resamples: int, rng: random.Random) -> dict:
    keys = sorted(set(a) & set(b))
    diffs = [100.0 * (a[k] - b[k]) for k in keys]
    n = len(diffs)
    boot = sorted(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(resamples))
    return {
        "subsets": n,
        "mean_diff": statistics.fmean(diffs),
        "ci95": [boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]],
        "win_rate": sum(d > 0 for d in diffs) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    report = {}
    for task in args.tasks:
        rec = paired(load_subsets(args.a, task), load_subsets(args.b, task), args.resamples, rng)
        report[task] = rec
        lo, hi = rec["ci95"]
        print(f"{task:20s} n={rec['subsets']:3d}  diff={rec['mean_diff']:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  win={100 * rec['win_rate']:.0f}%")
    (args.a / f"paired_vs_{args.b.name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
