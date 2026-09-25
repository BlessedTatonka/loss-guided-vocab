#!/usr/bin/env python3
"""Select the k most frequent candidate pairs exported by a frequency collector run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SELECTOR = "independent_frequency_count_min"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="dynamic_vocabulary_candidates.csv")
    parser.add_argument("--k", type=int, required=True, help="number of pairs to select")
    parser.add_argument("--output", type=Path, required=True, help="selection CSV for materialize_frequency_control.py")
    args = parser.parse_args()

    with args.candidates.open(encoding="utf-8", newline="") as handle:
        rows = [(int(row["pair_key"]), int(row["captured_support"])) for row in csv.DictReader(handle)]
    ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
    selected = ranked[: args.k]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pair_key", "selector", "support_estimate"))
        writer.writeheader()
        for key, support in sorted(selected):
            writer.writerow({"pair_key": key, "selector": SELECTOR, "support_estimate": support})
    report = {
        "candidate_rows": len(rows),
        "selected_rows": len(selected),
        "min_selected_support": selected[-1][1] if selected else None,
        "first_excluded_support": ranked[args.k][1] if len(ranked) > args.k else None,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
