"""Compare two complete runs on the same held-out sessions.

Example:
    python compare_results.py --before results/holdout/final_selected_N17_N18 \
        --after results/my_extra_trees
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_summary(folder: Path) -> dict[str, dict[str, str]]:
    with (folder / "summary.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or len({row["session"] for row in rows}) != len(rows):
        raise ValueError(f"Missing or repeated session rows in {folder}")
    return {row["session"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    before = read_summary(args.before)
    after = read_summary(args.after)
    if before.keys() != after.keys():
        raise ValueError("Results must contain exactly the same sessions")
    print("Session                          accuracy change    min-F1 change    score change")
    print("-" * 78)
    differences: dict[str, list[float]] = {"accuracy": [], "min_f1": [], "score": []}
    for session in sorted(before):
        delta = {key: float(after[session][key]) - float(before[session][key])
                 for key in differences}
        for key, value in delta.items():
            differences[key].append(value)
        print(f"{session:32} {delta['accuracy'] * 100:+8.2f} pp"
              f"       {delta['min_f1']:+8.4f}       {delta['score']:+8.4f}")
    print("-" * 78)
    n = len(before)
    print(f"Mean across {n} identical sessions: "
          f"{sum(differences['accuracy']) / n * 100:+.2f} pp accuracy, "
          f"{sum(differences['min_f1']) / n:+.4f} min-F1, "
          f"{sum(differences['score']) / n:+.4f} points")


if __name__ == "__main__":
    main()
