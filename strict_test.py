"""Re-run the independent N17/N18 test with a subject-excluding cache."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "results" / "holdout" / "artifacts_N17_N18"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing the public .mat sessions")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "user_strict_test")
    parser.add_argument("--classifier", type=Path, default=ROOT / "classifier.py",
                        help="standalone classifier file; default is the tested submission")
    args = parser.parse_args()
    data = args.data.resolve()
    expected = [p for p in data.glob("*.mat") if p.stem.split("_")[-1] in {"N17", "N18"}]
    if len(expected) != 8:
        parser.error(f"Expected eight N17/N18 sessions, found {len(expected)}")
    if not args.classifier.is_file():
        parser.error(f"Classifier file not found: {args.classifier}")
    with (ARTIFACT / "training_manifest.csv").open(newline="", encoding="utf-8") as f:
        excluded = {row["subject"] for row in csv.DictReader(f)}
    if excluded & {"N17", "N18"}:
        parser.error("The held-out people are present in the training cache")
    subprocess.run([sys.executable, str(ROOT / "evaluate_many.py"), "--data", str(data),
                    "--subjects", "N17", "N18", "--classifier", str(args.classifier.resolve()),
                    "--artifacts", str(ARTIFACT), "--out", str(args.out.resolve())], check=True)


if __name__ == "__main__":
    main()
