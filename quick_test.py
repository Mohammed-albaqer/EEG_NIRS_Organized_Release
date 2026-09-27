"""Run the supplied sample through the frozen chronological runner and scorer."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "samples" / "data" / "2025.12.01.16.04.01_N01.mat"
ARTIFACT = ROOT / "samples" / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "quick_test")
    args = parser.parse_args()
    out = args.out.resolve()
    with (ARTIFACT / "training_manifest.csv").open(newline="", encoding="utf-8") as f:
        assert SAMPLE.name not in {row["session"] for row in csv.DictReader(f)}
    subprocess.run([sys.executable, str(ROOT / "run.py"), "--input", str(SAMPLE),
                    "--output-dir", str(out), "--artifacts", str(ARTIFACT)], check=True)
    subprocess.run([sys.executable, str(ROOT / "score.py"), "--session", str(SAMPLE),
                    "--predictions", str(out / "predictions.csv"),
                    "--timings", str(out / "timings.csv"),
                    "--output", str(out / "metrics.json")], check=True)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "ok"
    print(f"Smoke test passed. Local score: {metrics['score']['total']}/130")


if __name__ == "__main__":
    main()
