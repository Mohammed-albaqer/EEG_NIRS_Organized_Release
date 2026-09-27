"""Score all 56 public sessions with the tested person excluded from cache.

This may take 20-40 minutes on a laptop. Each subject is evaluated in a
separate chronological runner, but all per-session results are finally
aggregated with the frozen score.py (no confusion-matrix pooling).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "metrics")
    args = ap.parse_args()
    data = args.data.resolve()
    out = args.out.resolve()
    files = sorted(data.glob("*_N*.mat"))
    if len(files) != 56:
        ap.error(f"Expected 56 public sessions, found {len(files)}")
    subjects = sorted({p.stem.rsplit("_", 1)[-1] for p in files})
    target = out / "per_session"
    target.mkdir(parents=True, exist_ok=True)
    for number, subject in enumerate(subjects, 1):
        paths = [p for p in files if p.stem.rsplit("_", 1)[-1] == subject]
        artifact = out / "fold_artifacts" / f"exclude_{subject}"
        manifest = artifact / "training_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as source:
            if subject in {row["subject"] for row in csv.DictReader(source)}:
                raise RuntimeError(f"Subject leakage in {artifact}")
        complete = all((target / p.stem / "metrics.json").is_file()
                       and (target / p.stem / "predictions.csv").is_file()
                       and (target / p.stem / "timings.csv").is_file()
                       for p in paths)
        if complete:
            print(f"[{number}/{len(subjects)}] {subject}: already complete", flush=True)
            continue
        print(f"[{number}/{len(subjects)}] {subject}: {len(paths)} sessions", flush=True)
        subprocess.run([sys.executable, str(ROOT / "evaluate_many.py"),
                        "--data", str(data), "--subjects", subject,
                        "--classifier", str(ROOT / "classifier.py"),
                        "--artifacts", str(artifact), "--out", str(target)], check=True)
    metrics = [target / p.stem / "metrics.json" for p in files]
    if any(not path.is_file() for path in metrics):
        raise RuntimeError("A per-session metric is missing")
    subprocess.run([sys.executable, str(ROOT / "score.py"), "--aggregate",
                    *(str(p) for p in metrics), "--output", str(out / "metrics.json")], check=True)
    summary = []
    for p, mpath in zip(files, metrics):
        result = json.loads(mpath.read_text(encoding="utf-8"))
        with (mpath.parent / "predictions.csv").open(encoding="utf-8", newline="") as source:
            predictions = list(csv.DictReader(source))
        summary.append({"session": p.name, "subject": p.stem.rsplit("_", 1)[-1],
                        "windows": len(predictions),
                        "accuracy": sum(r["y"] == r["true_y"] for r in predictions) / len(predictions),
                        "min_f1": result["hybrid"]["min_f1"],
                        "macro_recall": result["hybrid"]["macro_recall"],
                        "nirs_min_f1": result["nirs_only"]["min_f1"] if result["nirs_only"] else "",
                        "score": result["score"]["total"]})
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for name in ("metrics.json", "summary.csv"):
        (target / name).unlink(missing_ok=True)  # temporary per-fold aggregate
    aggregate = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    print(f"Finished {len(summary)} sessions across {len(subjects)} held-out people", flush=True)
    print(f"Mean accuracy: {sum(row['accuracy'] for row in summary) / len(summary):.4%}", flush=True)
    print(f"Official-style score: {aggregate['score']['total']}/130", flush=True)


if __name__ == "__main__":
    main()
