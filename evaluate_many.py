"""Evaluate one classifier/artifact set on multiple independent sessions.

The script always uses the frozen run.py + score.py logic from this package.
It writes per-session predictions/timings/metrics, summary.csv, and the official
unweighted mean aggregate metrics.json.

Examples:
    python evaluate_many.py --data data_all --subjects N17 N18 \
        --artifacts artifacts_holdout --out results/holdout

    python evaluate_many.py --data data_all --subjects N17 N18 \
        --classifier experiments/classifier_original.py --no-artifacts \
        --out results/original_holdout
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gather(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p.resolve())
        elif p.is_dir():
            out.extend(x.resolve() for x in sorted(p.glob("*.mat")) if "_N" in x.stem)
        else:
            out.extend(x.resolve() for x in sorted(Path().glob(raw)) if "_N" in x.stem)
    seen = set()
    ans = []
    for p in out:
        if p not in seen:
            ans.append(p)
            seen.add(p)
    return ans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--subjects", nargs="*", default=[])
    ap.add_argument("--classifier", type=Path, default=HERE / "classifier.py")
    ap.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "evaluation")
    args = ap.parse_args()

    sessions = gather(args.data)
    subs = {s.upper() for s in args.subjects}
    if subs:
        sessions = [p for p in sessions if p.stem.split("_")[-1].upper() in subs]
    if not sessions:
        ap.error("No sessions selected")
    if not args.classifier.is_file():
        ap.error(f"classifier file not found: {args.classifier}")

    args.out.mkdir(parents=True, exist_ok=True)
    metric_paths: list[Path] = []
    summary: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="eeg_eval_") as td:
        work = Path(td)
        for name in ("run.py", "score.py", "io_utils.py"):
            shutil.copy2(HERE / name, work / name)
        shutil.copy2(args.classifier, work / "classifier.py")

        for i, session in enumerate(sessions, 1):
            target = args.out / session.stem
            target.mkdir(parents=True, exist_ok=True)
            print(f"[{i}/{len(sessions)}] {session.name}")

            cmd = [
                sys.executable, str(work / "run.py"),
                "--input", str(session),
                "--output-dir", str(target),
            ]
            if not args.no_artifacts and args.artifacts.exists():
                cmd.extend(["--artifacts", str(args.artifacts.resolve())])
            subprocess.run(cmd, check=True)

            metrics_path = target / "metrics.json"
            subprocess.run(
                [
                    sys.executable, str(work / "score.py"),
                    "--session", str(session),
                    "--predictions", str(target / "predictions.csv"),
                    "--timings", str(target / "timings.csv"),
                    "--output", str(metrics_path),
                ],
                check=True,
            )
            metric_paths.append(metrics_path)
            m = json.loads(metrics_path.read_text(encoding="utf-8"))

            with (target / "predictions.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            acc = sum(r["y"] == r["true_y"] for r in rows) / max(len(rows), 1)
            nirs = m.get("nirs_only") or {}
            t = m.get("timings") or {}
            summary.append(
                {
                    "session": session.name,
                    "subject": session.stem.split("_")[-1],
                    "windows": len(rows),
                    "accuracy": acc,
                    "min_f1": m["hybrid"]["min_f1"],
                    "macro_recall": m["hybrid"]["macro_recall"],
                    "nirs_min_f1": nirs.get("min_f1", ""),
                    "predict_mean_s": t.get("predict_mean_s", ""),
                    "fit_mean_s": t.get("fit_mean_s", ""),
                    "penalty_total": t.get("penalty_total", ""),
                    "score": m["score"]["total"],
                }
            )

        aggregate_path = args.out / "metrics.json"
        subprocess.run(
            [
                sys.executable, str(work / "score.py"),
                "--aggregate", *(str(p) for p in metric_paths),
                "--output", str(aggregate_path),
            ],
            check=True,
        )

    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    aggregate = json.loads((args.out / "metrics.json").read_text(encoding="utf-8"))
    mean_acc = sum(float(x["accuracy"]) for x in summary) / len(summary)
    print("\n=== SUMMARY ===")
    print(f"sessions:      {len(summary)}")
    print(f"mean accuracy: {mean_acc:.4f} (diagnostic only)")
    print(f"mean min-F1:   {aggregate['hybrid']['min_f1']:.4f}")
    print(f"macro-recall:  {aggregate['hybrid']['macro_recall']:.4f}")
    print(f"score:         {aggregate['score']['total']:.4f}")
    print(f"written:       {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
