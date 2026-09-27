"""Check essential submission files, artifact provenance, and code identity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true",
                        help="check source/artifact identity without requiring 56 public scores")
    args = parser.parse_args()
    for name in ("classifier.py", "run.py", "score.py", "io_utils.py"):
        assert (ROOT / name).read_bytes() == (ROOT / "solution" / name).read_bytes(), name
    final = ROOT / "solution" / "artifacts"
    test = ROOT / "results" / "holdout" / "artifacts_N17_N18"
    assert (final / "pretrain_cache.npz").is_file()
    assert (test / "pretrain_cache.npz").is_file()
    final_sessions = rows(final / "training_manifest.csv")
    test_sessions = rows(test / "training_manifest.csv")
    assert len(final_sessions) == 54, len(final_sessions)
    assert len(test_sessions) == 46, len(test_sessions)
    assert not {r["subject"] for r in test_sessions} & {"N17", "N18"}
    final_meta = json.loads((final / "artifact_meta.json").read_text(encoding="utf-8"))
    test_meta = json.loads((test / "artifact_meta.json").read_text(encoding="utf-8"))
    assert final_meta["cache_samples"] == 1200
    assert test_meta["cache_samples"] == 1200
    assert len(rows(ROOT / "public_data_manifest.csv")) == 56
    target = ROOT / "results" / "holdout" / "final_selected_N17_N18"
    summary = rows(target / "summary.csv")
    assert len(summary) == 8
    assert {r["subject"] for r in summary} == {"N17", "N18"}
    public = ROOT / "results" / "metrics"
    subjects = {p.name.removeprefix("exclude_")
                for p in (public / "fold_artifacts").glob("exclude_*") if p.is_dir()}
    assert len(subjects) == 14, len(subjects)
    for subject in subjects:
        fold = public / "fold_artifacts" / f"exclude_{subject}" / "training_manifest.csv"
        assert all(row["subject"] != subject for row in rows(fold)), subject
    print("PASS: 54 public training sessions and 46 subject-excluding holdout sources")
    print("PASS: identical judge/root classifier and frozen runner/scorer copies")
    print("PASS: eight historical strict sessions and clean subject folds")
    if args.code_only:
        print("PENDING: code-only mode does not certify the 56 public scores")
        return
    if not (public / "summary.csv").is_file() or not (public / "metrics.json").is_file():
        raise SystemExit("PENDING: 56-session aggregate missing; run evaluate_public_loso.py")
    public_summary = rows(public / "summary.csv")
    assert len(public_summary) == 56
    assert json.loads((public / "metrics.json").read_text(encoding="utf-8"))["n_sessions"] == 56
    print("PASS: 56 public sessions with subject-excluding folds")


if __name__ == "__main__":
    main()
