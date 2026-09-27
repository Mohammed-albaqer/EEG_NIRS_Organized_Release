"""Build subject-excluding historical caches for every public test person.

The expensive chronological feature extraction is performed once. Each
saved artifact contains only sessions of *other* people. The temporary
in-memory feature bank itself is never loaded by the online classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import classifier as clf_mod
import io_utils as iu
from train import replay_extract, round_robin_balanced_indices

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "metrics" / "fold_artifacts")
    args = ap.parse_args()
    paths = sorted(args.data.resolve().glob("*_N*.mat"))
    if len(paths) != 56:
        ap.error(f"Expected 56 public sessions, found {len(paths)}")
    recordings: list[dict] = []
    for i, path in enumerate(paths, 1):
        start = time.perf_counter()
        session = iu.load_session(path)
        subject = path.stem.rsplit("_", 1)[-1]
        Xh, Xe, Xn, y = replay_extract(session, stride_sec=0.5)
        if Xn is None:
            print(f"[{i}/56] SKIP cache: {path.name}, no compatible NIRS", flush=True)
            continue
        counts = Counter(y.tolist())
        recordings.append({"path": path, "subject": subject, "Xh": Xh,
                           "Xe": Xe, "Xn": Xn, "y": y,
                           "manifest": {"session": path.name, "subject": subject,
                                        "samples_extracted": len(y),
                                        "class1": counts.get(1, 0),
                                        "class2": counts.get(2, 0),
                                        "class3": counts.get(3, 0),
                                        "seconds": round(time.perf_counter() - start, 3)}})
        print(f"[{i}/56] {path.name}: {len(y)} windows", flush=True)
    subjects = sorted({p.stem.rsplit("_", 1)[-1] for p in paths})
    if len(recordings) != 54 or len(subjects) != 14:
        raise RuntimeError(f"Unexpected usable sources or people: {len(recordings)}, {len(subjects)}")
    for subject in subjects:
        keep_records = [r for r in recordings if r["subject"] != subject]
        Xe = np.vstack([r["Xe"] for r in keep_records])
        Xh = np.vstack([r["Xh"] for r in keep_records])
        Xn = np.vstack([r["Xn"] for r in keep_records])
        y = np.concatenate([r["y"] for r in keep_records])
        sid = np.concatenate([np.full(len(r["y"]), i, dtype=np.int32)
                              for i, r in enumerate(keep_records)])
        idx = round_robin_balanced_indices(y, sid, 400, np.random.default_rng(2026))
        artifact = args.out / f"exclude_{subject}"
        artifact.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact / "pretrain_cache.npz",
                            X_hybrid=Xh[idx].astype(np.float32),
                            X_eeg=Xe[idx].astype(np.float32),
                            X_nirs=Xn[idx].astype(np.float32),
                            y=y[idx].astype(np.int32),
                            source_session_id=sid[idx].astype(np.int32))
        counts = Counter(y[idx].tolist())
        meta = {"version": 3,
                "classifier": "random_forest_multisession_cache_causal_sosfilt",
                "n_training_sessions": len(keep_records),
                "subjects": sorted({r["subject"] for r in keep_records}),
                "cache_samples": len(idx),
                "cache_class_counts": {str(k): counts.get(k, 0) for k in (1, 2, 3)},
                "eeg_feature_dim": Xe.shape[1], "nirs_feature_dim": Xn.shape[1],
                "hybrid_feature_dim": Xh.shape[1],
                "filter": {"type": "Butterworth bandpass + scipy.signal.sosfilt",
                           "low_hz": clf_mod.EEG_FILTER_LOW_HZ,
                           "high_hz": clf_mod.EEG_FILTER_HIGH_HZ,
                           "order": clf_mod.EEG_FILTER_ORDER, "causal": True,
                           "state_persisted_between_push_calls": True},
                "training": {"stride_sec": 0.5, "cache_per_class": 400,
                             "exclude_subjects": [subject], "exclude_sessions": [],
                             "seed": 2026}}
        (artifact / "artifact_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        with (artifact / "training_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(keep_records[0]["manifest"]))
            writer.writeheader()
            writer.writerows(r["manifest"] for r in keep_records)
        if any(r["subject"] == subject for r in keep_records):
            raise AssertionError("Subject leaked into its own fold")
        print(f"FOLD {subject}: {len(keep_records)} other-subject sessions, {len(idx)} windows", flush=True)


if __name__ == "__main__":
    main()
