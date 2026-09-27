"""Frozen online runner.

Controls timing and labels. The classifier sees only past data.

    python run.py --input /data/session.mat --output-dir /out [--artifacts solution/artifacts]

Do not modify this file. The judge replaces it with a reference copy.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier as clf_mod  # noqa: E402
import io_utils as iu  # noqa: E402

EXIT_OK, EXIT_DATA, EXIT_MODEL = 0, 2, 3


def fail(code: int, message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


def _empty_nirs(session: dict) -> tuple[np.ndarray, np.ndarray]:
    nch = int(session.get("n_nirs_ch") or 0)
    return (
        np.zeros((0, nch), dtype=np.float64),
        np.zeros((0, nch), dtype=np.float64),
    )


def run_session(
    session_path: Path,
    output_dir: Path,
    artifacts_dir: Path | None,
) -> dict:
    try:
        session = iu.load_session(session_path)
    except Exception as exc:
        fail(EXIT_DATA, f"could not read {session_path}: {exc}")

    meta = iu.session_meta(session)
    try:
        classifier = clf_mod.OnlineClassifier(meta)
        if artifacts_dir is not None and Path(artifacts_dir).exists():
            classifier.load(artifacts_dir)
    except Exception:
        fail(EXIT_MODEL, f"classifier initialization failed:\n{traceback.format_exc()}")

    eeg = session["eeg"]
    states = session["states"]
    blocks = session["blocks"]
    fs = session["fs_eeg"]
    n = eeg.shape[0]

    windows = list(iu.iter_scored_windows(states, blocks, fs))
    # fit_block is called for ALL blocks (including block 1 for initial training);
    # block 1 windows are not scored (see iter_scored_windows).
    block_ids = sorted({int(b) for b in np.unique(blocks) if int(b) >= 1})
    block_end: dict[int, int] = {}
    for b in block_ids:
        idxs = np.flatnonzero(blocks == b)
        if idxs.size:
            block_end[b] = int(idxs[-1]) + 1  # exclusive end sample

    pred_rows: list[dict] = []
    timing_rows: list[dict] = []
    fitted: set[int] = set()
    cursor = 0  # number of EEG samples already sent through push

    def push_upto(end_sample: int) -> None:
        nonlocal cursor
        if end_sample <= cursor:
            return
        eeg_chunk = eeg[cursor:end_sample]
        if session["has_nirs"]:
            hbo, hbr = iu.nirs_slice_for_eeg_range(session, cursor, end_sample)
            nirs_chunk = (hbo, hbr)
        else:
            nirs_chunk = _empty_nirs(session)
        try:
            classifier.push(eeg_chunk, nirs_chunk)
        except Exception:
            fail(EXIT_MODEL, f"push() failed at [{cursor},{end_sample}):\n{traceback.format_exc()}")
        cursor = end_sample

    # chronological pass: windows and fit_block in time order
    events: list[tuple[int, str, object]] = []
    for start, end, lab in windows:
        events.append((end, "predict", (start, end, lab)))
    for b, end_s in block_end.items():
        events.append((end_s, "fit", b))
    events.sort(key=lambda t: (t[0], 0 if t[1] == "predict" else 1))

    for end_sample, kind, payload in events:
        push_upto(end_sample)

        if kind == "fit":
            b = int(payload)
            if b in fitted:
                continue
            # completed-block labels: only samples with States>0 inside the block
            mask = (blocks == b) & (states > 0)
            labels = {
                "states": states[mask].copy(),
                "eeg_indices": np.flatnonzero(mask),
                "block": b,
            }
            t0 = time.perf_counter()
            try:
                classifier.fit_block(b, labels)
            except Exception:
                fail(EXIT_MODEL, f"fit_block({b}) failed:\n{traceback.format_exc()}")
            dt = time.perf_counter() - t0
            timing_rows.append({"event": "fit_block", "block": b, "seconds": round(dt, 6)})
            fitted.add(b)
            continue

        start, end, lab = payload  # type: ignore[misc]
        t0 = time.perf_counter()
        try:
            out = classifier.predict()
        except Exception:
            fail(EXIT_MODEL, f"predict() failed on window [{start},{end}):\n{traceback.format_exc()}")
        dt = time.perf_counter() - t0
        timing_rows.append(
            {"event": "predict", "end_sample": end, "seconds": round(dt, 6)}
        )

        if not isinstance(out, dict) or "y" not in out:
            fail(EXIT_MODEL, f"predict() must return a dict with key 'y', got: {type(out)}")
        y = int(out["y"])
        if y not in (1, 2, 3):
            fail(EXIT_MODEL, f"predict()['y'] must be 1|2|3, got: {y}")

        row = {
            "end_sample": end,
            "start_sample": start,
            "true_y": lab,
            "y": y,
        }
        if "y_eeg" in out and out["y_eeg"] is not None:
            row["y_eeg"] = int(out["y_eeg"])
        if "y_nirs" in out and out["y_nirs"] is not None:
            row["y_nirs"] = int(out["y_nirs"])
        pred_rows.append(row)

    # consume the session tail (if the last block ends after the last scored window)
    push_upto(n)
    for b in block_ids:
        if b not in fitted and b in block_end:
            push_upto(block_end[b])
            mask = (blocks == b) & (states > 0)
            labels = {
                "states": states[mask].copy(),
                "eeg_indices": np.flatnonzero(mask),
                "block": b,
            }
            t0 = time.perf_counter()
            try:
                classifier.fit_block(b, labels)
            except Exception:
                fail(EXIT_MODEL, f"fit_block({b}) failed:\n{traceback.format_exc()}")
            dt = time.perf_counter() - t0
            timing_rows.append({"event": "fit_block", "block": b, "seconds": round(dt, 6)})
            fitted.add(b)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.csv"
    time_path = output_dir / "timings.csv"

    pred_fields = ["end_sample", "start_sample", "true_y", "y"]
    if any("y_eeg" in r for r in pred_rows):
        pred_fields.append("y_eeg")
    if any("y_nirs" in r for r in pred_rows):
        pred_fields.append("y_nirs")
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pred_rows)

    time_fields = ["event", "seconds", "end_sample", "block"]
    with time_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=time_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(timing_rows)

    print(f"Session: {session_path.name}")
    print(f"Windows:   {len(pred_rows)}")
    print(f"fit:    {len(fitted)} blocks")
    print(f"Written: {pred_path}, {time_path}")
    return {"n_windows": len(pred_rows), "n_fits": len(fitted)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Online classifier run on one session")
    ap.add_argument("--input", required=True, help="path to a .mat session")
    ap.add_argument("--output-dir", default=".", help="where to write predictions.csv and timings.csv")
    ap.add_argument("--artifacts", default=None, help="directory of classifier artifacts (optional)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        fail(EXIT_DATA, f"file not found: {inp}")
    artifacts = Path(args.artifacts) if args.artifacts else (HERE / "artifacts")
    if not artifacts.exists():
        artifacts = None

    run_session(inp, Path(args.output_dir), artifacts)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
