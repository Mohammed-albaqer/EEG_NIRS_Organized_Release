"""Build a compact multi-session feature cache for the online classifier.

The judge does not run this script. It is used before submission to extract
causal EEG/NIRS features from many public *previous* sessions. The resulting
artifacts/pretrain_cache.npz is loaded by classifier.py and mixed with the
completed blocks of the live session inside fit_block().

Examples
--------
Validation without leakage:
    python train.py --data data_all --out artifacts_holdout --exclude-subjects N17 N18

Final public-data artifact:
    python train.py --data data_all --out solution/artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classifier as clf_mod  # noqa: E402
import io_utils as iu  # noqa: E402


def gather_sessions(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.glob("*.mat")))
        else:
            out.extend(sorted(Path().glob(raw)))
    out = [p.resolve() for p in out if "_N" in p.stem]
    seen: set[Path] = set()
    ans: list[Path] = []
    for p in out:
        if p not in seen:
            ans.append(p)
            seen.add(p)
    return ans


def _block_ends(blocks: np.ndarray) -> list[int]:
    ans = []
    for b in sorted({int(x) for x in np.unique(blocks) if int(x) >= 1}):
        idx = np.flatnonzero(blocks == b)
        if idx.size:
            ans.append(int(idx[-1]) + 1)
    return ans


def replay_extract(session: dict, stride_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Extract features by replaying the session through the exact causal path."""
    c = clf_mod.OnlineClassifier(iu.session_meta(session))
    windows = list(
        iu.iter_scored_windows(
            session["states"], session["blocks"], session["fs_eeg"], skip_block=-99999
        )
    )
    events: list[tuple[int, str, tuple[int, int, int] | None]] = []
    for w in windows:
        events.append((w[1], "window", w))
    for end in _block_ends(session["blocks"]):
        events.append((end, "boundary", None))
    events.sort(key=lambda z: (z[0], 0 if z[1] == "window" else 1))

    base_step = int(round(session["fs_eeg"] * 0.25))
    desired = max(base_step, int(round(session["fs_eeg"] * stride_sec)))
    every = max(1, int(round(desired / base_step)))

    fh_list: list[np.ndarray] = []
    fe_list: list[np.ndarray] = []
    fn_list: list[np.ndarray] = []
    y_list: list[int] = []
    cursor = 0
    window_i = 0
    eeg = session["eeg"]

    def push_upto(end: int) -> None:
        nonlocal cursor
        if end <= cursor:
            return
        if session["has_nirs"]:
            hbo, hbr = iu.nirs_slice_for_eeg_range(session, cursor, end)
            nirs_chunk = (hbo, hbr)
        else:
            nch = int(session.get("n_nirs_ch") or 0)
            nirs_chunk = (np.zeros((0, nch)), np.zeros((0, nch)))
        c.push(eeg[cursor:end], nirs_chunk)
        cursor = end

    for end, kind, payload in events:
        push_upto(end)
        if kind != "window" or payload is None:
            continue
        take = (window_i % every) == 0
        window_i += 1
        if not take:
            continue
        start, stop, lab = payload
        fh, fe, fn = c._features_for_window(start, stop)
        fh_list.append(fh)
        fe_list.append(fe)
        if session["has_nirs"]:
            fn_list.append(fn)
        y_list.append(int(lab))

    if not y_list:
        return (
            np.empty((0, c.hybrid_feature_dim), np.float32),
            np.empty((0, c.eeg_feature_dim), np.float32),
            None,
            np.empty(0, np.int32),
        )

    return (
        np.asarray(fh_list, dtype=np.float32),
        np.asarray(fe_list, dtype=np.float32),
        np.asarray(fn_list, dtype=np.float32) if session["has_nirs"] else None,
        np.asarray(y_list, dtype=np.int32),
    )


def round_robin_balanced_indices(
    y: np.ndarray,
    session_id: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Balanced cache while drawing from as many sessions as possible."""
    chosen: list[int] = []
    for cls in (1, 2, 3):
        by_session: dict[int, deque[int]] = {}
        for sid in np.unique(session_id):
            idx = np.flatnonzero((y == cls) & (session_id == sid))
            if len(idx):
                idx = idx.copy()
                rng.shuffle(idx)
                by_session[int(sid)] = deque(int(i) for i in idx)
        sids = list(by_session)
        rng.shuffle(sids)
        count = 0
        while count < per_class and sids:
            next_sids: list[int] = []
            for sid in sids:
                q = by_session[sid]
                if q and count < per_class:
                    chosen.append(q.popleft())
                    count += 1
                if q:
                    next_sids.append(sid)
            sids = next_sids
    return np.asarray(sorted(chosen), dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="solution/artifacts")
    ap.add_argument("--exclude-subjects", nargs="*", default=[])
    ap.add_argument("--exclude-sessions", nargs="*", default=[])
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--stride-sec", type=float, default=0.5)
    ap.add_argument("--cache-per-class", type=int, default=400,
                    help="historical cache size per class; total is about 3x this")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    all_paths = gather_sessions(args.data)
    excl_sub = {s.upper() for s in args.exclude_subjects}
    excl_names = {Path(s).name for s in args.exclude_sessions}
    paths = [
        p for p in all_paths
        if p.stem.split("_")[-1].upper() not in excl_sub and p.name not in excl_names
    ]
    if args.max_sessions > 0:
        paths = paths[: args.max_sessions]
    if not paths:
        ap.error("No training sessions selected")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Xh_all: list[np.ndarray] = []
    Xe_all: list[np.ndarray] = []
    Xn_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    sid_all: list[np.ndarray] = []
    manifest: list[dict] = []
    nirs_dim: int | None = None

    print(f"Selected {len(paths)} training sessions")
    t_start = time.perf_counter()
    for sid, p in enumerate(paths):
        t0 = time.perf_counter()
        try:
            s = iu.load_session(p)
            Xh, Xe, Xn, y = replay_extract(s, args.stride_sec)
        except Exception as exc:
            print(f"SKIP {p.name}: {exc}")
            continue
        if not len(y):
            continue
        if Xn is None:
            # EEG-only historical session can still help EEG, but to keep one
            # aligned label array for the compact cache we currently use only
            # sessions with both modalities when NIRS is expected.
            print(f"SKIP {p.name}: no NIRS")
            continue
        if nirs_dim is None:
            nirs_dim = Xn.shape[1]
        if Xn.shape[1] != nirs_dim:
            print(f"SKIP {p.name}: NIRS dim {Xn.shape[1]} != {nirs_dim}")
            continue

        Xh_all.append(Xh)
        Xe_all.append(Xe)
        Xn_all.append(Xn)
        y_all.append(y)
        sid_all.append(np.full(len(y), len(manifest), dtype=np.int32))
        counts = Counter(y.tolist())
        manifest.append({
            "session": p.name,
            "subject": s["subject"],
            "samples_extracted": int(len(y)),
            "class1": int(counts.get(1, 0)),
            "class2": int(counts.get(2, 0)),
            "class3": int(counts.get(3, 0)),
            "seconds": round(time.perf_counter() - t0, 3),
        })
        print(f"[{len(manifest):02d}] {p.name}: {len(y)} windows {dict(sorted(counts.items()))}")

    if not y_all:
        raise RuntimeError("No usable multi-session training data")

    Xh = np.vstack(Xh_all)
    Xe = np.vstack(Xe_all)
    Xn = np.vstack(Xn_all)
    y = np.concatenate(y_all)
    sid = np.concatenate(sid_all)

    keep = round_robin_balanced_indices(y, sid, args.cache_per_class, rng)
    Xh = Xh[keep]
    Xe = Xe[keep]
    Xn = Xn[keep]
    y_cache = y[keep]
    sid_cache = sid[keep]

    np.savez_compressed(
        out / "pretrain_cache.npz",
        X_hybrid=Xh.astype(np.float32),
        X_eeg=Xe.astype(np.float32),
        X_nirs=Xn.astype(np.float32),
        y=y_cache.astype(np.int32),
        source_session_id=sid_cache.astype(np.int32),
    )

    cache_counts = Counter(y_cache.tolist())
    meta = {
        "version": 3,
        "classifier": "random_forest_multisession_cache_causal_sosfilt",
        "n_training_sessions": len(manifest),
        "subjects": sorted({m["subject"] for m in manifest}),
        "cache_samples": int(len(y_cache)),
        "cache_class_counts": {str(k): int(cache_counts.get(k, 0)) for k in (1,2,3)},
        "eeg_feature_dim": int(Xe.shape[1]),
        "nirs_feature_dim": int(Xn.shape[1]),
        "hybrid_feature_dim": int(Xh.shape[1]),
        "filter": {
            "type": "Butterworth bandpass + scipy.signal.sosfilt",
            "low_hz": clf_mod.EEG_FILTER_LOW_HZ,
            "high_hz": clf_mod.EEG_FILTER_HIGH_HZ,
            "order": clf_mod.EEG_FILTER_ORDER,
            "causal": True,
            "state_persisted_between_push_calls": True,
        },
        "training": {
            "stride_sec": args.stride_sec,
            "cache_per_class": args.cache_per_class,
            "exclude_subjects": sorted(excl_sub),
            "exclude_sessions": sorted(excl_names),
            "seed": args.seed,
        },
    }
    (out / "artifact_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "training_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    print(f"\nExtracted before cache: {len(y)} windows")
    print(f"Saved cache: {len(y_cache)} windows {dict(sorted(cache_counts.items()))}")
    print(f"Sessions represented in cache: {len(set(sid_cache.tolist()))}/{len(manifest)}")
    print(f"Artifact: {(out/'pretrain_cache.npz').resolve()}")
    print(f"Total elapsed: {time.perf_counter() - t_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
