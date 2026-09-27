"""Experimental Extra Trees variant of the online EEG+NIRS classifier.

It reuses the same legal feature cache and interface as the tested Random
Forest submission. Its accuracy and timing have not yet been measured.

Upgrades over the original one-session package
----------------------------------------------
1. Causal EEG band-pass filtering with scipy.signal.sosfilt().  The IIR state
   is preserved between push() calls, so sample t never depends on the future.
2. Multi-session pretraining is represented by a compact *feature cache*
   (artifacts/pretrain_cache.npz) extracted from earlier sessions.  After each
   completed block, fit_block() trains on BOTH that historical cache and the
   labels legally released from the current session.  Thus the historical
   knowledge is not discarded after block 1.

The frozen judge still sees exactly the required contract:
    __init__ -> load? -> (push | predict | fit_block)*

Classes: 1=rest, 2=left-hand imagery, 3=right-hand imagery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.ensemble import ExtraTreesClassifier as RandomForestClassifier

EEG_FILTER_LOW_HZ = 1.0
EEG_FILTER_HIGH_HZ = 40.0
EEG_FILTER_ORDER = 4

# Historical samples are deliberately down-weighted versus current-session
# samples.  This gives cross-session prior knowledge without drowning out
# subject/session-specific adaptation.
HISTORICAL_SAMPLE_WEIGHT = 0.55
CURRENT_SAMPLE_WEIGHT = 1.50
MAX_CURRENT_TRAIN = 2600


class OnlineClassifier:
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.fs_eeg = float(meta["fs_eeg"])
        self.n_eeg_ch = int(meta["n_eeg_ch"])
        self.has_nirs = bool(meta.get("has_nirs", False))
        self.n_nirs_ch = int(meta.get("n_nirs_ch") or 0)
        if self.fs_eeg <= 2 * EEG_FILTER_LOW_HZ or self.n_eeg_ch <= 0:
            raise ValueError("Invalid EEG sample rate or channel count")
        if self.has_nirs and self.n_nirs_ch <= 0:
            raise ValueError("NIRS is enabled but its channel count is missing")

        self.win = int(round(self.fs_eeg * 1.0))
        self.step = int(round(self.fs_eeg * 0.25))
        self.c3_idx = int(meta.get("eeg_c3_idx", 18))
        self.c4_idx = int(meta.get("eeg_c4_idx", 0))
        if not (0 <= self.c3_idx < self.n_eeg_ch and 0 <= self.c4_idx < self.n_eeg_ch):
            raise ValueError("C3/C4 EEG channel indices are outside the input array")

        # Causal streaming Butterworth band-pass.
        nyq = self.fs_eeg / 2.0
        hi = min(EEG_FILTER_HIGH_HZ, nyq * 0.95)
        lo = min(EEG_FILTER_LOW_HZ, hi * 0.5)
        self.eeg_sos = butter(
            EEG_FILTER_ORDER,
            [lo, hi],
            btype="bandpass",
            fs=self.fs_eeg,
            output="sos",
        )
        self.eeg_filter_zi: np.ndarray | None = None

        self.eeg_buffer = np.empty((0, self.n_eeg_ch), dtype=np.float64)
        # Grow capacity geometrically: streaming chunks no longer copy the
        # entire recording on every push().  eeg_buffer remains the exact
        # visible prefix used by the feature extractor and legal block fits.
        self._eeg_storage = self.eeg_buffer
        self.nirs_records: list[dict[str, Any]] = []

        # Current-session training data (legal labels from completed blocks).
        self.X_hybrid: list[np.ndarray] = []
        self.X_eeg: list[np.ndarray] = []
        self.X_nirs: list[np.ndarray] = []
        self.y_train: list[int] = []

        # Historical multi-session feature cache loaded from artifacts/.
        self.pre_X_hybrid = np.empty((0, 0), dtype=np.float32)
        self.pre_X_eeg = np.empty((0, 0), dtype=np.float32)
        self.pre_X_nirs = np.empty((0, 0), dtype=np.float32)
        self.pre_y = np.empty(0, dtype=np.int32)

        self.model_hybrid: RandomForestClassifier | None = None
        self.model_eeg: RandomForestClassifier | None = None
        self.model_nirs: RandomForestClassifier | None = None

        self.eeg_feature_dim = self.n_eeg_ch * 5 + 4
        self.nirs_feature_dim = 6 * self.n_nirs_ch if self.has_nirs else 0
        self.hybrid_feature_dim = self.eeg_feature_dim + self.nirs_feature_dim

        # Window geometry is fixed for a session, so these FFT inputs can be
        # reused for every online prediction and every completed block.
        self._taper = np.hanning(self.win)[:, None]
        freqs = np.fft.rfftfreq(self.win, d=1.0 / self.fs_eeg)
        self._band_masks = tuple(
            (freqs >= lo) & (freqs < hi)
            for lo, hi in ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0))
        )

    @staticmethod
    def _new_rf(seed: int) -> RandomForestClassifier:
        # Extra Trees randomizes candidate split thresholds, providing a
        # different tree ensemble with the same features and artifact format.
        # This is an unvalidated candidate, not the measured submission.
        return RandomForestClassifier(
            n_estimators=90,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    def load(self, artifacts_dir: str | Path) -> None:
        """Load historical features; no test-session data are read here."""
        artifacts_dir = Path(artifacts_dir)
        cache = artifacts_dir / "pretrain_cache.npz"
        if not cache.exists():
            return

        meta_path = artifacts_dir / "artifact_meta.json"
        if not meta_path.exists():
            raise ValueError(f"Missing historical cache metadata: {meta_path}")
        artifact = json.loads(meta_path.read_text(encoding="utf-8"))
        filtering = artifact.get("filter", {})
        expected = (EEG_FILTER_LOW_HZ, EEG_FILTER_HIGH_HZ, EEG_FILTER_ORDER)
        actual = (filtering.get("low_hz"), filtering.get("high_hz"), filtering.get("order"))
        if actual != expected:
            raise ValueError(f"Historical cache filter {actual} does not match classifier {expected}")
        if int(artifact.get("eeg_feature_dim", -1)) != self.eeg_feature_dim:
            raise ValueError("Historical cache EEG feature dimension does not match the classifier")
        if self.has_nirs and (
            int(artifact.get("nirs_feature_dim", -1)) != self.nirs_feature_dim
            or int(artifact.get("hybrid_feature_dim", -1)) != self.hybrid_feature_dim
        ):
            raise ValueError("Historical cache NIRS feature dimensions do not match the classifier")

        with np.load(cache, allow_pickle=False) as z:
            Xe = np.asarray(z["X_eeg"], dtype=np.float32)
            y = np.asarray(z["y"], dtype=np.int32).ravel()
            Xh = np.asarray(z["X_hybrid"], dtype=np.float32) if "X_hybrid" in z else np.empty((0, 0), dtype=np.float32)
            Xn = np.asarray(z["X_nirs"], dtype=np.float32) if "X_nirs" in z else np.empty((0, 0), dtype=np.float32)

        if Xe.ndim != 2 or Xe.shape[1] != self.eeg_feature_dim or len(Xe) != len(y):
            raise ValueError(
                f"pretrain EEG cache shape {Xe.shape} incompatible with runtime feature dim {self.eeg_feature_dim}"
            )
        if not np.isfinite(Xe).all() or not np.isin(y, (1, 2, 3)).all():
            raise ValueError("Historical EEG cache contains nonfinite features or invalid labels")
        if self.has_nirs and (
            Xn.ndim != 2 or Xn.shape != (len(y), self.nirs_feature_dim)
            or Xh.ndim != 2 or Xh.shape != (len(y), self.hybrid_feature_dim)
            or not np.isfinite(Xn).all() or not np.isfinite(Xh).all()
        ):
            raise ValueError("Historical NIRS/hybrid cache is incomplete or nonfinite")

        self.pre_X_eeg = Xe
        self.pre_y = y

        if self.has_nirs:
            if Xn.ndim == 2 and Xn.shape[1] == self.nirs_feature_dim and len(Xn) == len(y):
                self.pre_X_nirs = Xn
            if Xh.ndim == 2 and Xh.shape[1] == self.hybrid_feature_dim and len(Xh) == len(y):
                self.pre_X_hybrid = Xh

    def push(self, eeg_chunk, nirs_chunk) -> None:
        eeg_chunk = np.asarray(eeg_chunk, dtype=np.float64)
        if eeg_chunk.ndim != 2:
            raise ValueError(f"eeg_chunk must be 2D, got {eeg_chunk.shape}")
        if eeg_chunk.shape[1] != self.n_eeg_ch:
            raise ValueError(
                f"expected {self.n_eeg_ch} EEG channels, got {eeg_chunk.shape[1]}"
            )

        eeg_start = len(self.eeg_buffer)
        eeg_end = eeg_start + len(eeg_chunk)

        if len(eeg_chunk):
            # A nonfinite sample must not poison the persistent IIR state.
            eeg_chunk = self._clean(eeg_chunk)
            if self.eeg_filter_zi is None:
                zi = sosfilt_zi(self.eeg_sos)[:, :, None]
                self.eeg_filter_zi = zi * eeg_chunk[0][None, None, :]
            eeg_filtered, self.eeg_filter_zi = sosfilt(
                self.eeg_sos,
                eeg_chunk,
                axis=0,
                zi=self.eeg_filter_zi,
            )
            needed = eeg_end
            if needed > len(self._eeg_storage):
                capacity = max(needed, 1024, 2 * len(self._eeg_storage))
                storage = np.empty((capacity, self.n_eeg_ch), dtype=np.float64)
                storage[:eeg_start] = self.eeg_buffer
                self._eeg_storage = storage
            self._eeg_storage[eeg_start:eeg_end] = eeg_filtered
            self.eeg_buffer = self._eeg_storage[:eeg_end]

        if not self.has_nirs or nirs_chunk is None:
            return

        hbo, hbr = nirs_chunk
        hbo = np.asarray(hbo, dtype=np.float64)
        hbr = np.asarray(hbr, dtype=np.float64)
        if hbo.size == 0 or hbr.size == 0:
            return
        if hbo.ndim == 1:
            hbo = hbo.reshape(-1, self.n_nirs_ch)
        if hbr.ndim == 1:
            hbr = hbr.reshape(-1, self.n_nirs_ch)

        n = min(len(hbo), len(hbr))
        hbo = hbo[:n]
        hbr = hbr[:n]
        if hbo.shape != (n, self.n_nirs_ch) or hbr.shape != (n, self.n_nirs_ch):
            raise ValueError("NIRS chunk channel count does not match session metadata")
        if n == 1:
            eeg_pos = np.array([(eeg_start + eeg_end - 1) / 2.0])
        else:
            eeg_pos = np.linspace(
                eeg_start,
                max(eeg_start, eeg_end - 1),
                n,
                dtype=np.float64,
            )

        self.nirs_records.append(
            {
                "eeg_start": eeg_start,
                "eeg_end": eeg_end,
                "eeg_pos": eeg_pos,
                "hbo": hbo,
                "hbr": hbr,
            }
        )

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(x, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _eeg_features(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (self.win, self.n_eeg_ch):
            return np.zeros(self.eeg_feature_dim, dtype=np.float32)

        x = self._clean(x)
        x = x - np.mean(x, axis=0, keepdims=True)
        log_var = np.log(np.var(x, axis=0) + 1e-12)

        spec = np.fft.rfft(x * self._taper, axis=0)
        power = (np.abs(spec) ** 2) / max(1, len(x))

        band_logs: list[np.ndarray] = []
        for mask in self._band_masks:
            if np.any(mask):
                bp = np.mean(power[mask], axis=0) + 1e-12
            else:
                bp = np.full(self.n_eeg_ch, 1e-12)
            band_logs.append(np.log(bp))

        per_channel = np.column_stack([log_var] + band_logs).reshape(-1)
        asym = []
        for band in band_logs:
            c3 = band[self.c3_idx] if self.c3_idx < len(band) else 0.0
            c4 = band[self.c4_idx] if self.c4_idx < len(band) else 0.0
            asym.append(c3 - c4)

        return self._clean(
            np.concatenate([per_channel, np.asarray(asym)])
        ).astype(np.float32)

    def _nirs_window(self, eeg_start: int, eeg_end: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.has_nirs or self.n_nirs_ch <= 0:
            return np.empty((0, 0)), np.empty((0, 0))

        hbo_parts: list[np.ndarray] = []
        hbr_parts: list[np.ndarray] = []
        for rec in reversed(self.nirs_records):
            if rec["eeg_end"] <= eeg_start:
                break
            if rec["eeg_start"] >= eeg_end:
                continue
            pos = rec["eeg_pos"]
            mask = (pos >= eeg_start) & (pos < eeg_end)
            if np.any(mask):
                hbo_parts.append(rec["hbo"][mask])
                hbr_parts.append(rec["hbr"][mask])

        if not hbo_parts:
            return (
                np.empty((0, self.n_nirs_ch)),
                np.empty((0, self.n_nirs_ch)),
            )
        return np.vstack(hbo_parts[::-1]), np.vstack(hbr_parts[::-1])

    def _nirs_features(self, eeg_start: int, eeg_end: int) -> np.ndarray:
        if not self.has_nirs or self.n_nirs_ch <= 0:
            return np.empty(0, dtype=np.float32)
        hbo, hbr = self._nirs_window(eeg_start, eeg_end)
        if len(hbo) == 0 or len(hbr) == 0:
            return np.zeros(self.nirs_feature_dim, dtype=np.float32)

        def stats(a: np.ndarray) -> np.ndarray:
            a = self._clean(a)
            mean = np.mean(a, axis=0)
            std = np.std(a, axis=0)
            delta = a[-1] - a[0] if len(a) >= 2 else np.zeros(a.shape[1])
            return np.column_stack((mean, std, delta)).reshape(-1)

        return self._clean(np.concatenate((stats(hbo), stats(hbr)))).astype(np.float32)

    def _features_for_window(
        self, eeg_start: int, eeg_end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eeg_feat = self._eeg_features(self.eeg_buffer[eeg_start:eeg_end])
        if self.has_nirs:
            nirs_feat = self._nirs_features(eeg_start, eeg_end)
            hybrid_feat = np.concatenate((eeg_feat, nirs_feat)).astype(np.float32)
        else:
            nirs_feat = np.empty(0, dtype=np.float32)
            hybrid_feat = eeg_feat
        return hybrid_feat, eeg_feat, nirs_feat

    def _fallback(self) -> int:
        labels = self.y_train if self.y_train else self.pre_y.tolist()
        if not labels:
            return 1
        counts = np.bincount(np.asarray(labels, dtype=int), minlength=4)
        return int(np.argmax(counts[1:4]) + 1)

    def predict(self) -> dict:
        end = len(self.eeg_buffer)
        start = end - self.win
        fallback = self._fallback()
        if start < 0:
            return {"y": fallback, "y_eeg": fallback, "y_nirs": fallback if self.has_nirs else None}

        hybrid_feat, eeg_feat, nirs_feat = self._features_for_window(start, end)

        def aligned(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
            result = np.zeros(3, dtype=np.float64)
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            for cls, probability in zip(model.classes_, probabilities):
                if int(cls) in (1, 2, 3):
                    result[int(cls) - 1] = probability
            return result

        pe = aligned(self.model_eeg, eeg_feat) if self.model_eeg is not None else None
        y_eeg = int(np.argmax(pe) + 1) if pe is not None else fallback
        y_nirs = None
        pn = None
        if self.has_nirs:
            pn = aligned(self.model_nirs, nirs_feat) if self.model_nirs is not None else None
            y_nirs = int(np.argmax(pn) + 1) if pn is not None else fallback

        if self.has_nirs and self.model_hybrid is not None and pe is not None and pn is not None:
            ph = aligned(self.model_hybrid, hybrid_feat)
            y = int(np.argmax(0.50 * ph + 0.35 * pe + 0.15 * pn) + 1)
        else:
            y = y_eeg

        if y not in (1, 2, 3):
            y = fallback
        if y_eeg not in (1, 2, 3):
            y_eeg = fallback
        if y_nirs is not None and y_nirs not in (1, 2, 3):
            y_nirs = fallback
        return {"y": y, "y_eeg": y_eeg, "y_nirs": y_nirs}

    def _iter_training_windows(self, eeg_indices: np.ndarray, states: np.ndarray):
        if len(eeg_indices) != len(states):
            raise ValueError("eeg_indices and states must have the same length")
        if len(eeg_indices) == 0:
            return
        eeg_indices = np.asarray(eeg_indices, dtype=np.int64)
        states = np.asarray(states, dtype=np.int32)
        order = np.argsort(eeg_indices)
        eeg_indices = eeg_indices[order]
        states = states[order]
        run_start = 0

        for i in range(1, len(eeg_indices) + 1):
            boundary = (
                i == len(eeg_indices)
                or eeg_indices[i] != eeg_indices[i - 1] + 1
                or states[i] != states[i - 1]
            )
            if not boundary:
                continue
            first_idx = int(eeg_indices[run_start])
            last_idx = int(eeg_indices[i - 1])
            label = int(states[run_start])
            seg_start, seg_end = first_idx, last_idx + 1
            if label in (1, 2, 3) and seg_end - seg_start >= self.win:
                min_end = seg_start + self.win
                end = ((min_end + self.step - 1) // self.step) * self.step
                while end <= seg_end:
                    start = end - self.win
                    if start >= 0 and end <= len(self.eeg_buffer):
                        yield start, end, label
                    end += self.step
            run_start = i

    @staticmethod
    def _combine_history_current(
        X_pre: np.ndarray,
        y_pre: np.ndarray,
        X_cur: np.ndarray,
        y_cur: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(X_pre) and X_pre.ndim == 2 and X_pre.shape[1] == X_cur.shape[1]:
            X = np.vstack((X_pre, X_cur))
            y = np.concatenate((y_pre, y_cur))
            w = np.concatenate((
                np.full(len(y_pre), HISTORICAL_SAMPLE_WEIGHT, dtype=np.float64),
                np.full(len(y_cur), CURRENT_SAMPLE_WEIGHT, dtype=np.float64),
            ))
        else:
            X, y = X_cur, y_cur
            w = np.full(len(y_cur), CURRENT_SAMPLE_WEIGHT, dtype=np.float64)
        return X, y, w

    def fit_block(self, block_idx: int, labels) -> None:
        states = np.asarray(labels.get("states", []), dtype=np.int32)
        eeg_indices = np.asarray(labels.get("eeg_indices", []), dtype=np.int64)

        new_count = 0
        for start, end, y in self._iter_training_windows(eeg_indices, states):
            hybrid_feat, eeg_feat, nirs_feat = self._features_for_window(start, end)
            self.X_hybrid.append(hybrid_feat)
            self.X_eeg.append(eeg_feat)
            if self.has_nirs:
                self.X_nirs.append(nirs_feat)
            self.y_train.append(int(y))
            new_count += 1
        if new_count == 0 or not self.y_train:
            return

        y_cur = np.asarray(self.y_train[-MAX_CURRENT_TRAIN:], dtype=np.int32)
        Xe_cur = np.asarray(self.X_eeg[-MAX_CURRENT_TRAIN:], dtype=np.float32)
        Xh_cur = np.asarray(self.X_hybrid[-MAX_CURRENT_TRAIN:], dtype=np.float32)

        Xe, ye, we = self._combine_history_current(self.pre_X_eeg, self.pre_y, Xe_cur, y_cur)
        self.model_eeg = self._new_rf(seed=42)
        self.model_eeg.fit(Xe, ye, sample_weight=we)
        self.model_eeg.set_params(n_jobs=1)

        if self.has_nirs:
            Xn_cur = np.asarray(self.X_nirs[-MAX_CURRENT_TRAIN:], dtype=np.float32)
            Xh, yh, wh = self._combine_history_current(self.pre_X_hybrid, self.pre_y, Xh_cur, y_cur)
            Xn, yn, wn = self._combine_history_current(self.pre_X_nirs, self.pre_y, Xn_cur, y_cur)

            self.model_hybrid = self._new_rf(seed=43)
            self.model_hybrid.fit(Xh, yh, sample_weight=wh)
            self.model_hybrid.set_params(n_jobs=1)

            self.model_nirs = self._new_rf(seed=44)
            self.model_nirs.fit(Xn, yn, sample_weight=wn)
            self.model_nirs.set_params(n_jobs=1)
