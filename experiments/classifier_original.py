"""Random-Forest implementation of the hackathon OnlineClassifier contract.

Contract:
    __init__ -> load? -> (push | predict | fit_block)*

Classes:
    1 = rest
    2 = left hand imagery
    3 = right hand imagery
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class OnlineClassifier:
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta

        self.fs_eeg = float(meta["fs_eeg"])
        self.n_eeg_ch = int(meta["n_eeg_ch"])
        self.has_nirs = bool(meta.get("has_nirs", False))
        self.n_nirs_ch = int(meta.get("n_nirs_ch") or 0)

        self.win = int(round(self.fs_eeg * 1.0))
        self.step = int(round(self.fs_eeg * 0.25))

        self.c3_idx = int(meta.get("eeg_c3_idx", 18))
        self.c4_idx = int(meta.get("eeg_c4_idx", 0))

        self.eeg_buffer = np.empty((0, self.n_eeg_ch), dtype=np.float64)
        self.nirs_records: list[dict[str, Any]] = []

        self.X_hybrid: list[np.ndarray] = []
        self.X_eeg: list[np.ndarray] = []
        self.X_nirs: list[np.ndarray] = []
        self.y_train: list[int] = []

        self.model_hybrid: RandomForestClassifier | None = None
        self.model_eeg: RandomForestClassifier | None = None
        self.model_nirs: RandomForestClassifier | None = None

        self._is_fitted_hybrid = False
        self._is_fitted_eeg = False
        self._is_fitted_nirs = False

        self.eeg_feature_dim = self.n_eeg_ch * 5 + 4
        self.nirs_feature_dim = 6 * self.n_nirs_ch if self.has_nirs else 0

    @staticmethod
    def _new_rf(seed: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    def load(self, artifacts_dir: str | Path) -> None:
        artifacts_dir = Path(artifacts_dir)

        p = artifacts_dir / "rf_hybrid.joblib"
        if p.exists():
            self.model_hybrid = joblib.load(p)
            self.model_hybrid.set_params(n_jobs=1)
            self._is_fitted_hybrid = True

        p = artifacts_dir / "rf_eeg.joblib"
        if p.exists():
            self.model_eeg = joblib.load(p)
            self.model_eeg.set_params(n_jobs=1)
            self._is_fitted_eeg = True

        p = artifacts_dir / "rf_nirs.joblib"
        if self.has_nirs and p.exists():
            self.model_nirs = joblib.load(p)
            self.model_nirs.set_params(n_jobs=1)
            self._is_fitted_nirs = True

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
            self.eeg_buffer = np.vstack((self.eeg_buffer, eeg_chunk))

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

        var = np.var(x, axis=0) + 1e-12
        log_var = np.log(var)

        taper = np.hanning(len(x))[:, None]
        spec = np.fft.rfft(x * taper, axis=0)
        power = (np.abs(spec) ** 2) / max(1, len(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.fs_eeg)

        bands = (
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 40.0),
        )

        band_logs: list[np.ndarray] = []
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
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

        feat = np.concatenate([per_channel, np.asarray(asym)])
        return self._clean(feat).astype(np.float32)

    def _nirs_window(
        self, eeg_start: int, eeg_end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.has_nirs or self.n_nirs_ch <= 0:
            return np.empty((0, 0)), np.empty((0, 0))

        hbo_parts = []
        hbr_parts = []

        for rec in self.nirs_records:
            if rec["eeg_end"] <= eeg_start:
                continue
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

        return np.vstack(hbo_parts), np.vstack(hbr_parts)

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

            if len(a) >= 2:
                delta = a[-1] - a[0]
            else:
                delta = np.zeros(a.shape[1])

            return np.column_stack((mean, std, delta)).reshape(-1)

        feat = np.concatenate((stats(hbo), stats(hbr)))
        return self._clean(feat).astype(np.float32)

    def _features_for_window(
        self,
        eeg_start: int,
        eeg_end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eeg_window = self.eeg_buffer[eeg_start:eeg_end]
        eeg_feat = self._eeg_features(eeg_window)

        if self.has_nirs:
            nirs_feat = self._nirs_features(eeg_start, eeg_end)
            hybrid_feat = np.concatenate((eeg_feat, nirs_feat)).astype(np.float32)
        else:
            nirs_feat = np.empty(0, dtype=np.float32)
            hybrid_feat = eeg_feat

        return hybrid_feat, eeg_feat, nirs_feat

    @staticmethod
    def _majority_fallback(y: list[int]) -> int:
        if not y:
            return 1

        counts = np.bincount(np.asarray(y, dtype=int), minlength=4)
        return int(np.argmax(counts[1:4]) + 1)

    def predict(self) -> dict:
        end = len(self.eeg_buffer)
        start = end - self.win

        fallback = self._majority_fallback(self.y_train)

        if start < 0:
            return {
                "y": fallback,
                "y_eeg": fallback,
                "y_nirs": fallback if self.has_nirs else None,
            }

        hybrid_feat, eeg_feat, nirs_feat = self._features_for_window(start, end)

        if self._is_fitted_eeg and self.model_eeg is not None:
            y_eeg = int(self.model_eeg.predict(eeg_feat.reshape(1, -1))[0])
        else:
            y_eeg = fallback

        y_nirs = None
        if self.has_nirs:
            if self._is_fitted_nirs and self.model_nirs is not None:
                y_nirs = int(
                    self.model_nirs.predict(nirs_feat.reshape(1, -1))[0]
                )
            else:
                y_nirs = fallback

        if self._is_fitted_hybrid and self.model_hybrid is not None:
            y = int(
                self.model_hybrid.predict(hybrid_feat.reshape(1, -1))[0]
            )
        else:
            y = y_eeg

        if y not in (1, 2, 3):
            y = fallback
        if y_eeg not in (1, 2, 3):
            y_eeg = fallback
        if y_nirs is not None and y_nirs not in (1, 2, 3):
            y_nirs = fallback

        # Trust agreement between sensors; otherwise keep the hybrid prediction.
        if y_nirs is not None and y_eeg == y_nirs:
            y = y_eeg

        return {
            "y": y,
            "y_eeg": y_eeg,
            "y_nirs": y_nirs,
        }

    def _iter_training_windows(
        self,
        eeg_indices: np.ndarray,
        states: np.ndarray,
    ):
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

            seg_start = first_idx
            seg_end = last_idx + 1

            if label in (1, 2, 3) and seg_end - seg_start >= self.win:
                min_end = seg_start + self.win
                end = ((min_end + self.step - 1) // self.step) * self.step

                while end <= seg_end:
                    start = end - self.win

                    if start >= 0 and end <= len(self.eeg_buffer):
                        yield start, end, label

                    end += self.step

            run_start = i

    def fit_block(self, block_idx: int, labels: dict) -> None:
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

        y = np.asarray(self.y_train, dtype=np.int32)
        X_h = np.asarray(self.X_hybrid, dtype=np.float32)
        X_e = np.asarray(self.X_eeg, dtype=np.float32)

        # Keep online fitting bounded in time.
        max_train = 4000
        if len(y) > max_train:
            y = y[-max_train:]
            X_h = X_h[-max_train:]
            X_e = X_e[-max_train:]

        self.model_eeg = self._new_rf(seed=42)
        self.model_eeg.fit(X_e, y)
        self.model_eeg.set_params(n_jobs=1)
        self._is_fitted_eeg = True

        self.model_hybrid = self._new_rf(seed=43)
        self.model_hybrid.fit(X_h, y)
        self.model_hybrid.set_params(n_jobs=1)
        self._is_fitted_hybrid = True

        if self.has_nirs and self.n_nirs_ch > 0 and self.X_nirs:
            X_n = np.asarray(self.X_nirs, dtype=np.float32)

            if len(X_n) > max_train:
                X_n = X_n[-max_train:]

            self.model_nirs = self._new_rf(seed=44)
            self.model_nirs.fit(X_n, y)
            self.model_nirs.set_params(n_jobs=1)
            self._is_fitted_nirs = True
