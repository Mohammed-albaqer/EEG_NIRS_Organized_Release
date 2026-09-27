"""Focused checks for the stateful online EEG filter and cache metadata."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from classifier import OnlineClassifier


def main() -> None:
    meta = {"fs_eeg": 250, "n_eeg_ch": 21, "has_nirs": False}
    rng = np.random.default_rng(2026)
    data = rng.standard_normal((1500, 21))

    whole = OnlineClassifier(meta)
    whole.push(data, None)
    chunked = OnlineClassifier(meta)
    for start in range(0, len(data), 125):
        chunked.push(data[start : start + 125], None)
    np.testing.assert_allclose(whole.eeg_buffer, chunked.eeg_buffer, atol=1e-11, rtol=0)

    changed_future = data.copy()
    changed_future[1000:] += 1000
    alternate = OnlineClassifier(meta)
    alternate.push(changed_future, None)
    np.testing.assert_allclose(whole.eeg_buffer[:1000], alternate.eeg_buffer[:1000], atol=1e-11, rtol=0)

    corrupted = data.copy()
    corrupted[200, 1] = np.nan
    corrupted[300, 2] = np.inf
    safe = OnlineClassifier(meta)
    safe.push(corrupted, None)
    assert np.isfinite(safe.eeg_buffer).all()
    assert np.isfinite(safe.eeg_filter_zi).all()

    with tempfile.TemporaryDirectory() as directory:
        artifacts = Path(directory)
        (artifacts / "pretrain_cache.npz").touch()
        (artifacts / "artifact_meta.json").write_text(
            json.dumps({"filter": {"low_hz": 1.0, "high_hz": 35.0, "order": 4}}),
            encoding="utf-8",
        )
        try:
            OnlineClassifier(meta).load(artifacts)
        except ValueError as error:
            assert "filter" in str(error)
        else:
            raise AssertionError("Mismatched historical filter was accepted")

    print("PASS: chunk boundaries preserve the causal filter output")
    print("PASS: changing future EEG does not change earlier output")
    print("PASS: nonfinite EEG cannot poison persistent filter state")
    print("PASS: incompatible historical filter metadata is rejected")


if __name__ == "__main__":
    main()
