"""Frozen I/O and window-generation utilities.

Do not modify this file. The judge replaces it with a reference copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.io import loadmat

# EEG electrode order (EEG.Raw columns), zero-based.
EEG_CHANNEL_NAMES: list[str] = [
    "c4", "rpa", "f8", "p8", "f4", "p4", "fp2", "o2", "cz", "pz",
    "fz", "o1", "fp1", "p3", "f3", "p7", "f7", "lpa", "c3", "c7", "c8",
]
EEG_C3_IDX = 18
EEG_C4_IDX = 0

WINDOW_SEC = 1.0
STEP_SEC = 0.25
SKIP_BLOCK = 1  # the first block is not scored


def load_session(path: str | Path) -> dict[str, Any]:
    """Load one .mat session into a standardized dictionary."""
    path = Path(path)
    mat = loadmat(str(path), simplify_cells=True)
    if "EEG" not in mat:
        raise KeyError(f"{path.name}: missing EEG field")

    eeg = mat["EEG"]
    raw = np.asarray(eeg["Raw"], dtype=np.float64)
    states = np.asarray(eeg["States"]).astype(np.int32).ravel()
    blocks = np.asarray(eeg["Blocks"]).astype(np.int32).ravel()
    fs_eeg = float(np.asarray(eeg["Frq"]).ravel()[0])

    if raw.ndim != 2:
        raise ValueError(f"{path.name}: EEG.Raw must be 2D, got {raw.shape}")
    if raw.shape[0] != states.shape[0]:
        raise ValueError(f"{path.name}: EEG.Raw and EEG.States lengths do not match")

    subject = path.stem.split("_")[-1]
    out: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "subject": subject,
        "eeg": raw,
        "states": states,
        "blocks": blocks,
        "fs_eeg": fs_eeg,
        "has_nirs": False,
        "nirs_hbo": None,
        "nirs_hbr": None,
        "nirs_states": None,
        "nirs_blocks": None,
        "nirs_frame": None,
        "fs_nirs": None,
        "n_nirs_ch": 0,
    }

    if "NIRS" in mat and mat["NIRS"] is not None:
        nirs = mat["NIRS"]
        if isinstance(nirs, dict) and "HbO" in nirs:
            hbo = np.asarray(nirs["HbO"], dtype=np.float64)
            hbr = np.asarray(nirs["HbR"], dtype=np.float64)
            out.update(
                has_nirs=True,
                nirs_hbo=hbo,
                nirs_hbr=hbr,
                nirs_states=np.asarray(nirs["States"]).astype(np.int32).ravel(),
                nirs_blocks=np.asarray(nirs["Blocks"]).astype(np.int32).ravel(),
                nirs_frame=np.asarray(nirs["Frame"]).astype(np.int32).ravel(),
                fs_nirs=float(np.asarray(nirs["Frq"]).ravel()[0]),
                n_nirs_ch=int(hbo.shape[1]),
            )
    return out


def session_meta(session: dict[str, Any]) -> dict[str, Any]:
    """Metadata for OnlineClassifier.__init__."""
    return {
        "fs_eeg": session["fs_eeg"],
        "fs_nirs": session["fs_nirs"],
        "n_eeg_ch": int(session["eeg"].shape[1]),
        "n_nirs_ch": int(session["n_nirs_ch"]),
        "has_nirs": bool(session["has_nirs"]),
        "subject": session["subject"],
        "session_name": session["name"],
        "eeg_channel_names": list(EEG_CHANNEL_NAMES),
        "eeg_c3_idx": EEG_C3_IDX,
        "eeg_c4_idx": EEG_C4_IDX,
        "n_samples_eeg": int(session["eeg"].shape[0]),
    }


def iter_scored_windows(
    states: np.ndarray,
    blocks: np.ndarray,
    fs_eeg: float,
    window_sec: float = WINDOW_SEC,
    step_sec: float = STEP_SEC,
    skip_block: int = SKIP_BLOCK,
) -> Iterator[tuple[int, int, int]]:
    """Iterator over scored windows.

    Yields
    ------
    start, end, true_label
        EEG indices [start, end) (zero-based, exclusive end) and true class.
    """
    win = int(round(fs_eeg * window_sec))
    step = int(round(fs_eeg * step_sec))
    n = int(len(states))
    if win <= 0 or step <= 0:
        raise ValueError(f"invalid window/step: win={win}, step={step}")

    for end in range(win, n + 1, step):
        start = end - win
        seg_s = states[start:end]
        seg_b = blocks[start:end]
        if int(seg_b[0]) == skip_block:
            continue
        if not np.all(seg_b == seg_b[0]):
            continue
        lab = int(seg_s[0])
        if lab <= 0:
            continue
        if not np.all(seg_s == lab):
            continue
        yield start, end, lab


def nirs_slice_for_eeg_range(
    session: dict[str, Any],
    eeg_start: int,
    eeg_end: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Select NIRS samples corresponding to EEG interval [eeg_start, eeg_end).

    Frame in the file is a one-based EEG sample index.
    """
    if not session["has_nirs"]:
        return None, None
    frame = session["nirs_frame"]
    # Frame may repeat; take samples with one-based Frame in (eeg_start, eeg_end]
    # i.e. frame is in [eeg_start+1, eeg_end]
    lo = eeg_start + 1
    hi = eeg_end
    mask = (frame >= lo) & (frame <= hi)
    if not np.any(mask):
        z1 = session["nirs_hbo"][:0]
        z2 = session["nirs_hbr"][:0]
        return z1, z2
    return session["nirs_hbo"][mask], session["nirs_hbr"][mask]


def list_mat_files(path: str | Path) -> list[Path]:
    """List .mat files in a directory or return a single file."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.mat"))
