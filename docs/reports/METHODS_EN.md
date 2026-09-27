# Model and evaluation method

## Task and interface

The classifier predicts rest (1), imagined left-hand movement (2), and
imagined right-hand movement (3). `solution/run.py` supplies EEG samples and
optional NIRS samples in chronological order. Its one-second windows advance
by 250 ms. Block 1 is training only; `fit_block` receives labels only after
the corresponding block ends. The frozen runner, scorer, and I/O utilities
were not changed in this submission.

## Signal processing

EEG is cleaned of nonfinite values and passed through a fourth-order causal
Butterworth bandpass at 1-40 Hz. `scipy.signal.sosfilt` keeps its IIR state
between calls, including between 250 ms updates. One-second EEG features are
channel log variance and log power in 4-8, 8-13, 13-30, and 30-40 Hz. Four
additional features compare C3 and C4 band power. The C3/C4 features are
continuous inputs to the classifier; no fixed laterality threshold assigns
hand labels.

When NIRS exists, the classifier gathers only the HbO/HbR samples that have
arrived for the current EEG window. Per-channel mean, standard deviation,
and first-to-last change are appended to the EEG features. Three separate
random forests are trained for EEG only, NIRS only, and combined features.
The three models' class probabilities are aligned and combined with weights
0.50 for hybrid, 0.35 for EEG, and 0.15 for NIRS. The separate NIRS
prediction is returned for the competition's NIRS confusion matrix. With no
NIRS, the algorithm uses EEG alone.

Each forest has 90 trees, depth at most 12, minimum leaf size 2, and the
`balanced_subsample` class weighting mode. On each finished block the model
refits from the legally released current-session labels and a compact cache
of previous sessions. Historical examples have sample weight 0.55; current
session examples have 1.50. The cache is balanced to at most 400 examples
per class and sampled across as many sessions as possible.

## Data separation

The public bucket listed 56 `.mat` sessions from 14 people. The strict
evaluation cache excludes N17 and N18 entirely: 48 possible source files,
46 usable recordings with both modalities, and 1,200 cached windows.
The strict test contains eight N17/N18 sessions. A separate development
cache excludes N15, N16, N17, and N18, and was used only to compare the
fusion choices. Every scored session still learns from its own completed
blocks, as the competition interface allows.

The planned aggregate metrics on all 56 public sessions use a separate
leave-one-subject-out cache for each of the 14 people. The cache for person
Nxx contains only other people's recordings. The frozen scorer evaluates
each session separately before the unweighted session-level aggregate. The
14 caches are included; the full 56-session replay has not yet completed.

The final `solution/artifacts` cache is retrained from all public files with
compatible NIRS data for use on future hidden sessions. It must never be
used to claim an independent public-session test score. The downloaded raw
data remain outside the submission ZIP. The included public-data manifest
records names, sizes, and SHA256 hashes, and `download_sessions.py` can fetch
the files again.

## Metrics and runtime

The frozen scorer computes worst-class F1 and macro recall separately for
each session and averages those values over sessions. It adds a NIRS bonus
and subtracts penalties when inference or block fitting exceeds the stated
limits. We report accuracy for interpretation only. Engineering/report
points, up to 20, are not claimed here because the jury awards them.

The implementation uses Python, NumPy, SciPy, pandas, scikit-learn, and
joblib. Versions are pinned in `requirements.txt`. The supplied Dockerfile
specifies Python 3.11 and a `jupyter` user with UID 1000. The measured runs
in this workspace used Python 3.12; a Docker or Windows build was not
available to test here. See `DEPLOY.md` for exact commands.

## Limits

Eight held-out sessions come from only two people, so the number of truly
independent subjects is small. Data from the hidden competition set are
unavailable. The NIRS statistics use a one-second local window; they may
not capture the full delayed hemodynamic response. Quality and timing on
the hidden environment can differ from the public test.
