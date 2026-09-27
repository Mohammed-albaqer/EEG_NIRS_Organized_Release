# How the classifier works

This guide follows the actual code in `solution/classifier.py`. The files
`solution/run.py`, `solution/score.py`, and `solution/io_utils.py` are frozen
competition utilities. The root copies of those utilities support local
experiments. Change the classifier and artifacts, not the scorer, when
trying a new model.

## 1. What arrives and what must be returned

One `.mat` recording contains EEG samples, labels, block numbers, and
possibly NIRS HbO/HbR samples. The runner reads that file; the classifier
receives only streamed chunks and metadata. It predicts three labels:

| Number | Meaning |
| --- | --- |
| 1 | Rest |
| 2 | Imagined left-hand movement |
| 3 | Imagined right-hand movement |

At 250 EEG samples per second, a one-second scored window has 250 samples.
The next window ends about 250 ms later, so successive windows overlap.
The runner skips scoring block 1, but releases its labels when it finishes.
Only windows entirely inside one block and one labeled state are scored.

The classifier implements `OnlineClassifier(meta)`, `load(artifacts_dir)`,
`push(eeg_chunk, nirs_chunk)`, `predict()`, and `fit_block(block_idx, labels)`.
Its prediction dictionary contains `y`, `y_eeg`, and `y_nirs` when NIRS is
available. The scorer primarily uses `y`, and the NIRS bonus uses `y_nirs`.

## 2. Chronological execution

1. `io_utils.load_session()` reads a file and identifies C3 at channel
   index 18 and C4 at index 0 for the supplied EEG layout.
2. `run.py` creates the classifier and calls `load()` once. The artifact
   provides historical **features and labels**, not fitted trees.
3. `push()` receives only current and past chunks. EEG passes through a
   stateful fourth-order 1-40 Hz Butterworth bandpass implemented with
   SciPy `sosfilt`. Its state is retained across chunks.
4. `predict()` uses the latest one-second EEG window and only NIRS samples
   already received for that window. It returns a class before that block's
   true labels are disclosed.
5. At the end of a block, `run.py` passes its labeled sample indices to
   `fit_block()`. The classifier extracts labeled one-second windows and
   refits its forests. The next block can benefit from these new labels.
6. After the chronological pass, `score.py` reads predictions and timings
   and writes confusion matrices, F1, recall, and score. It is local
   evaluation code; the classifier never asks it for hidden labels.

The event order matters. If a window ends exactly at a block boundary,
the runner predicts before calling `fit_block` for that boundary. This
prevents that block's labels from being used to predict its own windows.

## 3. EEG features

The filter is **causal**: sample `t` is filtered using samples no later
than `t`. `filtfilt` would use future samples and is not used in the online
classifier. NaN/Inf input is replaced before filtering so the persistent
filter state stays finite.

For each EEG channel, a one-second window contributes five numbers:

- Log variance of the centered signal.
- Mean log spectral power in 4-8 Hz, 8-13 Hz, 13-30 Hz, and 30-40 Hz.

The spectrum uses a Hann taper and an FFT. Four further values subtract
C4 log band power from C3 log band power. The formula is a **continuous
feature**, not a fixed decision threshold: a forest may learn that a small
asymmetry is meaningful when other features support it. With 21 EEG
channels, the total is `21 * 5 + 4 = 109` EEG features.

The optimized implementation creates the taper and frequency masks once
per session. It also grows its EEG storage array geometrically instead of
copying the complete growing recording on every `push()` call.

## 4. NIRS and fusion

If NIRS exists, the classifier collects HbO and HbR samples corresponding
to the current EEG window. For each NIRS channel and each of HbO/HbR it
computes mean, standard deviation, and change from first to last sample.
With the supplied 18 NIRS channels, that is `18 * 2 * 3 = 108` NIRS
features. The combined feature vector has `109 + 108 = 217` numbers.

Three forests are fitted when NIRS exists:

| Forest | Input | Output |
| --- | --- | --- |
| EEG | 109 EEG features | EEG-only class probabilities |
| NIRS | 108 NIRS features | NIRS-only class probabilities |
| Hybrid | 217 combined features | Combined class probabilities |

Each forest uses 90 trees, depth at most 12, minimum leaf size 2,
`max_features="sqrt"`, and `class_weight="balanced_subsample"`.
The three probability vectors are aligned to classes 1, 2, and 3, since
an early block may lack one class. The final class is the largest element
of this weighted vector:

```text
0.50 * hybrid probabilities
+ 0.35 * EEG probabilities
+ 0.15 * NIRS probabilities
```

The values are class **probabilities**, not numeric label averages. The
separate EEG and NIRS classes also come from those same probabilities, so
each forest is queried only once per prediction. If NIRS is absent, the
EEG forest is used alone.

## 5. Historical cache and online fitting

`train.py` replays public recordings through the same causal feature
extractor and selects up to 400 windows per class, drawing across many
sessions. Its `.npz` file contains arrays `X_eeg`, `X_nirs`,
`X_hybrid`, `y`, and `source_session_id`. The manifest lists the recordings;
the JSON metadata records feature dimensions, filter settings, classes,
subjects, and cache-building options.

This feature-cache step is distinct from fitting Random Forest trees. At
each completed block, `fit_block()` combines the cache with legally
released current-session windows. Historical samples receive weight 0.55;
current-session samples receive weight 1.50, with at most the latest 2,600
current-session windows used in a fit. It fits the EEG forest and, when
NIRS exists, the hybrid and NIRS forests. Forests are rebuilt at every
completed block; this code does **not** use `warm_start`.

The eight-session independent-person check uses
`results/holdout/artifacts_N17_N18/`: 46 compatible other-person sessions
and no N17/N18 source. The final `solution/artifacts/` cache instead uses
54 compatible public sessions, including N17/N18. Its purpose is later
independent hidden recordings. Do not use it to claim an N17/N18 holdout
score. The two public files without compatible NIRS were omitted from the
joint feature cache; an EEG-only evaluation path still exists.

## 6. How the reported percentage is calculated

`evaluate_many.py` runs each of the eight test sessions with the frozen
chronological runner, then `score.py` evaluates each separately. The
aggregate gives every session equal weight. Mean accuracy counts correct
scored windows within each session and averages the eight session rates;
the user's latest console showed **65.12%**.

The competition-style computed points use the average of each session's
**worst-class F1**, average macro recall, a NIRS-only bonus, and timing
penalties. The latest reported values were F1 **0.4181**, recall **0.5881**,
NIRS bonus **8.7541**, no penalty, and computed score **51.4810/130**.
See `USER_MAC_RESULTS_EN.md` for the comparison and caveats. A jury may
separately award design/report points; the hidden score is unknown.

## 7. What each historical model change did

| Stage | Change | What was measured |
| --- | --- | --- |
| Original | A smaller one-session Random Forest baseline | 58.12% on an older seven-session split; not directly paired with the later eight-session tests |
| Causal and multi-session | Persistent causal EEG filter, C3/C4 and NIRS features, balanced cache and legal block adaptation | Created the comparable N17/N18 testing setup |
| Hard agreement | EEG/NIRS label agreement could override hybrid | 63.67% accuracy and 50.2353/130 on N17/N18 |
| Probability fusion | Weighted probabilities replaced the hard override | 65.18% and 51.2448/130 on the same N17/N18 split |
| Faster implementation | Reused FFT geometry, amortized EEG storage, one probability query per forest, stronger artifact checks | User's Mac: 65.12% and 51.4810/130; one-sample prediction parity and faster local prediction measured separately |
| Extra Trees candidate | Different randomized tree splits | One-sample accuracy increased, but competition score fell; not selected |

The Mac and earlier eight-session results used the same named test people
but different machines. Their small score difference cannot be assigned
entirely to the speed refactor. Only the original hard-agreement versus
probability-fusion comparison was controlled in one environment.
