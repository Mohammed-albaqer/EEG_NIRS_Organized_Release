# Validation results and submission decision

## Version boundary

All numeric comparisons below were measured on the archived source
`experiments/classifier_probability_fusion.py`, before the speed and cache
validation changes in the present `solution/classifier.py`. The speed refactor
preserves the filter, feature definitions, forest parameters, and fusion
weights by design. The user later reported an eight-session Mac run of the
optimized source: 65.12% and 51.4810/130. Its console-derived comparison
and different-machine limitation are in `../USER_MAC_RESULTS_EN.md`. The
optional `experiments/classifier_extra_trees.py` is a
new algorithm with only a single-session smoke measurement and **no
independent-person result**. Do not attribute the 65.18% number or the
timing values below to either newer source. See `SPEED_REVIEW_EN.md` for
the short sample comparison.

The 56 public leave-one-subject-out sessions have **not** all been scored:
the long evaluation was stopped at user request. The 14 fold feature caches
are provided, but `results/metrics/metrics.json` is pending. This archive
cannot claim a complete official-style public aggregate or hidden score.

## Decision

The selected classifier uses probability fusion of the three trained Random
Forests. On a separate eight-session test from N17 and N18, it achieved
**65.18% mean accuracy**, **0.4152 mean worst-class F1**,
**0.5888 mean macro recall**, and **51.2448/130 computed points** with jury
engineering points set to zero. The eight sessions represent **two people**.
The hidden competition result remains unknown.

## Development and independent test

The development cache excluded N15, N16, N17, and N18. It used 38 of the
40 available other-subject sessions; two lacked compatible NIRS. The model
choice was made on eight N15/N16 sessions. The independent N17/N18 cache
excluded those two people, used 46 of 48 eligible other-subject sessions,
and held eight N17/N18 sessions for the last evaluation. Every session
still learned from its own completed blocks through the legal `fit_block`
interface. All evaluation uses the frozen chronological runner and scorer.

| Evaluation set and model | Mean accuracy | Worst-class F1 | Macro recall | NIRS worst-class F1 | Computed points / 130 |
|---|---:|---:|---:|---:|---:|
| N15/N16 development, hard agreement | 45.57% | 0.3062 | 0.4314 | 0.2381 | 39.8895 |
| N15/N16 development, probability fusion | **48.70%** | **0.3077** | **0.4490** | 0.2381 | **40.5059** |
| N17/N18 independent, hard agreement | 63.67% | 0.4050 | 0.5754 | 0.2447 | 50.2353 |
| N17/N18 independent, probability fusion | **65.18%** | **0.4152** | **0.5888** | 0.2447 | **51.2448** |

The selected model improved the independent mean by **1.51 percentage
points in accuracy**, **0.0101 in worst-class F1**, and **1.0095 computed
points**. Its NIRS-only model was unchanged. Its mean of per-session
prediction times was approximately **0.0154 s**, compared with **0.0100 s**
for hard agreement. The mean of per-session block-fit times was approximately
**0.959 s**, versus **0.984 s**. These times were measured on this machine;
both aggregates had zero official timing penalties here. The jury awards
up to 20 design/report points separately, so these computed figures are
not full official final scores.
The selected model's largest observed individual `predict` call was
0.0664 s and its largest `fit_block` call was 1.1792 s across these
eight sessions.

| N17/N18 session | Previous accuracy | Selected accuracy | Previous points | Selected points |
|---|---:|---:|---:|---:|
| 2026.03.23 14:24 N17 | 58.58% | 60.00% | 42.4747 | 44.5078 |
| 2026.03.24 18:00 N18 | 66.84% | 68.46% | 56.8363 | 58.3093 |
| 2026.03.25 12:49 N18 | 69.75% | 74.42% | 58.6647 | 63.4620 |
| 2026.03.25 13:04 N18 | 58.32% | 60.00% | 47.5072 | 49.3031 |
| 2026.03.25 14:35 N17 | 60.79% | 62.33% | 49.1332 | 51.2803 |
| 2026.03.25 14:46 N17 | 60.66% | 61.63% | 49.6808 | 48.1972 |
| 2026.03.26 12:37 N18 | 67.53% | 67.01% | 46.0171 | 42.9874 |
| 2026.03.26 14:21 N17 | 66.88% | 67.59% | 51.5684 | 51.9111 |

Six of eight sessions improved in computed points and two declined.
The four N17 sessions averaged a 0.7598-point gain; the four N18 sessions
averaged a 1.2591-point gain. These are still only two independent people.
Resampling the eight paired session score differences 20,000 times with
seed 2026 gives an illustrative percentile interval of **-0.58 to +2.52
points** for the mean change. It includes zero; there is no reliable proof
that a similar gain will hold for new people.

## Precisely what changed in the model

The previous version let the EEG and NIRS class labels override the hybrid
prediction whenever those two labels agreed. The selected version instead
aligns the forests' probabilities to classes 1, 2, and 3, then chooses:

```python
score = 0.50 * hybrid_probability + 0.35 * eeg_probability + 0.15 * nirs_probability
y = int(np.argmax(score) + 1)
```

The filter, features, forest architecture, historical sample weighting,
and block retraining remained the same. The probability weights were fixed
before the N17/N18 evaluation after checking the development set; we did
not fit them to those test sessions. The tested code is preserved in
`experiments/classifier_probability_fusion.py`. The optimized current source
is identical in `classifier.py` and `solution/classifier.py`. A reference
copy is kept under `experiments/classifier_optimized_rf_reference.py`.

The received model was previously measured on seven sessions from three
people at 58.12% accuracy and 46.1005/130 with a leave-one-subject-out
cache. That was a different data split and is not a paired comparison to
the 65.18% number above. The directly comparable numbers are the N17/N18
rows of the table.

## Final artifact and public subject holdout metrics

`solution/artifacts/` contains 1,200 balanced cached windows from **54
usable public recordings**. Two of the 56 public recordings had no
compatible NIRS and were skipped by the cache builder. This final artifact
is intended for an independent hidden test. To produce the public submission
metrics without reusing a tested person's own data, `build_public_loso.py`
creates 14 other-subject caches, and `evaluate_public_loso.py` replays all
56 recordings with the corresponding cache. `results/metrics/metrics.json`
will be the aggregate of those 56 per-session online scores once generated.
This retrospective
leave-one-subject-out protocol is distinct from the separately selected
N17/N18 test: every public person participates in the aggregate, and some
of the other-subject sessions have later calendar dates. Do not present it
as the hidden competition score.
The two recordings without NIRS are still scored through the EEG-only
fallback; they have no NIRS-only confusion matrix. The frozen aggregator
averages the NIRS-only metrics over the sessions where that modality exists.

## Reproducibility and limits

- `results/holdout/development_baseline/` and
  `results/holdout/development_probability_fusion/` retain eight predictions,
  timings, confusion matrices, per-session metrics, and aggregates each.
- `results/holdout/baseline_N17_N18/` and
  `results/holdout/final_selected_N17_N18/` retain the same evidence on the
  independent-person test. Their corresponding training manifests are in
  `results/holdout/artifacts_dev_N15_N16/` and
  `results/holdout/artifacts_N17_N18/`.
- `strict_test.py` refuses to run if N17 or N18 appears in its cache's
  manifest. `verify_runtime.py` checks chunk invariance, future independence,
  nonfinite EEG handling, and cache filter metadata.
- `quick_test.py` checks a bundled sample without downloading data. It is
  a smoke test and uses a cache containing other sessions of the same person.
- Results were produced with Python 3.12 and pinned package versions.
  The Python 3.11 Docker build and a Windows run were not available for
  verification in this environment. Timing may vary there.
- A previous user-supplied console log reported 63.19% and 49.4304/130
  for a related 46-session run. Its trained artifact and prediction files
  were not supplied. This package retrained independently from the public
  recordings, so the older logged number is context rather than a score
  claimed for the included model.

Run `python strict_test.py --data data_all --out results/my_strict_test`
after downloading the eight N17/N18 public sessions, or execute the Windows
batch file. The scorer's `metrics.json` is the authoritative numeric output.
