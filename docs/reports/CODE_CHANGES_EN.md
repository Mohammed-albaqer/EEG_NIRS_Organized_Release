# Exact model change from hard agreement to probability fusion

## Later source review and faster inference

The measured comparison below used
`experiments/classifier_probability_fusion.py`. The current submission keeps
its learned feature representation, 90-tree Random Forest settings, and
0.50/0.35/0.15 probability fusion. A later source review made these changes:

| Area | Previously | Current code | Expected effect |
| --- | --- | --- | --- |
| EEG recording | `np.vstack` copied the growing recording on every push | A geometric-capacity array grows only when full | Less copying in long streams; same sample order |
| Online inference | EEG and NIRS `.predict` followed by `.predict_proba` for fusion | One probability call per model supplies both class and fusion | Fewer forest traversals per prediction |
| Spectral setup | Hann taper and frequency masks recreated per window | Construct once in `__init__` | Less repeated work; same frequency bins |
| Cache input | EEG shape and filter checked | Validate metadata, all feature shapes, labels, and finite values | A broken artifact fails early |
| NIRS input | Channel count could be inconsistent | Validate the expected channel shape | Clear failure before model training |

The old inference path could traverse the EEG and NIRS forests twice. The
current path aligns each forest's `predict_proba` result to classes 1, 2, 3,
then uses the largest value for its individual output. With the same trained
trees, this matches scikit-learn classification's probability argmax. The
change was timed on one bundled sample and later scored on eight sessions
in a user-reported Mac console run: 65.12% and 51.4810/130. Its eight-run
per-call timing files were not supplied. See `../USER_MAC_RESULTS_EN.md` for
the score comparison and cross-machine limitation; source review and one
sample alone cannot prove the same whole-dataset speed or score.

An independent candidate, `experiments/classifier_extra_trees.py`, replaces
the Random Forests with randomized Extra Trees while keeping the interface,
feature cache, and class-fusion rule. This is a new algorithm to evaluate;
neither an independent score gain nor a cross-subject timing gain is claimed.
The one-sample observation is in `SPEED_REVIEW_EN.md`. Run
`03_TEST_EXTRA_TREES_WINDOWS.bat` to compare it on N17/N18 before deciding
whether to use it for submission.

The public 56-session aggregate was not finished. Its 14 held-out-person
feature caches are included; `results/metrics/metrics.json` is pending.

## Earlier measured fusion change

The earlier probability-fusion selection changed only the decision rule inside
`OnlineClassifier.predict`. Training, causal EEG filtering, feature
dimensions, forest settings, artifact format, and NIRS-only output stayed
the same. The old source is retained in
`experiments/classifier_hard_agreement.py`; its tested successor is
`experiments/classifier_probability_fusion.py`, with later speed changes in
`solution/classifier.py`. The literal earlier source diff is
`docs/reports/classifier_fusion.patch`. The short excerpts below omit the existing
fallback guards for readability.

## Before

```python
y_eeg = int(self.model_eeg.predict(eeg_feat.reshape(1, -1))[0])
y_nirs = int(self.model_nirs.predict(nirs_feat.reshape(1, -1))[0])
y = int(self.model_hybrid.predict(hybrid_feat.reshape(1, -1))[0])
if y_eeg == y_nirs:
    y = y_eeg
```

This makes matching EEG and NIRS class labels override the hybrid model,
even when those two models are uncertain.

## After

```python
ph = aligned(self.model_hybrid, hybrid_feat)
pe = aligned(self.model_eeg, eeg_feat)
pn = aligned(self.model_nirs, nirs_feat)
y = int(np.argmax(0.50 * ph + 0.35 * pe + 0.15 * pn) + 1)
```

`aligned` maps each model's `predict_proba` output onto positions for
classes 1, 2, and 3. This matters when a model trained after an early
block has not yet seen every class. The final output is a weighted mean
of **probabilities**, not an arithmetic mean of class numbers. The
separate NIRS-only output `y_nirs` remains available to the scorer.

## Controlled comparison

On N15/N16 development sessions, the previous and new decision rules
shared the same subject-excluding artifact, Random Forest configuration,
runner, and scorer. The new rule scored 40.5059 vs 39.8895/130. We then
evaluated N17/N18 with a different artifact that excluded them both.
There, the final rule scored 51.2448 vs 50.2353/130 and improved average
accuracy from 63.67% to 65.18%. See `VALIDATION_EN.md` for the F1,
macro recall, session table, time measurements, and uncertainty.

The previously supplied classifier also received two robustness fixes:
EEG NaN/Inf values are cleaned before the stateful filter, and loaded
artifact filter metadata must match the runtime filter. These fixes do
not change clean, finite input predictions. `verify_runtime.py` exercises
both cases plus filter causality and chunk invariance.
