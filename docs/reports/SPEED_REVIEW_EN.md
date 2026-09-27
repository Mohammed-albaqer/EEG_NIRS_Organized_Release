# Faster streaming review and short functional comparison

The submission uses the same EEG/NIRS features, 90-tree Random Forests,
historical cache, and probability-fusion weights as the earlier measured
classifier. It now avoids repeated EEG buffer copies, builds FFT masks once,
and obtains each forest's probabilities once per prediction. The separate
Extra Trees candidate uses the same inputs with randomized split thresholds.

## One bundled sample session

The sample `2025.12.01.16.04.01_N01.mat` supplied 1,550 scored windows and
seven legal block fits. All three runs used the same `samples/artifacts` cache.
That cache includes **other sessions by N01**; this is a functional smoke
comparison and does not estimate performance on an unseen person.

| Model | Accuracy | Worst-class F1 | Macro recall | Score / 130 | Mean predict | Mean block fit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier probability-fusion Random Forest | 44.45% | 0.3454 | 0.4561 | 43.5812 | 15.71 ms | 1.027 s |
| Optimized Random Forest submission | 44.45% | 0.3454 | 0.4561 | 43.5812 | 9.08 ms | 0.955 s |
| Experimental Extra Trees | 49.87% | 0.3285 | 0.4470 | 42.3154 | 8.86 ms | 0.459 s |

The two Random Forest runs wrote **byte-identical predictions for all 1,550
windows** and had the same score. The optimized run's average `predict`
time was about **42% lower**, or **1.73 times faster**, on this machine and
sample. Single-run timings can vary. The gain came from doing less repeated
work, not a demonstrated increase in accuracy.

Extra Trees had higher accuracy on this one sample, but weaker worst-class
F1 and a **1.2658-point lower competition score**. Because the competition
rewards the weakest class, it is not promoted to the submission. It also
has not been tested on the independent N17/N18 sessions.

## Readiness

Fast checks passed: syntax compilation, causal filtering across chunk
boundaries, no influence from future EEG, finite persistent filter state,
artifact metadata mismatch rejection, and source/artifact layout checks.
One sample session completed end-to-end with no timing penalty.

The earlier **65.18% on N17/N18** belongs to the archived pre-refactor
probability-fusion code. The user later reported **65.12% and 51.4810/130**
for the optimized source over the same eight named sessions on a Mac; see
`../USER_MAC_RESULTS_EN.md`. The full original Mac prediction and timing
files were not supplied, so cross-machine timing and per-window parity on
those eight sessions cannot be confirmed from the console text. The
56-session public leave-one-subject-out aggregate is still pending, and the
hidden competition score is unknown. To repeat the current source, run
`02_STRICT_TEST_WINDOWS.bat`,
download all recordings with `python download_sessions.py --out data_all`,
and run `python evaluate_public_loso.py --data data_all --out results/metrics`.

## Source fragments

Previously, every `push` copied the entire EEG recording:

```python
self.eeg_buffer = np.vstack((self.eeg_buffer, eeg_filtered))
```

Now the recording grows geometrically and only the new chunk is written:

```python
if needed > len(self._eeg_storage):
    capacity = max(needed, 1024, 2 * len(self._eeg_storage))
    storage = np.empty((capacity, self.n_eeg_ch), dtype=np.float64)
    storage[:eeg_start] = self.eeg_buffer
    self._eeg_storage = storage
self._eeg_storage[eeg_start:eeg_end] = eeg_filtered
self.eeg_buffer = self._eeg_storage[:eeg_end]
```

The earlier `predict` used `model.predict` for EEG and NIRS labels and then
called `predict_proba` again for their fusion weights. The current version
calls `predict_proba` once per available model and uses those same aligned
probabilities for the individual labels and the fused result.
