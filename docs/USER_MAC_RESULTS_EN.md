# Independent-person test reported from macOS

**Source:** terminal output supplied by the user on 2026-09-27 after running
`caffeinate python strict_test.py --data data_all --out results/my_strict_test`.
The command evaluated the optimized Random Forest classifier on eight N17/N18
recordings. The package includes only a transcription of this terminal text;
the original Mac `predictions.csv` and `timings.csv` were not supplied. The
previous comparison is the archived run at
`results/holdout/final_selected_N17_N18/` from another machine.

## Aggregate comparison

| Metric | Earlier probability fusion | Optimized model on user's Mac | Change |
| --- | ---: | ---: | ---: |
| Sessions / independent people | 8 / 2 | 8 / 2 | Same split |
| Mean accuracy (diagnostic) | 65.18% | 65.12% | -0.06 percentage points |
| Mean weakest-class F1 | 0.4152 | 0.4181 | +0.0029 |
| Mean macro recall | 0.5888 | 0.5881 | -0.0007 |
| NIRS weakest-class F1 | 0.2447 | 0.2503 | +0.0056 |
| Base-quality points | 42.5746 | 42.7269 | +0.1523 |
| NIRS bonus points | 8.6702 | 8.7541 | +0.0839 |
| Timing penalties | 0 | 0 | 0 |
| Computed score / 130 | 51.2448 | **51.4810** | **+0.2362** |

The scorer uses an **unweighted mean over sessions** for the two main
quality metrics. Its computed total is approximately
`60 * mean_min_F1 + 30 * mean_macro_recall + (5 + 15 * NIRS_min_F1)
- timing_penalties`. The submitted console numbers yield 42.7269 base points
and 8.7541 NIRS points. Accuracy is helpful for interpretation but is not
the primary score. Jury design/report points are set to zero here and the
hidden test result is unknown.

## Session-by-session computed score

| Recording | Windows | Earlier | Mac base | Mac NIRS bonus | Mac total | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026.03.23.14.24.35_N17 | 1,550 | 44.5078 | 37.7424 | 6.3166 | 44.0590 | -0.4488 |
| 2026.03.24.18.00.22_N18 | 1,547 | 58.3093 | 48.5773 | 9.8364 | 58.4137 | +0.1044 |
| 2026.03.25.12.49.29_N18 | 1,544 | 63.4620 | 51.9151 | 10.0410 | 61.9561 | -1.5059 |
| 2026.03.25.13.04.32_N18 | 1,545 | 49.3031 | 39.1914 | 10.0000 | 49.1914 | -0.1117 |
| 2026.03.25.14.35.20_N17 | 1,553 | 51.2803 | 43.5714 | 8.6683 | 52.2397 | +0.9594 |
| 2026.03.25.14.46.04_N17 | 1,548 | 48.1972 | 40.2298 | 9.6625 | 49.8923 | +1.6951 |
| 2026.03.26.12.37.48_N18 | 1,552 | 42.9874 | 36.0675 | 7.8922 | 43.9597 | +0.9723 |
| 2026.03.26.14.21.39_N17 | 1,546 | 51.9111 | 44.5207 | 7.6154 | 52.1361 | +0.2250 |

Five sessions improved and three declined. The largest increase was 1.6951
points; the largest decline was 1.5059 points. The console reported about
12,385 scored windows in total and seven completed-block fits per session,
or 56 block fits over the eight sessions. The first block provides legal
training labels after it finishes and is not scored.
The NIRS-only weakest-class F1 varied from **0.0878** to **0.3361** across
sessions; exact per-session NIRS values and bonus points are retained in
`results/reported_mac_2026-09-27/session_scores_from_console.csv`.

## How the training and test are separated

The `results/holdout/artifacts_N17_N18/` cache was extracted from **46
compatible recordings by people other than N17 and N18**. Two other eligible
public recordings lacked compatible NIRS and were skipped by the joint-cache
builder. The cache keeps 1,200 windows: 400 rest, 400 left imagery, and 400
right imagery. It is a feature cache rather than a saved trained forest.

During each test session, the classifier receives only samples that have
arrived. After a block ends, the runner releases that block's labels and
`fit_block()` trains three 90-tree Random Forests from the historical
cache plus eligible labeled windows from the current session. Historical
examples weigh 0.55 and current-session examples weigh 1.50. The forests
are refitted after each of the seven blocks. The scored windows of a block
are predicted before its labels are released.

The public recording filenames span 2025-12-01 through 2026-03-27. This is
a **retrospective person holdout**, not a strictly calendar-ordered replay
across people: some other-person cache recordings have later dates than
the N17/N18 sessions. N17 and N18 themselves are absent from their holdout
cache. The final `solution/artifacts/` cache differs: it was extracted from
54 compatible public recordings, including N17/N18, and must be reserved
for truly independent future sessions.

The original cache-building logs recorded **89.65 seconds for the 46-session
holdout cache** and **102.69 seconds for the 54-session final cache** on the
earlier workspace. Those figures measure feature extraction/cache creation,
not the user's Mac test duration or the total training time of its forests.
The pasted Mac output contains no wall-clock duration or per-call times.
The eight-session run is shorter than the planned 56-session public replay
because it processes only eight recordings and reuses an existing cache.
The code also avoids repeated EEG buffer copies, reuses FFT masks, and makes
one probability call per forest; the separate one-sample smoke comparison
measured 15.71 ms versus 9.08 ms per prediction on the earlier machine.
That local timing cannot be transferred directly to the Mac.

## Decision and limits

Keep the optimized Random Forest as the default classifier. The computed
score and weakest-class F1 rose slightly, accuracy and macro recall fell
slightly, and there were no time penalties. The +0.2362-point difference
is small and was measured across different computers; it does not prove
that the source refactor alone caused the change or that it will generalize
to new people. The alternative Extra Trees model raised ordinary accuracy
on one included sample but lowered that sample's competition score, so it
remains experimental. A full 56-session subject-holdout aggregate is still
pending.

For exact Mac timing and per-window comparison, use the files already
written on the user's computer in `results/my_strict_test/`; they are not
reconstructed from this console transcript. The transcribed per-session
score table is also in `results/reported_mac_2026-09-27/`.
