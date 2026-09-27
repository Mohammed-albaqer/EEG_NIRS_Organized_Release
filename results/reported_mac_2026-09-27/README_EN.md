# Mac run reported by the user on 2026-09-27

The CSV transcribes the eight per-session scores, window counts, and NIRS
details pasted from a terminal run of:

```sh
caffeinate python strict_test.py --data data_all --out results/my_strict_test
```

The earlier comparison values came from
`../holdout/final_selected_N17_N18/summary.csv`. The user reported aggregate
accuracy 0.6512 (rounded), mean weakest-class F1 0.4181, macro recall
0.5881, NIRS weakest-class F1 0.2502701490923945, base quality 42.7269,
NIRS bonus 8.7541, no timing penalties, and total 51.4810/130.

This directory contains a transcription of the console output, **not** the
Mac-generated `predictions.csv`, `timings.csv`, `summary.csv`, or
`metrics.json`. Those full files remain on the user's Mac under
`results/my_strict_test/`. Individual hybrid F1, recall, accuracy, and
timings were not in the pasted text and are not inferred here.
