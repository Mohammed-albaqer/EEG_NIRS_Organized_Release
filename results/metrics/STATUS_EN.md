# Public subject-holdout evaluation status

Fourteen subject-excluding feature caches are in `fold_artifacts/`. The
complete online replay and its `summary.csv` / `metrics.json` have not been
generated. Do not treat the eight-session N17/N18 result or the included
single-session smoke check as the required 56-session public aggregate.

After downloading all 56 recordings into `data_all/`, run:

```sh
python evaluate_public_loso.py --data data_all --out results/metrics
python verify_submission.py
```

The replay scores each session chronologically using its subject-excluding
cache. It can take a long time, and its timing measurements depend on the
computer used.
