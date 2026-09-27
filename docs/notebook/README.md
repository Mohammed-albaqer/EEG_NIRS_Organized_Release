# Experiment notebook

`reproduce_experiments.ipynb` runs from the package root. It downloads and
verifies the public dataset, repeats the eight-session N17/N18 test with
the subject-excluding cache, reads the computed metrics, and shows the
command for rebuilding the final public artifact.

Install the pinned dependencies from `../../requirements.txt` first. The
notebook was produced for review; the real command-line evaluations and
their outputs are retained under `../../results/holdout/` and
`../../results/metrics/`. The latter currently contains fold caches but not
the completed 56-session aggregate. Keep the final public artifact separate
from any subject holdout test.
