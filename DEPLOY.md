# Deployment and verification

## Python environment

The supplied Dockerfile uses Python 3.11 and creates the required `jupyter`
user with UID 1000. This package was executed here with Python 3.12 and the
exact versions in `requirements.txt`:

```text
numpy==2.3.5
scipy==1.17.0
pandas==2.2.3
scikit-learn==1.8.0
joblib==1.5.3
```

On Windows, install Python 3.11 x64. In Command Prompt in the extracted
project directory, run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verify_runtime.py
.venv\Scripts\python.exe quick_test.py --out results\my_quick_test
```

The included sample artifact excludes the supplied sample session by exact
filename. The quick test is a smoke test; it includes other sessions of the
same person in the cache and is not an independent subject test.

On macOS or Linux with Python 3.11 installed:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_runtime.py
python quick_test.py --out results/my_quick_test
```

## Judge interface

The judge uses `solution/run.py`, which imports `solution/classifier.py` and
calls `OnlineClassifier.__init__`, `load`, `push`, `predict`, and `fit_block`.
The final public historical cache is in `solution/artifacts/`. It is intended
for independent hidden sessions of the same format. The judge's runner,
scorer, and I/O code were retained byte-for-byte from the supplied package.

```sh
python solution/run.py --input /data/new_session.mat --output-dir /out \
  --artifacts solution/artifacts
python solution/score.py --session /data/new_session.mat \
  --predictions /out/predictions.csv --timings /out/timings.csv \
  --output /out/metrics.json
```

`score.py` needs the session's true labels and is for local evaluation. The
classifier itself never loads the session file or its true labels. The runner
releases labels after a completed block. File paths can be outside the
project. No Yandex bucket name is embedded in the classifier.

## Reproduce independent public validation

Download all 56 public recordings (approximately 1.2 GB) and verify them:

```sh
python download_sessions.py --out data_all
python verify_public_data.py --data data_all
python strict_test.py --data data_all --out results/my_strict_test
```

The bundled `results/holdout/artifacts_N17_N18/` was trained only from
other subjects. `strict_test.py` checks its training manifest and requires
the eight N17/N18 recordings. It writes per-session predictions, timings,
confusion matrices, and aggregate metrics. Timing penalties can differ across
machines. `02_STRICT_TEST_WINDOWS.bat` downloads just the eight needed files
and runs the same test without full-dataset SHA256 verification.

To rebuild the held-out artifact after downloading every file:

```sh
python train.py --data data_all --out rebuilt_holdout \
  --exclude-subjects N17 N18 --cache-per-class 400 --stride-sec 0.5 --seed 2026
```

To rebuild the final artifact for independent hidden sessions, use a **new**
output path and review its training manifest before replacing the bundled
artifact:

```sh
python train.py --data data_all --out rebuilt_public \
  --cache-per-class 400 --stride-sec 0.5 --seed 2026
```

To reproduce the required public-data metrics for every person without
using that person's recordings as historical training data:

```sh
python build_public_loso.py --data data_all \
  --out results/metrics/fold_artifacts
python evaluate_public_loso.py --data data_all --out results/metrics
```

This creates 14 subject-excluding artifacts and scores all 56 recordings.
It takes substantially longer than the eight-session strict test. The
aggregate `results/metrics/metrics.json` averages per-session metrics.
The 14 fold artifacts are already included; rebuilding them is optional when
the same data and feature format are used. The full 56-session evaluation
was stopped on request and the aggregate is not yet included. Run
`python verify_submission.py` to check completion, or add `--code-only` for
a source/artifact check without the full public results.

To rerun the optimized default model on independent subjects:

```sh
python strict_test.py --data data_all --out results/optimized_N17_N18
```

To evaluate the new algorithm without changing the submission:

```sh
python strict_test.py --data data_all --out results/extra_trees_N17_N18 \
  --classifier experiments/classifier_extra_trees.py
python compare_results.py \
  --before results/holdout/final_selected_N17_N18 \
  --after results/extra_trees_N17_N18
```

The raw public recordings are not included in the submission ZIP; the sample
recording is included for an immediate smoke test. The full public artifact
contains features from all usable public recordings. Scoring one of those
same recordings with it is an in-sample metric, not a subject holdout test.

## Container

```sh
docker compose build
docker compose run --rm eeg-nirs-evaluator
```

The Compose example uses a public file for a functional run. It does not
produce an independent performance estimate when the final public artifact
is selected. Docker, Windows, and the judge container were unavailable in
the execution environment used to produce this package; verify those
platforms before final platform submission.
