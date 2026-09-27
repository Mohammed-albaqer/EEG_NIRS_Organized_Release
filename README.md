# EEG + NIRS hand-imagery classifier

This project predicts **rest (1), imagined left hand (2), or imagined right
hand (3)** from a stream of EEG and optional NIRS data. It contains the
selected competition classifier, its historical feature caches, executable
evaluation tools, archived comparisons, a detailed English explanation,
and a blank personal results form. All written package files are in English.

**Read this first:** `docs/FILE_GUIDE_EN.md` explains every source file and
every family of secondary results; `docs/FILE_INVENTORY_EN.md` lists the
individual packaged files. `docs/CODE_WALKTHROUGH_EN.md` explains the actual
code step by step. `docs/SETUP_WINDOWS_MAC_EN.md` gives complete installation
and usage commands for Windows and macOS.

## Current measured status

The user ran the optimized Random Forest on eight independent-person N17/N18
sessions on macOS and supplied its terminal output. This was an **online
evaluation**, including legal retraining after each completed block, rather
than a training-set accuracy report.

| Eight-session measure | Earlier probability-fusion run | Optimized run on user's Mac |
| --- | ---: | ---: |
| Mean accuracy | 65.18% | **65.12%** |
| Mean F1 of each session's weakest class | 0.4152 | **0.4181** |
| Mean macro recall | 0.5888 | **0.5881** |
| NIRS-only weakest-class F1 | 0.2447 | **0.2503** |
| Timing penalties | 0 | **0** |
| Computed score / 130 | 51.2448 | **51.4810** |

The computed score rose **0.2362 points** and weakest-class F1 rose about
0.0029; accuracy fell about **0.06 percentage points**. Five sessions gained
score and three lost score. This is a small difference from a different
computer, so it does not prove an accuracy gain caused by the refactor.
Keep `solution/classifier.py` as the selected default. The one-sample Extra
Trees candidate had a lower competition score and remains in `experiments/`.
See `docs/USER_MAC_RESULTS_EN.md` for every pasted session score, the scorer
formula, training separation, runtime context, and limits. The original Mac
prediction and timing CSV files were not uploaded; this package includes a
clearly labeled transcription of the console output only.

The complete 56-session subject-holdout evaluation and hidden competition
score are **not available yet**. Fourteen per-person feature caches are
included to enable the long public replay. The jury's separate design/report
points are not included in the score above.

## What is the essential submission?

| Location | Why it matters |
| --- | --- |
| `solution/classifier.py` | Active online prediction and block-training algorithm. |
| `solution/artifacts/` | Final historical feature cache, metadata, and training-source manifest. |
| `solution/run.py`, `solution/io_utils.py`, `solution/score.py` | Frozen reference-style runner, data reader, and local scorer. |
| `requirements.txt` | Exact Python package versions. |

`solution/train.py` documents how the cache is rebuilt. The root files
`classifier.py`, `run.py`, `score.py`, `io_utils.py`, and `train.py` are
local-working copies for experiments. Do not switch the final artifact for
the N17/N18 holdout artifact: the final cache contains those people and
must be used only on independent hidden/new sessions.

## Requirements

Use **64-bit Python 3.11** for the intended deployment environment. The
same code was smoke-tested in the earlier workspace with Python 3.12; the
provided container uses 3.11. `requirements.txt` pins:

| Package | Version | Why it is needed |
| --- | --- | --- |
| NumPy | 2.3.5 | Signal arrays, FFT, window features, saved caches. |
| SciPy | 1.17.0 | `.mat` reading and causal Butterworth filtering. |
| pandas | 2.2.3 | Reading prediction/timing tables in the scorer. |
| scikit-learn | 1.8.0 | Random Forests, Extra Trees candidate, model probabilities. |
| joblib | 1.5.3 | Dependency used by scikit-learn for parallel tree fitting. |

Windows requires an installed Python 3.11 x64 launcher; macOS requires
`python3.11`. The standard-library modules (`pathlib`, `json`, `csv`, etc.)
come with Python. Docker is optional; no additional Python package is
required just to run the command-line tests.

## Windows: install and test

Open **Command Prompt** in the extracted project folder (the folder
containing this README). Run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
01_QUICK_TEST_WINDOWS.bat
```

The quick test uses the bundled sample and writes
`results\my_quick_test\metrics.json`. For an independent-person test run:

```bat
02_STRICT_TEST_WINDOWS.bat
```

The Windows script obtains the eight N17/N18 recordings if they are absent
and writes `results\my_strict_test\summary.csv` and `metrics.json`. To
evaluate the optional Extra Trees candidate, run
`03_TEST_EXTRA_TREES_WINDOWS.bat` after setup. It does not replace the
selected submission. Full commands and troubleshooting are in
`docs/SETUP_WINDOWS_MAC_EN.md`.

## macOS: install or reuse an environment, then test

In Terminal, change into the extracted project folder and run:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
bash 04_QUICK_TEST_MAC.sh
```

If you already have a working virtual environment from a previous project,
you may call its `bin/python` with an absolute path instead of copying the
environment. Copying a Python virtual environment can retain old internal
paths. You may copy a previous `data_all/` folder into this project rather
than downloading the 56 recordings again; see the exact `rsync` example in
`docs/SETUP_WINDOWS_MAC_EN.md`.

With eight N17/N18 `.mat` files in `data_all/`, run:

```sh
caffeinate bash 05_STRICT_TEST_MAC.sh
```

Or use the direct command on **one line**:

```sh
caffeinate .venv/bin/python strict_test.py --data data_all --out results/my_strict_test
```

The shell scripts choose `.venv/bin/python` when present; otherwise they
try `python3.11`. Always start commands in the project root. This package
does **not** contain a file called `strict_generalization_experiment.py`;
the supported eight-session entry point is `strict_test.py`.

## Data, training period, and where the score comes from

The public source lists **56 recordings from 14 people**, with filenames
dated from **2025-12-01 to 2026-03-27**. Two recordings lacked compatible
NIRS for the joint feature-cache builder. The final cache contains
balanced features from **54 compatible sessions**; its build log recorded
**102.69 s** in the earlier workspace. The separate N17/N18 holdout cache
contains **46 compatible other-person sessions**, excludes both test
people, and took **89.65 s** to extract there. Each saves 1,200 feature
windows, 400 from each class. These times measure feature extraction and
cache creation on that machine, not the elapsed duration of your Mac run.

The cache is **not a pre-fitted forest**. During each tested recording the
runner sends live EEG/NIRS chunks, scores one-second windows every 250 ms,
and releases labels only after a full block. The first block trains but is
not scored. `fit_block()` rebuilds the forests from the historical feature
cache plus released current-session labels after each of seven blocks.
Thus the eight-session test performed **56 block-fit calls**. The Mac
console did not show the per-call or total elapsed training seconds;
`results/my_strict_test/*/timings.csv` on your Mac contains the call times.

Each EEG channel supplies log variance and four band powers (4-8, 8-13,
13-30, 30-40 Hz). C3/C4 band differences add four features. Available HbO
and HbR samples add mean, spread, and change for each NIRS channel. Three
90-tree Random Forests learn EEG-only, NIRS-only, and combined predictions.
The selected decision averages their class probabilities with weights
**0.35 EEG + 0.15 NIRS + 0.50 combined**. The EEG filter is causal and
stateful; labels from the current block cannot reach its earlier predictions.

The 65.12% diagnostic accuracy averages eight separate session accuracies.
The competition-style score uses weakest-class F1, macro recall, a NIRS
bonus, and timing penalties; see the exact formula and per-session table in
`docs/USER_MAC_RESULTS_EN.md`. These eight sessions represent only **two
independent people**, and the cache is a retrospective other-person split,
not a strict calendar-ordered training experiment.

## Version history and evidence

| Version | Algorithmic change | Measured result and proper interpretation |
| --- | --- | --- |
| Early baseline | Single-session Random Forest prototype | 58.12% and 46.1005/130 in a previously reported **different seven-session split**; do not compare this directly with the eight-session test. |
| Causal multi-session model | Streaming 1-40 Hz Butterworth `sosfilt`, balanced other-session feature cache, legal block refitting; EEG/NIRS/hybrid forests | Set up the later subject-excluding evaluations. |
| Hard agreement | EEG and NIRS class agreement could override the hybrid result | 63.67% accuracy and 50.2353/130 on the eight N17/N18 sessions. |
| Probability fusion | Uses weighted class probabilities instead of a hard label override | 65.18% and 51.2448/130 on **the same eight sessions in the earlier workspace**: +1.51 accuracy points and +1.0095 competition points. |
| Optimized Random Forest (selected) | Fewer repeated forest calls, cached FFT masks, amortized EEG storage, stricter input/artifact validation; core feature and fusion definitions retained | Mac console: 65.12%, 0.4181 weakest-class F1, 51.4810/130. Score difference vs previous: +0.2362, but machines differ. A bundled one-session test gave byte-identical predictions to the previous probability-fusion source and about 1.73x faster mean prediction there. |
| Extra Trees (optional) | Randomized candidate tree splits using the same cache and fusion | One sample had higher ordinary accuracy but a lower competition score; not selected and no independent-person result. |

The exact archives and sample checks are in `docs/reports/` and `results/`.
The measured gain from hard agreement to probability fusion was tested on
the same split; the small Mac-versus-earlier score difference is not a
controlled code-only comparison. Do not promise the same performance on
hidden people.

## Results directories and next commands

- `results/holdout/`: earlier controlled development and N17/N18 runs and
  their subject-excluding caches.
- `results/reported_mac_2026-09-27/`: scores transcribed from the user's
  console, clearly separated from original local output files.
- `results/smoke/`: three one-session functional and timing comparisons.
- `results/metrics/fold_artifacts/`: 14 other-person caches. The complete
  56-session `results/metrics/metrics.json` is still pending.
- `results/MY_RESULTS_EN.md`: blank template to fill with your own results.

After copying all 56 public recordings into `data_all/`, the longer public
subject-holdout replay is:

```sh
python evaluate_public_loso.py --data data_all --out results/metrics
python verify_submission.py
```

Use `python verify_submission.py --code-only` for a fast code/artifact check
before that replay. The full check deliberately reports pending status until
all 56 results exist. Refer to `DEPLOY.md` for direct judge and Docker
commands.
