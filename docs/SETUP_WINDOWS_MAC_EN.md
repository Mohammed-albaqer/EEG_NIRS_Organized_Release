# Installation and use on Windows and macOS

Run every relative command from the **extracted project root**, the folder
that contains `README.md`, `requirements.txt`, and `strict_test.py`. The
recordings used by a command belong in a sibling `data_all/` folder inside
that root; downloaded data and virtual environments are not in the ZIP.

## Required software and versions

Use 64-bit **Python 3.11** for the intended competition environment. The
supplied container also uses Python 3.11. The earlier development smoke
checks ran with Python 3.12, which does not substitute for a final test of
the target environment. The pinned Python libraries are:

```text
numpy==2.3.5
scipy==1.17.0
pandas==2.2.3
scikit-learn==1.8.0
joblib==1.5.3
```

NumPy supplies arrays and the FFT; SciPy reads MATLAB `.mat` files and
performs causal filtering; pandas supports the local scorer's CSV tables;
scikit-learn supplies the tree ensembles; joblib supports their parallel
work. `venv` and `pip` come with a standard Python installation. Installing
the pinned requirements is more reliable than copying a virtual environment
between folders or operating systems. `data_all/` can be copied safely.

Official installation references: [Python 3.11 Windows guidance](https://docs.python.org/3.11/using/windows.html),
[Python virtual environments](https://docs.python.org/3.11/library/venv.html),
and [Homebrew's Python 3.11 formula](https://formulae.brew.sh/formula/python@3.11).

## Windows 10/11: clean installation with Command Prompt

1. Install 64-bit Python 3.11 and ensure the Python launcher `py` works.
2. Extract the ZIP. In File Explorer, open the extracted project folder,
   type `cmd` in the address bar, and press Enter.
3. Run these commands:

```bat
py -3.11 --version
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
01_QUICK_TEST_WINDOWS.bat
```

The quick check uses the bundled one-session sample. Look in
`results\my_quick_test\metrics.json`. For the independent-person replay:

```bat
02_STRICT_TEST_WINDOWS.bat
```

The Windows shortcut downloads the eight N17/N18 files that it needs if
they are missing, checks that the two people are excluded from its cache,
and writes `results\my_strict_test\summary.csv` and `metrics.json`. To
try an optional, not selected alternative:

```bat
03_TEST_EXTRA_TREES_WINDOWS.bat
```

If you already downloaded `data_all` on Windows, copy its entire folder
from the earlier project into the extracted project root before the strict
test. The download helper checks existing public files; you do not need to
download the same files again.

To run a single new `.mat` session using the final all-public cache:

```bat
.venv\Scripts\python.exe solution\run.py --input C:\path\new_session.mat --output-dir results\new_session --artifacts solution\artifacts
.venv\Scripts\python.exe solution\score.py --session C:\path\new_session.mat --predictions results\new_session\predictions.csv --timings results\new_session\timings.csv --output results\new_session\metrics.json
```

The second command needs the session's true labels and is for a locally
labeled recording. Do not score a public training file with the final
all-public cache and call the result independent.

## macOS: clean installation

If `python3.11 --version` does not work and you already use Homebrew,
install it with `brew install python@3.11`. Open Terminal, move into the
extracted project folder, and run:

```sh
python3.11 --version
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
bash 04_QUICK_TEST_MAC.sh
```

The script automatically chooses `.venv/bin/python` when present. You can
also use the interpreter directly without activating the environment:

```sh
.venv/bin/python verify_runtime.py
.venv/bin/python quick_test.py --out results/my_quick_test
```

### Reuse the user's already-downloaded data

The old project folder provided by the user is:

```text
/Users/mohammedalbaqerabboodei/Downloads/EEG_RF_Strict_Generalization
```

For a new release extracted at the assumed Downloads path below, copy the
existing recordings without network access:

```sh
cd "/Users/mohammedalbaqerabboodei/Downloads/EEG_NIRS_Organized_Release"
mkdir -p data_all
rsync -avh --progress "/Users/mohammedalbaqerabboodei/Downloads/EEG_RF_Strict_Generalization/data_all/" "data_all/"
find data_all -maxdepth 1 -name '*.mat' | wc -l
```

If the extracted folder has a different name or location, change only the
first `cd` path. A complete public collection has 56 `.mat` files. To check
their sizes and SHA256 values against the manifest, run:

```sh
.venv/bin/python verify_public_data.py --data data_all
```

### Reuse a working old virtual environment without copying it

An existing environment can run code in the new project as long as it has
the required packages. From the new project root, use its Python executable:

```sh
"/Users/mohammedalbaqerabboodei/Downloads/EEG_RF_Strict_Generalization/.venv/bin/python" -m pip check
"/Users/mohammedalbaqerabboodei/Downloads/EEG_RF_Strict_Generalization/.venv/bin/python" verify_runtime.py
"/Users/mohammedalbaqerabboodei/Downloads/EEG_RF_Strict_Generalization/.venv/bin/python" quick_test.py --out results/my_quick_test
```

If the old environment is used this way, use the same full Python path in
the strict-test command below rather than the shell shortcut. A copied
`.venv/` may contain scripts pointing back to the old folder; a clean new
environment is the self-contained choice. Never copy a macOS environment
to Windows or the reverse.

### Eight-session test on macOS

With N17/N18 data available in `data_all/`:

```sh
caffeinate bash 05_STRICT_TEST_MAC.sh
```

Or, with the original project's environment:

```sh
caffeinate "/Users/mohammedalbaqerabboodei/Downloads/EEG_RF_Strict_Generalization/.venv/bin/python" strict_test.py --data data_all --out results/my_strict_test
```

Use **one shell line**, or place `\` at the very end of a line with no
spaces after it. The result files are `results/my_strict_test/summary.csv`,
`metrics.json`, and a subfolder for each of the eight recordings. The
script `strict_generalization_experiment.py` belongs to an earlier project
and is not included in this release. `strict_test.py` is the supported
eight-session entry point here.

### Complete public subject-holdout replay

Only after all 56 `.mat` files are present, run:

```sh
caffeinate .venv/bin/python evaluate_public_loso.py --data data_all --out results/metrics
.venv/bin/python verify_submission.py
```

This takes substantially longer than the eight-session test. It uses the
included 14 subject-excluding feature caches and writes the aggregate
`results/metrics/metrics.json`. If you need to rebuild those caches from
the public files, see `DEPLOY.md`. `verify_submission.py --code-only` checks
the packaged source and artifacts without requiring the long aggregate.

## Reading the output

`predictions.csv` has true and predicted labels for each scored window;
`timings.csv` records individual prediction and block-fit durations;
per-session `metrics.json` includes confusion matrices and points;
`summary.csv` lists one line per recording; aggregate `metrics.json` gives
an unweighted mean across sessions. Copy your measured values into the
intentionally blank `results/MY_RESULTS_EN.md` if you want a separate
personal report.

If `strict_test.py` reports fewer than eight N17/N18 files, confirm that
`data_all/` is in the project root. If `py` or `python3.11` is missing,
install or select Python 3.11 before creating `.venv/`. If imports fail,
install `requirements.txt` using the **same Python executable** that runs
the scripts.
