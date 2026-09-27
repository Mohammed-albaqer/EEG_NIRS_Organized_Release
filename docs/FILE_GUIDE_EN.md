# File guide: what matters and why it is included

Start with the root `README.md`, then `docs/SETUP_WINDOWS_MAC_EN.md` for your
computer and `docs/CODE_WALKTHROUGH_EN.md` for the model. Run commands from
the extracted **project root**, which contains `classifier.py` and
`requirements.txt`. A file marked "optional" is still useful for analysis;
it is not needed to call the saved classifier in a judge run.

## A. Essential submission files

| Path | Contents and role |
| --- | --- |
| `solution/classifier.py` | The active `OnlineClassifier` implementation: causal filter, features, three Random Forests, probability fusion, block adaptation. This is the model to submit. |
| `solution/artifacts/pretrain_cache.npz` | Balanced saved EEG, NIRS, hybrid feature arrays and class labels from 54 compatible public recordings. Required to use the historical cache. It does **not** contain fitted forests. |
| `solution/artifacts/artifact_meta.json` | Artifact version, filter, dimensions, class counts, source subjects, and extraction settings. The classifier checks compatibility when loading. |
| `solution/artifacts/training_manifest.csv` | Exact source recordings used to make the final cache; use it to audit provenance. |
| `solution/run.py` | Frozen online runner supplied for this package; sequences `push`, `predict`, and `fit_block` without giving future labels to the classifier. |
| `solution/io_utils.py` | Frozen `.mat` reader, C3/C4 electrode indices, scored-window creation, and alignment of incoming NIRS samples. |
| `solution/score.py` | Frozen local scorer for confusion matrices, F1, recall, NIRS bonus, and timing penalties. It needs true labels, so it is not part of the classifier's prediction logic. |
| `requirements.txt` | Exact pinned versions of the required Python packages. |

`solution/train.py` is an optional copy of the cache builder. It exists to
show how the final artifact was produced; the judge does not need to run it.

## B. Root-level scripts: run these from the project root

| Path | Role |
| --- | --- |
| `classifier.py` | Local copy of the active `solution/classifier.py`; local training/evaluation helpers import this file. Keep the two in sync when experimenting. |
| `run.py`, `score.py`, `io_utils.py` | Local copies of the frozen runner/scorer/I/O code. `evaluate_many.py` copies them with a chosen classifier to an isolated temporary folder for a fair test. |
| `quick_test.py` | Smoke test on the included sample and the sample-only artifact; checks operation, not independent-person accuracy. |
| `strict_test.py` | Independent-person N17/N18 test; checks the holdout training manifest and invokes the frozen runner/scorer for eight recordings. Accepts `--classifier` for an optional candidate. |
| `evaluate_many.py` | Generic session-by-session evaluator. It writes predictions, timings, per-session metrics, `summary.csv`, and aggregate `metrics.json`. |
| `train.py` | Rebuilds an historical feature cache from specified `.mat` files. `--exclude-subjects` prevents a held-out person entering training. |
| `build_public_loso.py` | Builds 14 caches, one excluding each public person, for the 56-session retrospective evaluation. The supplied caches are already included. |
| `evaluate_public_loso.py` | Replays all 56 public recordings with the matching other-person cache; currently the complete aggregate is pending. This is much longer than `strict_test.py`. |
| `download_sessions.py` | Optional public-data downloader. Skip it if you already copied your `data_all/` directory. |
| `verify_public_data.py` | Compares local `.mat` files against names, sizes, and SHA256 hashes in `public_data_manifest.csv`. |
| `compare_results.py` | Compares matched per-session `summary.csv` files for accuracy, weakest-class F1, and points. |
| `verify_runtime.py` | Quick checks for filter causality, chunk invariance, nonfinite input, and incompatible artifact filter metadata. |
| `verify_submission.py` | Checks code copies, training manifests, caches, and 14 subject folds. By default it also requires a completed 56-session aggregate; `--code-only` stops after code/artifact checks. |

## C. Setup, entry points, and documentation

| Path | Role |
| --- | --- |
| `README.md` | Main overview, status, version history, and where to start. |
| `DEPLOY.md` | Detailed judge, container, and validation commands. |
| `docs/SETUP_WINDOWS_MAC_EN.md` | Python version, libraries, clean installation, reusing an existing Mac environment/data folder, and the main Windows/macOS commands. |
| `docs/CODE_WALKTHROUGH_EN.md` | Beginner-friendly explanation of each processing stage, online labels, features, training, fusion, and metrics. |
| `docs/USER_MAC_RESULTS_EN.md` | Detailed interpretation of the user's reported eight-session Mac run and comparison with the archived run. |
| `docs/FILE_GUIDE_EN.md` | This hand-written map of important and secondary files. |
| `docs/FILE_INVENTORY_EN.md` | Generated one-row-per-file listing with roles and sizes, including repeated result and cache files. |
| `public_data_manifest.csv` | Public data filenames, sizes, source URLs, and SHA256 digests; no raw 56-session collection is bundled. |
| `SHA256SUMS.txt` | Hashes of distributed files in this release for checking ZIP extraction. It is regenerated when the archive is built. |
| `.gitignore` | Keeps local environments, downloaded public data, temporary files, and personal runs out of Git. |
| `01_QUICK_TEST_WINDOWS.bat` | Windows Command Prompt shortcut for filter verification and the sample test. |
| `02_STRICT_TEST_WINDOWS.bat` | Windows shortcut that obtains the eight required public recordings if needed and runs `strict_test.py`. |
| `03_TEST_EXTRA_TREES_WINDOWS.bat` | Windows shortcut for the optional Extra Trees candidate and its matched comparison. |
| `04_QUICK_TEST_MAC.sh` | macOS shell shortcut for the quick sample test. Run with `bash`. |
| `05_STRICT_TEST_MAC.sh` | macOS shell shortcut for the eight-session test using existing `data_all/`. Run with `caffeinate bash` to prevent sleep. |
| `Dockerfile` | Optional Python 3.11 container with pinned packages and the `solution/` folder. |
| `compose.yaml` | Optional example container invocation. Its example public session is an operational check, not a subject-held-out score. |

## D. Samples, experiment history, and review material

| Path | Role |
| --- | --- |
| `samples/data/2025.12.01.16.04.01_N01.mat` | Single bundled EEG/NIRS recording for the immediate smoke test. |
| `samples/artifacts/pretrain_cache.npz` | Smaller sample-test feature cache; excludes that exact recording, but includes other N01 recordings. |
| `samples/artifacts/artifact_meta.json`, `training_manifest.csv` | Sample-cache settings and source provenance. |
| `experiments/classifier_original.py` | Earlier baseline for historical comparison; not the chosen submission. |
| `experiments/classifier_causal_filter_only.py` | Earlier causal-filter candidate; historical reference. |
| `experiments/classifier_hard_agreement.py` | Former decision rule using class-label agreement. |
| `experiments/classifier_probability_fusion.py` | Exact probability-fusion implementation used for the previous 65.18% N17/N18 run before the speed refactor. |
| `experiments/classifier_optimized_rf_reference.py` | Reference copy of the active optimized Random Forest classifier. |
| `experiments/classifier_extra_trees.py` | Optional alternative tree algorithm; one-sample competition score was lower, so it is not the default. |
| `docs/reports/VALIDATION_EN.md` | Archived controlled N15/N16 and N17/N18 comparisons and limitations. |
| `docs/reports/SPEED_REVIEW_EN.md` | One-sample prediction parity and timing measurements plus Extra Trees comparison. |
| `docs/reports/METHODS_EN.md` | Competition task, features, cache separation, and validation method. |
| `docs/reports/CODE_CHANGES_EN.md` | Source change notes from hard agreement to probability fusion and later runtime refactor. |
| `docs/reports/classifier_fusion.patch` | Literal earlier source diff for the probability-fusion change. |
| `docs/presentation/EEG_NIRS_Final_7min.pptx` | Seven-slide presentation. It shows the archived controlled 65.18% result and labels pending work. |
| `docs/presentation/SPEAKER_NOTES_EN.md` | Speaking notes and limits for the slides. |
| `docs/notebook/reproduce_experiments.ipynb` | Optional notebook that launches or describes command-line reproductions from the project root. |
| `docs/notebook/README.md` | How to use the notebook. |

## E. Results and logs

| Path or pattern | Role |
| --- | --- |
| `results/MY_RESULTS_EN.md` | Intentionally blank personal results form. Fill it with your future runs; it does not overwrite measured evidence. |
| `results/reported_mac_2026-09-27/README_EN.md` | Provenance and limits of the pasted Mac terminal output. |
| `results/reported_mac_2026-09-27/session_scores_from_console.csv` | Eight transcribed Mac per-session scores and the archived paired scores. It is not the original Mac output directory. |
| `results/holdout/artifacts_N17_N18/` | Historical feature cache excluding both independent test people: `pretrain_cache.npz`, metadata JSON, and source manifest. |
| `results/holdout/artifacts_dev_N15_N16/` | Development cache excluding N15, N16, N17, and N18, with the same three artifact files. |
| `results/holdout/development_baseline/` and `development_probability_fusion/` | Development comparisons: session folders, aggregate `metrics.json`, and `summary.csv`. |
| `results/holdout/baseline_N17_N18/` and `final_selected_N17_N18/` | Earlier paired independent test outputs for hard agreement and probability fusion. |
| `results/smoke/previous_random_forest/`, `optimized_random_forest/`, `extra_trees/` | One-sample runs used to check matching predictions and approximate speed; these are not new-person estimates. |
| `results/metrics/fold_artifacts/exclude_Nxx/` | One historical feature cache per person, with `.npz`, JSON settings, and CSV source manifest that excludes Nxx. There are 14 such directories. |
| `results/metrics/STATUS_EN.md` | Explains why `results/metrics/metrics.json` and `summary.csv` are pending. |
| `results/build_logs/*.log` | Text logs of cache extraction, fold-cache building, baseline and selected experimental runs. Logs are for audit; the runner reads artifacts instead. |

Within a per-session result directory, `predictions.csv` lists window
boundaries, true labels, and predicted classes; `timings.csv` lists each
prediction and fit duration; `metrics.json` contains confusion matrices,
class F1, macro recall, NIRS measurements, penalties, and points. At the
parent level, `summary.csv` lists each session and `metrics.json` gives the
unweighted aggregate. No Mac per-window files were inserted because they
were not supplied in the message.

## F. Generated locally and intentionally absent

`data_all/` is your downloaded 56-session data directory and is intentionally
excluded from the ZIP (about 1.2 GB). `.venv/` is a local Python environment,
`results/my_*` contains personal test runs, and `__pycache__/` contains
temporary Python bytecode. Recreate or copy these on your own computer;
they are not required source files.
