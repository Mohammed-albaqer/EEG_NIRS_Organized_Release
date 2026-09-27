# Individual file inventory

This index covers each file distributed in the release. Repeated output files are
listed individually; their role definitions are explained in `FILE_GUIDE_EN.md`.
File sizes are approximate and can change with a future run.

| Tier | File | Size | Contents and purpose |
| --- | --- | ---: | --- |
| Optional setup | `.gitignore` | 0.1 KiB | Ignores environments, downloaded data, personal results, and bytecode. |
| Windows shortcut | `01_QUICK_TEST_WINDOWS.bat` | 0.3 KiB | Creates sample result after fast filter checks. |
| Windows shortcut | `02_STRICT_TEST_WINDOWS.bat` | 0.4 KiB | Obtains eight N17/N18 files if needed and runs strict test. |
| Windows shortcut | `03_TEST_EXTRA_TREES_WINDOWS.bat` | 0.6 KiB | Tests alternative Extra Trees on N17/N18 and compares results. |
| Mac shortcut | `04_QUICK_TEST_MAC.sh` | 0.3 KiB | Runs fast filter checks and bundled sample test. |
| Mac shortcut | `05_STRICT_TEST_MAC.sh` | 0.3 KiB | Runs the eight-session N17/N18 test on local data. |
| Guide | `DEPLOY.md` | 5.3 KiB | Judge/container commands and detailed reproduction steps. |
| Optional setup | `Dockerfile` | 0.4 KiB | Python 3.11 container recipe for the solution. |
| Start here | `README.md` | 11.1 KiB | Project overview, status, version history, training provenance, and main commands. |
| Training tool | `build_public_loso.py` | 5.0 KiB | Rebuilds the fourteen other-person caches for public validation. |
| Local core | `classifier.py` | 19.6 KiB | Active Random Forest classifier used by local evaluators. |
| Analysis tool | `compare_results.py` | 1.9 KiB | Shows matched per-session changes in accuracy, F1, and points. |
| Optional setup | `compose.yaml` | 0.3 KiB | Example container command with data and output mounts. |
| Guide | `docs/CODE_WALKTHROUGH_EN.md` | 8.3 KiB | Stepwise explanation of streaming, features, training, fusion, and score. |
| Guide | `docs/FILE_GUIDE_EN.md` | 10.9 KiB | Narrative map of essential, supporting, and optional paths. |
| Guide | `docs/SETUP_WINDOWS_MAC_EN.md` | 7.7 KiB | Python and libraries installation, data reuse, and commands for both systems. |
| Evidence review | `docs/USER_MAC_RESULTS_EN.md` | 6.0 KiB | Detailed eight-session user console comparison and limits. |
| Optional notebook | `docs/notebook/README.md` | 0.7 KiB | How to use the experiment notebook from project root. |
| Optional notebook | `docs/notebook/reproduce_experiments.ipynb` | 5.0 KiB | Notebook walkthrough calling the command-line reproduction tools. |
| Optional presentation | `docs/presentation/EEG_NIRS_Final_7min.pptx` | 27.7 KiB | Seven-slide jury summary with earlier controlled results. |
| Optional presentation | `docs/presentation/SPEAKER_NOTES_EN.md` | 3.2 KiB | Speaker notes and the limits of the slide results. |
| Historical report | `docs/reports/CODE_CHANGES_EN.md` | 4.8 KiB | Source changes from hard agreement through speed refactor. |
| Historical report | `docs/reports/METHODS_EN.md` | 4.2 KiB | Model features, subject separation, and scoring methodology. |
| Historical report | `docs/reports/SPEED_REVIEW_EN.md` | 3.7 KiB | One-sample timing and prediction parity versus the earlier classifier. |
| Historical report | `docs/reports/VALIDATION_EN.md` | 8.3 KiB | Controlled development and N17/N18 comparisons from earlier workspace. |
| Historical source | `docs/reports/classifier_fusion.patch` | 1.2 KiB | Literal diff of the earlier probability-fusion rule. |
| Optional tool | `download_sessions.py` | 4.4 KiB | Downloads missing public recordings; unnecessary with an existing data_all folder. |
| Test tool | `evaluate_many.py` | 5.6 KiB | Runs a selected classifier over sessions and aggregates per-session metrics. |
| Long test | `evaluate_public_loso.py` | 4.1 KiB | Scores 56 sessions with a different excluded-person cache for each person. |
| Historical/experimental | `experiments/classifier_causal_filter_only.py` | 13.4 KiB | Earlier causal-filter-only candidate. |
| Historical/experimental | `experiments/classifier_extra_trees.py` | 19.8 KiB | Optional randomized Extra Trees candidate, not selected. |
| Historical/experimental | `experiments/classifier_hard_agreement.py` | 16.5 KiB | Earlier agreement override comparator. |
| Historical/experimental | `experiments/classifier_optimized_rf_reference.py` | 19.6 KiB | Reference copy of selected optimized Random Forest source. |
| Historical/experimental | `experiments/classifier_original.py` | 12.8 KiB | Original early baseline source. |
| Historical/experimental | `experiments/classifier_probability_fusion.py` | 16.9 KiB | Archived exact source for earlier 65.18% result. |
| Local frozen | `io_utils.py` | 5.1 KiB | MATLAB input reader, C3/C4 channel map, windows, and NIRS alignment. |
| Data verification | `public_data_manifest.csv` | 10.7 KiB | 56 public recording names, sizes, SHA256 values, and source URLs. |
| Test tool | `quick_test.py` | 1.4 KiB | Runs the bundled sample through the frozen runner and scorer. |
| Essential | `requirements.txt` | 0.2 KiB | Pinned NumPy, SciPy, pandas, scikit-learn, and joblib versions. |
| Blank template | `results/MY_RESULTS_EN.md` | 1.0 KiB | Empty personal form for future results; no scores filled in. |
| Build/run log | `results/build_logs/baseline.log` | 5.5 KiB | Earlier N17/N18 hard-agreement run log. |
| Build/run log | `results/build_logs/development_baseline.log` | 5.6 KiB | Earlier N15/N16 baseline run log. |
| Build/run log | `results/build_logs/development_probability_fusion.log` | 5.9 KiB | Earlier N15/N16 fusion run log. |
| Build/run log | `results/build_logs/final_cache.log` | 4.1 KiB | 54-session final cache extraction log with elapsed time. |
| Build/run log | `results/build_logs/final_selected.log` | 5.7 KiB | Earlier selected N17/N18 fusion run log. |
| Build/run log | `results/build_logs/subject_folds.log` | 3.4 KiB | Fourteen fold-cache build log. |
| Build/run log | `results/build_logs/train.log` | 3.6 KiB | 46-session N17/N18 holdout cache extraction log. |
| Build/run log | `results/build_logs/train_dev.log` | 3.0 KiB | Development cache extraction log. |
| Test feature cache | `results/holdout/artifacts_N17_N18/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/holdout/artifacts_N17_N18/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/holdout/artifacts_N17_N18/training_manifest.csv` | 2.5 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/holdout/artifacts_dev_N15_N16/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/holdout/artifacts_dev_N15_N16/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/holdout/artifacts_dev_N15_N16/training_manifest.csv` | 2.1 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.23.14.24.35_N17/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.23.14.24.35_N17/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.23.14.24.35_N17/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.24.18.00.22_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.24.18.00.22_N18/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.24.18.00.22_N18/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.12.49.29_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.12.49.29_N18/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.12.49.29_N18/timings.csv` | 38.3 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.13.04.32_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.13.04.32_N18/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.13.04.32_N18/timings.csv` | 38.3 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.14.35.20_N17/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.14.35.20_N17/predictions.csv` | 33.1 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.14.35.20_N17/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.14.46.04_N17/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.14.46.04_N17/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.25.14.46.04_N17/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.26.12.37.48_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.26.12.37.48_N18/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.26.12.37.48_N18/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.26.14.21.39_N17/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.26.14.21.39_N17/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/2026.03.26.14.21.39_N17/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/metrics.json` | 0.9 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/baseline_N17_N18/summary.csv` | 1.4 KiB | One row per evaluated session with accuracy, F1, recall, timing, and score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.24.16.07.40_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.24.16.07.40_N15/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.24.16.07.40_N15/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.27.13.11.59_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.27.13.11.59_N15/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.27.13.11.59_N15/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.27.13.24.42_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.27.13.24.42_N15/predictions.csv` | 33.1 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.02.27.13.24.42_N15/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.04.12.51.22_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.04.12.51.22_N15/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.04.12.51.22_N15/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.10.17.02.53_N16/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.10.17.02.53_N16/predictions.csv` | 33.1 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.10.17.02.53_N16/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.11.19.32.02_N16/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.11.19.32.02_N16/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.11.19.32.02_N16/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.11.19.45.53_N16/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.11.19.45.53_N16/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.11.19.45.53_N16/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.12.12.31.02_N16/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.12.12.31.02_N16/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_baseline/2026.03.12.12.31.02_N16/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_baseline/metrics.json` | 0.9 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_baseline/summary.csv` | 1.4 KiB | One row per evaluated session with accuracy, F1, recall, timing, and score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.24.16.07.40_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.24.16.07.40_N15/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.24.16.07.40_N15/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.27.13.11.59_N15/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.27.13.11.59_N15/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.27.13.11.59_N15/timings.csv` | 38.3 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.27.13.24.42_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.27.13.24.42_N15/predictions.csv` | 33.1 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.02.27.13.24.42_N15/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.04.12.51.22_N15/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.04.12.51.22_N15/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.04.12.51.22_N15/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.10.17.02.53_N16/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.10.17.02.53_N16/predictions.csv` | 33.1 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.10.17.02.53_N16/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.11.19.32.02_N16/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.11.19.32.02_N16/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.11.19.32.02_N16/timings.csv` | 38.3 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.11.19.45.53_N16/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.11.19.45.53_N16/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.11.19.45.53_N16/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.12.12.31.02_N16/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.12.12.31.02_N16/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/development_probability_fusion/2026.03.12.12.31.02_N16/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/development_probability_fusion/metrics.json` | 0.9 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/development_probability_fusion/summary.csv` | 1.4 KiB | One row per evaluated session with accuracy, F1, recall, timing, and score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.23.14.24.35_N17/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.23.14.24.35_N17/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.23.14.24.35_N17/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.24.18.00.22_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.24.18.00.22_N18/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.24.18.00.22_N18/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.12.49.29_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.12.49.29_N18/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.12.49.29_N18/timings.csv` | 38.3 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.13.04.32_N18/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.13.04.32_N18/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.13.04.32_N18/timings.csv` | 38.3 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.14.35.20_N17/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.14.35.20_N17/predictions.csv` | 33.1 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.14.35.20_N17/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.14.46.04_N17/metrics.json` | 1.5 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.14.46.04_N17/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.25.14.46.04_N17/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.26.12.37.48_N18/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.26.12.37.48_N18/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.26.12.37.48_N18/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.26.14.21.39_N17/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.26.14.21.39_N17/predictions.csv` | 32.9 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/2026.03.26.14.21.39_N17/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/metrics.json` | 0.9 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/holdout/final_selected_N17_N18/summary.csv` | 1.4 KiB | One row per evaluated session with accuracy, F1, recall, timing, and score. |
| Status | `results/metrics/STATUS_EN.md` | 0.7 KiB | Notes that the 56-session aggregate still needs to run. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N01/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N01/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N01/training_manifest.csv` | 2.8 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N02/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N02/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N02/training_manifest.csv` | 2.8 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N03/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N03/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N03/training_manifest.csv` | 2.8 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N06/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N06/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N06/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N08/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N08/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N08/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N09/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N09/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N09/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N10/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N10/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N10/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N11/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N11/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N11/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N13/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N13/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N13/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N14/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N14/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N14/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N15/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N15/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N15/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N16/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N16/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N16/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N17/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N17/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N17/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N18/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N18/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `results/metrics/fold_artifacts/exclude_N18/training_manifest.csv` | 2.7 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| User evidence | `results/reported_mac_2026-09-27/README_EN.md` | 0.9 KiB | Provenance and limits of transcribed Mac console numbers. |
| User evidence | `results/reported_mac_2026-09-27/session_scores_from_console.csv` | 1.0 KiB | Eight per-session Mac scores and corresponding earlier scores. |
| Evaluation evidence | `results/smoke/extra_trees/2025.12.01.16.04.01_N01/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/smoke/extra_trees/2025.12.01.16.04.01_N01/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/smoke/extra_trees/2025.12.01.16.04.01_N01/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/smoke/extra_trees/metrics.json` | 0.9 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/smoke/extra_trees/summary.csv` | 0.3 KiB | One row per evaluated session with accuracy, F1, recall, timing, and score. |
| Evaluation evidence | `results/smoke/optimized_random_forest/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/smoke/optimized_random_forest/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/smoke/optimized_random_forest/timings.csv` | 38.4 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/smoke/previous_random_forest/2025.12.01.16.04.01_N01/metrics.json` | 1.6 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/smoke/previous_random_forest/2025.12.01.16.04.01_N01/predictions.csv` | 33.0 KiB | True and predicted class for every scored EEG window. |
| Evaluation evidence | `results/smoke/previous_random_forest/2025.12.01.16.04.01_N01/timings.csv` | 38.5 KiB | Seconds spent on individual predictions and completed-block fits. |
| Evaluation evidence | `results/smoke/previous_random_forest/metrics.json` | 0.9 KiB | Confusion matrices, weakest-class F1, recall, NIRS bonus, penalties, or aggregate score. |
| Evaluation evidence | `results/smoke/previous_random_forest/summary.csv` | 0.3 KiB | One row per evaluated session with accuracy, F1, recall, timing, and score. |
| Local frozen | `run.py` | 7.7 KiB | Chronological streaming runner and legal block-label release. |
| Test feature cache | `samples/artifacts/artifact_meta.json` | 0.7 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Test feature cache | `samples/artifacts/pretrain_cache.npz` | 1.35 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Test feature cache | `samples/artifacts/training_manifest.csv` | 0.4 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Bundled sample | `samples/data/2025.12.01.16.04.01_N01.mat` | 22.02 MiB | One raw EEG/NIRS recording for the immediate smoke test. |
| Local frozen | `score.py` | 10.4 KiB | Local confusion matrices, class F1/recall, bonuses, and penalties. |
| Submission artifact | `solution/artifacts/artifact_meta.json` | 0.8 KiB | Source subjects, class counts, feature dimensions, and filter/build settings. |
| Submission artifact | `solution/artifacts/pretrain_cache.npz` | 1.81 MiB | Compressed EEG, NIRS, hybrid features, labels, and source IDs; forests are fitted online. |
| Submission artifact | `solution/artifacts/training_manifest.csv` | 3.0 KiB | Exact source-session list and extracted window counts; audit person exclusions here. |
| Essential submission | `solution/classifier.py` | 19.6 KiB | Judge-area selected online Random Forest model. |
| Essential submission | `solution/io_utils.py` | 5.1 KiB | Judge-area MATLAB reader and windows. |
| Essential submission | `solution/run.py` | 7.7 KiB | Judge-area chronological runner. |
| Essential submission | `solution/score.py` | 10.4 KiB | Judge-area local scoring code. |
| Optional rebuild | `solution/train.py` | 10.6 KiB | Judge-area historical cache builder. |
| Test tool | `strict_test.py` | 1.7 KiB | Scores eight N17/N18 sessions against a cache excluding those people. |
| Training tool | `train.py` | 10.6 KiB | Builds a balanced historical feature cache and provenance manifest. |
| Check | `verify_public_data.py` | 1.3 KiB | Compares all public files with manifest sizes and SHA256 hashes. |
| Check | `verify_runtime.py` | 2.0 KiB | Quick causal-filter, chunk, nonfinite EEG, and metadata checks. |
| Check | `verify_submission.py` | 3.0 KiB | Confirms code identity, cache provenance, fold cleanliness, and public metric completion. |
| Guide | `docs/FILE_INVENTORY_EN.md` | generated | This complete per-file inventory. |
| Verification | `SHA256SUMS.txt` | generated | SHA256 hashes of release files; generated immediately before zipping. |
