@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Create the virtual environment first. See README.md
  exit /b 1
)
.venv\Scripts\python.exe download_sessions.py --out data_all --subjects N17 N18 || exit /b 1
.venv\Scripts\python.exe strict_test.py --data data_all --out results\my_extra_trees --classifier experiments\classifier_extra_trees.py || exit /b 1
.venv\Scripts\python.exe compare_results.py --before results\holdout\final_selected_N17_N18 --after results\my_extra_trees || exit /b 1
echo Done. The comparison above measures the experimental candidate on the same eight sessions.
pause
