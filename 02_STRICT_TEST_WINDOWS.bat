@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Create the virtual environment first. See README.md
  exit /b 1
)
.venv\Scripts\python.exe download_sessions.py --out data_all --subjects N17 N18 || exit /b 1
.venv\Scripts\python.exe strict_test.py --data data_all --out results\my_strict_test || exit /b 1
echo Done. Read results\my_strict_test\metrics.json
pause
