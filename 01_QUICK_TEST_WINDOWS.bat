@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Create the virtual environment first. See README.md
  exit /b 1
)
.venv\Scripts\python.exe verify_runtime.py || exit /b 1
.venv\Scripts\python.exe quick_test.py --out results\my_quick_test || exit /b 1
echo Done. Read results\my_quick_test\metrics.json
pause
