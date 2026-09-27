#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  PYTHON_EXE=.venv/bin/python
else
  PYTHON_EXE=python3.11
fi
"$PYTHON_EXE" verify_runtime.py
"$PYTHON_EXE" quick_test.py --out results/my_quick_test
printf 'Results: results/my_quick_test/metrics.json\n'
