#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  PYTHON_EXE=.venv/bin/python
else
  PYTHON_EXE=python3.11
fi
"$PYTHON_EXE" strict_test.py --data data_all --out results/my_strict_test
printf 'Results: results/my_strict_test/summary.csv and metrics.json\n'
