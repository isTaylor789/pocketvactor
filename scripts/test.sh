#!/usr/bin/env bash
# Tests sin GPU (validación + sobre). Uso: ./scripts/test.sh
set -u
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
  elif [ -x env/bin/python ]; then
    PY=env/bin/python
  else
    PY=python
  fi
fi

"$PY" -m pytest tests/ -q
