#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/fetch_upstream.sh
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 --allow-existing .venv
  uv pip install --python .venv/bin/python -c requirements.lock -e ./upstream -e '.[dev]'
else
  PYTHON_BIN="${AGENTLAB_PYTHON:-python3.11}"
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install -c requirements.lock -e ./upstream -e '.[dev]'
fi
.venv/bin/python -c 'import openjudge; from agentlab.graders import StateOutcomeGrader; print("OpenJudge source:", openjudge.__file__)'
echo 'Ready: source .venv/bin/activate'
