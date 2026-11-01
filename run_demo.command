#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/agentlab ]; then bash scripts/setup.sh; fi
.venv/bin/agentlab demo --out runs/v3-demo
open runs/v3-demo/report.html
