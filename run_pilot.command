#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/agentlab ]; then bash scripts/setup.sh; fi
export AGENTLAB_BASE_URL='https://api.deepseek.com'
export AGENTLAB_MODEL='deepseek-flash'
export AGENTLAB_OPTIMIZER_MODEL='deepseek-flash'
printf '%s\n' 'DeepSeek官方API预实验：24个任务执行，最多150次HTTP尝试；只跑预实验，不自动调优或测试。'
# Provider reads the local .env file; existing environment variables take precedence.

.venv/bin/agentlab pilot --out runs/ds-pilot-v3 --concurrency 1 --task-timeout 300 --max-api-calls 150 --max-reserved-tokens 1000000
open -a TextEdit runs/ds-pilot-v3/pilot.json
