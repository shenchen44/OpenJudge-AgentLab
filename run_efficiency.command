#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/agentlab efficiency-plan --pilot runs/ds-pilot-v3 --out runs/ds-pilot-v3
export AGENTLAB_BASE_URL='https://api.deepseek.com'
export AGENTLAB_MODEL='deepseek-flash'
printf '%s\n' '只跑12条短Prompt候选；复用原对照；沿用原150次累计请求上限，不会自动扩容。'
# Provider reads the local .env file; existing environment variables take precedence.

.venv/bin/agentlab efficiency --pilot runs/ds-pilot-v3 --out runs/ds-pilot-v3 --max-api-calls 150 --max-reserved-tokens 1000000
open -a TextEdit runs/ds-pilot-v3/efficiency.json
