#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
upstream_ref='c2fdc38314cd971b03d4d9333cd2c9f24d85c27f'
if [ ! -d upstream/.git ]; then
  if [ -e upstream ]; then
    echo 'upstream/ exists without Git metadata; refusing to overwrite it.' >&2
    exit 1
  fi
  git clone --no-checkout https://github.com/agentscope-ai/OpenJudge.git upstream
  git -C upstream fetch origin "$upstream_ref"
  git -C upstream checkout --detach "$upstream_ref"
fi
if [ "$(git -C upstream rev-parse HEAD)" != "$upstream_ref" ]; then
  echo 'upstream/ is at a different revision; refusing to reset local work.' >&2
  exit 1
fi
