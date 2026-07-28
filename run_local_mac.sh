#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ ! -x .venv/bin/python ]]; then
  echo "未找到 .venv，请先使用 Python 3.12 创建虚拟环境并安装 requirements.txt。" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source ./.env
  set +a
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/.runtime/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

exec ./.venv/bin/python dashboard.py
