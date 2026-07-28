#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ ! -x .venv/bin/gunicorn ]]; then
  echo "未找到 Gunicorn，请先使用 Python 3.12 创建 .venv 并安装 requirements.txt。" >&2
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  source ./.env
  set +a
fi

host="${SG_QUANT_HOST:-0.0.0.0}"
port="${SG_QUANT_PORT:-5000}"
exec ./.venv/bin/gunicorn --workers "${SG_QUANT_WORKERS:-1}" --bind "$host:$port" dashboard:app
