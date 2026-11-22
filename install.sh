#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
set -euo pipefail
release_root="$(dirname "$(realpath "$0")")"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [[ "${1:-}" == "--help" ]]; then
    printf 'winintel AI winintel-vllm\nUsage: bash install.sh [local-model-directory]\n'
    exit 0
fi
for tool in curl tar git g++ nvcc nvidia-smi; do
    command -v "$tool" >/dev/null || { echo "Install the required tool: $tool" >&2; exit 2; }
done
if ! command -v uv >/dev/null; then
    uv_setup="$(mktemp)"
    curl --fail --location --max-time 60 \
        "${WININTEL_UV_INSTALL_URL:-https://astral.sh/uv/install.sh}" -o "$uv_setup"
    sh "$uv_setup"
    rm "$uv_setup"
fi
uv venv --python 3.12 "$release_root/.installer.venv"
exec uv run --no-project --python "$release_root/.installer.venv/bin/python" \
    "$release_root/scripts/install_qwen.py" "$@"
