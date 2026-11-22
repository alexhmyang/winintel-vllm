#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
set -euo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$(pwd)/.venv/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS="${MAX_JOBS:-4}"
source_root="$(pwd)"
engine_python="$source_root/.venv/bin/python"
"$engine_python" - <<'PY'
import torch
assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
assert "sm_70" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
print("Building WinIntel with official Torch", torch.__version__)
PY
".venv/bin/cmake" -S . -B build/sm70 -G Ninja \
    -DVLLM_PYTHON_EXECUTABLE="$engine_python" \
    -DCMAKE_INSTALL_PREFIX="$source_root" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWININTEL_SM70_ONLY=ON \
    -DVLLM_TARGET_DEVICE=cuda
".venv/bin/cmake" --build build/sm70 \
    --target _C _moe_C cumem_allocator --parallel "$MAX_JOBS"
for component in _C _moe_C cumem_allocator; do
    ".venv/bin/cmake" --install build/sm70 --component "$component"
done
"$engine_python" - <<'PY'
import vllm._C
import vllm._moe_C
from vllm.platforms import current_platform
print("Native extension:", vllm._C.__file__)
print("Platform:", current_platform)
PY
