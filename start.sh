#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
set -euo pipefail
release_root="$(dirname "$(realpath "$0")")"
runtime="${WININTEL_INSTALL_DIR:-$release_root/runtime}"
export SM70_TENSOR_PARALLEL_SIZE="${SM70_TENSOR_PARALLEL_SIZE:-8}"
case "$SM70_TENSOR_PARALLEL_SIZE" in
    2)
        export SM70_MAX_NUM_SEQS="${SM70_MAX_NUM_SEQS:-1}"
        export SM70_MEMORY_UTILIZATION="${SM70_MEMORY_UTILIZATION:-0.92}"
        export SM70_MAX_BATCHED_TOKENS="${SM70_MAX_BATCHED_TOKENS:-512}"
        ;;
    4)
        export SM70_MEMORY_UTILIZATION="${SM70_MEMORY_UTILIZATION:-0.80}"
        ;;
    8) ;;
    *) echo "SM70_TENSOR_PARALLEL_SIZE must be 2, 4, or 8" >&2; exit 2 ;;
esac
case "${WININTEL_PROFILE:-standard}" in
    standard)
        export SM70_MAX_MODEL_LEN="${SM70_MAX_MODEL_LEN:-8192}"
        export SM70_KV_CACHE_DTYPE="${SM70_KV_CACHE_DTYPE:-auto}"
        export SM70_MEMORY_UTILIZATION="${SM70_MEMORY_UTILIZATION:-0.40}"
        ;;
    throughput|long)
        export SM70_KV_CACHE_DTYPE="${SM70_KV_CACHE_DTYPE:-fp8_e4m3}"
        export SM70_MEMORY_UTILIZATION="${SM70_MEMORY_UTILIZATION:-0.80}"
        if [[ "${WININTEL_PROFILE:-}" == long ]]; then
            export SM70_MAX_MODEL_LEN="${SM70_MAX_MODEL_LEN:-262144}"
            export WININTEL_SM70_TORCH_PREFILL="${WININTEL_SM70_TORCH_PREFILL:-1}"
        else
            export SM70_MAX_MODEL_LEN="${SM70_MAX_MODEL_LEN:-8192}"
        fi
        ;;
    *) echo "WININTEL_PROFILE must be standard, throughput, or long" >&2; exit 2 ;;
esac
export ENGINE_ROOT="$runtime"
export VLLM_TARGET_DEVICE=cuda
exec bash "$runtime/scripts/serve_sm70_qwen_tp8.sh" "${1:?Provide a local model directory}"
