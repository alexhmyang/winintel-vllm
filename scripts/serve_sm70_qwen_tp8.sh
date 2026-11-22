#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
# NVFP4 checkpoint compatibility: weights are decoded once to FP16 on V100.
set -euo pipefail
engine_root="${ENGINE_ROOT:-$(dirname "$(dirname "$(realpath "$0")")")}"
model="${1:?Provide the Qwen NVFP4 checkpoint directory}"
tp="${SM70_TENSOR_PARALLEL_SIZE:-8}"
case "$tp" in
    2|4|8) ;;
    *) echo "SM70_TENSOR_PARALLEL_SIZE must be 2, 4, or 8" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=0
    for ((device = 1; device < tp; device++)); do
        CUDA_VISIBLE_DEVICES+=",$device"
    done
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#devices[@]}" -lt "$tp" ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose at least $tp GPUs" >&2
    exit 2
fi
draft_tokens="${SM70_SPECULATIVE_TOKENS:-7}"
case "$draft_tokens" in
    3|5|7|9|11) ;;
    *) echo "SM70_SPECULATIVE_TOKENS must be 3, 5, 7, 9, or 11" >&2; exit 2 ;;
esac
verify_tokens=$((draft_tokens + 1))
method="${SM70_SPECULATIVE_METHOD:-mtp}"
draft_model="${SM70_DRAFT_MODEL:-}"
scheduler_args=()
case "$method" in
    mtp) ;;
    dflash2)
        if [[ "${WININTEL_ENABLE_EXPERIMENTAL_DFLASH2:-0}" != 1 ]]; then
            echo "DFlash2 is experimental and has no validated speedup on V100. Set WININTEL_ENABLE_EXPERIMENTAL_DFLASH2=1 for development." >&2
            exit 2
        fi
        if [[ "$draft_tokens" != 7 || ! -f "$draft_model/config.json" ]]; then
            echo "DFlash2 requires seven draft tokens and SM70_DRAFT_MODEL with config.json" >&2
            exit 2
        fi
        export WININTEL_SM70_DFLASH_GRAPH="${WININTEL_SM70_DFLASH_GRAPH:-0}"
        scheduler_args+=(--no-async-scheduling)
        ;;
    *) echo "SM70_SPECULATIVE_METHOD must be mtp or dflash2" >&2; exit 2 ;;
esac
speculative_config="$("$engine_root/.venv/bin/python" -c '
import json, sys
config = {
    "method": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "enforce_eager": False,
    "use_local_argmax_reduction": True,
}
if sys.argv[1] == "dflash2":
    config["model"] = sys.argv[3]
print(json.dumps(config))
' "$method" "$draft_tokens" "$draft_model")"
max_seqs="${SM70_MAX_NUM_SEQS:-8}"
if ! [[ "$max_seqs" =~ ^[1-9][0-9]*$ ]]; then
    echo "SM70_MAX_NUM_SEQS must be positive" >&2
    exit 2
fi
capture_sizes=""
for ((batch = 1; batch <= max_seqs; batch++)); do
    capture_sizes+="${capture_sizes:+,}$((batch * verify_tokens))"
done
export PYTHONPATH="$engine_root${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$engine_root/.venv/bin${CUDA_HOME:+:$CUDA_HOME/bin}:$PATH"
export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WININTEL_SM70_CUDAGRAPH=1
export WININTEL_SM70_GDN_HF_ROUNDING=1
export WININTEL_SM70_NVFP4_GPU_DEQUANT=1
export WININTEL_SM70_MTP_GRAPH=1
export WININTEL_SM70_MTP_GRAPH_VERIFY=1
export WININTEL_SM70_NARROW_BLASLT=1
export WININTEL_SM70_GATE_BLASLT=1
export NCCL_ALGO="${NCCL_ALGO:-allreduce:Tree}"
export NCCL_PROTO="${NCCL_PROTO:-allreduce:LL}"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS="${MAX_JOBS:-4}"
exec "$engine_root/.venv/bin/python" -m vllm.winintel.cli serve "$model" \
    "${scheduler_args[@]}" \
    --served-model-name "${SM70_SERVED_MODEL_NAME:-winintel-qwen27-nvfp4-tp$tp}" \
    --host 0.0.0.0 --port "${SM70_PORT:-8037}" \
    --tensor-parallel-size "$tp" --dtype float16 \
    --kv-cache-dtype "${SM70_KV_CACHE_DTYPE:-auto}" \
    --attention-backend WININTEL_SM70 --no-enforce-eager \
    --compilation-config \
    "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$capture_sizes]}" \
    --speculative-config \
    "$speculative_config" \
    --max-model-len "${SM70_MAX_MODEL_LEN:-8192}" --max-num-seqs "$max_seqs" \
    --max-num-batched-tokens "${SM70_MAX_BATCHED_TOKENS:-6144}" --gpu-memory-utilization \
    "${SM70_MEMORY_UTILIZATION:-0.40}" --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}'
