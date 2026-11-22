# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Dispatcher for the WinIntel SM70 kernels.

The CUDA extension is optional: when it is not available the dispatcher falls
back to the reference implementation so that the rest of the engine keeps
working.  The fallback is correct but only suitable for development and for
small batches.

Set ``WININTEL_SM70_JIT=0`` to disable the on-demand extension build.
``WININTEL_SM70_TILED_PREFILL=1`` enables experimental paged
WMMA prefill for at least sixteen query tokens and head sizes 64/128.
It rounds softmax probabilities to FP16 and requires model-quality validation.
``WININTEL_SM70_MULTI_SEQUENCE_PREFILL=1`` additionally enables multi-sequence
tiles. Trained OLMoE has shown output differences with that experimental path.
``WININTEL_SM70_MOE_HF_ROUNDING=1`` enables experimental FP16 activation,
routing-weight and expert-order accumulation rounding for Transformers checks.
``WININTEL_SM70_ATTENTION_HF_ROUNDING=1`` enables experimental rounded scores
and normalized probabilities. It uses scalar prefill and two attention passes.
``WININTEL_SM70_COOPERATIVE_DECODE=1`` enables four-warp rounded decode for
head size 128, up to three query tokens and a table capacity of at most 4096.
It changes FP32 value summation order and requires model-quality validation.
"""

from __future__ import annotations

import logging
import os

import torch

from . import reference

logger = logging.getLogger(__name__)

_extension_state: bool | None = None
_warned = False


def _register_extension(build: bool = False, retry: bool = False) -> bool:
    global _extension_state, _warned
    if _extension_state is True:
        return True

    from .build import already_loaded, load_extension

    try:
        if already_loaded() is None:
            if _extension_state is False and not retry:
                return False
            if not build or os.environ.get("WININTEL_SM70_JIT", "1") == "0":
                if not _warned:
                    _warned = True
                    logger.warning(
                        "WinIntel SM70 CUDA extension is not built; using the "
                        "slow reference path. Build it with "
                        "`.venv/bin/python -m winintel_kernels.build`."
                    )
                return False
            load_extension()
        if not all(
            hasattr(torch.ops.winintel_sm70, name)
            for name in ("paged_attention", "reshape_and_cache")
        ):
            raise RuntimeError("extension did not register torch.ops.winintel_sm70")
        _extension_state = True
    except Exception as exc:  # pragma: no cover - depends on local toolchain
        _extension_state = False
        if not _warned:
            _warned = True
            logger.warning(
                "WinIntel SM70 CUDA extension is unavailable (%s); "
                "falling back to the slow reference path.",
                exc,
            )
    return _extension_state


def is_extension_available(build: bool = True) -> bool:
    """Return whether the CUDA kernels can be used on this machine."""
    if not torch.cuda.is_available():
        return False
    return _register_extension(build=build, retry=build)


def fused_moe(
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    weight_on_input: bool = False,
    out: torch.Tensor | None = None,
    gated: torch.Tensor | None = None,
    projected: torch.Tensor | None = None,
    hf_rounding: bool | None = None,
) -> torch.Tensor:
    """Compute routed FP16 SwiGLU experts using native CUDA only."""
    if not _register_extension(build=True) or not hasattr(
        torch.ops.winintel_sm70, "fused_moe"
    ):
        raise RuntimeError("Build the WinIntel SM70 extension with MoE support")
    m, h = input.shape
    k = ids.shape[1]
    if out is None:
        out = torch.empty_like(input)
    if gated is None:
        gated = input.new_empty((m, k, w2.shape[2]))
    if projected is None:
        projected = input.new_empty((m, k, h))
    if hf_rounding is None:
        hf_rounding = os.environ.get("WININTEL_SM70_MOE_HF_ROUNDING", "0") == "1"
    torch.ops.winintel_sm70.fused_moe(
        out,
        input,
        w13,
        w2,
        weights.float().contiguous(),
        ids.int().contiguous(),
        None if expert_map is None else expert_map.int().contiguous(),
        gated,
        projected,
        weight_on_input,
        hf_rounding,
    )
    return out


def _can_use_kernel(query: torch.Tensor) -> bool:
    return (
        query.is_cuda
        and query.dtype == torch.float16
        and query.shape[-1] in (64, 128, 256)
        and _register_extension(build=True)
    )


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Store ``key``/``value`` into the packed NHD KV cache."""
    if not _can_use_kernel(key):
        reference.reshape_and_cache(
            key, value, kv_cache, slot_mapping, k_scale, v_scale
        )
        return

    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    torch.ops.winintel_sm70.reshape_and_cache(
        key.contiguous(),
        value.contiguous(),
        kv_cache,
        slot_mapping.contiguous(),
        float(k_scale),
        float(v_scale),
    )


def paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    token_seq: torch.Tensor,
    token_pos: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    softcap: float = 0.0,
    sliding_window: int = -1,
    num_partitions: int = 1,
    out: torch.Tensor | None = None,
    hf_rounding: bool | None = None,
    cooperative_decode: bool | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Paged decode / chunked-prefill attention.

    Args:
        query: ``[num_tokens, num_heads, head_size]``.
        kv_cache: ``[num_blocks, block_size, num_kv_heads, 2 * head_size]``.
        block_tables: ``[num_seqs, max_blocks_per_seq]`` int32.
        token_seq/token_pos: per-token sequence index and absolute position.
        scale: softmax scale.
        num_kv_heads: number of KV heads.
        softcap/sliding_window: optional attention modifiers.
        num_partitions: KV split factor for long-context decoding.
        out: optional pre-allocated output.
        hf_rounding: round scores and normalized probabilities to FP16;
            defaults to ``WININTEL_SM70_ATTENTION_HF_ROUNDING``.
        cooperative_decode: use experimental four-warp rounded decode;
            defaults to ``WININTEL_SM70_COOPERATIVE_DECODE``.
        k_scale/v_scale: dequantization scales for uint8 E4M3 KV caches.

    Returns:
        The attention output, same shape and dtype as ``query``.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if hf_rounding is None:
        hf_rounding = os.environ.get("WININTEL_SM70_ATTENTION_HF_ROUNDING", "0") == "1"
    if hf_rounding:
        num_partitions = max(2, num_partitions)
    if cooperative_decode is None:
        cooperative_decode = (
            os.environ.get("WININTEL_SM70_COOPERATIVE_DECODE", "0") == "1"
        )

    if out is None:
        out = torch.empty_like(query)

    if not _can_use_kernel(query) or kv_cache.dtype not in (
        torch.float16,
        torch.uint8,
    ):
        result = reference.paged_attention(
            query,
            kv_cache,
            block_tables,
            token_seq,
            token_pos,
            scale,
            num_kv_heads,
            softcap,
            sliding_window,
            hf_rounding=hf_rounding,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        out.copy_(result)
        return out

    num_tokens, num_heads, head_size = query.shape
    empty = torch.empty(0, dtype=torch.float32, device=query.device)
    if num_partitions > 1:
        partial_acc = torch.empty(
            (num_tokens, num_heads, num_partitions, head_size),
            dtype=torch.float32,
            device=query.device,
        )
        partial_max = torch.empty(
            (num_tokens, num_heads, num_partitions),
            dtype=torch.float32,
            device=query.device,
        )
        partial_sum = torch.empty_like(partial_max)
    else:
        partial_acc = partial_max = partial_sum = empty

    torch.ops.winintel_sm70.paged_attention(
        out,
        query,
        kv_cache,
        block_tables,
        token_seq,
        token_pos,
        partial_acc,
        partial_max,
        partial_sum,
        float(scale),
        int(kv_cache.shape[1]),
        int(num_kv_heads),
        int(num_partitions),
        float(softcap),
        int(sliding_window),
        (
            os.environ.get("WININTEL_SM70_TILED_PREFILL", "0") == "1"
            and not hf_rounding
            and num_partitions == 1
            and num_tokens >= 16
            and (
                block_tables.shape[0] == 1
                or os.environ.get("WININTEL_SM70_MULTI_SEQUENCE_PREFILL", "0") == "1"
            )
            and head_size in (64, 128)
        ),
        hf_rounding,
        cooperative_decode,
        float(k_scale),
        float(v_scale),
    )
    return out
