# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Reference implementation of the standalone WinIntel SM70 primitives.

The functions in this module mirror the CUDA kernels in
``csrc/winintel/sm70`` exactly (packed NHD KV cache, causal decoding,
optional sliding window and soft cap).  They are used as the fallback when the
extension has not been built and as the oracle in the unit tests.
"""

from __future__ import annotations

import torch


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Write ``key``/``value`` into the packed KV cache at ``slot_mapping``."""
    block_size = kv_cache.shape[1]
    head_size = key.shape[-1]
    target_dtype = kv_cache.dtype
    if target_dtype == torch.uint8:
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    for token in range(key.shape[0]):
        slot = int(slot_mapping[token].item())
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        if target_dtype == torch.uint8:
            kv_cache[block, offset, :, :head_size] = (
                (key[token].float() / k_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            )
            kv_cache[block, offset, :, head_size:] = (
                (value[token].float() / v_scale)
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            continue
        kv_cache[block, offset, :, :head_size] = (
            key[token].to(torch.float32) * k_scale
        ).to(target_dtype)
        kv_cache[block, offset, :, head_size:] = (
            value[token].to(torch.float32) * v_scale
        ).to(target_dtype)


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
    hf_rounding: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Reference paged attention.

    Args:
        query: ``[num_tokens, num_heads, head_size]``.
        kv_cache: ``[num_blocks, block_size, num_kv_heads, 2 * head_size]``.
        block_tables: ``[num_seqs, max_blocks_per_seq]`` int32.
        token_seq: sequence index of every query token.
        token_pos: absolute position of every query token.
        scale: softmax scale, normally ``head_size ** -0.5``.
        num_kv_heads: number of KV heads (grouped-query attention aware).
        softcap: if positive, apply ``softcap * tanh(score / softcap)``.
        sliding_window: if positive, only attend to the last
            ``sliding_window`` tokens.
        hf_rounding: round scores and normalized probabilities to FP16.

    Returns:
        ``[num_tokens, num_heads, head_size]`` with the same dtype as ``query``.
    """
    num_tokens, num_heads, head_size = query.shape
    block_size = kv_cache.shape[1]
    group_size = num_heads // num_kv_heads
    head_map = torch.arange(num_heads, device=query.device) // group_size

    out = torch.zeros_like(query)
    for token in range(num_tokens):
        seq = int(token_seq[token].item())
        pos = int(token_pos[token].item())
        start = max(0, pos - sliding_window + 1) if sliding_window > 0 else 0
        if pos < start:
            continue

        positions = torch.arange(start, pos + 1, device=query.device)
        blocks = block_tables[seq, positions // block_size].to(torch.long)
        offsets = positions % block_size
        packed = kv_cache[blocks, offsets]
        if kv_cache.dtype == torch.uint8:
            packed = packed.view(torch.float8_e4m3fn).float()
            packed[..., :head_size] *= k_scale
            packed[..., head_size:] *= v_scale
        else:
            packed = packed.float()
        keys = packed[..., :head_size].index_select(1, head_map)
        values = packed[..., head_size:].index_select(1, head_map)

        q = query[token].to(torch.float32)
        scores = torch.einsum("hd,nhd->hn", q, keys)
        if hf_rounding:
            scores = scores.half().float()
        scores = scores * scale
        if hf_rounding:
            scores = scores.half().float()
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        probs = torch.softmax(scores, dim=-1)
        if hf_rounding:
            probs = probs.half().float()
        out[token] = torch.einsum("hn,nhd->hd", probs, values).to(query.dtype)

    return out
