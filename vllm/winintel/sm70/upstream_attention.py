# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Optional adapter for vLLM's Apache-2.0 paged attention on V100."""

from __future__ import annotations

import os

import torch

from vllm import _custom_ops
from winintel_kernels import ops


def cache_views(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, blocks, heads, chunks, block_size, vector = cache.shape
    return cache[0], cache[1].view(blocks, heads, chunks * vector, block_size)


def cutlass_prefill(query, keys, values, metadata, capacity, scale, out):
    """Reuse PyTorch CUTLASS with paged prefixes and bottom-right causal masking."""
    kv_heads = keys.shape[2]
    keys = keys.reshape(-1, kv_heads, query.shape[-1])
    values = values.reshape_as(keys)
    repeats = query.shape[1] // kv_heads
    if repeats > 1:
        keys = keys.repeat_interleave(repeats, dim=1)
        values = values.repeat_interleave(repeats, dim=1)
    key_starts = (
        torch.arange(
            metadata.seq_lens.shape[0] + 1,
            device=query.device,
            dtype=torch.int32,
        )
        * capacity
    )
    result = torch.ops.aten._efficient_attention_forward(
        query.unsqueeze(0),
        keys.unsqueeze(0),
        values.unsqueeze(0),
        None,
        metadata.query_start_loc,
        key_starts,
        metadata.max_query_len,
        capacity,
        0.0,
        2,
        False,
        scale=scale,
        seqlen_k=metadata.seq_lens,
    )[0]
    out.copy_(result.squeeze(0))


def packed_cutlass_prefill(query, cache, table, metadata, scale, ks, vs, out):
    """Gather packed FP16/E4M3 pages only for prefill, retaining native decode."""
    block_size = cache.shape[1]
    columns = min(table.shape[1], (metadata.max_seq_len + block_size - 1) // block_size)
    packed = cache.index_select(0, table[:, :columns].long().reshape(-1))
    if cache.dtype == torch.uint8:
        packed = packed.view(torch.float8_e4m3fn).half()
        packed[..., : query.shape[-1]].mul_(ks)
        packed[..., query.shape[-1] :].mul_(vs)
    cutlass_prefill(
        query,
        packed[..., : query.shape[-1]],
        packed[..., query.shape[-1] :],
        metadata,
        columns * block_size,
        scale,
        out,
    )


class UpstreamAttention:
    """Use existing decode and optional PyTorch CUTLASS with one KV cache."""

    def __init__(self) -> None:
        self.workspaces: dict[tuple, tuple[torch.Tensor, ...]] = {}
        self.scales: dict[torch.device, torch.Tensor] = {}
        self.cutlass_prefill = os.environ.get("WININTEL_SM70_TORCH_PREFILL", "0") == "1"

    def scale_tensor(self, device: torch.device) -> torch.Tensor:
        if device not in self.scales:
            self.scales[device] = torch.ones((), device=device, dtype=torch.float32)
        return self.scales[device]

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        key_cache, value_cache = cache_views(cache)
        scale = self.scale_tensor(cache.device)
        _custom_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slots, "auto", scale, scale
        )

    def forward(
        self,
        query: torch.Tensor,
        cache: torch.Tensor,
        table: torch.Tensor,
        metadata,
        scale: float,
        num_kv_heads: int,
        out: torch.Tensor,
    ) -> None:
        key_cache, value_cache = cache_views(cache)
        block_size = cache.shape[4]
        if metadata.max_query_len > 1 or table.shape[0] != query.shape[0]:
            columns = min(
                table.shape[1], (metadata.max_seq_len + block_size - 1) // block_size
            )
            selected = table[:, :columns].long().reshape(-1)
            keys = key_cache.index_select(0, selected).permute(0, 3, 1, 2, 4)
            keys = keys.reshape(-1, block_size, num_kv_heads, query.shape[-1])
            values = value_cache.index_select(0, selected).permute(0, 3, 1, 2)
            if (
                self.cutlass_prefill
                and not metadata.dflash_window
                and metadata.max_query_len > 1
                and metadata.query_start_loc is not None
                and metadata.seq_lens is not None
                and int(metadata.query_start_loc[-1]) == query.shape[0]
            ):
                capacity = columns * block_size
                cutlass_prefill(query, keys, values, metadata, capacity, scale, out)
                return
            native_cache = torch.cat((keys, values), dim=-1).contiguous()
            native_table = torch.arange(
                selected.numel(), device=table.device, dtype=torch.int32
            ).view(table.shape[0], columns)
            ops.paged_attention(
                query,
                native_cache,
                native_table,
                metadata.token_seq,
                metadata.token_pos,
                scale,
                num_kv_heads,
                sliding_window=metadata.dflash_window or -1,
                num_partitions=16 if metadata.dflash_window else 1,
                out=out,
            )
            return

        # Table capacity stays constant throughout a decode graph replay.
        max_context = table.shape[1] * block_size
        partitions = (max_context + 511) // 512
        tokens, heads, dimension = query.shape
        workspace_key = (query.device, tokens, heads, dimension, partitions)
        if workspace_key not in self.workspaces:
            sums = torch.empty(
                tokens, heads, partitions, device=query.device, dtype=torch.float32
            )
            maxima = torch.empty_like(sums)
            partial = torch.empty(
                tokens,
                heads,
                partitions,
                dimension,
                device=query.device,
                dtype=query.dtype,
            )
            self.workspaces[workspace_key] = (sums, maxima, partial)
        sums, maxima, partial = self.workspaces[workspace_key]
        lengths = metadata.seq_lens
        if lengths is None:
            lengths = metadata.token_pos + 1
        scale_tensor = self.scale_tensor(query.device)
        _custom_ops.paged_attention_v2(
            out,
            sums,
            maxima,
            partial,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            table,
            lengths,
            block_size,
            max_context,
            None,
            "auto",
            scale_tensor,
            scale_tensor,
        )
        out.masked_fill_((lengths <= 0).view(-1, 1, 1), 0)
