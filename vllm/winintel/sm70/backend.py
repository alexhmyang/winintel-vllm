# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""WinIntel SM70 attention backend for the v1 engine.

The backend is a thin scheduling layer around
``csrc/winintel/sm70/winintel_sm70_paged_attention.cu``.  It keeps the KV
cache in the packed NHD layout

    [num_blocks, block_size, num_kv_heads, 2 * head_size]

so K and V live next to each other for every (block, slot, head) tuple and a
single warp can stream them with fully coalesced loads.
FP8 E4M3 uses the same layout with byte storage and layer K/V scales.

``WININTEL_SM70_UPSTREAM_DECODE=1`` instead stores upstream paged K/V and
uses vLLM's existing decode kernel. Native prefill gathers the active pages.
``WININTEL_SM70_TORCH_PREFILL=1`` additionally uses PyTorch's existing
CUTLASS prefill on those pages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from winintel_kernels import ops

_MAX_PARTITIONS = 128
_SMALL_BATCH_LIMIT = 64
_PARTITION_CONTEXT = 128
_MIN_DECODE_PARTITIONS = 8


@dataclass
class WinIntelSM70Metadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table: torch.Tensor
    token_seq: torch.Tensor
    token_pos: torch.Tensor
    fixed_num_partitions: int | None = None
    seq_lens: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    dflash_window: int = 0


class WinIntelSM70MetadataBuilder(AttentionMetadataBuilder[WinIntelSM70Metadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport:
        from vllm.config.compilation import CompilationMode, CUDAGraphMode

        if (
            os.environ.get("WININTEL_SM70_CUDAGRAPH") == "1"
            and not vllm_config.model_config.enforce_eager
            and vllm_config.compilation_config.mode == CompilationMode.NONE
            and vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_DECODE_ONLY
        ):
            if vllm_config.speculative_config is not None:
                return AttentionCGSupport.UNIFORM_BATCH
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.reorder_batch_threshold = 1
        self.graph_decode = (
            self.get_cudagraph_support(vllm_config, kv_cache_spec)
            != AttentionCGSupport.NEVER
        )
        self.decode_query_len = (
            1
            if vllm_config.speculative_config is None
            else 1 + vllm_config.speculative_config.num_speculative_tokens
        )
        if self.graph_decode:
            if not ops.is_extension_available(build=True):
                raise RuntimeError("SM70 CUDA Graph requires the native extension")
            capacity = vllm_config.scheduler_config.max_num_seqs
            self.decode_token_seq = torch.arange(
                capacity, dtype=torch.int32, device=device
            )
            self.spec_token_seq = torch.repeat_interleave(
                self.decode_token_seq, self.decode_query_len
            )
            self.spec_token_offset = torch.arange(
                self.decode_query_len, dtype=torch.int32, device=device
            ).repeat(capacity)
            self.decode_token_pos = torch.empty(
                capacity * self.decode_query_len, dtype=torch.int32, device=device
            )
            self.decode_partitions = max(
                _MIN_DECODE_PARTITIONS,
                min(
                    _MAX_PARTITIONS,
                    (vllm_config.model_config.max_model_len + _PARTITION_CONTEXT - 1)
                    // _PARTITION_CONTEXT,
                ),
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> WinIntelSM70Metadata:
        if common_prefix_len > 0:
            raise NotImplementedError(
                "WinIntel SM70 does not support cascade attention"
            )
        causal = common_attn_metadata.causal
        dflash = (
            not causal
            and self.vllm_config.speculative_config is not None
            and self.vllm_config.speculative_config.method == "dflash2"
        )
        if not isinstance(causal, bool) or (not causal and not dflash):
            raise NotImplementedError(
                "WinIntel SM70 only supports causal decoder attention"
            )

        num_reqs = common_attn_metadata.num_reqs
        # ``num_actual_tokens`` in the common metadata can include padding that
        # is only used by graph capture.  The real token count is the end of
        # the query start locations.
        num_tokens = int(common_attn_metadata.query_start_loc_cpu[num_reqs])
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        device = query_start_loc.device

        fixed_num_partitions = None
        query_len = common_attn_metadata.max_query_len
        query_lens_cpu = (
            common_attn_metadata.query_start_loc_cpu[1 : num_reqs + 1]
            - common_attn_metadata.query_start_loc_cpu[:num_reqs]
        )
        uniform_decode = query_len in (1, self.decode_query_len) and bool(
            ((query_lens_cpu == query_len) | (query_lens_cpu == 0)).all()
        )
        if self.graph_decode and uniform_decode:
            # Graphs also execute padded rows, whose seq_lens are zero.
            # Keep addresses and partition count stable across all replays.
            num_tokens = num_reqs * query_len
            token_pos = self.decode_token_pos[:num_tokens]
            if query_len == 1:
                token_seq = self.decode_token_seq[:num_reqs]
                torch.sub(seq_lens[:num_reqs], 1, out=token_pos)
            else:
                token_seq = self.spec_token_seq[:num_tokens]
                positions = torch.repeat_interleave(seq_lens[:num_reqs], query_len)
                torch.sub(positions, query_len, out=token_pos)
                token_pos.add_(self.spec_token_offset[:num_tokens])
            fixed_num_partitions = (
                self.decode_partitions if num_reqs <= _SMALL_BATCH_LIMIT else 1
            )
        elif common_attn_metadata.max_query_len == 1 and num_tokens == num_reqs:
            token_seq = torch.arange(num_reqs, dtype=torch.int32, device=device)
            token_pos = seq_lens[:num_reqs].to(torch.int32) - 1
        else:
            query_lens = (
                query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
            ).to(torch.int32)
            starts = query_start_loc[:num_reqs].to(torch.int32)
            token_seq = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int32, device=device),
                query_lens,
            )
            token_pos = (
                torch.arange(num_tokens, dtype=torch.int32, device=device)
                - torch.repeat_interleave(starts, query_lens)
                + torch.repeat_interleave(
                    seq_lens[:num_reqs].to(torch.int32) - query_lens, query_lens
                )
            )
            if token_seq.numel() != num_tokens:
                raise RuntimeError(
                    "WinIntel SM70 metadata expects query_start_loc to cover "
                    f"all {num_tokens} tokens, got {token_seq.numel()}"
                )
        dflash_window = 0
        if dflash:
            config = self.vllm_config.speculative_config.draft_model_config.hf_config
            block = config.dflash_config["block_size"]
            if query_len != block or not bool((query_lens_cpu == block).all()):
                raise ValueError("DFlash2 attention requires complete draft blocks")
            token_pos = torch.repeat_interleave(
                seq_lens[:num_reqs].to(torch.int32) - 1, block
            )
            dflash_window = config.sliding_window + block
        return WinIntelSM70Metadata(
            num_actual_tokens=num_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table=common_attn_metadata.block_table_tensor,
            token_seq=token_seq,
            token_pos=token_pos,
            fixed_num_partitions=fixed_num_partitions,
            seq_lens=seq_lens[:num_reqs],
            query_start_loc=query_start_loc[: num_reqs + 1],
            dflash_window=dflash_window,
        )


class WinIntelSM70Backend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "fp8",
        "fp8_e4m3",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "WININTEL_SM70"

    @staticmethod
    def get_impl_cls() -> type[WinIntelSM70Impl]:
        return WinIntelSM70Impl

    @staticmethod
    def get_builder_cls() -> type[WinIntelSM70MetadataBuilder]:
        return WinIntelSM70MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        if os.environ.get("WININTEL_SM70_UPSTREAM_DECODE", "0") == "1":
            return [16, 32]
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if os.environ.get("WININTEL_SM70_UPSTREAM_DECODE", "0") == "1":
            return (2, num_blocks, num_kv_heads, head_size // 8, block_size, 8)
        return (num_blocks, block_size, num_kv_heads, 2 * head_size)

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        return "NHD"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype == torch.float16

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7 and capability.minor == 0

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return os.environ.get("WININTEL_SM70_UPSTREAM_DECODE", "0") != "1"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args: Any, **kwargs: Any) -> bool:
        return False


class WinIntelSM70Impl(AttentionImpl[WinIntelSM70Metadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("WinIntel SM70 does not support ALiBi")
        if kwargs.get("sinks") is not None:
            raise NotImplementedError("WinIntel SM70 does not support attention sinks")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"WinIntel SM70 only supports decoder attention, got {attn_type}"
            )
        if head_size not in (64, 128, 256):
            raise ValueError(
                f"WinIntel SM70 supports head sizes 64/128/256, got {head_size}"
            )
        if kv_cache_dtype not in ("auto", "float16", "fp8", "fp8_e4m3"):
            raise ValueError(
                f"Unsupported WinIntel SM70 KV cache dtype: {kv_cache_dtype}"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = -1 if sliding_window is None else int(sliding_window)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = float(logits_soft_cap or 0.0)
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.upstream_attention = None
        if os.environ.get("WININTEL_SM70_UPSTREAM_DECODE", "0") == "1":
            if kv_cache_dtype not in ("auto", "float16"):
                raise NotImplementedError(
                    "FP8 KV requires the native packed NHD attention path"
                )
            if sliding_window is not None or self.logits_soft_cap:
                raise NotImplementedError(
                    "Upstream SM70 decode does not support sliding window or softcap"
                )
            from .upstream_attention import UpstreamAttention

            self.upstream_attention = UpstreamAttention()

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if os.environ.get("WININTEL_SM70_FP32_REDUCTION", "0") == "1":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    def do_kv_cache_update(
        self,
        layer: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        k_scale = float(getattr(layer, "_k_scale_float", 1.0))
        v_scale = float(getattr(layer, "_v_scale_float", 1.0))
        if self.upstream_attention is not None:
            self.upstream_attention.update(key, value, kv_cache, slot_mapping)
            return
        ops.reshape_and_cache(
            key, value, kv_cache, slot_mapping, k_scale=k_scale, v_scale=v_scale
        )

    def _num_partitions(self, metadata: WinIntelSM70Metadata) -> int:
        if metadata.fixed_num_partitions is not None:
            return metadata.fixed_num_partitions
        if metadata.max_query_len > 1:
            return 1
        if metadata.num_actual_tokens > _SMALL_BATCH_LIMIT:
            return 1
        return max(
            _MIN_DECODE_PARTITIONS,
            min(
                _MAX_PARTITIONS,
                (metadata.max_seq_len + _PARTITION_CONTEXT - 1) // _PARTITION_CONTEXT,
            ),
        )

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: WinIntelSM70Metadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "WinIntel SM70 does not support fused output quantization"
            )
        if attn_metadata is None:
            return output.fill_(0)

        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens == 0:
            return output
        query = query[:num_tokens]
        if not query.is_contiguous():
            query = query.contiguous()

        out = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        block_table = attn_metadata.block_table
        if block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)
        if self.upstream_attention is not None:
            self.upstream_attention.forward(
                query,
                kv_cache,
                block_table,
                attn_metadata,
                self.scale,
                self.num_kv_heads,
                out,
            )
            return output
        if (
            os.environ.get("WININTEL_SM70_TORCH_PREFILL", "0") == "1"
            and attn_metadata.fixed_num_partitions is None
            and attn_metadata.max_query_len > 1
            and not attn_metadata.dflash_window
            and self.sliding_window <= 0
            and not self.logits_soft_cap
            and attn_metadata.query_start_loc is not None
            and attn_metadata.seq_lens is not None
        ):
            from .upstream_attention import packed_cutlass_prefill

            packed_cutlass_prefill(
                query,
                kv_cache,
                block_table,
                attn_metadata,
                self.scale,
                float(getattr(layer, "_k_scale_float", 1.0)),
                float(getattr(layer, "_v_scale_float", 1.0)),
                out,
            )
            return output
        ops.paged_attention(
            query,
            kv_cache,
            block_table,
            attn_metadata.token_seq,
            attn_metadata.token_pos,
            self.scale,
            self.num_kv_heads,
            softcap=self.logits_soft_cap,
            sliding_window=attn_metadata.dflash_window or self.sliding_window,
            num_partitions=self._num_partitions(attn_metadata),
            out=out,
            k_scale=float(getattr(layer, "_k_scale_float", 1.0)),
            v_scale=float(getattr(layer, "_v_scale_float", 1.0)),
        )
        return output
