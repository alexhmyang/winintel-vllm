# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Independent SM70 convolution, gated delta recurrence, and gated RMSNorm."""

import os
from functools import lru_cache

import torch

from .build import source_dir


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SM70 GDN requires V100")
    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name="winintel_sm70_gdn_cuda",
        sources=[str(source_dir() / "winintel_sm70_gdn.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
    )


def forward_core(layer, mixed_qkv, b, a, output):
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    if context.attn_metadata is None:
        return
    metadata = context.attn_metadata[layer.prefix]
    if layer.head_k_dim != 128 or layer.head_v_dim != 128:
        raise NotImplementedError("SM70 GDN requires 128-dimensional heads")
    if layer.conv_kernel_size != 4 or layer.conv1d.bias is not None:
        raise NotImplementedError(
            "SM70 GDN requires a bias-free width-four convolution"
        )
    if layer.activation not in ("silu", "swish"):
        raise NotImplementedError("SM70 GDN requires SiLU activation")
    conv, ssm = layer.kv_cache[context.virtual_engine]
    tokens = metadata.num_actual_tokens
    extension = load_extension()

    def run(qkv, av, bv, starts, indices, initial, result, accepted=None):
        if accepted is not None:
            accepted = accepted.to(dtype=torch.int32).contiguous()
        extension.forward(
            qkv,
            layer.conv1d.weight,
            conv,
            ssm,
            av,
            bv,
            layer.A_log,
            layer.dt_bias,
            starts,
            indices,
            initial,
            result,
            layer.num_k_heads // layer.tp_size,
            os.environ.get("WININTEL_SM70_GDN_HF_ROUNDING", "0") == "1",
            accepted,
        )

    if metadata.spec_sequence_masks is None:
        run(
            mixed_qkv[:tokens],
            a[:tokens],
            b[:tokens],
            metadata.non_spec_query_start_loc,
            metadata.non_spec_state_indices_tensor,
            metadata.has_initial_state,
            output,
        )
        return
    only_spec = metadata.num_prefills == 0 and metadata.num_decodes == 0
    indices = metadata.spec_token_indx
    spec_qkv = mixed_qkv[:tokens] if only_spec else mixed_qkv.index_select(0, indices)
    spec_a = a[:tokens] if only_spec else a.index_select(0, indices)
    spec_b = b[:tokens] if only_spec else b.index_select(0, indices)
    spec_output = (
        output
        if only_spec
        else torch.empty(
            (spec_qkv.shape[0], *output.shape[1:]),
            device=output.device,
            dtype=output.dtype,
        )
    )
    run(
        spec_qkv,
        spec_a.contiguous(),
        spec_b.contiguous(),
        metadata.spec_query_start_loc[: metadata.num_spec_decodes + 1],
        metadata.spec_state_indices_tensor[: metadata.num_spec_decodes],
        None,
        spec_output,
        metadata.num_accepted_tokens[: metadata.num_spec_decodes],
    )
    if not only_spec:
        output.index_copy_(0, indices, spec_output)
        indices = metadata.non_spec_token_indx
        result = torch.empty(
            (indices.numel(), *output.shape[1:]),
            device=output.device,
            dtype=output.dtype,
        )
        run(
            mixed_qkv.index_select(0, indices),
            a.index_select(0, indices).contiguous(),
            b.index_select(0, indices).contiguous(),
            metadata.non_spec_query_start_loc,
            metadata.non_spec_state_indices_tensor,
            metadata.has_initial_state,
            result,
        )
        output.index_copy_(0, indices, result)


def gated_norm(layer, x, z):
    if (
        z is None
        or not layer.norm_before_gate
        or layer.group_size is not None
        or layer.activation not in ("silu", "swish")
        or x.shape[-1] != 128
    ):
        return layer.forward_native(x, z)
    return load_extension().norm(x, z, layer.weight, layer.eps)
