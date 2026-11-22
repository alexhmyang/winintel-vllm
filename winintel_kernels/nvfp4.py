# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Load NVFP4 weights through compressed-tensors' Apache-2.0 decoder."""

import os
from functools import lru_cache

import torch

from .build import source_dir


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SM70 NVFP4 requires V100")
    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name="winintel_sm70_nvfp4_cuda",
        sources=[str(source_dir() / "winintel_sm70_nvfp4.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
    )


def prepare_scales(scales, global_scale, widths):
    """Fold checkpoint divisors in FP32 without changing decoded weights."""
    if sum(widths) != scales.shape[0] or global_scale.numel() != len(widths):
        raise ValueError("NVFP4 fused widths and global scales do not match")
    divisors = global_scale.detach().float().cpu().reshape(-1)
    if not torch.isfinite(divisors).all() or (divisors <= 0).any():
        raise ValueError("NVFP4 global scale divisors must be finite and positive")
    values = scales.detach().float().cpu()
    offset = 0
    for width, divisor in zip(widths, divisors):
        values[offset : offset + width].div_(divisor)
        offset += width
    if not torch.isfinite(values).all():
        raise ValueError("NVFP4 group scales must be finite")
    return values.to(device=scales.device).contiguous()


def prepare_lookup(scales, global_scale, widths):
    """Precompute exact FP16 weights for each checkpoint scale and FP4 code."""
    if sum(widths) != scales.shape[0] or global_scale.numel() != len(widths):
        raise ValueError("NVFP4 fused widths and global scales do not match")
    if scales.dtype != torch.float8_e4m3fn or len(widths) > 128:
        raise ValueError("NVFP4 lookup requires E4M3 group scales and <=128 widths")
    divisors = global_scale.detach().float().cpu().reshape(-1)
    if not torch.isfinite(divisors).all() or (divisors <= 0).any():
        raise ValueError("NVFP4 global scale divisors must be finite and positive")
    codes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    tables = ((values[None, :, None] / divisors[:, None, None]) * codes).half()
    ids = scales.detach().view(torch.uint8).cpu().to(torch.int32)
    offset = 0
    for index, width in enumerate(widths):
        ids[offset : offset + width].add_(index * 256)
        offset += width
    used = tables.reshape(-1, 16)[ids.unique().long()]
    if not torch.isfinite(used).all():
        raise ValueError("NVFP4 decoded weights contain NaN or infinity")
    return (
        ids.to(device=scales.device, dtype=torch.int16).contiguous(),
        tables.to(device=scales.device).contiguous(),
    )


def lookup_linear(x, packed, scale_ids, table, bias=None):
    """Benchmark packed WMMA with precomputed checkpoint weight values."""
    shape = x.shape
    inputs = x.reshape(-1, shape[-1]).contiguous()
    extension = load_extension()
    if inputs.shape[0] <= 16:
        result = extension.gemm_lookup(inputs, packed, scale_ids, table)
    else:
        result = torch.nn.functional.linear(
            inputs, extension.dequant_lookup(packed, scale_ids, table)
        )
    if bias is not None:
        result.add_(bias)
    return result.reshape(*shape[:-1], packed.shape[0])


def packed_linear(x, packed, scales, bias=None):
    """Read packed weights during decode; unpack only the current prefill layer."""
    shape = x.shape
    inputs = x.reshape(-1, shape[-1]).contiguous()
    extension = load_extension()
    if os.environ.get("WININTEL_SM70_NVFP4_WMMA") == "1" and inputs.shape[0] <= 16:
        result = extension.gemm(inputs, packed, scales)
    elif inputs.shape[0] <= 8:
        result = extension.gemv(inputs, packed, scales)
    else:
        result = torch.nn.functional.linear(inputs, extension.dequant(packed, scales))
    if bias is not None:
        result.add_(bias)
    return result.reshape(*shape[:-1], packed.shape[0])


def dequantize(packed, scales, global_scale, widths):
    """Restore fused projections with each projection's own scale divisor."""
    from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
        unpack_fp4_from_uint8,
    )

    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    if columns % 16 or tuple(scales.shape) != (rows, columns // 16):
        raise ValueError("NVFP4 requires one group scale per 16 columns")
    if sum(widths) != rows or global_scale.numel() != len(widths):
        raise ValueError("NVFP4 fused widths and global scales do not match")
    divisors = global_scale.detach().float().cpu().reshape(-1)
    if not torch.isfinite(divisors).all() or (divisors <= 0).any():
        raise ValueError("NVFP4 global scale divisors must be finite and positive")
    # Decode on CPU: upstream's compiled GPU decoder requires newer CUDA GPUs.
    unpack = getattr(
        unpack_fp4_from_uint8,
        "_torchdynamo_orig_callable",
        unpack_fp4_from_uint8,
    )
    result = torch.empty((rows, columns), dtype=torch.float16, device=packed.device)
    offset = 0
    for width, divisor in zip(widths, divisors):
        for start in range(offset, offset + width, 128):
            end = min(start + 128, offset + width)
            values = unpack(
                packed[start:end].cpu(), end - start, columns, torch.float32
            )
            group_scales = scales[start:end].float().cpu() / divisor
            values = (
                (
                    values.reshape(end - start, columns // 16, 16)
                    * group_scales.unsqueeze(-1)
                )
                .reshape(end - start, columns)
                .half()
            )
            if not torch.isfinite(values).all():
                raise ValueError("NVFP4 decoded weights contain NaN or infinity")
            result[start:end].copy_(values)
        offset += width
    return result
