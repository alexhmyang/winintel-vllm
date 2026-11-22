# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Opt-in group-128 W4A16 experiments; requires model-quality validation."""

import os
from functools import lru_cache

import torch

from .build import source_dir


def enabled() -> bool:
    return os.environ.get("WININTEL_SM70_EXPERIMENTAL_W4", "0") == "1"


def parallel_enabled() -> bool:
    return os.environ.get("WININTEL_SM70_EXPERIMENTAL_W4_TP", "0") == "1"


def down_combine_enabled() -> bool:
    return os.environ.get("WININTEL_SM70_W4_DOWN_COMBINE", "0") == "1"


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("SM70 W4 requires CUDA")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Experimental SM70 W4 is restricted to V100")
    os.environ.setdefault("MAX_JOBS", "4")
    return load(
        name="winintel_sm70_w4_cuda",
        sources=[str(source_dir() / "winintel_sm70_w4.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
    )


def pack(weight: torch.Tensor):
    if weight.ndim != 3 or weight.shape[-1] % 128:
        raise ValueError("W4 requires 3D weights with columns divisible by 128")
    packed = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] // 4),
        dtype=torch.uint16,
        device=weight.device,
    )
    scales = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] // 128),
        dtype=torch.float16,
        device=weight.device,
    )
    load_extension().pack(weight.contiguous(), packed, scales)
    return packed, scales


def prepare_dense(layer) -> None:
    if not enabled():
        return
    weight = layer.weight
    if (
        not weight.is_cuda
        or torch.cuda.get_device_capability(weight.device) != (7, 0)
        or weight.dtype != torch.float16
        or weight.ndim != 2
        or weight.shape[0] <= 128
        or weight.shape[1] % 128
    ):
        return
    packed, scales = pack(weight.unsqueeze(0))
    layer.register_buffer("winintel_w4_packed", packed)
    layer.register_buffer("winintel_w4_scales", scales)
    layer.register_buffer(
        "winintel_w4_ids",
        torch.zeros((1, 1), dtype=torch.int32, device=weight.device),
    )


def apply_dense(layer, x, bias):
    if not hasattr(layer, "winintel_w4_packed") or x.numel() != x.shape[-1]:
        return None
    output = torch.empty(
        (*x.shape[:-1], layer.weight.shape[0]), dtype=x.dtype, device=x.device
    )
    load_extension().linear(
        x.contiguous(),
        layer.winintel_w4_packed,
        layer.winintel_w4_scales,
        layer.winintel_w4_ids,
        output,
        False,
        True,
    )
    if bias is not None:
        output.add_(bias)
    return output
