# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
"""Optional normalization directly from rank-interleaved Q/K gathers."""

import os

import torch


def enabled(tp_size: int, q_size: int, kv_size: int) -> bool:
    if (
        os.environ.get("WININTEL_SM70_FUSED_QK_NORM", "0") != "1"
        or tp_size <= 1
        or q_size != kv_size
        or q_size % 8
        or q_size * tp_size > 65536
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (7, 0)
    ):
        return False
    from .build import load_extension

    load_extension()
    return True


def normalize(
    gathered: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(
        (gathered.shape[0], qw.numel() // tp_size),
        dtype=gathered.dtype,
        device=gathered.device,
    )
    k = torch.empty_like(q)
    torch.ops.winintel_sm70.gathered_qk_norm(gathered, qw, kw, q, k, tp_rank, epsilon)
    return q, k
