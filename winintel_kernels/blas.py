# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Select existing cuBLASLt kernels for measured SM70 decode projections."""

import os

import torch


def prepare_lm_head(layer):
    """Keep measured vocabulary projections by column with identical values."""
    weight = layer.weight
    if (
        os.environ.get("WININTEL_SM70_COLUMN_LM_HEAD") == "1"
        and weight.is_cuda
        and weight.dtype == torch.float16
        and weight.is_contiguous()
        and torch.cuda.get_device_capability(weight.device) == (7, 0)
        and tuple(weight.shape) in ((31040, 5120), (62080, 5120))
    ):
        layer.winintel_sm70_lm_head_row_weight = weight.detach()
        weight.data = weight.t().contiguous().t()


def apply_lm_head(layer, x, bias=None):
    weight = layer.weight
    if (
        os.environ.get("WININTEL_SM70_COLUMN_LM_HEAD") == "1"
        and weight.is_cuda
        and weight.dtype == torch.float16
        and not weight.is_contiguous()
        and tuple(weight.shape) in ((31040, 5120), (62080, 5120))
        and torch.cuda.get_device_capability(weight.device) == (7, 0)
    ):
        if x.ndim == 2 and x.shape[0] == 1:
            return torch.nn.functional.linear(
                x, layer.winintel_sm70_lm_head_row_weight, bias
            )
        previous = torch.backends.cuda.preferred_blas_library()
        torch.backends.cuda.preferred_blas_library("cublaslt")
        try:
            return torch.nn.functional.linear(x, weight, bias)
        finally:
            torch.backends.cuda.preferred_blas_library(previous)
    return None


def prepare_weight(weight):
    """Store measured TP4 projections by column without changing their values."""
    if os.environ.get("WININTEL_SM70_COLUMN_WEIGHTS") == "1" and tuple(
        weight.shape
    ) in ((8704, 5120), (5120, 4352)):
        return weight.t().contiguous().t()
    return weight


def linear(x, weight, bias=None):
    if (
        x.ndim != 2
        or not 1 < x.shape[0] <= 24
        or weight.dtype != torch.float16
        or not weight.is_cuda
        or torch.cuda.get_device_capability(weight.device) != (7, 0)
    ):
        return torch.nn.functional.linear(x, weight, bias)
    narrow = (
        os.environ.get("WININTEL_SM70_NARROW_BLASLT") == "1"
        and weight.shape[0] <= 16
        and x.shape[0] <= 12
    )
    gate = (
        os.environ.get("WININTEL_SM70_GATE_BLASLT") == "1"
        and tuple(weight.shape) == (4352, 5120)
        and x.shape[0] <= 8
    )
    column = (
        os.environ.get("WININTEL_SM70_COLUMN_WEIGHTS") == "1"
        and not weight.is_contiguous()
        and tuple(weight.shape) in ((8704, 5120), (5120, 4352))
    )
    if not (narrow or gate or column):
        return torch.nn.functional.linear(x, weight, bias)
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        return torch.nn.functional.linear(x, weight, bias)
    finally:
        torch.backends.cuda.preferred_blas_library(previous)
