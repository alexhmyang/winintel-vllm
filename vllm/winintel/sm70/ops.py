# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Compatibility exports for the standalone kernel dispatcher."""

from winintel_kernels.ops import (
    fused_moe,
    is_extension_available,
    paged_attention,
    reshape_and_cache,
)

__all__ = [
    "fused_moe",
    "is_extension_available",
    "paged_attention",
    "reshape_and_cache",
]
