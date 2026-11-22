# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""WinIntel SM70 attention kernels and backend."""

from vllm.winintel.sm70.ops import (
    is_extension_available,
    paged_attention,
    reshape_and_cache,
)

__all__ = [
    "is_extension_available",
    "paged_attention",
    "reshape_and_cache",
]
