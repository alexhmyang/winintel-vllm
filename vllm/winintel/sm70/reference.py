# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Compatibility exports for the standalone reference implementation."""

from winintel_kernels.reference import paged_attention, reshape_and_cache

__all__ = ["paged_attention", "reshape_and_cache"]
