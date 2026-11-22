# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Plugin hooks for the WinIntel distribution.

The SM70 backend is part of the tree, so the plugin mainly exists to make the
registration explicit and idempotent for processes that discover backends
through entry points instead of the enum table.
"""

from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)


def register_winintel() -> None:
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError:  # pragma: no cover - v0-only checkouts
        return

    backend = getattr(AttentionBackendEnum, "WININTEL_SM70", None)
    if backend is None:
        logger.debug("WININTEL_SM70 backend is not present in this build")
        return

    register_backend(backend, "vllm.winintel.sm70.backend.WinIntelSM70Backend")
    logger.debug("Registered the WinIntel SM70 attention backend")
