# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Compatibility entry point for the standalone kernel builder."""

from winintel_kernels.build import (
    already_loaded,
    load_extension,
    main,
    source_dir,
)

__all__ = ["already_loaded", "load_extension", "main", "source_dir"]

if __name__ == "__main__":
    raise SystemExit(main())
