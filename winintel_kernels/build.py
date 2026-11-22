# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""JIT build helper for the WinIntel SM70 CUDA extension.

Building only this extension is cheap compared to a full vLLM build, which
makes it the default developer path on Windows and on machines that already
have a working vLLM installation.

Usage:
    .venv/bin/python -m winintel_kernels.build
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_MODULE_NAME = "winintel_sm70_cuda"


def source_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "csrc" / "winintel" / "sm70"


def already_loaded():
    """Return the compiled extension if it is already importable."""
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        try:
            module = importlib.import_module(_MODULE_NAME)
        except ImportError:
            return None
    # ``torch_extensions`` keeps the build directory next to the binary.  If
    # that directory happens to be on ``sys.path``, a plain import can resolve
    # to an empty namespace package instead of the compiled module.
    path = getattr(module, "__file__", None)
    if not path:
        return None
    if Path(path).suffix.lower() not in (".pyd", ".so", ".dylib"):
        return None
    return module


def load_extension(verbose: bool = False):
    """Build (or load from the torch extension cache) the SM70 extension."""
    module = already_loaded()
    if module is not None:
        return module

    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("the WinIntel SM70 extension requires CUDA")

    directory = source_dir()
    sources = [
        str(directory / "winintel_sm70_paged_attention.cu"),
        str(directory / "winintel_sm70_moe.cu"),
        str(directory / "winintel_sm70_qk_norm.cu"),
        str(directory / "winintel_sm70_torch_bindings.cpp"),
    ]
    for source in sources:
        if not Path(source).is_file():
            raise FileNotFoundError(f"missing kernel source: {source}")

    # Default to a pure Volta build.  sm_70 SASS is used on V100 and the
    # compute_70 PTX is JIT-compiled on newer cards, which keeps the same
    # artifact usable for local development on Ampere/Ada/Blackwell.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0+PTX")
    os.environ.setdefault("MAX_JOBS", str(max(1, (os.cpu_count() or 4) // 2)))

    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    standard = "c++20" if torch_version >= (2, 11) else "c++17"
    cuda_flags = [
        "-O3",
        f"-std={standard}",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_70,code=compute_70",
    ]
    cxx_flags: list[str] = []
    if sys.platform == "win32":
        cxx_flags.append(f"/std:{standard}")
        # Older CUDA toolkits do not know about the newest MSVC releases.
        cuda_flags.append("-allow-unsupported-compiler")
    else:
        cxx_flags.append(f"-std={standard}")

    return load(
        name=_MODULE_NAME,
        sources=sources,
        extra_include_paths=[str(directory)],
        extra_cflags=cxx_flags,
        extra_cuda_cflags=cuda_flags,
        verbose=verbose,
    )


def main() -> int:
    extension = load_extension(verbose=True)
    print(f"built {_MODULE_NAME}: {extension.__file__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
