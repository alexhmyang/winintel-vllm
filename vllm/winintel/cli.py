# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""``winintel`` command line entry point.

It wraps the upstream vLLM CLI and, on Volta hardware, applies the settings
that the SM70 kernels need: float16 weights, the WinIntel attention backend and
execution without Triton/inductor compilation.
"""

from __future__ import annotations

import os
import sys

from vllm.winintel.version import BUILD_FLAVOR, UPSTREAM_VERSION, __version__

_SM70_SUBCOMMANDS = {"serve", "run-batch"}
_BANNER = (
    f"winintel-vllm {__version__} ({BUILD_FLAVOR} edition, "
    f"upstream vLLM {UPSTREAM_VERSION})"
)


def _is_sm70() -> bool:
    override = os.environ.get("WININTEL_SM70_DEFAULTS")
    if override is not None:
        return override == "1"
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(0) == (7, 0)
    except Exception:
        return False


def _has_flag(argv: list[str], flag: str) -> bool:
    return flag in argv or any(arg.startswith(flag + "=") for arg in argv)


def _inject_sm70_defaults(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in _SM70_SUBCOMMANDS or not _is_sm70():
        return argv

    additions: list[str] = []
    if not _has_flag(argv, "--attention-backend"):
        additions += ["--attention-backend", "WININTEL_SM70"]
    if not _has_flag(argv, "--dtype"):
        additions += ["--dtype", "float16"]
    if not _has_flag(argv, "--enforce-eager") and not _has_flag(
        argv, "--no-enforce-eager"
    ):
        additions.append(
            "--no-enforce-eager"
            if os.environ.get("WININTEL_SM70_CUDAGRAPH", "0") == "1"
            else "--enforce-eager"
        )
    if (
        os.environ.get("WININTEL_SM70_CUDAGRAPH", "0") == "1"
        and not _has_flag(argv, "--enforce-eager")
        and not _has_flag(argv, "--compilation-config")
    ):
        additions += [
            "--compilation-config",
            (
                '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",'
                '"cudagraph_capture_sizes":[1,3]}'
            ),
        ]

    return [*argv, *additions]


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in ("-h", "--help"):
        print(_BANNER, file=sys.stderr)

    arguments = _inject_sm70_defaults(arguments)
    sys.argv = ["winintel", *arguments]

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
