# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Run a Linux evaluation with continuous GPU temperature and memory records."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def sample_gpus(indices: set[int]) -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-gpu=index,memory.used,temperature.gpu,power.draw,"
                "clocks.current.sm,utilization.gpu"
            ),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=3,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, memory, temperature, power, clock, utilization = line.split(",")
        if int(index) in indices:
            rows.append(
                {
                    "index": int(index),
                    "memory_mib": int(memory),
                    "temperature_c": int(temperature),
                    "power_w": float(power),
                    "sm_clock_mhz": int(clock),
                    "utilization_percent": int(utilization),
                }
            )
    if {row["index"] for row in rows} != indices:
        raise RuntimeError("A requested GPU is missing from nvidia-smi")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Physical GPU indices, e.g. 4,5")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-c", type=int, default=80)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Optional wall-clock limit for the entire test, separate from TTFT",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Provide the evaluation command after --")
    indices = {int(index) for index in args.gpus.split(",")}
    records = []
    process = None
    error = None
    started = time.monotonic()
    try:
        initial = sample_gpus(indices)
        records.append({"elapsed_seconds": 0.0, "gpus": initial})
        if any(row["temperature_c"] >= args.limit_c for row in initial):
            raise RuntimeError("GPU temperature limit reached before evaluation")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = args.gpus
        process = subprocess.Popen(command, start_new_session=True, env=environment)
        while process.poll() is None:
            if (
                args.timeout_seconds is not None
                and time.monotonic() - started >= args.timeout_seconds
            ):
                raise TimeoutError("Evaluation wall-clock limit reached")
            rows = sample_gpus(indices)
            records.append(
                {"elapsed_seconds": time.monotonic() - started, "gpus": rows}
            )
            if any(row["temperature_c"] >= args.limit_c for row in rows):
                raise RuntimeError("GPU temperature limit reached during evaluation")
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)
        if process.returncode:
            raise RuntimeError(f"Evaluation exited with code {process.returncode}")
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        args.report.write_text(
            json.dumps(
                {
                    "command": command,
                    "cuda_visible_devices": args.gpus,
                    "temperature_limit_c": args.limit_c,
                    "timeout_seconds": args.timeout_seconds,
                    "exit_code": process.returncode if process else None,
                    "error": error,
                    "samples": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
