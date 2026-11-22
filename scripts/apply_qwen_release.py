# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Apply the Qwen release to a separate, verified vLLM source directory."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[1]
    upstream, output = args.upstream.resolve(), args.output.resolve()
    if (
        output.exists()
        or upstream == output
        or upstream in output.parents
        or output == release
        or output in release.parents
    ):
        parser.error("Use a new output directory that does not overwrite input sources")
    manifest = json.loads((release / "release-manifest.json").read_text("utf-8"))
    for name, expected in manifest["baseline_files"].items():
        path = upstream / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Expected pristine official vLLM 0.17.0: {name}")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((release / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release hash mismatch: {name}")
    shutil.copytree(upstream, output)
    for name in manifest["runtime_files"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release / name, destination)
    shutil.copy2(release / "NOTICE", output / "NOTICE-WININTEL")
    print(f"Qwen runtime sources prepared: {output}")


if __name__ == "__main__":
    main()
