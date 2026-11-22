# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Install the packaged Qwen SM70 runtime using uv and official build tools."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def run(command, cwd=None, environment=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def validate_model(directory):
    config = json.loads((directory / "config.json").read_text("utf-8"))
    architectures = config.get("architectures", [])
    if "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError("Select a Qwen3.8-27B checkpoint with qwen3_5 architecture")
    text_config = config.get("text_config", config)
    if text_config.get("hidden_size") != 5120:
        raise ValueError("This Qwen 27B profile requires hidden_size=5120")
    quantization = config.get("quantization_config", {})
    if quantization:
        method = quantization.get("quant_method")
        if method not in ("compressed-tensors", "compressed_tensors", "modelopt"):
            raise ValueError("Use a compressed-tensors or ModelOpt NVFP4 checkpoint")
        if method == "modelopt" and quantization.get("quant_algo") != "MIXED_PRECISION":
            raise ValueError("ModelOpt requires mixed FP8/NVFP4 MIXED_PRECISION")
        if method == "modelopt":
            nvfp4 = any(
                layer.get("quant_algo") == "NVFP4" and layer.get("group_size", 16) == 16
                for layer in quantization.get("quantized_layers", {}).values()
            )
        else:
            nvfp4 = any(
                group.get("weights", {}).get("num_bits") == 4
                and group.get("weights", {}).get("group_size") == 16
                for group in quantization.get("config_groups", {}).values()
            )
        if not nvfp4:
            raise ValueError("NVFP4 requires 4-bit weights with group_size=16")
    if not any(directory.glob("*.safetensors")):
        raise ValueError("The model directory must contain safetensors weight shards")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?")
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[1]
    model = args.model.resolve() if args.model else None
    if model:
        validate_model(model)
    runtime = Path(
        os.environ.get("WININTEL_INSTALL_DIR", release / "runtime")
    ).resolve()
    manifest = json.loads((release / "release-manifest.json").read_text("utf-8"))
    archive = release / "vendor/vllm-0.17.0.tar.gz"
    if (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        != manifest["baseline_archive_sha256"]
    ):
        raise ValueError("Base source archive SHA256 verification failed")
    if not runtime.exists():
        with tempfile.TemporaryDirectory(prefix="winintel-qwen-") as temporary:
            stage = Path(temporary)
            with tarfile.open(archive) as packed:
                packed.extractall(stage, filter="data")
            run(
                [
                    "uv",
                    "run",
                    "--offline",
                    "--no-project",
                    "--python",
                    sys.executable,
                    str(release / "scripts/apply_qwen_release.py"),
                    str(stage / "vllm-0.17.0"),
                    str(runtime),
                ]
            )
    for name in manifest["runtime_files"]:
        installed_file = runtime / name
        if (
            not installed_file.is_file()
            or hashlib.sha256(installed_file.read_bytes()).hexdigest()
            != manifest["files"][name]
        ):
            raise ValueError(f"Installed file does not match this release: {name}")
    marker = runtime / ".winintel-installed.json"
    if not marker.exists():
        runtime_python = runtime / ".venv/bin/python"
        environment = os.environ.copy()
        environment["CUDA_HOME"] = environment.get("CUDA_HOME", "/usr/local/cuda-12.8")
        environment["PATH"] = (
            f"{runtime / '.venv/bin'}:{environment['CUDA_HOME']}/bin:"
            + environment["PATH"]
        )
        environment["TORCH_CUDA_ARCH_LIST"] = "7.0"
        environment["PYTHONPATH"] = str(runtime)
        environment["VLLM_TARGET_DEVICE"] = "cuda"
        environment["VLLM_USE_PRECOMPILED"] = "0"
        environment.pop("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
        environment["MAX_JOBS"] = environment.get("MAX_JOBS", "4")
        environment["UV_INDEX_URL"] = environment.get(
            "UV_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        torch_index = environment.get(
            "WININTEL_TORCH_INDEX_URL", "https://mirrors.nju.edu.cn/pytorch/whl/cu128"
        )
        run(["uv", "venv", "--python", "3.12", str(runtime / ".venv")])
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(runtime_python),
                "--index",
                torch_index,
                "torch==2.10.0+cu128",
                "torchvision==0.25.0+cu128",
                "torchaudio==2.10.0+cu128",
            ],
            environment=environment,
        )
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(runtime_python),
                "--index",
                torch_index,
                "-r",
                str(runtime / "requirements/build.txt"),
                "-r",
                str(runtime / "requirements/cuda.txt"),
                "-r",
                str(runtime / "requirements/lint.txt"),
            ],
            environment=environment,
        )
        run(["git", "init", str(runtime)])
        run(
            [str(runtime / ".venv/bin/pre-commit"), "install"],
            cwd=runtime,
            environment=environment,
        )
        run(
            ["bash", str(runtime / "scripts/build_sm70_linux.sh")],
            cwd=runtime,
            environment=environment,
        )
        metadata_environment = environment | {
            "VLLM_TARGET_DEVICE": "empty",
            "VLLM_USE_PRECOMPILED": "0",
        }
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(runtime_python),
                "--no-build-isolation",
                "--no-deps",
                "-e",
                str(runtime),
            ],
            environment=metadata_environment,
        )
        run(
            [
                "uv",
                "run",
                "--offline",
                "--no-project",
                "--python",
                str(runtime_python),
                str(runtime_python),
                "-c",
                (
                    "import torch; "
                    "assert 'sm_70' in torch.cuda.get_arch_list(); "
                    "import vllm._C; import vllm._moe_C; "
                    "from winintel_kernels.build import load_extension; "
                    "load_extension(); "
                    "from winintel_kernels.gdn import load_extension as gdn; gdn(); "
                    "from winintel_kernels.nvfp4 import load_extension as nvfp4; "
                    "nvfp4(); print('winintel AI SM70 extensions are ready')"
                ),
            ],
            cwd=runtime,
            environment=environment,
        )
        marker.write_text(
            json.dumps({"version": "0.1.0", "profile": "qwen3.8"}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"winintel AI winintel-vllm installed: {runtime}", flush=True)
    if model:
        environment = os.environ | {"WININTEL_INSTALL_DIR": str(runtime)}
        os.execvpe(
            "bash",
            ["bash", str(release / "start.sh"), str(model)],
            environment,
        )


if __name__ == "__main__":
    main()
