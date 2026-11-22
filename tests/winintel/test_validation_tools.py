# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""CPU checks for accepting correct model code without allowing arbitrary calls."""

import runpy
from pathlib import Path

import pytest

check_function = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/check_sm70_answers.py")
)["check_function"]


@pytest.mark.parametrize("strategy", ["tensor", "channel", "modelopt"])
def test_fp8_fused_linear_preserves_each_projection_scale(strategy):
    """FP8 expansion must retain different Q/K/V scales in FP16 outputs."""
    import torch
    from compressed_tensors.quantization import QuantizationArgs

    from vllm.winintel.sm70.fp8 import WinIntelSM70FP8

    layer = torch.nn.Module()
    layer.logical_widths = [2, 1]
    layer.orig_dtype = torch.float16
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]]).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    scales = [0.25, 0.25, 2.0] if strategy == "channel" else [0.25, 2.0]
    layer.weight_scale = torch.nn.Parameter(torch.tensor(scales), requires_grad=False)
    if strategy == "modelopt":
        from vllm.winintel.sm70.modelopt import SM70ModelOptFP8

        layer.input_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        scheme = SM70ModelOptFP8(None)
        scheme.process_weights_after_loading(layer)
        actual = scheme.apply(layer, torch.tensor([[2.0, 1.0]]).half())
    else:
        scheme = WinIntelSM70FP8(
            QuantizationArgs(num_bits=8, type="float", strategy=strategy),
            is_static_input_scheme=False,
        )
        scheme.process_weights_after_loading(layer)
        actual = scheme.apply_weights(layer, torch.tensor([[2.0, 1.0]]).half())
    torch.testing.assert_close(
        actual, torch.tensor([[0.0, 1.625, 4.0]]).half(), rtol=0, atol=0
    )


def test_stream_heartbeats_do_not_reset_first_token_deadline(monkeypatch):
    """The 180-second budget includes all stream events before the first token."""
    from types import SimpleNamespace

    import scripts.benchmark_sm70_http as benchmark

    clock = [0.0]
    timeouts = []

    class Response:
        fp = SimpleNamespace(
            raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=timeouts.append))
        )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def readline(self):
            clock[0] += 100
            return b": heartbeat\n"

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(TimeoutError, match="First token"):
        benchmark.request("http://localhost/v1", "model", [1], 16)
    assert timeouts == [180, 80]


def test_modelopt_nvfp4_global_scales_multiply_each_fused_projection(monkeypatch):
    """ModelOpt scales multiply weights; treating them as divisors corrupts output."""
    import torch

    from vllm.winintel.sm70.modelopt import SM70ModelOptNVFP4

    monkeypatch.delenv("WININTEL_SM70_NVFP4_GPU_DEQUANT", raising=False)
    layer = torch.nn.Module()
    layer.logical_widths = [2, 1]
    layer.weight = torch.nn.Parameter(
        torch.full((3, 8), 0x22, dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((3, 1), 4.0).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_2 = torch.nn.Parameter(
        torch.tensor([0.5, 0.25]), requires_grad=False
    )
    layer.input_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    scheme = SM70ModelOptNVFP4(None)
    scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(
        scheme.apply(layer, torch.ones(1, 16).half()),
        torch.tensor([[32.0, 32.0, 16.0]]).half(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("prefix", ["language_model.lm_head", "lm_head"])
def test_modelopt_quantized_lm_head_loads_vocab_shard_and_scalar_scales(
    monkeypatch, prefix
):
    """The official NVFP4 output head must use its scales and local vocabulary."""
    from types import SimpleNamespace

    import torch
    import vllm.model_executor.layers.vocab_parallel_embedding as vocab
    import vllm.model_executor.parameter as parameter
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.winintel.sm70.modelopt import (
        SM70ModelOptNVFP4,
        WinIntelSM70ModelOptMixed,
    )

    monkeypatch.delenv("WININTEL_SM70_NVFP4_GPU_DEQUANT", raising=False)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 2)
    config = SimpleNamespace(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
    )
    quant = WinIntelSM70ModelOptMixed(
        None, [], {"lm_head": {"quant_algo": "NVFP4"}}, None, config
    )
    quant.apply_vllm_mapper(
        WeightsMapper(orig_to_new_prefix={"lm_head.": "language_model.lm_head."})
    )
    with set_current_vllm_config(VllmConfig()):
        layer = vocab.ParallelLMHead(
            64,
            16,
            params_dtype=torch.float16,
            quant_config=quant,
            prefix=prefix,
        )
    assert isinstance(layer.quant_method, SM70ModelOptNVFP4)
    for name, value in (
        ("weight", torch.full((64, 8), 0x22, dtype=torch.uint8)),
        ("weight_scale", torch.full((64, 1), 4.0).to(torch.float8_e4m3fn)),
        ("weight_scale_2", torch.tensor(0.5)),
        ("input_scale", torch.tensor(1.0)),
    ):
        param = getattr(layer, name)
        param.weight_loader(param, value)
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(
        layer.quant_method.apply(layer, torch.ones(1, 16).half()),
        torch.full((1, 32), 32.0).half(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "body",
    [
        "return max(a, b)",
        "if a > b:\n        return a\n    return b",
    ],
)
def test_generated_comparison_accepts_correct_answers(body):
    result = check_function(f"```python\ndef larger(a, b):\n    {body}\n```")
    assert result["passed"], result
    assert all(result["outcomes"])


@pytest.mark.parametrize(
    "body",
    [
        "return eval(a)",
        "return max(a, b, key=a)",
        "return max(a.real, b)",
        "return a",
    ],
)
def test_generated_comparison_rejects_other_calls_and_wrong_answers(body):
    result = check_function(f"```python\ndef larger(a, b):\n    {body}\n```")
    assert not result["passed"]


def test_nvfp4_fused_projections_preserve_independent_scale_divisors():
    """Different fused global scales must not be replaced with their maximum."""
    import torch

    from winintel_kernels.nvfp4 import dequantize

    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 2,
        dtype=torch.uint8,
    )
    scales = torch.tensor([[4.0], [8.0]]).to(torch.float8_e4m3fn)
    divisors = torch.tensor([2.0, 8.0])
    decoded = dequantize(packed, scales, divisors, [1, 1])
    expected = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float16,
    )
    torch.testing.assert_close(decoded[0], expected * 2, rtol=0, atol=0)
    torch.testing.assert_close(decoded[1], expected, rtol=0, atol=0)


def test_nvfp4_qkv_tuple_loads_all_three_scale_slots(monkeypatch):
    """Qwen's combined QKV checkpoint scale must reach Q, K and V under TP8."""
    import torch
    import vllm.model_executor.parameter as parameter

    import vllm.model_executor.layers.linear as linear
    from vllm.winintel.sm70.nvfp4 import WinIntelSM70NVFP4

    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 8)
    layer = linear.MergedColumnParallelLinear(
        32, [16] * 4, bias=False, params_dtype=torch.float16
    )
    del layer.weight
    scheme = WinIntelSM70NVFP4()
    scheme.create_weights(
        layer,
        output_partition_sizes=[2] * 4,
        input_size_per_partition=32,
        params_dtype=torch.float16,
        weight_loader=layer.weight_loader_v2,
    )
    layer.weight_global_scale.data.zero_()
    for suffix, value in (
        ("weight_packed", torch.full((48, 16), 0x22, dtype=torch.uint8)),
        ("weight_scale", torch.full((48, 2), 4.0).to(torch.float8_e4m3fn)),
        ("weight_global_scale", torch.tensor([2.0])),
    ):
        param = getattr(layer, suffix)
        param.weight_loader(param, value, (0, 1, 2))
    for suffix, value in (
        ("weight_packed", torch.full((16, 16), 0x22, dtype=torch.uint8)),
        ("weight_scale", torch.full((16, 2), 8.0).to(torch.float8_e4m3fn)),
        ("weight_global_scale", torch.tensor([8.0])),
    ):
        param = getattr(layer, suffix)
        param.weight_loader(param, value, 3)
    scheme.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight[:6], torch.full((6, 32), 2.0).half())
    torch.testing.assert_close(layer.weight[6:], torch.ones((2, 32)).half())
