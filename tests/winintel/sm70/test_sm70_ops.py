# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Tests for the WinIntel SM70 reference path and CUDA kernels."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from winintel_kernels import ops, reference


def test_dflash_convolution_does_not_mix_requests():
    """The first token of a block must not read the preceding request's token."""
    from vllm.winintel.sm70.dflash2_layers import grouped_dynamic_conv

    hidden = torch.arange(16 * 32, dtype=torch.float32).view(16, 32)
    delta = torch.ones(16, 2, 2)
    base = torch.ones(2, 32)
    actual = grouped_dynamic_conv(hidden, delta, base, block_size=8, group_size=16)
    expected = hidden * 2
    for position in range(16):
        if position % 8:
            expected[position] += hidden[position - 1] * 2
    torch.testing.assert_close(actual, expected)


def test_dflash_selector_uses_selected_predecessor():
    """A transition-selected token must influence the next draft position."""
    from vllm.winintel.sm70.dflash2_layers import CandidateSelector, select_draft_path

    selector = CandidateSelector(vocab_size=4, hidden_size=2, rank=2, top_k=2)
    with torch.no_grad():
        selector.hidden_projection.weight.copy_(torch.eye(2))
        selector.predecessor_codebook.zero_()
        selector.successor_codebook.zero_()
        selector.predecessor_codebook[0, 0] = 1
        selector.predecessor_codebook[2, 1] = 1
        selector.successor_codebook[2, 0] = 5
        selector.successor_codebook[3, 1] = 5
    logits = torch.tensor([[[0.0, 2.0, 1.0, -5.0], [0.0, 2.0, -5.0, 1.0]]])
    hidden = torch.ones(1, 2, 2)
    selected = select_draft_path(selector, logits, hidden, torch.tensor([0]))
    assert selected.tolist() == [[2, 3]]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="DFlash2 paged attention requires V100",
)
@pytest.mark.parametrize("upstream", [False, True])
def test_dflash_block_sees_full_noise_block_and_only_recent_context(
    monkeypatch, upstream
):
    """All eight queries see the same 2048-prefix-plus-eight-noise-token mask."""
    from vllm.winintel.sm70.backend import WinIntelSM70Impl, WinIntelSM70Metadata
    from vllm.winintel.sm70.upstream_attention import cache_views

    monkeypatch.setenv("WININTEL_SM70_UPSTREAM_DECODE", str(int(upstream)))
    monkeypatch.setenv("WININTEL_SM70_TORCH_PREFILL", "1")
    lengths = [2300, 2180]
    cache, tables, _, _, _ = _paged_inputs(2, 8, 2, 128, 16, lengths, "cuda")
    source_cache = cache
    if upstream:
        packed = torch.empty(
            2, cache.shape[0], 2, 16, 16, 8, device="cuda", dtype=torch.float16
        )
        keys, values = cache_views(packed)
        keys.copy_(
            cache[..., :128]
            .reshape(cache.shape[0], 16, 2, 16, 8)
            .permute(0, 2, 3, 1, 4)
        )
        values.copy_(cache[..., 128:].permute(0, 2, 3, 1))
        cache = packed
    query = torch.randn(16, 8, 128, device="cuda").half()
    seq = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(8)
    pos = (
        torch.tensor(lengths, device="cuda", dtype=torch.int32).repeat_interleave(8) - 1
    )
    metadata = WinIntelSM70Metadata(
        num_actual_tokens=16,
        max_query_len=8,
        max_seq_len=max(lengths),
        block_table=tables,
        token_seq=seq,
        token_pos=pos,
        dflash_window=2056,
    )
    impl = WinIntelSM70Impl(8, 128, 128**-0.5, 2)
    actual = torch.empty_like(query)
    impl.forward(None, query, None, None, cache, metadata, actual)
    for request, length in enumerate(lengths):
        prefix_start = length - 8 - 2048
        visible = torch.arange(prefix_start, length, device="cuda")
        pages = tables[request, visible // 16].long()
        kv = source_cache[pages, visible % 16].float()
        keys = kv[..., :128].repeat_interleave(4, dim=1).transpose(0, 1)
        values = kv[..., 128:].repeat_interleave(4, dim=1).transpose(0, 1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query[request * 8 : (request + 1) * 8].float().transpose(0, 1),
            keys,
            values,
            is_causal=False,
        ).transpose(0, 1)
        torch.testing.assert_close(
            actual[request * 8 : (request + 1) * 8].float(),
            expected,
            rtol=0.005,
            atol=0.005,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 CUDA is required",
)
def test_nvfp4_lookup_preserves_each_fused_projection_divisor():
    """Fused projections must decode with their own global scale divisors."""
    from winintel_kernels.nvfp4 import (
        dequantize,
        load_extension,
        lookup_linear,
        prepare_lookup,
    )

    torch.manual_seed(871)
    packed = torch.randint(0, 256, (32, 128), device="cuda", dtype=torch.uint8)
    scales = torch.randint(1, 8, (32, 16), device="cuda").to(torch.float8_e4m3fn)
    divisors = torch.tensor([0.7, 1.3], device="cuda")
    widths = [16, 16]
    ids, table = prepare_lookup(scales, divisors, widths)
    weight = dequantize(packed, scales, divisors, widths)
    assert torch.equal(load_extension().dequant_lookup(packed, ids, table), weight)
    x = torch.randn(3, 256, device="cuda").half()
    for _ in range(3):
        lookup_linear(x, packed, ids, table)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = lookup_linear(x, packed, ids, table)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(x, weight), rtol=0.02, atol=0.02
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 CUDA is required",
)
@pytest.mark.parametrize("tokens", [3, 4, 8, 12])
@pytest.mark.parametrize("rows", [12, 4352, 8704])
def test_selective_blas_projection_replays_new_inputs_and_restores_backend(
    tokens, rows, monkeypatch
):
    """Changing decode inputs must preserve linear results and the BLAS preference."""
    from winintel_kernels.blas import linear, prepare_weight

    monkeypatch.setenv("WININTEL_SM70_NARROW_BLASLT", "1")
    monkeypatch.setenv("WININTEL_SM70_GATE_BLASLT", "1")
    monkeypatch.setenv("WININTEL_SM70_COLUMN_WEIGHTS", "1")
    torch.manual_seed(417)
    weight = (torch.randn(rows, 5120, device="cuda") * 0.02).half()
    original = weight
    weight = prepare_weight(weight)
    torch.testing.assert_close(weight, original, atol=0, rtol=0)
    x = torch.randn(tokens, 5120, device="cuda").half()
    bias = torch.randn(rows, device="cuda").half()
    previous = torch.backends.cuda.preferred_blas_library()
    for _ in range(3):
        linear(x, weight, bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear(x, weight, bias)
    assert torch.backends.cuda.preferred_blas_library() == previous
    x.mul_(0.5)
    graph.replay()
    expected = torch.nn.functional.linear(x, original, bias)
    torch.testing.assert_close(captured, expected, atol=0.02, rtol=0.02)
    assert torch.backends.cuda.preferred_blas_library() == previous


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 CUDA is required",
)
@pytest.mark.parametrize("accepted_count", [1, 2, 3])
@pytest.mark.parametrize("use_adapter", [False, True])
def test_gdn_rejected_draft_states_do_not_affect_next_round(
    accepted_count, use_adapter, monkeypatch
):
    """Rollback convolution and recurrence to exactly the accepted draft prefix."""
    from winintel_kernels.gdn import load_extension

    extension = load_extension()
    torch.manual_seed(812)
    kh, vh, channels = 1, 2, 512
    x = torch.randn(6, channels, device="cuda").half()
    a = torch.randn(6, vh, device="cuda").half()
    b = torch.randn_like(a)
    weights = (torch.randn(channels, 1, 4, device="cuda") * 0.1).half()
    alog = torch.zeros(vh, device="cuda").half()
    dt = torch.zeros_like(alog)
    initial_conv = torch.randn(1, 3, channels, device="cuda").half()
    initial_ssm = torch.randn(1, vh, 128, 128, device="cuda") * 0.1
    conv = torch.zeros(3, 5, channels, device="cuda").half()
    conv[0, :3].copy_(initial_conv[0])
    ssm = initial_ssm.expand(3, -1, -1, -1).clone()
    starts = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    indices = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    accepted = torch.ones(1, device="cuda", dtype=torch.int32)
    output = torch.empty(3, vh, 128, device="cuda").half()

    def run(lo, hi, conv, ssm, starts, indices, out, accepted=None):
        if accepted is not None and use_adapter:
            from winintel_kernels.gdn import forward_core

            metadata = SimpleNamespace(
                num_actual_tokens=hi - lo,
                spec_sequence_masks=True,
                num_prefills=0,
                num_decodes=0,
                num_spec_decodes=1,
                spec_token_indx=None,
                spec_query_start_loc=starts,
                spec_state_indices_tensor=indices,
                num_accepted_tokens=accepted.to(torch.int64),
            )
            context = SimpleNamespace(attn_metadata={"gdn": metadata}, virtual_engine=0)
            monkeypatch.setattr(
                "vllm.forward_context.get_forward_context", lambda: context
            )
            monkeypatch.setenv("WININTEL_SM70_GDN_HF_ROUNDING", "1")
            layer = SimpleNamespace(
                prefix="gdn",
                head_k_dim=128,
                head_v_dim=128,
                conv_kernel_size=4,
                conv1d=SimpleNamespace(weight=weights, bias=None),
                activation="silu",
                kv_cache=[(conv, ssm)],
                A_log=alog,
                dt_bias=dt,
                num_k_heads=kh,
                tp_size=1,
            )
            forward_core(layer, x[lo:hi], b[lo:hi], a[lo:hi], out)
            return
        extension.forward(
            x[lo:hi],
            weights,
            conv,
            ssm,
            a[lo:hi],
            b[lo:hi],
            alog,
            dt,
            starts,
            indices,
            None,
            out,
            kh,
            True,
            accepted,
        )

    run(0, 3, conv, ssm, starts, indices, output, accepted)
    reference_conv = initial_conv.clone()
    reference_ssm = initial_ssm.clone()
    reference_out = torch.empty(1, vh, 128, device="cuda").half()
    one_start = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    one_index = torch.zeros(1, device="cuda", dtype=torch.int32)
    for token in range(accepted_count):
        run(
            token,
            token + 1,
            reference_conv,
            reference_ssm,
            one_start,
            one_index,
            reference_out,
        )
        torch.testing.assert_close(output[token], reference_out[0], atol=0, rtol=0)
        torch.testing.assert_close(ssm[token], reference_ssm[0], atol=0, rtol=0)
    accepted.fill_(accepted_count)
    run(3, 6, conv, ssm, starts, indices, output, accepted)
    for token in range(3):
        run(
            token + 3,
            token + 4,
            reference_conv,
            reference_ssm,
            one_start,
            one_index,
            reference_out,
        )
        torch.testing.assert_close(output[token], reference_out[0], atol=0, rtol=0)
        torch.testing.assert_close(ssm[token], reference_ssm[0], atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 CUDA is required",
)
@pytest.mark.parametrize("is_neox_style", [True, False])
def test_rotary_without_flash_attention_replays_changed_inputs(is_neox_style):
    """V100 rotary encoding must run and replay without FlashAttention binaries."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    with set_current_vllm_config(VllmConfig()):
        operation = ApplyRotaryEmb(is_neox_style=is_neox_style)
    x = torch.randn(3, 2, 64, device="cuda").half()
    angles = torch.randn(3, 32, device="cuda")
    cos, sin = angles.cos().half(), angles.sin().half()
    operation.forward_cuda(x, cos, sin)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation.forward_cuda(x, cos, sin)
    x.copy_(torch.randn_like(x))
    graph.replay()
    expected = operation.forward_static(x.cpu(), cos.cpu(), sin.cpu(), is_neox_style)
    torch.testing.assert_close(output.cpu(), expected, rtol=0.002, atol=0.002)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 CUDA is required",
)
@pytest.mark.parametrize(
    "tokens,ranks,rank",
    [(1, 2, 1), (3, 4, 0), (1, 4, 3), (1, 8, 7), (3, 8, 0)],
)
def test_gathered_qk_norm_preserves_global_rms_and_graph_replay(tokens, ranks, rank):
    """Rank-interleaved normalization must match full-width vLLM RMSNorm."""
    from vllm import _custom_ops
    from winintel_kernels.build import load_extension
    from winintel_kernels.qk_norm import normalize

    load_extension()
    torch.manual_seed(739)
    local = 2048 // ranks
    packed = torch.randn(tokens, ranks * local * 2, device="cuda").half()
    qw = torch.randn(2048, device="cuda").half()
    kw = torch.randn_like(qw)

    def oracle():
        views = packed.view(tokens, ranks, 2, local)
        result = []
        for index, weight in enumerate((qw, kw)):
            full = views[:, :, index, :].reshape(tokens, -1).contiguous()
            out = torch.empty_like(full)
            _custom_ops.rms_norm(out, full, weight, 1e-5)
            result.append(out[:, rank * local : (rank + 1) * local])
        return result

    actual = normalize(packed, qw, kw, ranks, rank, 1e-5)
    for out, expected in zip(actual, oracle()):
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = normalize(packed, qw, kw, ranks, rank, 1e-5)
    for magnitude in (0.1, 3.0):
        packed.copy_(torch.randn_like(packed) * magnitude)
        graph.replay()
        for out, expected in zip(actual, oracle()):
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _gdn_reference(
    x,
    weights,
    conv,
    ssm,
    a,
    b,
    alog,
    dt,
    starts,
    ids,
    initial,
    kh,
    hf_rounding=False,
):
    """Compute the public convolution/recurrence contract in FP32."""
    vh = ssm.shape[1]
    result = torch.zeros(x.shape[0], vh, 128, device=x.device, dtype=x.dtype)
    for seq, page in enumerate(ids.tolist()):
        if page < 0 or starts[seq] == starts[seq + 1]:
            continue
        history = conv[page].float()
        state = ssm[page].clone()
        if initial is not None and not initial[seq]:
            history.zero_()
            state.zero_()
        for t in range(int(starts[seq]), int(starts[seq + 1])):
            raw = x[t].float()
            values = torch.cat((history, raw[None]), dim=0)
            mixed = (values.T * weights.float()).sum(-1)
            if hf_rounding:
                mixed = mixed.half().float()
            mixed = torch.nn.functional.silu(mixed)
            mixed = mixed.half().float()
            history = values[1:]
            q, k, v = mixed.split([kh * 128, kh * 128, vh * 128])
            q, k = q.reshape(kh, 128), k.reshape(kh, 128)
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q = q.repeat_interleave(vh // kh, 0) / 128**0.5
            k = k.repeat_interleave(vh // kh, 0)
            decay = torch.exp(
                -alog.float().exp()
                * torch.nn.functional.softplus(a[t].float() + dt.float())
            )
            state *= decay[:, None, None]
            delta = v.reshape(vh, 128) - (state * k[:, None]).sum(-1)
            beta = b[t].float().sigmoid()
            if hf_rounding:
                beta = beta.half().float()
            delta *= beta[:, None]
            state += delta[:, :, None] * k[:, None]
            result[t] = (state * q[:, None]).sum(-1).half()
        conv[page].copy_(history.half())
        ssm[page].copy_(state)
    return result


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native GDN requires V100",
)
@pytest.mark.parametrize("hf_rounding", [False, True])
def test_gdn_reset_continuation_and_padded_graph(hf_rounding):
    """Reset pages and graph continuation agree with the recurrence equations."""
    from winintel_kernels import gdn

    torch.manual_seed(92)
    kh, vh, channels = 2, 6, 1280
    x = torch.randn(5, channels + 128, device="cuda", dtype=torch.float16)[:, :channels]
    weights = torch.randn(channels, 4, device="cuda", dtype=torch.float16) * 0.2
    conv = torch.randn(3, 3 * channels + 32, device="cuda", dtype=torch.float16)
    conv = conv[:, : 3 * channels].view(3, 3, channels)
    ssm = torch.randn(3, vh * 128 * 128 + 64, device="cuda")
    ssm = ssm[:, : vh * 128 * 128].view(3, vh, 128, 128)
    a, b = [torch.randn(5, vh, device="cuda", dtype=torch.float16) for _ in range(2)]
    alog, dt = [torch.randn(vh, device="cuda", dtype=torch.float16) for _ in range(2)]
    starts = torch.tensor([0, 3, 5, 5], device="cuda", dtype=torch.int32)
    ids = torch.tensor([2, 99, 0, 99, -1, 99], device="cuda", dtype=torch.int32)[::2]
    initial = torch.tensor([False, True, False], device="cuda")
    out = torch.zeros(5, vh, 128, device="cuda", dtype=torch.float16)
    extension = gdn.load_extension()
    reference_conv, reference_ssm = conv.clone(), ssm.clone()
    expected = _gdn_reference(
        x,
        weights,
        reference_conv,
        reference_ssm,
        a,
        b,
        alog,
        dt,
        starts,
        ids,
        initial,
        kh,
        hf_rounding,
    )
    extension.forward(
        x,
        weights,
        conv,
        ssm,
        a,
        b,
        alog,
        dt,
        starts,
        ids,
        initial,
        out,
        kh,
        hf_rounding,
    )
    torch.testing.assert_close(out, expected, atol=0.0002, rtol=0.005)
    torch.testing.assert_close(conv, reference_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm, reference_ssm, atol=0.00002, rtol=0.005)

    x, a, b, out = x[:2], a[:2], b[:2], out[:2]
    starts.copy_(torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int32))
    initial.fill_(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        extension.forward(
            x,
            weights,
            conv,
            ssm,
            a,
            b,
            alog,
            dt,
            starts,
            ids,
            initial,
            out,
            kh,
            hf_rounding,
        )
    for pages in ([0, 2, -1], [2, 0, -1]):
        ids.copy_(torch.tensor(pages, device="cuda", dtype=torch.int32))
        reference_conv.copy_(conv)
        reference_ssm.copy_(ssm)
        expected = _gdn_reference(
            x,
            weights,
            reference_conv,
            reference_ssm,
            a,
            b,
            alog,
            dt,
            starts,
            ids,
            initial,
            kh,
            hf_rounding,
        )
        graph.replay()
        torch.testing.assert_close(out, expected, atol=0.0002, rtol=0.005)
        torch.testing.assert_close(ssm, reference_ssm, atol=0.00002, rtol=0.005)
        torch.testing.assert_close(conv, reference_conv, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Native GDN requires V100",
)
def test_gdn_gated_norm_matches_fp32_equations():
    from winintel_kernels import gdn

    torch.manual_seed(93)
    x = torch.randn(9, 128, device="cuda", dtype=torch.float16)
    z = torch.randn(9, 256, device="cuda", dtype=torch.float16)[:, :128]
    weight = torch.randn(128, device="cuda", dtype=torch.float16)
    expected = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
        * torch.nn.functional.silu(z.float())
    ).half()
    actual = gdn.load_extension().norm(x, z, weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.0001)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Experimental W4 requires V100",
)
def test_w4_packing_and_changed_routes_in_graph():
    """Quantization matches its contract and graph replay reads changed routes."""
    from winintel_kernels import w4

    torch.manual_seed(72)
    weight = torch.randn(3, 128, 128, device="cuda", dtype=torch.float16) * 0.1
    packed, scales = w4.pack(weight)
    groups = weight.reshape(3, 128, 1, 128).float()
    expected_scale = (groups.abs().amax(-1) / 7).half()
    torch.testing.assert_close(scales, expected_scale, rtol=0, atol=0)
    expected_codes = torch.round(groups / expected_scale.float().unsqueeze(-1)).clamp(
        -7, 7
    )
    bits = packed.to(torch.int32)
    codes = torch.stack([(bits >> (4 * k)) & 15 for k in range(4)], dim=-1)
    dequantized = (
        (codes.reshape_as(weight).float() - 8)
        * scales.repeat_interleave(128, dim=-1).float()
    ).half()
    expected = (expected_codes * expected_scale.float().unsqueeze(-1)).half()
    torch.testing.assert_close(dequantized, expected.reshape_as(weight), rtol=0, atol=0)

    x = torch.randn(1, 128, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor([[2, 0, 2, -1]], dtype=torch.int32, device="cuda")
    weights = torch.tensor([[0.2, 0.3, 0.4, 0.1]], device="cuda")
    projected = torch.empty((1, 4, 128), dtype=torch.float16, device="cuda")
    output = torch.empty_like(x)
    extension = w4.load_extension()

    def run():
        extension.linear(x.repeat(4, 1), packed, scales, ids, projected, False, True)
        extension.combine(projected, weights, ids, output, 3)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for routing in ([2, 0, 2, -1], [1, 2, -1, 1]):
        ids.copy_(torch.tensor([routing], device="cuda", dtype=torch.int32))
        graph.replay()
        actual = output.clone()
        torch.cuda.synchronize()
        expected_output = torch.zeros_like(output)
        for k in sorted(range(4), key=lambda k: routing[k]):
            if routing[k] < 0:
                continue
            value = (dequantized[routing[k]].float() @ x[0].float()).half()
            expected_output += (value * weights[0, k].half()).half()
        torch.testing.assert_close(actual, expected_output, rtol=0.003, atol=0.0002)
        run()
        torch.testing.assert_close(output, actual, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="route count"):
        invalid_ids = torch.zeros((1, 33), device="cuda", dtype=torch.int32)
        extension.linear(x, packed, scales, invalid_ids, output, False, True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Experimental W4 requires V100",
)
@pytest.mark.parametrize(
    "tokens,columns,rows", [(1, 256, 33), (3, 128, 35), (0, 128, 33)]
)
def test_w4_fused_down_preserves_rounding_and_changed_graph(tokens, columns, rows):
    """Fusion preserves stable route order and each FP16 rounding boundary."""
    from winintel_kernels import w4

    torch.manual_seed(747)
    extension = w4.load_extension()
    packed, scales = w4.pack(
        torch.randn(8, rows, columns, device="cuda", dtype=torch.float16) * 0.1
    )
    storage = torch.empty(
        tokens * (8 * columns + rows), device="cuda", dtype=torch.float16
    )
    x = storage[: tokens * 8 * columns].view(tokens * 8, columns).normal_()
    output = storage[tokens * 8 * columns :].view(tokens, rows)
    ids = torch.empty(tokens, 8, device="cuda", dtype=torch.int32)
    weights = torch.empty(tokens, 8, device="cuda")
    projected = torch.empty(tokens * 8, rows, device="cuda", dtype=torch.float16)
    expected = torch.empty_like(output)

    def baseline():
        extension.linear(x, packed, scales, ids, projected, False, True)
        extension.combine(projected, weights, ids, expected, 8)

    def fused():
        extension.down_combine(x, packed, scales, ids, weights, output)

    ids.zero_()
    weights.fill_(1)
    fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    for routing in (
        [7, 2, 7, -1, 8, 0, 6, 2],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [-1] * 8,
    ):
        x.normal_()
        weights.uniform_(-1, 1)
        ids.copy_(
            torch.tensor(routing, device="cuda", dtype=torch.int32).expand_as(ids)
        )
        baseline()
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        fused()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
    if tokens:
        with pytest.raises(RuntimeError, match="nonoverlapping"):
            extension.down_combine(
                x,
                packed,
                scales,
                ids,
                weights,
                x.flatten()[: tokens * rows].view_as(output),
            )
        with pytest.raises(RuntimeError, match="shapes"):
            extension.down_combine(
                x, packed, scales, ids, weights, output[:, :-1].contiguous()
            )


def _moe_reference(x, w13, w2, weights, ids, expert_map, weight_on_input, hf_rounding):
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        routes = range(ids.shape[1])
        if hf_rounding:
            routes = sorted(routes, key=lambda k: int(ids[row, k]))
        for route in routes:
            expert = int(ids[row, route])
            if expert < 0 or expert >= (
                len(expert_map) if expert_map is not None else len(w13)
            ):
                continue
            if expert_map is not None:
                expert = int(expert_map[expert])
            if expert < 0 or expert >= len(w13):
                continue
            value = x[row].float()
            if weight_on_input:
                value = (value * weights[row, route]).half().float()
            gate, up = (w13[expert].float() @ value).half().float().chunk(2)
            activated = torch.nn.functional.silu(gate)
            if hf_rounding:
                activated = activated.half().float()
            gated = (activated * up).half().float()
            down = (w2[expert].float() @ gated).half().float()
            weight = weights.new_tensor(1) if weight_on_input else weights[row, route]
            if hf_rounding:
                contribution = (down * weight.half().float()).half().float()
                result[row] = (result[row] + contribution).half().float()
            else:
                result[row] += down * weight
    return result.half()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 64, 32, 1), (3, 67, 35, 2), (5, 128, 96, 3)])
@pytest.mark.parametrize("weight_on_input", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("hf_rounding", [False, True])
def test_native_moe_routing_weighting_and_graph_replay(
    shape, weight_on_input, mapped, hf_rounding
):
    """Protect mapping, rounding and changing routes during graph replay."""
    assert ops.is_extension_available(), "Native kernel build is required"
    torch.manual_seed(42)
    m, h, i, k = shape
    x = torch.randn(m, h, device="cuda", dtype=torch.float16) * 0.2
    w13 = torch.randn(3, 2 * i, h, device="cuda", dtype=torch.float16) * 0.1
    w2 = torch.randn(3, h, i, device="cuda", dtype=torch.float16) * 0.1
    weights = torch.rand(m, k, device="cuda")
    ids = torch.arange(m * k, device="cuda", dtype=torch.int32).view(m, k) % 5
    expert_map = torch.tensor([2, -1, 0, 1, -1], device="cuda", dtype=torch.int32)
    if not mapped:
        expert_map = None
    # The modular engine reuses the gate workspace for its final output.
    shared = x.new_empty(max(m * h, m * k * i))
    output = shared[: m * h].view(m, h)
    gated = shared[: m * k * i].view(m, k, i)
    projected = x.new_empty((m, k, h))

    def run():
        return ops.fused_moe(
            x,
            w13,
            w2,
            weights,
            ids,
            expert_map,
            weight_on_input,
            output,
            gated,
            projected,
            hf_rounding=hf_rounding,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for changed in (False, True):
        if changed:
            x.mul_(0.7)
            weights.mul_(0.3)
            ids.copy_(torch.flip(ids, [0, 1]))
            ids[0, 0] = 99
            ids[-1, -1] = -1
        expected = _moe_reference(
            x, w13, w2, weights, ids, expert_map, weight_on_input, hf_rounding
        )
        graph.replay()
        torch.testing.assert_close(output, expected, atol=2e-4, rtol=3e-3)
        torch.testing.assert_close(run(), expected, atol=2e-4, rtol=3e-3)


def _paged_inputs(
    num_seqs: int,
    heads: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    seq_lens: list[int],
    device: str,
    padded: bool = False,
):
    torch.manual_seed(0)
    max_len = max(seq_lens)
    max_blocks = (max_len + block_size - 1) // block_size + 1
    total_blocks = num_seqs * max_blocks
    block_tables = (
        torch.randperm(total_blocks, device=device)
        .view(num_seqs, max_blocks)
        .to(torch.int32)
    )

    dtype = torch.float16
    k = (torch.randn(num_seqs, max_len, kv_heads, head_size, device=device) * 0.2).to(
        dtype
    )
    v = torch.randn(num_seqs, max_len, kv_heads, head_size, device=device).to(dtype)
    page_size = block_size * kv_heads * 2 * head_size
    backing = torch.zeros(
        total_blocks, page_size + (64 if padded else 0), dtype=dtype, device=device
    )
    cache = backing[:, :page_size].view(
        total_blocks, block_size, kv_heads, 2 * head_size
    )

    slots, keys, values = [], [], []
    for seq, length in enumerate(seq_lens):
        for pos in range(length):
            block = int(block_tables[seq, pos // block_size])
            slots.append(block * block_size + pos % block_size)
            keys.append(k[seq, pos])
            values.append(v[seq, pos])

    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)
    ops.reshape_and_cache(torch.stack(keys), torch.stack(values), cache, slot_mapping)
    return cache, block_tables, slot_mapping, k, v


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="PyTorch CUTLASS prefill requires V100",
)
@pytest.mark.parametrize("head_size,block_size", [(64, 16), (128, 16), (256, 32)])
@pytest.mark.parametrize("query_lens", [[17], [5, 1, 0], [5, 17, 33]])
@pytest.mark.parametrize("cache_mode", ["upstream", "packed", "fp8"])
def test_cutlass_prefill_preserves_prefix_and_request_boundaries(
    monkeypatch, head_size, block_size, query_lens, cache_mode
):
    """Unequal requests, cached prefixes, and unused page tails stay isolated."""
    from vllm.winintel.sm70.backend import WinIntelSM70Impl, WinIntelSM70Metadata
    from vllm.winintel.sm70.upstream_attention import cache_views

    monkeypatch.setenv(
        "WININTEL_SM70_UPSTREAM_DECODE", str(int(cache_mode == "upstream"))
    )
    monkeypatch.setenv("WININTEL_SM70_TORCH_PREFILL", "1")
    lengths = [q + 19 + i * 13 if q else 0 for i, q in enumerate(query_lens)]
    heads, kv_heads = 8, 2
    cache, tables, slots, _, _ = _paged_inputs(
        len(lengths), heads, kv_heads, head_size, block_size, lengths, "cuda"
    )
    unused = torch.ones(cache.shape[:2], device="cuda", dtype=torch.bool)
    unused.view(-1)[slots] = False
    cache.masked_fill_(unused[:, :, None, None], 100)
    ks, vs = (0.017, 0.029) if cache_mode == "fp8" else (1.0, 1.0)
    if cache_mode == "fp8":
        quantized = cache.float()
        quantized[..., :head_size].div_(ks)
        quantized[..., head_size:].div_(vs)
        cache = quantized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    packed = torch.empty(
        2,
        cache.shape[0],
        kv_heads,
        head_size // 8,
        block_size,
        8,
        device="cuda",
        dtype=torch.float16,
    )
    keys, values = cache_views(packed)
    keys.copy_(
        cache[..., :head_size]
        .reshape(cache.shape[0], block_size, kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
    )
    values.copy_(cache[..., head_size:].permute(0, 2, 3, 1))
    if cache_mode != "upstream":
        packed = cache
    starts = torch.tensor([0] + query_lens, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    seq = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda", dtype=torch.int32),
        torch.tensor(query_lens, device="cuda"),
    )
    pos = torch.cat(
        [
            torch.arange(length - q, length, device="cuda", dtype=torch.int32)
            for q, length in zip(query_lens, lengths)
        ]
    )
    query = torch.randn(sum(query_lens), heads, head_size, device="cuda").half()
    metadata = WinIntelSM70Metadata(
        num_actual_tokens=query.shape[0],
        max_query_len=max(query_lens),
        max_seq_len=max(lengths),
        block_table=tables,
        token_seq=seq,
        token_pos=pos,
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        query_start_loc=starts,
    )
    impl = WinIntelSM70Impl(
        heads,
        head_size,
        head_size**-0.5,
        kv_heads,
        kv_cache_dtype="fp8_e4m3" if cache_mode == "fp8" else "auto",
    )
    output = torch.empty_like(query)
    impl.forward(
        SimpleNamespace(_k_scale_float=ks, _v_scale_float=vs),
        query,
        None,
        None,
        packed,
        metadata,
        output,
    )
    expected = reference.paged_attention(
        query,
        cache,
        tables,
        seq,
        pos,
        head_size**-0.5,
        kv_heads,
        k_scale=ks,
        v_scale=vs,
    )
    torch.testing.assert_close(output, expected, rtol=0.005, atol=0.005)


def test_reference_matches_dense_attention() -> None:
    seq_lens = [17, 40]
    device = "cpu"
    cache, block_tables, _, k, v = _paged_inputs(2, 8, 2, 64, 16, seq_lens, device)
    token_seq = torch.tensor([0, 1], dtype=torch.int32)
    token_pos = torch.tensor([seq_lens[0] - 1, seq_lens[1] - 1], dtype=torch.int32)
    q = torch.randn(2, 8, 64, dtype=torch.float16)

    out = reference.paged_attention(
        q, cache, block_tables, token_seq, token_pos, 64**-0.5, 2
    )

    expected = torch.empty_like(q)
    for index, seq in enumerate((0, 1)):
        length = seq_lens[seq]
        keys = k[seq, :length].repeat_interleave(4, dim=1)
        values = v[seq, :length].repeat_interleave(4, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[index].float(), keys.float())
        probs = torch.softmax(scores * 64**-0.5, dim=-1)
        expected[index] = torch.einsum("hn,nhd->hd", probs, values.float()).to(q.dtype)

    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("modifiers", [(0.0, -1), (1.5, 31)])
@pytest.mark.parametrize("hf_rounding", [False, True])
def test_kernel_matches_reference(
    head_size: int, padded: bool, modifiers: tuple[float, int], hf_rounding, monkeypatch
) -> None:
    """Protect paged score/probability rounding and mixed prefill/decode."""
    monkeypatch.setattr(ops, "_extension_state", False)
    if not ops.is_extension_available():
        pytest.skip("WinIntel SM70 extension is not available")

    seq_lens = [33, 64, 91, 257]
    device = "cuda"
    cache, block_tables, _, _, _ = _paged_inputs(
        4, 8, 2, head_size, 16, seq_lens, device, padded
    )
    token_seq = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32, device=device)
    token_pos = torch.tensor([2, 32, 63, 90, 256], dtype=torch.int32, device=device)
    q = (torch.randn(5, 8, head_size, device=device) * 0.2).to(torch.float16)
    softcap, sliding_window = modifiers

    for partitions in (1, 3, 8, 32, 128):
        out = ops.paged_attention(
            q,
            cache,
            block_tables,
            token_seq,
            token_pos,
            head_size**-0.5,
            2,
            softcap=softcap,
            sliding_window=sliding_window,
            num_partitions=partitions,
            hf_rounding=hf_rounding,
        )
        expected = reference.paged_attention(
            q,
            cache,
            block_tables,
            token_seq,
            token_pos,
            head_size**-0.5,
            2,
            softcap=softcap,
            sliding_window=sliding_window,
            hf_rounding=hf_rounding,
        )
        torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("padding,offset", [(64, 0), (65, 0), (64, 1)])
def test_cooperative_decode_cache_alignment_preserves_output(
    monkeypatch, padding, offset
):
    """Padded pages and shifted storage must preserve paired-load results."""
    monkeypatch.setenv("WININTEL_SM70_COOPERATIVE_DECODE", "1")
    if not ops.is_extension_available():
        pytest.skip("WinIntel SM70 extension is not available")
    cache, tables, _, _, _ = _paged_inputs(3, 8, 2, 128, 16, [257, 257, 257], "cuda")
    pages = cache.shape[0]
    page_size = cache[0].numel()
    backing = torch.full(
        (pages, page_size + padding), 7, dtype=cache.dtype, device="cuda"
    )
    shifted = backing[:, offset : offset + page_size].view_as(cache)
    shifted.copy_(cache)
    query = torch.randn(3, 8, 128, dtype=cache.dtype, device="cuda")
    sequence = torch.arange(3, dtype=torch.int32, device="cuda")
    position = torch.tensor([32, 128, 256], dtype=torch.int32, device="cuda")

    def run(values):
        return ops.paged_attention(
            query,
            values,
            tables,
            sequence,
            position,
            128**-0.5,
            2,
            num_partitions=8,
            hf_rounding=True,
        )

    expected = run(cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(shifted)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "head_size,mode", [(64, "scalar"), (128, "rounded"), (128, "tiled")]
)
@pytest.mark.parametrize("sequences", [1, 3])
def test_fp8_cache_graph_preserves_scales_pages_and_new_inputs(
    head_size, mode, sequences, monkeypatch
):
    """E4M3 bytes and scaled attention must match PyTorch after graph replay."""
    assert ops.is_extension_available()
    monkeypatch.setenv("WININTEL_SM70_TILED_PREFILL", str(int(mode == "tiled")))
    monkeypatch.setenv("WININTEL_SM70_MULTI_SEQUENCE_PREFILL", "1")
    source, tables, slots, k, v = _paged_inputs(
        sequences, 8, 2, head_size, 16, [257] * sequences, "cuda"
    )
    width = 16 * 2 * 2 * head_size
    backing = torch.full(
        (source.shape[0], width + 64), 17, device="cuda", dtype=torch.uint8
    )
    cache = backing[:, :width].view_as(source)
    key, value = k.flatten(0, 1).clone(), v.flatten(0, 1).clone()
    slots = slots.clone()
    slots[-1] = -1
    query = torch.randn(16 * sequences, 8, head_size, device="cuda").half()
    seq = torch.arange(sequences, device="cuda", dtype=torch.int32).repeat_interleave(
        16
    )
    pos = torch.arange(241, 257, device="cuda", dtype=torch.int32).repeat(sequences)
    ks, vs = 0.013, 0.027

    def forward():
        ops.reshape_and_cache(key, value, cache, slots, ks, vs)
        return ops.paged_attention(
            query,
            cache,
            tables,
            seq,
            pos,
            head_size**-0.5,
            2,
            num_partitions=8 if mode == "rounded" else 1,
            hf_rounding=mode == "rounded",
            k_scale=ks,
            v_scale=vs,
        )

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    for multiplier in (0.5, -1.0):
        key.mul_(multiplier)
        value.mul_(multiplier)
        expected_backing = torch.full_like(backing, 17)
        expected_cache = expected_backing[:, :width].view_as(source)
        reference.reshape_and_cache(key, value, expected_cache, slots, ks, vs)
        graph.replay()
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
        expected = reference.paged_attention(
            query,
            expected_cache,
            tables,
            seq,
            pos,
            head_size**-0.5,
            2,
            hf_rounding=mode == "rounded",
            k_scale=ks,
            v_scale=vs,
        )
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cache_update_handles_qkv_views_and_skips_negative_slots() -> None:
    """QKV 分割产生的非连续张量仍按槽位写入，保留未使用页和填充。"""
    if not ops.is_extension_available():
        pytest.skip("WinIntel SM70 extension is not available")
    packed = torch.randn(3, 2, 192, dtype=torch.float16, device="cuda")
    key, value, _ = packed.split(64, dim=-1)
    backing = torch.full((2, 2 * 2 * 128 + 64), 7.0, device="cuda", dtype=torch.float16)
    cache = backing[:, : 2 * 2 * 128].view(2, 2, 2, 128)
    expected = backing.clone()
    expected_cache = expected[:, : 2 * 2 * 128].view(2, 2, 2, 128)
    slots = torch.tensor([0, -1, 3], device="cuda", dtype=torch.int64)
    reference.reshape_and_cache(key, value, expected_cache, slots, 0.5, 2.0)
    ops.reshape_and_cache(key, value, cache, slots, 0.5, 2.0)
    torch.testing.assert_close(backing, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("modifiers", [(0.0, -1), (1.5, 31), (0.0, 257)])
@pytest.mark.parametrize(
    "sequences,multi_sequence", [(1, False), (3, False), (3, True)]
)
def test_tiled_prefill_masks_pages_tail_and_graph_positions(
    head_size, modifiers, sequences, multi_sequence, monkeypatch
) -> None:
    """Compare tiled prefill with FP32 attention across irregular paged rows."""
    monkeypatch.setenv("WININTEL_SM70_TILED_PREFILL", "1")
    monkeypatch.setenv("WININTEL_SM70_MULTI_SEQUENCE_PREFILL", str(int(multi_sequence)))
    assert ops.is_extension_available(), "Native kernel build is required"
    cache, tables, _, _, _ = _paged_inputs(
        sequences, 8, 2, head_size, 16, [517] * sequences, "cuda", padded=True
    )
    query = torch.randn(19, 8, head_size, device="cuda", dtype=torch.float16)
    seq = torch.zeros(19, device="cuda", dtype=torch.int32)
    positions = torch.tensor(
        [
            -1,
            0,
            1,
            15,
            16,
            17,
            31,
            32,
            63,
            64,
            127,
            128,
            255,
            256,
            511,
            516,
            3,
            257,
            500,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    if sequences == 3:
        # Unaligned boundaries exercise both homogeneous and mixed WMMA tiles.
        positions = positions.repeat(3)
        query = query.repeat(3, 1, 1)
        seq = torch.tensor(
            [2] * 17 + [0] * 18 + [1] * 22, device="cuda", dtype=torch.int32
        )
    softcap, window = modifiers

    def run():
        return ops.paged_attention(
            query,
            cache,
            tables,
            seq,
            positions,
            head_size**-0.5,
            2,
            softcap=softcap,
            sliding_window=window,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for changed in (False, True):
        if changed:
            positions.copy_(positions.flip(0))
            seq.copy_(seq.flip(0))
            tables.copy_(tables.flip(0))
            query.mul_(0.7)
        expected = reference.paged_attention(
            query,
            cache,
            tables,
            seq,
            positions,
            head_size**-0.5,
            2,
            softcap,
            window,
        )
        graph.replay()
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(run(), output, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_empty_attention_does_not_launch_a_zero_block_grid() -> None:
    if not ops.is_extension_available():
        pytest.skip("WinIntel SM70 extension is not available")
    query = torch.empty(0, 8, 64, device="cuda", dtype=torch.float16)
    cache = torch.empty(1, 16, 2, 128, device="cuda", dtype=torch.float16)
    tables = torch.zeros(0, 1, device="cuda", dtype=torch.int32)
    token = torch.empty(0, device="cuda", dtype=torch.int32)
    result = ops.paged_attention(query, cache, tables, token, token, 0.125, 2)
    torch.cuda.synchronize()
    assert result.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("upstream", "hf_rounding", "head_size", "block_size"),
    [
        (False, False, 64, 16),
        (False, True, 64, 16),
        (False, False, 128, 16),
        (False, True, 128, 16),
        (True, False, 64, 16),
        (True, False, 128, 16),
        (True, False, 256, 16),
        (True, False, 128, 32),
    ],
)
@pytest.mark.parametrize("query_len", [1, 4, 6, 8])
def test_decode_graph_replays_positions_pages_and_padded_requests(
    monkeypatch, upstream, hf_rounding, head_size, block_size, query_len
) -> None:
    """Replay the backend across page/partition boundaries and 3->2->1->3 rows."""
    if upstream and query_len > 1:
        pytest.skip("Upstream decode only supports one token per request")
    if not ops.is_extension_available():
        pytest.skip("WinIntel SM70 extension is not available")
    from vllm.config.compilation import CompilationConfig, CUDAGraphMode
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.winintel.sm70.backend import (
        WinIntelSM70Impl,
        WinIntelSM70MetadataBuilder,
    )

    monkeypatch.setenv("WININTEL_SM70_CUDAGRAPH", "1")
    monkeypatch.setenv("WININTEL_SM70_UPSTREAM_DECODE", str(int(upstream)))
    monkeypatch.setenv("WININTEL_SM70_TORCH_PREFILL", "1")
    monkeypatch.setenv("WININTEL_SM70_ATTENTION_HF_ROUNDING", str(int(hf_rounding)))
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False, max_model_len=2048),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=query_len - 1)
            if query_len > 1
            else None
        ),
    )
    builder = WinIntelSM70MetadataBuilder(
        FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=2,
            head_size=head_size,
            dtype=torch.float16,
        ),
        ["test"],
        config,
        torch.device("cuda"),
    )
    scale = head_size**-0.5
    impl = WinIntelSM70Impl(8, head_size, scale, num_kv_heads=2)
    cache, source_tables, _, _, _ = _paged_inputs(
        3, 8, 2, head_size, block_size, [1500, 1500, 1500], "cuda"
    )
    tables = source_tables.clone()
    expected_cache = cache.clone()
    if upstream:
        from vllm.winintel.sm70.upstream_attention import cache_views

        blocks = cache.shape[0]
        packed = torch.empty(
            2,
            blocks,
            2,
            head_size // 8,
            block_size,
            8,
            device="cuda",
            dtype=torch.float16,
        )
        keys, values = cache_views(packed)
        keys.copy_(
            cache[..., :head_size]
            .reshape(blocks, block_size, 2, head_size // 8, 8)
            .permute(0, 2, 3, 1, 4)
        )
        values.copy_(cache[..., head_size:].permute(0, 2, 3, 1))
        cache = packed
    seq_lens = torch.full((3,), query_len, device="cuda", dtype=torch.int32)
    starts_cpu = torch.arange(4, dtype=torch.int32) * query_len
    common = CommonAttentionMetadata(
        query_start_loc=starts_cpu.cuda(),
        query_start_loc_cpu=starts_cpu,
        seq_lens=seq_lens,
        num_reqs=3,
        num_actual_tokens=3 * query_len,
        max_query_len=query_len,
        max_seq_len=2048,
        block_table_tensor=tables,
        slot_mapping=torch.full((3 * query_len,), -1, device="cuda", dtype=torch.int64),
    )
    query = torch.randn(3 * query_len, 8, head_size, device="cuda", dtype=torch.float16)
    qkv = torch.randn(
        3 * query_len, 3, 2, head_size, device="cuda", dtype=torch.float16
    )
    _, key, value = qkv.unbind(1)
    output = torch.empty_like(query)
    metadata = builder.build_for_cudagraph_capture(common)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.do_kv_cache_update(None, key, value, cache, common.slot_mapping)
            impl.forward(None, query, key, value, cache, metadata, output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        impl.do_kv_cache_update(None, key, value, cache, common.slot_mapping)
        impl.forward(None, query, key, value, cache, metadata, output)

    for lengths, order in (
        ([16, 128, 256], [0, 1, 2]),
        ([17, 129, 0], [2, 0, 1]),
        ([257, 0, 0], [1, 2, 0]),
        ([300, 260, 130], [0, 2, 1]),
        ([1024, 1025, 129], [2, 1, 0]),
    ):
        seq_lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        tables.copy_(source_tables[order])
        for row, length in enumerate(lengths):
            for offset in range(query_len):
                position = length - query_len + offset
                common.slot_mapping[row * query_len + offset] = (
                    tables[row, position // block_size].long() * block_size
                    + position % block_size
                    if length
                    else -1
                )
        active = sum(length > 0 for length in lengths)
        starts_cpu.copy_(
            torch.tensor([min(row, active) * query_len for row in range(4)])
        )
        common.max_seq_len = max(lengths)
        updated = builder.build(0, common)
        assert updated.token_pos.data_ptr() == metadata.token_pos.data_ptr()
        reference.reshape_and_cache(key, value, expected_cache, common.slot_mapping)
        graph.replay()
        expected = reference.paged_attention(
            query,
            expected_cache,
            tables,
            updated.token_seq,
            updated.token_pos,
            scale,
            2,
            hf_rounding=hf_rounding,
        )
        if upstream:
            keys, values = cache_views(cache)
            restored = torch.cat(
                (
                    keys.permute(0, 3, 1, 2, 4).reshape(
                        blocks, block_size, 2, head_size
                    ),
                    values.permute(0, 3, 1, 2),
                ),
                dim=-1,
            )
        else:
            restored = cache
        torch.testing.assert_close(restored, expected_cache, atol=0, rtol=0)
        torch.testing.assert_close(output, expected, atol=2e-3, rtol=2e-3)
        assert torch.count_nonzero(output[active * query_len :]) == 0
        eager_output = torch.empty_like(query)
        impl.forward(
            None,
            query,
            key,
            value,
            cache,
            replace(updated, fixed_num_partitions=None),
            eager_output,
        )
        # A larger graph launch must not change the active KV reduction tree.
        tolerance = 0 if query_len == 1 else 2e-3
        torch.testing.assert_close(output, eager_output, atol=tolerance, rtol=tolerance)

    if upstream:
        prefill_query = torch.randn(6, 8, head_size, device="cuda", dtype=torch.float16)
        prefill_metadata = replace(
            metadata,
            num_actual_tokens=6,
            max_query_len=2,
            max_seq_len=257,
            token_seq=torch.tensor(
                [0, 0, 1, 1, 2, 2], device="cuda", dtype=torch.int32
            ),
            token_pos=torch.tensor(
                [14, 15, 127, 128, 255, 256], device="cuda", dtype=torch.int32
            ),
            fixed_num_partitions=None,
        )
        prefill_output = torch.empty_like(prefill_query)
        impl.forward(
            None, prefill_query, key, value, cache, prefill_metadata, prefill_output
        )
        expected = reference.paged_attention(
            prefill_query,
            expected_cache,
            tables,
            prefill_metadata.token_seq,
            prefill_metadata.token_pos,
            scale,
            2,
        )
        torch.testing.assert_close(prefill_output, expected, atol=2e-3, rtol=2e-3)
        # Eager decode can omit empty rows without compacting the block table.
        sparse_metadata = replace(
            prefill_metadata,
            num_actual_tokens=2,
            max_query_len=1,
            token_seq=prefill_metadata.token_seq[[0, 4]],
            token_pos=prefill_metadata.token_pos[[0, 4]],
        )
        sparse_output = torch.empty_like(prefill_query[:2])
        impl.forward(
            None, prefill_query[:2], key, value, cache, sparse_metadata, sparse_output
        )
        expected = reference.paged_attention(
            prefill_query[:2],
            expected_cache,
            tables,
            sparse_metadata.token_seq,
            sparse_metadata.token_pos,
            scale,
            2,
        )
        torch.testing.assert_close(sparse_output, expected, atol=2e-3, rtol=2e-3)
