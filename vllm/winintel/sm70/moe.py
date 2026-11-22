# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 WinIntel contributors.
# Author: 数萌AI服务器 (winintel)
# Email: service@winintel.com
# Website: https://winintel.com
"""Native unquantized SwiGLU experts for Volta."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.platforms import current_platform
from winintel_kernels import ops


class WinIntelSM70PrepareFinalize(MoEPrepareAndFinalizeNoDPEPModular):
    """Leave per-route input weighting to the native expert kernel."""

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        return a1, None, None, None, None

    def topk_indices_dtype(self):
        return torch.int32


class WinIntelSM70Experts(mk.FusedMoEExpertsModular):
    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        if (
            moe_config.in_dtype != torch.float16
            or not self._supports_activation(moe_config.activation)
            or not self._supports_parallel_config(moe_config.moe_parallel_config)
            or moe_config.has_bias
            or getattr(moe_config, "is_lora_enabled", False)
            or quant_config.quant_dtype is not None
            or quant_config.w1_bias is not None
            or quant_config.w2_bias is not None
            or getattr(quant_config, "gemm1_clamp_limit", None) is not None
            or getattr(quant_config, "gemm1_alpha", None) not in (None, 1.0)
            or getattr(quant_config, "gemm1_beta", None) not in (None, 0.0)
            or getattr(moe_config, "swiglu_limit", None) is not None
            or getattr(moe_config, "swiglu_alpha", None) not in (None, 1.0)
            or getattr(moe_config, "swiglu_beta", None) not in (None, 0.0)
        ):
            raise NotImplementedError(
                "WININTEL_SM70 MoE requires unquantized FP16 SwiGLU, "
                "TP without DP/EP, and no bias, LoRA or activation clamp"
            )
        if not ops.is_extension_available():
            raise RuntimeError("WININTEL_SM70 MoE requires the native CUDA extension")

    @staticmethod
    def is_supported_config(
        cls,
        moe_config,
        weight_key,
        activation_key,
        activation_format,
    ):
        if moe_config.in_dtype != torch.float16:
            return False, "kernel requires FP16 activations"
        if moe_config.has_bias or getattr(moe_config, "is_lora_enabled", False):
            return False, "kernel does not support bias or LoRA"
        if (
            getattr(moe_config, "swiglu_limit", None) is not None
            or getattr(moe_config, "swiglu_alpha", None) not in (None, 1.0)
            or getattr(moe_config, "swiglu_beta", None) not in (None, 0.0)
        ):
            return False, "kernel requires standard unclamped SwiGLU"
        return mk.FusedMoEExpertsModular.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        return current_platform.is_cuda() and current_platform.is_device_capability(70)

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return getattr(activation, "value", activation) == "silu"

    @staticmethod
    def _supports_parallel_config(config):
        return (
            config.dp_size == 1
            and config.ep_size == 1
            and not config.use_ep
            and not getattr(config, "is_sequence_parallel", False)
            and getattr(config, "pcp_size", 1) == 1
        )

    @staticmethod
    def _supports_batch_invariance():
        from winintel_kernels import w4

        return not w4.enabled()

    def supports_chunking(self):
        return True

    def supports_expert_map(self):
        return True

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        if self._uses_w4_down_combine(topk):
            return (M, K), (M, topk, N // 2), (M, K)
        # vLLM aliases workspace13 with the output. The gate buffer is dead
        # before combine; down-projection results must remain separate.
        return (M, topk, N // 2), (M, topk, K), (M, K)

    def _uses_w4_down_combine(self, topk):
        from winintel_kernels import w4

        return (
            hasattr(self, "w4_weights")
            and topk == 8
            and self.w4_weights[1].shape[-1] <= 64
            and w4.down_combine_enabled()
        )

    def finalize_weight_and_reduce_impl(self):
        return TopKWeightAndReduceNoOP()

    def prepare_w4(self, layer):
        from winintel_kernels import w4

        if not w4.enabled():
            return
        if (
            self.moe_config.moe_parallel_config.tp_size != 1
            and not w4.parallel_enabled()
        ):
            raise NotImplementedError(
                "Experimental SM70 W4 TP requires WININTEL_SM70_EXPERIMENTAL_W4_TP=1"
            )
        if any(
            getattr(layer, name).shape[-1] % 128 for name in ("w13_weight", "w2_weight")
        ):
            raise NotImplementedError("SM70 W4 requires group-aligned TP shards")
        self.w4_extension = w4.load_extension()
        self.w4_weights, self.w4_scales = [], []
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(layer, name)
            packed, scales = w4.pack(weight)
            layer.register_buffer(f"winintel_{name}_packed", packed)
            layer.register_buffer(f"winintel_{name}_scales", scales)
            self.w4_weights.append(packed)
            self.w4_scales.append(scales)
            # Preserve logical shapes for the modular workspace allocator.
            placeholder = weight.new_empty(1).expand(weight.shape)
            setattr(layer, name, torch.nn.Parameter(placeholder, requires_grad=False))

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        if not self._supports_activation(activation):
            raise NotImplementedError("WININTEL_SM70 MoE only supports SwiGLU")
        if hasattr(self, "w4_weights"):
            if expert_map is not None or apply_router_weight_on_input:
                raise NotImplementedError(
                    "Experimental W4 requires local experts and unweighted inputs"
                )
            ids = topk_ids.int().contiguous()
            weights = topk_weights.float().contiguous()
            if self._uses_w4_down_combine(ids.shape[1]):
                self.w4_extension.linear(
                    hidden_states.contiguous(),
                    self.w4_weights[0],
                    self.w4_scales[0],
                    ids,
                    workspace2,
                    True,
                    True,
                )
                self.w4_extension.down_combine(
                    workspace2,
                    self.w4_weights[1],
                    self.w4_scales[1],
                    ids,
                    weights,
                    output,
                )
                return
            self.w4_extension.linear(
                hidden_states.contiguous(),
                self.w4_weights[0],
                self.w4_scales[0],
                ids,
                workspace13,
                True,
                True,
            )
            self.w4_extension.linear(
                workspace13,
                self.w4_weights[1],
                self.w4_scales[1],
                ids,
                workspace2,
                False,
                True,
            )
            self.w4_extension.combine(
                workspace2,
                weights,
                ids,
                output,
                self.w4_weights[0].shape[0],
            )
            return
        ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_map,
            apply_router_weight_on_input,
            output,
            workspace13,
            workspace2,
        )
