# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Reuse official ModelOpt mixed checkpoint loaders with V100 FP16 execution."""

import os

import torch
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8LinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4LinearMethod,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import replace_parameter

from vllm.model_executor.layers.linear import LinearBase
from winintel_kernels.blas import linear, prepare_weight
from winintel_kernels.nvfp4 import dequantize, load_extension


class SM70ModelOptLoader:
    def create_weights(self, layer, *args, weight_loader, params_dtype, **kwargs):
        def load(param, loaded_weight, shard_id=None):
            if isinstance(layer, ParallelLMHead) and isinstance(
                param, PerTensorScaleParameter
            ):
                weight_loader(param, loaded_weight.reshape(param.shape))
            elif isinstance(param, PerTensorScaleParameter) and isinstance(
                shard_id, tuple
            ):
                for index in shard_id:
                    weight_loader(param, loaded_weight, index)
            elif shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)

        super().create_weights(
            layer, *args, weight_loader=load, params_dtype=params_dtype, **kwargs
        )
        layer.orig_dtype = params_dtype

    def apply(self, layer, x, bias=None):
        return linear(x, layer.weight, bias)


class SM70ModelOptFP8(SM70ModelOptLoader, ModelOptFp8LinearMethod):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer):
        scales = convert_to_channelwise(layer.weight_scale, layer.logical_widths)
        weight = (layer.weight.float() * scales.float().reshape(-1, 1)).to(
            layer.orig_dtype
        )
        replace_parameter(layer, "weight", prepare_weight(weight))
        del layer.weight_scale
        del layer.input_scale


class SM70ModelOptNVFP4(SM70ModelOptLoader, ModelOptNvFp4LinearMethod):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer):
        # ModelOpt global scales multiply; compressed-tensors scales divide.
        global_scales = convert_to_channelwise(
            layer.weight_scale_2, layer.logical_widths
        ).float()
        scales = layer.weight_scale.float() * global_scales.reshape(-1, 1)
        if os.environ.get("WININTEL_SM70_NVFP4_GPU_DEQUANT") == "1":
            weight = load_extension().dequant(
                layer.weight.contiguous(), scales.contiguous()
            )
        else:
            weight = dequantize(
                layer.weight,
                scales,
                torch.ones(len(layer.logical_widths)),
                layer.logical_widths,
            )
        replace_parameter(layer, "weight", prepare_weight(weight))
        del layer.weight_scale
        del layer.weight_scale_2
        del layer.input_scale


class WinIntelSM70ModelOptMixed(ModelOptMixedPrecisionConfig):
    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        modules = {name + ".": info for name, info in self.quantized_layers.items()}
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        self.quantized_layers = {
            name.removesuffix("."): info
            for name, info in hf_to_vllm_mapper.apply_dict(modules).items()
        }

    @classmethod
    def from_upstream(cls, config):
        return cls(
            config.kv_cache_quant_method,
            config.exclude_modules,
            config.quantized_layers,
            config.fp8_config,
            config.nvfp4_config,
        )

    @classmethod
    def get_min_capability(cls):
        return 70

    def get_quant_method(self, layer, prefix):
        if isinstance(
            layer, (LinearBase, ParallelLMHead)
        ) and not self.is_layer_excluded(prefix):
            algorithm = self._resolve_quant_algo(prefix)
            if algorithm is None and isinstance(layer, ParallelLMHead):
                algorithm = self._resolve_quant_algo("language_model." + prefix)
            if algorithm == "FP8":
                return SM70ModelOptFP8(self.fp8_config)
            if algorithm == "NVFP4":
                return SM70ModelOptNVFP4(self.nvfp4_config)
        return super().get_quant_method(layer, prefix)
