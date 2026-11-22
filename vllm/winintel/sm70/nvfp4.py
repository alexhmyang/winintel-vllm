# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""vLLM 0.17 NVFP4 checkpoint compatibility using FP16 arithmetic on V100."""

import os

from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.parameter import PerTensorScaleParameter
from winintel_kernels.blas import linear, prepare_weight
from winintel_kernels.nvfp4 import (
    dequantize,
    load_extension,
    packed_linear,
    prepare_scales,
)


class WinIntelSM70NVFP4(CompressedTensorsW4A4Fp4):
    """Reuse upstream TP weight loading and decompress once before inference."""

    def __init__(self, use_a16=False):
        self.group_size = 16
        self.use_a16 = use_a16
        self.kernel = self

    @staticmethod
    def input_quant_key():
        return None

    @classmethod
    def get_min_capability(cls):
        return 70

    def create_weights(self, layer, *args, weight_loader, **kwargs):
        def load(param, loaded_weight, shard_id=None):
            if isinstance(param, PerTensorScaleParameter) and isinstance(
                shard_id, tuple
            ):
                for index in shard_id:
                    weight_loader(param, loaded_weight, index)
            elif shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)

        super().create_weights(layer, *args, weight_loader=load, **kwargs)

    def process_weights_after_loading(self, layer):
        if os.environ.get("WININTEL_SM70_NVFP4_PACKED", "0") == "1":
            load_extension()
            layer.weight_packed = Parameter(
                layer.weight_packed.detach().contiguous(), requires_grad=False
            )
            layer.weight_scale = Parameter(
                prepare_scales(
                    layer.weight_scale,
                    layer.weight_global_scale,
                    layer.logical_widths,
                ),
                requires_grad=False,
            )
            for name in ("weight_global_scale", "input_global_scale"):
                if hasattr(layer, name):
                    delattr(layer, name)
            return
        if os.environ.get("WININTEL_SM70_NVFP4_GPU_DEQUANT") == "1":
            weight = load_extension().dequant(
                layer.weight_packed.contiguous(),
                prepare_scales(
                    layer.weight_scale,
                    layer.weight_global_scale,
                    layer.logical_widths,
                ),
            )
        else:
            weight = dequantize(
                layer.weight_packed,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.logical_widths,
            )
        layer.weight = Parameter(prepare_weight(weight), requires_grad=False)
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        ):
            if hasattr(layer, name):
                delattr(layer, name)

    def apply_weights(self, layer, x, bias=None):
        if hasattr(layer, "weight_packed"):
            return packed_linear(x, layer.weight_packed, layer.weight_scale, bias)
        return linear(x, layer.weight, bias)
