# SPDX-License-Identifier: Apache-2.0
# Author: 数萌AI服务器 (winintel)
"""Reuse compressed-tensors FP8 loading with FP16 linear execution on V100."""

from compressed_tensors.quantization import QuantizationStrategy
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A16Fp8,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import replace_parameter

from winintel_kernels.blas import linear, prepare_weight


class WinIntelSM70FP8(CompressedTensorsW8A16Fp8):
    """Expand tensor/channel FP8 weights once, retaining upstream TP loaders."""

    @classmethod
    def get_min_capability(cls):
        return 70

    def create_weights(self, layer, *args, weight_loader, **kwargs):
        if self.strategy not in (
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
        ):
            raise NotImplementedError("SM70 FP8 requires tensor or channel scales")

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
        scales = layer.weight_scale
        if self.strategy == QuantizationStrategy.TENSOR:
            scales = convert_to_channelwise(scales, layer.logical_widths)
        weight = (layer.weight.float() * scales.float().reshape(-1, 1)).to(
            layer.orig_dtype
        )
        replace_parameter(layer, "weight", prepare_weight(weight))
        del layer.weight_scale
        if hasattr(layer, "input_scale"):
            del layer.input_scale

    def apply_weights(self, layer, x, bias=None):
        return linear(x, layer.weight, bias)
