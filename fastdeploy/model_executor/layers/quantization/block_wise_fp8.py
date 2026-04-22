"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from typing import Optional

import paddle

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVGateParallelLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    deep_gemm,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from fastdeploy.model_executor.utils import (
    TensorTracker,
    process_weight_transpose,
    set_weight_attrs,
)
from fastdeploy.platforms import current_platform
from fastdeploy.utils import register_custom_python_op

from ..utils import get_sm_version, get_tensor, per_block_cast_to_fp8
from .quant_base import QuantConfigBase, QuantMethodBase

if current_platform.is_cuda():
    try:
        fp8_gemm_nt = deep_gemm.fp8_gemm_nt
    except:
        fp8_gemm_nt = deep_gemm.gemm_fp8_fp8_bf16_nt
else:
    fp8_gemm_nt = None

# Detect whether fp8_gemm_nt accepts a 'bias' keyword argument
_fp8_gemm_nt_has_bias_kwarg = False
if fp8_gemm_nt is not None:
    import inspect

    try:
        _sig = inspect.signature(fp8_gemm_nt)
        _fp8_gemm_nt_has_bias_kwarg = "bias" in _sig.parameters
    except (ValueError, TypeError):
        # pybind11 functions may not expose signatures via inspect;
        # fall back to a cheap probe call to determine support.
        pass


class BlockWiseFP8Config(QuantConfigBase):
    """
    block wise quantization config, only support fp8 quant and only supports loading weights in BF16 format.
    After loading the weights, it will automatically compute quantization sparsity and dynamically perform
    per-token quantization of activations during inference.
    """

    def __init__(self, weight_block_size: list = [-1, -1], is_checkpoint_bf16: bool = False) -> None:
        super().__init__()
        self.weight_block_size = weight_block_size
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.use_deep_gemm = bool(envs.FD_USE_DEEP_GEMM)
        self.use_blackwell_gemm = bool(envs.FD_USE_BLACKWELL_GEMM)
        self.is_checkpoint_bf16 = is_checkpoint_bf16
        self.deepgemm_scale_ue8m0 = True if get_sm_version() >= 100 else False

    def name(self) -> str:
        return "block_wise_fp8"

    @classmethod
    def from_config(cls, config: dict) -> "BlockWiseFP8Config":
        weight_block_size = config.get("weight_block_size", [128, 128])
        is_checkpoint_bf16 = not config.get("is_quantized", False)
        return cls(weight_block_size, is_checkpoint_bf16)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        Get quantization method.
        """
        if isinstance(layer, FusedMoE):
            if self.use_blackwell_gemm:
                assert (
                    self.use_deep_gemm
                ), "Blackwell gemm is supported only for prefill moe, please set FD_USE_DEEP_GEMM=1 as well"
                from fastdeploy.model_executor.layers.moe.fused_moe_blackwell_backend import (
                    BlackwellGemmFusedMoeMethod,
                )

                return BlackwellGemmFusedMoeMethod(self)
            elif layer.ep_size > 1 or self.use_deep_gemm:
                from fastdeploy.model_executor.layers.moe.fused_moe_deepgemm_backend import (
                    DeepGemmFusedMoeMethod,
                )

                return DeepGemmFusedMoeMethod(self)
            else:
                from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
                    BlockWiseFP8MoEMethod,
                )
            return BlockWiseFP8MoEMethod(self)
        else:
            return BlockWiseFP8LinearMethod(self)


def deep_gemm_fp8_gemm_nt_infer_meta(
    x_meta: "paddle.static.MetaTensor",
    x_scale_tensor_meta: "paddle.static.MetaTensor",
    layer_weight_meta: "paddle.static.MetaTensor",
    layer_weight_scale_inv_meta: "paddle.static.MetaTensor",
    linear_out_meta: "paddle.static.MetaTensor",
    layer_output_size: int,
):
    return paddle.static.MetaTensor(shape=[x_meta.shape[0], layer_output_size], dtype=paddle.bfloat16)


@register_custom_python_op(
    name="deep_gemm_fp8_gemm_nt",
    infer_meta=deep_gemm_fp8_gemm_nt_infer_meta,
    input_names=["x", "x_scale_tensor", "layer_weight", "layer_weight_scale_inv", "linear_out_empty"],
    output_names=["linear_out"],
    inplace_map={},
)
def deep_gemm_fp8_gemm_nt(
    x: paddle.Tensor,
    x_scale_tensor: paddle.Tensor,
    layer_weight: paddle.Tensor,
    layer_weight_scale_inv: paddle.Tensor,
    linear_out: paddle.Tensor,
    layer_output_size: int,
    bias: paddle.Tensor = None,
):
    sm_version = get_sm_version()
    if sm_version >= 100 and current_platform.is_cuda():
        # disable_ue8m0_cast is default False for SM100
        if _fp8_gemm_nt_has_bias_kwarg:
            fp8_gemm_nt(
                (x, x_scale_tensor),
                (layer_weight, layer_weight_scale_inv),
                linear_out,
                bias=bias,
            )
        else:
            fp8_gemm_nt(
                (x, x_scale_tensor),
                (layer_weight, layer_weight_scale_inv),
                linear_out,
            )
            if bias is not None:
                linear_out = paddle.add(linear_out, bias)
    else:
        fp8_gemm_nt(
            (x, x_scale_tensor),
            (layer_weight, layer_weight_scale_inv),
            linear_out,
        )
    return linear_out


class BlockWiseFP8LinearMethod(QuantMethodBase):
    """
    block wise quantization method for linear
    """

    def __init__(
        self,
        quant_config: BlockWiseFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer, **extra_weight_attrs):
        # TODO(bukejiyu): remove v1 loader check when v0 loader is removed
        self.model_format = extra_weight_attrs.get("model_format")
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            weight_shape = layer.weight_shape[::-1] if self.model_format == "torch" else layer.weight_shape
            layer.weight = layer.create_parameter(
                shape=weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            quant_attrs = extra_weight_attrs
            if (
                isinstance(layer, MergedColumnParallelLinear)
                or isinstance(layer, QKVParallelLinear)
                or isinstance(layer, MergedReplicatedLinear)
                or isinstance(layer, QKVGateParallelLinear)
            ):
                tensor_output_dim = (self.model_format == "torch") ^ quant_attrs.get("output_dim", True)
                quant_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=weight_shape, output_dim=tensor_output_dim),
                }
            if self.model_format == "torch" and "output_dim" in quant_attrs:
                quant_attrs["output_dim"] = not quant_attrs["output_dim"]
            set_weight_attrs(
                layer.weight,
                quant_attrs,
            )
        else:
            layer.weight_shape.reverse()
            if not self.quant_config.deepgemm_scale_ue8m0:
                weight_scale_inv_shape = [
                    (layer.weight_shape[0] + self.quant_config.weight_block_size[0] - 1)
                    // self.quant_config.weight_block_size[0],
                    (layer.weight_shape[1] + self.quant_config.weight_block_size[1] - 1)
                    // self.quant_config.weight_block_size[1],
                ]
            else:
                num_scales = (
                    layer.weight_shape[1] + self.quant_config.weight_block_size[1] - 1
                ) // self.quant_config.weight_block_size[1]
                num_scale_packs = (num_scales + 3) // 4
                weight_scale_inv_shape = [
                    layer.weight_shape[0],
                    num_scale_packs,
                ]

            if self.model_format != "torch" and layer.fd_config.load_config.load_choices == "default_v1":
                weight_shape = layer.weight_shape[::-1]
                weight_scale_inv_shape = weight_scale_inv_shape[::-1]
            else:
                # v0 loader or torch model format
                weight_shape = layer.weight_shape
                weight_scale_inv_shape = weight_scale_inv_shape
                extra_weight_attrs["output_dim"] = (
                    not extra_weight_attrs["output_dim"]
                    if extra_weight_attrs.get("output_dim", None) is not None
                    else None
                )

            layer.weight_dtype = "float8_e4m3fn"
            layer.weight = layer.create_parameter(
                shape=weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            if not self.quant_config.deepgemm_scale_ue8m0:
                layer.weight_scale_inv = layer.create_parameter(
                    shape=weight_scale_inv_shape,
                    dtype="float32",
                    is_bias=False,
                )
            else:
                layer.weight_scale_inv = layer.create_parameter(
                    shape=weight_scale_inv_shape,
                    dtype="int32",
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0),
                )

            set_weight_attrs(
                layer.weight,
                extra_weight_attrs,
            )
            set_weight_attrs(
                layer.weight_scale_inv,
                {
                    **extra_weight_attrs,
                    "is_scale": True,
                },
            )

    def process_weights_after_loading(self, layer) -> None:
        def _process_quantize():
            weight_tensor = layer.weight.transpose([1, 0])

            if not self.quant_config.deepgemm_scale_ue8m0:
                quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)
            else:
                quanted_weight_tensor, weight_block_scale_tensor = quant_weight_ue8m0(weight_tensor, [128, 128])
                weight_block_scale_tensor = transform_scale_ue8m0(
                    weight_block_scale_tensor,
                    mn=quanted_weight_tensor.shape[-2],
                    weight_block_size=[128, 128],
                )
            if hasattr(layer.weight, "tensor_track"):
                layer.weight.tensor_track = None
            layer.weight.value().get_tensor()._clear()
            del layer.weight

            layer.weight = layer.create_parameter(
                shape=quanted_weight_tensor.shape,
                dtype="float8_e4m3fn",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.weight_scale_inv = layer.create_parameter(
                shape=weight_block_scale_tensor.shape,
                dtype=weight_block_scale_tensor.dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.weight.copy_(quanted_weight_tensor, False)
            layer.weight_scale_inv.data = weight_block_scale_tensor

        if self.quant_config.is_checkpoint_bf16:
            if self.model_format == "torch":
                process_weight_transpose(layer, "weight")
            _process_quantize()
        else:
            if self.model_format != "torch":
                process_weight_transpose(layer, "weight")
                process_weight_transpose(layer, "weight_scale_inv")
            if self.quant_config.deepgemm_scale_ue8m0:
                new_weight_scale_inv = paddle.empty(
                    layer.weight_scale_inv.shape[::-1], dtype=layer.weight_scale_inv.dtype
                )
                new_weight_scale_inv = new_weight_scale_inv.transpose([1, 0])
                layer.weight_scale_inv.data = new_weight_scale_inv

    def process_loaded_weights(self, layer, weights) -> None:
        weight_tensor = weights.transpose([1, 0])
        if not self.quant_config.deepgemm_scale_ue8m0:
            quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)
        else:
            weight_block_size = self.quant_config.weight_block_size
            assert weight_block_size == [
                128,
                128,
            ], f"weight_block_size must be [128, 128] for ue8m0, but got {weight_block_size}"
            quanted_weight_tensor, weight_block_scale_tensor = quant_weight_ue8m0(weight_tensor, weight_block_size)
            weight_block_scale_tensor = transform_scale_ue8m0(
                weight_block_scale_tensor,
                mn=quanted_weight_tensor.shape[-2],
                weight_block_size=weight_block_size,
            )
        layer.weight.copy_(quanted_weight_tensor, False)
        layer.weight_scale_inv.data = weight_block_scale_tensor

    def process_prequanted_weights(self, layer, state_dict, is_rearrange: bool = False):
        """
        process_prequanted_weights
        """
        quant_weight = get_tensor(state_dict.pop(layer.weight_key))
        weight_scale = get_tensor(state_dict.pop(layer.weight_scale_key))

        quant_weight = quant_weight.transpose([1, 0]).contiguous()
        layer.weight.copy_(quant_weight.view("float8_e4m3fn"), False)

        weight_scale = weight_scale.transpose([1, 0])
        layer.weight_scale_inv.set_value(weight_scale)

    def apply(self, layer, x):
        linear_out = paddle.empty((x.shape[0], layer.output_size), dtype=paddle.bfloat16)
        if x.shape[0] == 0:
            return linear_out
        if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
            x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
                x, self.quant_config.weight_block_size[0], self.quant_config.deepgemm_scale_ue8m0
            )
            x_scale_tensor = x_scale_tensor[: x.shape[0], ...]
        else:
            x, x_scale_tensor = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                using_pow2_scale=self.quant_config.deepgemm_scale_ue8m0 or fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE,
                output_scale_transpose=True,
                using_ue8m0_scale=self.quant_config.deepgemm_scale_ue8m0,
            )
            x_scale_tensor = x_scale_tensor.T[: x.shape[0], ...]

        if get_sm_version() == 100 and current_platform.is_cuda():
            deep_gemm_fp8_gemm_nt(
                x,
                x_scale_tensor,
                layer.weight,
                layer.weight_scale_inv,
                linear_out,
                layer_output_size=layer.output_size,
                bias=layer.bias if layer.with_bias else None,
            )
        else:
            deep_gemm_fp8_gemm_nt(
                x,
                x_scale_tensor,
                layer.weight,
                layer.weight_scale_inv,
                linear_out,
                layer_output_size=layer.output_size,
            )
            if layer.with_bias:
                linear_out = paddle.add(linear_out, layer.bias)

        return linear_out
