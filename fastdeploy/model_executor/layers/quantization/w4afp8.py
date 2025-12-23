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

from ..moe import FusedMoE
from .quant_base import QuantConfigBase, QuantMethodBase

QUANT_SCALING_FACTOR = 448

from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.utils import (
    TensorTracker,
    process_weight_transpose,
    set_weight_attrs,
)


class W4AFP8Config(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, weight_scale_dict, act_scale_dict, is_permuted, hadamard_block_size, is_quantized) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.is_permuted = is_permuted
        self.hadamard_block_size = hadamard_block_size
        self.is_quantized = is_quantized

    def name(self) -> str:
        return "w4afp8"

    @classmethod
    def from_config(cls, config: dict) -> "W4AFP8Config":
        weight_scale_dict = config.get("weight_scale_dict", None)
        act_scale_dict = config.get("act_scale_dict", None)
        is_permuted = config.get("is_permuted", True)
        hadamard_block_size = config.get("hadamard_block_size", 128)
        is_quantized = config.get("is_quantized", False)
        # print("weight_scale_dict",weight_scale_dict)
        # print("act_scale_dict",act_scale_dict)
        return cls(weight_scale_dict, act_scale_dict, is_permuted, hadamard_block_size, is_quantized)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            print("use W4AFP8MoEMethod for MoE layer--------")
            from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
                CutlassW4AFP8MoEMethod,
            )

            return CutlassW4AFP8MoEMethod(self)
        print("use w4afp8linear")
        return W4AFP8LinearMethod(self)


class W4AFP8LinearMethod(QuantMethodBase):
    """
    W4 AFP8 quant method for linear
    """

    def __init__(
        self,
        quant_config: W4AFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer, **extra_weight_attrs):
        self.model_format = extra_weight_attrs.get("model_format")

        # =======================================================
        # 分支 1: 加载原始 BF16/FP16 权重 (后续进行即时量化)
        # =======================================================
        # 注意：这里使用 quant_config.is_checkpoint_bf16 更准确，与 BlockWise 保持一致
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            # 如果是 Torch 格式 [Out, In]，为了适配 Paddle API 需要转为 [In, Out] (反转)
            weight_shape = layer.weight_shape[::-1] if self.model_format == "torch" else layer.weight_shape

            layer.weight = layer.create_parameter(
                shape=weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            # 处理并行切分逻辑 (Tensor Parallel)
            quant_attrs = extra_weight_attrs
            if (
                isinstance(layer, MergedColumnParallelLinear)
                or isinstance(layer, QKVParallelLinear)
                or isinstance(layer, MergedReplicatedLinear)
            ):
                # 异或逻辑：处理 Torch 和 Paddle 在 output_dim 定义上的差异
                tensor_output_dim = (self.model_format == "torch") ^ quant_attrs.get("output_dim", True)
                quant_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=weight_shape, output_dim=tensor_output_dim),
                }

            # 修正 Torch 格式下的 output_dim 属性
            if self.model_format == "torch" and "output_dim" in quant_attrs:
                quant_attrs["output_dim"] = not quant_attrs["output_dim"]

            set_weight_attrs(
                layer.weight,
                quant_attrs,
            )

        # =======================================================
        # 分支 2: 加载预量化 (Pre-quantized) 权重
        # =======================================================
        else:
            # 1. 以 [Out, In] 为基准计算形状
            layer.weight_shape.reverse()

            # INT4 Packing: [Out, In] -> [Out/2, In] (假设沿着 Out 维度打包)
            # 注意：具体是 Out/2 还是 In/2 取决于 Kernel 实现，
            # 参照旧代码 layer.weight_shape[0] //= 2，这里假设是 Out 维度被压缩
            weight_packed_shape = [layer.weight_shape[0] // 2, layer.weight_shape[1]]

            # Scale Shape: Per-Channel 量化，通常是 [Out]
            weight_scale_shape = [layer.weight_shape[0]]

            # 2. 如果是 Paddle 格式，需要反转回 [In, Out/2] 以匹配文件
            if self.model_format != "torch" and layer.fd_config.load_config.load_choices == "default_v1":
                weight_packed_shape = weight_packed_shape[::-1]
                # Scale 是 1D Tensor，通常不需要反转，除非是 [Out, 1] 这种 2D 形式。
                # 如果是 1D [N]，[::-1] 只是倒序数据，不是转置维度，所以这里不动 Scale。

            layer.weight_dtype = "int8"  # 用于存储 packed int4
            layer.weight = layer.create_parameter(
                shape=weight_packed_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.weight_scale = layer.create_parameter(
                shape=weight_scale_shape,
                dtype="float16",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )

            set_weight_attrs(layer.weight, extra_weight_attrs)
            set_weight_attrs(layer.weight_scale, {**extra_weight_attrs, "is_scale": True})

    def process_weights_after_loading(self, layer) -> None:
        def _process_quantize():
            # 获取原始权重
            weight_tensor = layer.weight

            # 关键修正：确保输入给量化算子的形状是 [Out, In]
            # 因为 per-channel 量化通常是沿着 Out 维度进行的
            if self.model_format != "torch":
                # 如果是 Paddle 格式 [In, Out]，转置为 [Out, In]
                weight_tensor = weight_tensor.transpose([1, 0])

            # 此时 weight_tensor 应该是 [Out, In]
            (
                quanted_weight_tensor,
                weight_scale_tensor,
            ) = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize(
                paddle.cast(weight_tensor, "float32"),
                groupsize=-1,  # -1 表示 per-channel
                scale_dtype="float16",
            )

            # 清理旧权重
            if hasattr(layer.weight, "tensor_track"):
                layer.weight.tensor_track = None
            layer.weight.value().get_tensor()._clear()
            del layer.weight

            # 创建新参数 (Packed INT8 Weight)
            layer.weight = layer.create_parameter(
                shape=quanted_weight_tensor.shape,
                dtype="int8",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            # 创建 Scale 参数
            layer.weight_scale = layer.create_parameter(
                shape=weight_scale_tensor.shape,
                dtype="float16",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )

            layer.weight.copy_(quanted_weight_tensor, False)
            layer.weight_scale.copy_(weight_scale_tensor, False)

        # =======================================================
        # 逻辑修正：这里必须清晰地区分 "即时量化" 和 "直接加载"
        # =======================================================
        if self.quant_config.is_checkpoint_bf16:
            # === 分支 A: 需要即时量化 ===

            # 1. 如果是 Torch 格式，先处理物理转置，使其在内存中变为 [Out, In]
            if self.model_format == "torch":
                process_weight_transpose(layer, "weight")

            # 2. 无论是不是 Torch，只要是 BF16 checkpoint，都需要执行量化
            _process_quantize()

        else:
            # === 分支 B: 预量化加载 (Pre-quantized) ===

            # 如果不是 Torch 格式 (即 Paddle 格式)，需要转置权重以适配 Kernel 要求
            if self.model_format != "torch":
                process_weight_transpose(layer, "weight")
                # 注意：Scale 通常是 1D Tensor，不需要 process_weight_transpose。
                # 只有当 Scale 被定义为 2D [N, 1] 时才需要转置。
                # W4A8 的 Scale 通常是一维向量，所以这里去掉 Scale 的转置操作。
            else:
                return

    def process_loaded_weights(self, layer, weights) -> None:
        """保持原有的 V0 Loader 支持"""
        (
            quanted_weight_tensor,
            weight_scale_tensor,
        ) = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize(
            paddle.cast(weights, "float32").cpu(),
            groupsize=-1,
            scale_dtype="float16",
        )
        weight_scale_tensor = paddle.view(weight_scale_tensor, layer._dtype)
        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor)

    def apply(self, layer, x):
        # ... 保持不变 ...
        linear_out = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16(
            x,
            layer.weight,
            layer.weight_scale,
            zero_points=None,
            bias=layer.bias if layer.with_bias else None,
            out_scale=self.quant_config.weight_scale_dict.get(layer.prefix + ".weight_scale")
            / (
                self.quant_config.act_scale_dict.get(layer.prefix + ".activation_scale")
                * QUANT_SCALING_FACTOR
                * QUANT_SCALING_FACTOR
            ),
            groupsize=0,
            out_dtype=layer._dtype,
        )
        return linear_out
