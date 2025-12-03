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

from fastdeploy import envs
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.utils import (
    TensorTracker,
    process_weight_transpose,
    set_weight_attrs,
)

from ..utils import get_tensor, per_block_cast_to_fp8
from .quant_base import QuantConfigBase, QuantMethodBase


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
        self.is_checkpoint_bf16 = is_checkpoint_bf16

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
            if layer.ep_size > 1 or self.use_deep_gemm:
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
            weight_scale_inv_shape = [
                (layer.weight_shape[0] + self.quant_config.weight_block_size[0] - 1)
                // self.quant_config.weight_block_size[0],
                (layer.weight_shape[1] + self.quant_config.weight_block_size[1] - 1)
                // self.quant_config.weight_block_size[1],
            ]

            if self.model_format != "torch" and layer.fd_config.load_config.load_choices == "default_v1":
                weight_shape = layer.weight_shape[::-1]
                weight_scale_inv_shape = weight_scale_inv_shape[::-1]
            else:
                # v0 loader or torch model format
                weight_shape = layer.weight_shape
                weight_scale_inv_shape = weight_scale_inv_shape
                extra_weight_attrs["output_dim"] = (
                    not extra_weight_attrs["output_dim"] if extra_weight_attrs["output_dim"] is not None else None
                )

            layer.weight_dtype = "float8_e4m3fn"
            layer.weight = layer.create_parameter(
                shape=weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.weight_scale_inv = layer.create_parameter(
                shape=weight_scale_inv_shape,
                dtype="float32",
                is_bias=False,
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
            quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)
            from fastdeploy.model_executor.ops.gpu import deep_gemm

            # quanted_weight_tensor, weight_block_scale_tensor = deep_gemm.utils.math.per_block_cast_to_fp8(
            #     weight_tensor, use_ue8m0=True
            # )
            # ======新加的
            # =======aaaaaaaaaaaaaaaaaaaaa 以下是调试代码新加的

            def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
                x: paddle.Tensor,
            ):
                from deep_gemm.utils import align, get_tma_aligned_size

                assert x.dtype == paddle.float and x.dim() in (2, 3)

                # First, convert into UE8M0 `uint8_t`
                ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

                # Second, make padded packed tensors
                mn, k = x.shape[-2], x.shape[-1]
                remove_dim = False
                if x.dim() == 2:
                    x, remove_dim = x.unsqueeze(0), True
                b = x.shape[0]
                aligned_mn = get_tma_aligned_size(mn, 4)
                aligned_k = align(k, 4)
                padded = paddle.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8)
                padded[:, :mn, :k] = ue8m0_tensor
                padded = padded.view(-1).view(dtype=paddle.int).view(b, aligned_mn, aligned_k // 4)

                # Finally, transpose
                transposed = paddle.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int).mT
                transposed[:, :, :] = padded
                aligned_x = transposed[:, :mn, :]
                return aligned_x.squeeze(0) if remove_dim else aligned_x

            def block_quant_dequant(
                x_q_block,
                x_s,
                block_size,
                dtype,
            ):
                """This function converts block-wise quantization to unquantized.
                The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
                and the block size.
                The output is an unquantized tensor with dtype.
                """
                block_n, block_k = block_size[0], block_size[1]
                *_, n, k = x_q_block.shape

                # ... n_scale k_scale -> ... (n_scale block_n) (k_scale block_k)
                x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(block_k, dim=-1)
                x_scale_repeat = x_scale_repeat[..., :n, :k]

                return (x_q_block.to(paddle.float32) * x_scale_repeat).to(dtype)

            def requant_weight_ue8m0(
                weight,
                weight_scale_inv,
                weight_block_size,
            ):
                assert weight_block_size == [128, 128]

                *_, n, k = weight.shape

                weight_dequant = block_quant_dequant(
                    weight,
                    weight_scale_inv,
                    weight_block_size,
                    paddle.bfloat16,
                )

                out_w, out_s = quant_weight_ue8m0(
                    weight_dequant=weight_dequant,
                    weight_block_size=weight_block_size,
                )

                out_s = transform_scale_ue8m0(out_s, mn=out_w.shape[-2])

                return out_w, out_s

            def quant_weight_ue8m0(weight_dequant, weight_block_size):
                assert weight_block_size == [128, 128]
                assert weight_dequant.dtype == paddle.bfloat16, f"{weight_dequant.dtype=} {weight_dequant.shape=}"

                *batch_dims, n, k = weight_dequant.shape

                weight_dequant_flat = weight_dequant.view((-1, k))
                out_w_flat, out_s_flat = deep_gemm.utils.math.per_block_cast_to_fp8(
                    weight_dequant_flat, use_ue8m0=True
                )

                out_w = out_w_flat.view((*batch_dims, n, k))
                out_s = out_s_flat.view(
                    (
                        *batch_dims,
                        deep_gemm.utils.math.ceil_div(n, weight_block_size[0]),
                        deep_gemm.utils.math.ceil_div(k, weight_block_size[1]),
                    )
                )

                return out_w, out_s

            # NOTE copy and modified from DeepGEMM
            def transform_scale_ue8m0(sf, mn, use_torch_impl: bool = False):
                # get_mn_major_tma_aligned_packed_ue8m0_tensor = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor
                get_mn_major_tma_aligned_packed_ue8m0_tensor = _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl

                sf = sf.index_select(-2, paddle.arange(mn, device=sf.device) // 128)
                sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
                return sf

            # ======aaaaaaaaa 以上是调试代码新加的
            quanted_weight_tensor, weight_block_scale_tensor = requant_weight_ue8m0(
                quanted_weight_tensor.to(weight_block_scale_tensor.place), weight_block_scale_tensor, [128, 128]
            )
            # ======新加的

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
            # layer.weight_scale_inv = layer.create_parameter(
            #     shape=weight_block_scale_tensor.shape,
            #     dtype="float32",
            #     is_bias=False,
            #     default_initializer=paddle.nn.initializer.Constant(0),
            # )
            layer.weight_scale_inv = weight_block_scale_tensor
            # layer.weight_scale_inv = layer.create_parameter(
            #     shape=weight_block_scale_tensor.shape,
            #     dtype=weight_block_scale_tensor.dtype,
            #     is_bias=False,
            #     default_initializer=paddle.nn.initializer.Constant(0),
            # )

            layer.weight.copy_(quanted_weight_tensor, False)
            layer.weight_scale_inv.copy_(weight_block_scale_tensor, False)

        if self.quant_config.is_checkpoint_bf16:
            if self.model_format == "torch":
                process_weight_transpose(layer, "weight")
            _process_quantize()
        else:
            if self.model_format != "torch":
                process_weight_transpose(layer, "weight")
                process_weight_transpose(layer, "weight_scale_inv")
            else:
                return

    def process_loaded_weights(self, layer, weights) -> None:
        weight_tensor = weights.transpose([1, 0])
        quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)
        layer.weight.copy_(quanted_weight_tensor, False)
        layer.weight_scale_inv.set_value(weight_block_scale_tensor)

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
        # print(f"[FP8Linear] x: {x}")
        # print(f"[FP8Linear] block_tables 1: {block_tables}")
        # import os

        # debug_dir = "/workspace2/FastDeploy/debug"
        # debug_data = {
        #     "x": x,
        #     "weight": layer.weight,
        #     "weight_scale_inv": layer.weight_scale_inv,
        # }
        # debug_file = os.path.join(debug_dir, f"fp8_linear_debug.pdparam")
        # paddle.save(debug_data, debug_file)
        # print(f"[FP8Linear] Debug data saved to {debug_file}")

        # x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
        #     x, self.quant_config.weight_block_size[0]
        # )
        # print(f"[FP8Linear] block_tables 2: {block_tables}")
        from fastdeploy.model_executor.ops.gpu import deep_gemm

        x, x_scale_tensor = deep_gemm.utils.math.per_token_cast_to_fp8(x, use_ue8m0=True)

        linear_out: paddle.Tensor = paddle.empty((x.shape[0], layer.output_size), dtype=paddle.bfloat16)

        # print(f"[FP8Linear] x_quantized: {x}")
        # print(f"[FP8Linear] x_scale_tensor: {x_scale_tensor}")
        # print(f"[FP8Linear] layer.weight: {layer.weight}")
        # print(f"[FP8Linear] layer.weight_scale_inv: {layer.weight_scale_inv}")
        deep_gemm.fp8_gemm_nt(
            (x, x_scale_tensor),
            (layer.weight, layer.weight_scale_inv),
            linear_out,
            # disable_ue8m0_cast=True,
        )
        # print(f"[FP8Linear] block_tables 3: {block_tables}")
        # print(f"[FP8Linear] linear_out: {linear_out}")
        if layer.with_bias:
            linear_out = paddle.add(linear_out, layer.bias)
        # print(f"[FP8Linear] linear_out after bias: {linear_out}")
        return linear_out
