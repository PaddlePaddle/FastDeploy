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

import importlib
import importlib.util
import math
from enum import Enum
from typing import Callable, Optional

import paddle
from paddle import nn

from fastdeploy import envs
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase
from fastdeploy.model_executor.utils import set_weight_attrs
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import moe_expert_dispatch
from fastdeploy.utils import get_logger

from ..moe import FusedMoE
from .quant_base import QuantConfigBase, QuantMethodBase

paddle.compat.enable_torch_proxy(scope={"flashinfer"})

logger = get_logger("config", "config.log")


class Mxfp4Backend(Enum):
    NONE = 0

    # FlashInfer Backend
    SM90_FI_MXFP4_BF16 = 1

    # Triton Backend
    TRITON = 2


def check_device_capability(num):
    if paddle.is_compiled_with_cuda():
        device = paddle.device.get_device()
        major, minor = paddle.device.cuda.get_device_capability(device)
        return major * 10 + minor >= num
    else:
        return False


def has_flashinfer():
    return importlib.util.find_spec("flashinfer") is not None


def round_up(a, b):
    return ((a + b - 1) // b) * b


def get_mxfp4_backend():
    if current_platform.is_cuda():
        if check_device_capability(90) and has_flashinfer() and envs.FD_MOE_MXFP4_BACKEND == "flashinfer":
            logger.info("FastDeploy Using FlashInfer MXFP4 BF16 backend for SM90 in MoE")
            return Mxfp4Backend.SM90_FI_MXFP4_BF16
        elif envs.FD_MOE_MXFP4_BACKEND == "triton":
            logger.info("FastDeploy Using Triton backend in MoE")
            return Mxfp4Backend.TRITON
    raise NotImplementedError


def get_padding_weight(param, shape) -> paddle.Tensor:
    if len(param.shape) == 4:
        param = param.reshape([param.shape[0], param.shape[1], param.shape[2] * param.shape[3]])

    if len(shape) == 3:
        weight = paddle.nn.functional.pad(
            param.cast("int32"),
            pad=[0, shape[-1] - param.shape[-1], 0, shape[-2] - param.shape[-2]],
            mode="constant",
            value=0,
        ).cast(param.dtype)
    elif len(shape) == 2:
        weight = paddle.nn.functional.pad(
            param,
            pad=[0, shape[-1] - param.shape[-1]],
            mode="constant",
            value=0,
        )
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    return weight


def _interleave_mxfp4_cutlass_sm90(w):
    w_shape = w.shape
    w_interleaved = w.reshape([w_shape[0], w_shape[1], (w_shape[2] // 4), 4])
    w_interleaved = w_interleaved.permute([0, 2, 1, 3])
    w_interleaved = w_interleaved.reshape([w_shape[0], w_shape[2] // 4, w_shape[1] * 4])
    return w_interleaved


class MXFP4Config(QuantConfigBase):
    """Base class for quantization configs."""

    def __init__(self, is_checkpoint_bf16: bool = False):
        super().__init__()
        self.is_checkpoint_bf16 = is_checkpoint_bf16

    def name(self) -> str:
        return "mxfp4"

    @classmethod
    def from_config(cls, config: dict) -> "MXFP4Config":
        is_checkpoint_bf16 = not config.get("is_quantized", False)
        return cls(is_checkpoint_bf16)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            return MXFP4MoeMethod(self)
        else:
            raise NotImplementedError


class MXFP4MoeMethod(MoEMethodBase):
    def __init__(
        self,
        quant_config: MXFP4Config,
    ) -> None:
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.mxfp4_backend = get_mxfp4_backend()

    def create_weights(self, layer, **extra_weight_attrs):
        self.extra_weight_attrs = extra_weight_attrs

        block_size = 32

        self.intermediate_size = layer.fd_config.model_config.intermediate_size
        self.hidden_size = layer.fd_config.model_config.hidden_size
        self.num_experts = layer.fd_config.model_config.num_local_experts

        self.tp_rank = layer.tp_rank
        self.tp_size = layer.tp_size
        self.ep_size = layer.ep_size
        self.ep_rank = layer.ep_rank

        if self.ep_size > 1:
            raise NotImplementedError("EP has not yet been implemented in MXFP4.")
            assert self.num_experts % self.ep_size == 0, "only support num_experts divisible by ep_size"
        self.num_local_experts = self.num_experts // self.ep_size

        self.up_gate_proj_weight_shape = [
            self.num_experts,
            self.intermediate_size * 2,
            self.hidden_size // block_size,
            block_size // 2,
        ]

        self.down_proj_weight_shape = [
            self.num_experts,
            self.hidden_size,
            self.intermediate_size // block_size,
            block_size // 2,
        ]

        self.up_gate_proj_scale_shape = [
            self.num_experts,
            self.intermediate_size * 2,
            self.hidden_size // block_size,
        ]

        self.down_proj_scale_shape = [
            self.num_experts,
            self.hidden_size,
            self.intermediate_size // block_size,
        ]

        self.weight_dtype = "uint8"

        setattr(
            layer,
            "up_gate_proj_weight",
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_weight",
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        setattr(
            layer,
            "up_gate_proj_scale",
            layer.create_parameter(
                shape=self.up_gate_proj_scale_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        setattr(
            layer,
            "down_proj_scale",
            layer.create_parameter(
                shape=self.down_proj_scale_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        extra_weight_attrs["weight_need_transpose"] = not extra_weight_attrs.get("model_format") == "torch"

        set_weight_attrs(layer.up_gate_proj_weight, extra_weight_attrs)
        set_weight_attrs(layer.down_proj_weight, extra_weight_attrs)
        set_weight_attrs(layer.up_gate_proj_scale, extra_weight_attrs)
        set_weight_attrs(layer.down_proj_scale, extra_weight_attrs)

        if layer.with_bias:
            layer.up_gate_proj_bias = layer.create_parameter(
                shape=[self.num_experts, self.intermediate_size * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_bias = layer.create_parameter(
                shape=[self.num_experts, self.hidden_size],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            set_weight_attrs(
                layer.up_gate_proj_bias,
                extra_weight_attrs,
            )
            set_weight_attrs(
                layer.down_proj_bias,
                extra_weight_attrs,
            )

        if layer.activation == "swigluoai":
            gemm1_alpha = layer.create_parameter(
                shape=[self.num_local_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(1.702),
            )
            gemm1_alpha.initialize()
            setattr(layer, "gemm1_alpha", gemm1_alpha)

            gemm1_beta = layer.create_parameter(
                shape=[self.num_local_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )
            gemm1_beta.initialize()
            setattr(layer, "gemm1_beta", gemm1_beta)

            gemm1_clamp_limit = layer.create_parameter(
                shape=[self.num_local_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(7.0),
            )
            gemm1_clamp_limit.initialize()
            setattr(layer, "gemm1_clamp_limit", gemm1_clamp_limit)

    def process_weights_after_loading(self, layer) -> None:
        extra_weight_attrs = self.extra_weight_attrs

        block_size = 32

        intermediate_size = self.intermediate_size
        intermediate_size_block = intermediate_size // block_size
        per_rank_intermediate_size_block = math.ceil(intermediate_size_block / self.tp_size)
        per_rank_intermediate_size = per_rank_intermediate_size_block * block_size

        intermediate_size_pad = per_rank_intermediate_size
        hidden_size_pad = self.hidden_size

        if self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16:
            intermediate_size_pad = round_up(intermediate_size_pad, 128)
            hidden_size_pad = round_up(hidden_size_pad, 128)
        else:
            intermediate_size_pad = round_up(intermediate_size_pad, 64)

        self.intermediate_size_pad = intermediate_size_pad
        self.hidden_size_pad = hidden_size_pad

        tp_rank_start = self.tp_rank * intermediate_size_pad
        tp_rank_end = min((self.tp_rank + 1) * intermediate_size_pad, intermediate_size)

        ep_rank_start = self.ep_rank * self.num_local_experts
        ep_rank_end = (self.ep_rank + 1) * self.num_local_experts

        self.up_gate_proj_weight_shape = [
            self.num_local_experts,
            intermediate_size_pad * 2,
            hidden_size_pad // 2,  # uint8
        ]

        self.down_proj_weight_shape = [
            self.num_local_experts,
            hidden_size_pad,
            intermediate_size_pad // 2,  # uint8
        ]

        self.up_gate_proj_scale_shape = [
            self.num_local_experts,
            intermediate_size_pad * 2,
            hidden_size_pad // block_size,
        ]

        self.down_proj_scale_shape = [
            self.num_local_experts,
            hidden_size_pad,
            intermediate_size_pad // block_size,
        ]

        self.weight_dtype = "uint8"

        up_gate_proj_weight_padding = layer.create_parameter(
            shape=self.up_gate_proj_weight_shape,
            dtype=self.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        weight = layer.up_gate_proj_weight.reshape([self.num_experts, self.intermediate_size * 2, -1])
        if self.ep_size > 1:
            weight = weight[ep_rank_start:ep_rank_end, ...]
        else:
            weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
        weight = get_padding_weight(weight, self.up_gate_proj_weight_shape)
        gate_w, up_w = weight[:, ::2, :], weight[:, 1::2, :]
        up_gate_proj_weight_padding.copy_(paddle.concat([up_w, gate_w], axis=1), False)
        layer.up_gate_proj_weight._clear()
        layer.up_gate_proj_weight = up_gate_proj_weight_padding

        down_proj_weight_padding = layer.create_parameter(
            shape=self.down_proj_weight_shape,
            dtype=self.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        weight = layer.down_proj_weight.reshape([self.num_experts, self.hidden_size, -1])
        if self.ep_size > 1:
            weight = weight[ep_rank_start:ep_rank_end, ...]
        else:
            weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]
        weight = get_padding_weight(weight, self.down_proj_weight_shape)
        down_proj_weight_padding.copy_(weight, False)
        layer.down_proj_weight._clear()
        layer.down_proj_weight = down_proj_weight_padding

        up_gate_proj_scale_padding = layer.create_parameter(
            shape=self.up_gate_proj_scale_shape,
            dtype=self.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        weight = layer.up_gate_proj_scale
        if self.ep_size > 1:
            weight = weight[ep_rank_start:ep_rank_end, ...]
        else:
            weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]
        weight = get_padding_weight(weight, self.up_gate_proj_scale_shape)
        gate_s, up_s = weight[:, ::2, :], weight[:, 1::2, :]
        up_gate_proj_scale = paddle.concat([up_s, gate_s], axis=1)
        up_gate_proj_scale_interleaved = _interleave_mxfp4_cutlass_sm90(up_gate_proj_scale)
        up_gate_proj_scale_padding.copy_(up_gate_proj_scale_interleaved, False)
        layer.up_gate_proj_scale._clear()
        layer.up_gate_proj_scale = up_gate_proj_scale_padding

        down_proj_scale_padding = layer.create_parameter(
            shape=self.down_proj_scale_shape,
            dtype=self.weight_dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        weight = layer.down_proj_scale
        if self.ep_size > 1:
            weight = weight[ep_rank_start:ep_rank_end, ...]
        else:
            weight = weight[..., tp_rank_start // block_size : tp_rank_end // block_size]
        weight = get_padding_weight(weight, self.down_proj_scale_shape)
        down_proj_scale = weight
        down_proj_scale_interleaved = _interleave_mxfp4_cutlass_sm90(down_proj_scale)
        down_proj_scale_padding.copy_(down_proj_scale_interleaved, False)
        layer.down_proj_scale._clear()
        layer.down_proj_scale = down_proj_scale_padding

        extra_weight_attrs["weight_need_transpose"] = not extra_weight_attrs.get("model_format") == "torch"

        set_weight_attrs(layer.up_gate_proj_weight, extra_weight_attrs)
        set_weight_attrs(layer.down_proj_weight, extra_weight_attrs)
        set_weight_attrs(layer.up_gate_proj_scale, extra_weight_attrs)
        set_weight_attrs(layer.down_proj_scale, extra_weight_attrs)

        if layer.with_bias:
            up_gate_proj_bias_padding = layer.create_parameter(
                shape=[self.num_local_experts, intermediate_size_pad * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            weight = layer.up_gate_proj_bias
            if self.ep_size > 1:
                weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
            weight = get_padding_weight(weight, [self.num_local_experts, self.intermediate_size_pad * 2])
            gate_b, up_b = weight[:, ::2].cast("bfloat16"), weight[:, 1::2].cast("bfloat16")
            up_gate_proj_bias_padding.copy_(paddle.concat([up_b, gate_b], axis=-1), False)
            layer.up_gate_proj_bias._clear()
            layer.up_gate_proj_bias = up_gate_proj_bias_padding

            down_proj_bias_padding = layer.create_parameter(
                shape=[self.num_local_experts, hidden_size_pad],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            weight = layer.down_proj_bias
            if self.ep_size > 1:
                weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                if self.tp_rank != 0:
                    weight = paddle.zeros_like(weight)
            weight = get_padding_weight(weight, [self.num_local_experts, self.hidden_size_pad])
            down_proj_bias_padding.copy_(weight.cast("bfloat16"), False)
            layer.down_proj_bias._clear()
            layer.down_proj_bias = down_proj_bias_padding

            set_weight_attrs(
                layer.up_gate_proj_bias,
                extra_weight_attrs,
            )
            set_weight_attrs(
                layer.down_proj_bias,
                extra_weight_attrs,
            )

    def apply(
        self, layer: nn.Layer, x: paddle.Tensor, router: nn.Layer, topk_ids_hookfunc: Callable = None
    ) -> paddle.Tensor:
        router_out = router(x.cast("float32"))

        if self.mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16:

            (
                _,
                _,
                _,
                topk_weights,
                topk_idx,
                *_,
            ) = moe_expert_dispatch(
                x,
                router_out,
                layer.gate_correction_bias,
                (
                    layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None
                ),  # if set, permute_input will be int8_t
                layer.top_k,
                False,
                self.quant_config.name(),
                topk_only_mode=False,
            )

            if topk_ids_hookfunc is not None:
                topk_ids_hookfunc(topk_ids=topk_idx)

            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            quant_scales = [
                layer.up_gate_proj_scale,
                layer.down_proj_scale,
            ]
            extra_kwargs = dict(
                use_w4_group_scaling=True,
                fc1_expert_weights=layer.up_gate_proj_weight,
                fc2_expert_weights=layer.down_proj_weight,
            )

            from flashinfer.fused_moe import (
                cutlass_fused_moe as flashinfer_cutlass_fused_moe,
            )

            # if x.shape[0] == 0:
            #     return paddle.zeros([0, layer.hidden_size], dtype="bfloat16")

            x = paddle.nn.functional.pad(x, pad=[0, self.hidden_size_pad - x.shape[-1]], mode="constant", value=0)

            output = paddle.zeros_like(x, dtype="bfloat16")

            _ = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_idx,
                token_final_scales=topk_weights,
                output_dtype=paddle.bfloat16,
                output=output,
                quant_scales=quant_scales,
                fc1_expert_biases=layer.up_gate_proj_bias,
                fc2_expert_biases=layer.down_proj_bias,
                swiglu_alpha=layer.gemm1_alpha,
                swiglu_beta=layer.gemm1_beta,
                swiglu_limit=layer.gemm1_clamp_limit,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                tune_max_num_tokens=8192,
                **extra_kwargs,
            )

            return output[..., : layer.hidden_size].clone()

    def process_loaded_weights(self, layer, weights):
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return

    def apply_tp(self, layer, x, gate, topk_ids_hookfunc=None):
        return self.apply(layer, x, gate, topk_ids_hookfunc)

    def apply_ep_prefill(self, layer, x, gate, topk_ids_hookfunc=None):
        raise NotImplementedError("EP 尚未在 MXFP4 中实现")

    def apply_ep_decode(self, layer, x, gate, topk_ids_hookfunc=None):
        raise NotImplementedError("EP 尚未在 MXFP4 中实现")
