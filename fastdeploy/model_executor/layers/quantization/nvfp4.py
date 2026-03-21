"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Callable, Optional

import paddle
from paddle import nn
from paddleformers.utils.log import logger

import os
import fastdeploy
from fastdeploy.model_executor.layers.moe.ep import deep_ep
from fastdeploy import envs
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase
from fastdeploy.model_executor.utils import (
    create_parameter_and_copy,
    free_tensor,
    set_weight_attrs,
)

from fastdeploy.model_executor.ops.gpu import (
    depermute_prefill_combine,
    prefill_permute_to_masked_gemm,
)

from .quant_base import QuantConfigBase, QuantMethodBase

paddle.compat.enable_torch_proxy(scope={"flashinfer"})

from flashinfer import (
    scaled_fp4_grouped_quantize,
    silu_and_mul_scaled_nvfp4_experts_quantize,
)
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked


def call_prefill_permute_to_masked_gemm(
    x: paddle.Tensor,
    scale: paddle.Tensor,
    topk_ids: paddle.Tensor,
    num_local_experts: int,
    max_token_num: int,
):
    """
    Permute input tokens and scales from token-major to expert-major layout
    for MoE masked GEMM operations.

    Args:
        x: Input hidden states [num_tokens, hidden].
        scale: Input scales [num_tokens, hidden_scale].
        topk_ids: Expert routing indices [num_tokens, topk] (int64 or int32).
        num_local_experts: Number of local experts on this device.
        max_token_num: Maximum tokens per expert buffer.

    Returns:
        tuple: (permute_x, permute_scale, permuted_indice_map, token_nums_per_expert)
    """
    if topk_ids.dtype != paddle.int64:
        topk_ids = topk_ids.cast(paddle.int64)

    results = prefill_permute_to_masked_gemm(x, scale, topk_ids, num_local_experts, max_token_num)

    return results[0], results[1], results[2], results[3]


def call_depermute_prefill_combine(
    x: paddle.Tensor,
    indice_map: paddle.Tensor,
    topk_weights: paddle.Tensor,
    num_worst_tokens: int,
):
    """
    Depermute and combine expert outputs back to token-major layout.

    Args:
        x: Expert outputs [num_local_experts, max_tokens_per_expert, hidden].
        indice_map: Flat index tensor [num_worst_tokens, topk] (int32).
        topk_weights: Combination weights [num_worst_tokens, topk] (float32).
        num_worst_tokens: Number of output tokens to produce.

    Returns:
        depermuted_x: Combined output [num_worst_tokens, hidden].
    """
    results = depermute_prefill_combine(x, indice_map, topk_weights, num_worst_tokens)

    return results


def _perm(tensor, *dims):
    try:
        return tensor.transpose(list(dims))
    except TypeError:
        return tensor.permute(*dims)


def _get_cute_dtype(input_tensor) -> str:
    s = str(input_tensor.dtype).split(".")[-1]
    if s == "bfloat16":
        return "bfloat16"
    if s == "float16":
        return "float16"
    if s == "float32":
        return "float32"
    raise ValueError(f"Unsupported cute dtype {input_tensor.dtype}")



def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _process_scale_interleaved(scales):
    scale_dim = len(scales.shape)
    if scale_dim == 2:
        scales = scales.unsqueeze(0)
    assert len(scales.shape) == 3
    B, M, K = scales.shape
    round_up_multiple = lambda x, m: (x + m - 1) // m * m
    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    padded_scales = paddle.empty([B, M_padded, K_padded], dtype=scales.dtype)
    padded_scales[:B, :M, :K].copy_(scales)
    batches, rows, cols = padded_scales.shape
    assert rows % 128 == 0
    assert cols % 4 == 0
    padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
    padded_scales = padded_scales.transpose([0, 1, 4, 3, 2, 5])
    # [batches, rows // 128, cols // 4, 32, 4, 4]

    padded_scales = padded_scales.contiguous().to(paddle.device.get_device())
    padded_scales = (
        padded_scales.reshape(M_padded, K_padded) if scale_dim == 2 else padded_scales.reshape(B, M_padded, K_padded)
    )
    return padded_scales


class ModelOptNvFp4Config(QuantConfigBase):
    """
    quantization config for ModelOpt Nvfp4 datatype
    """

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool,
        kv_cache_quant_algo: str | None,
        exclude_modules: list[str],
        group_size: int = 16,
        is_checkpoint_bf16: bool = False,
    ) -> None:
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint. Please note that"
                " the format is experimental and could change in future."
            )

            self.group_size = group_size
            self.kv_cache_quant_algo = kv_cache_quant_algo
            self.exclude_modules = exclude_modules

        self.quant_max_bound = 6
        self.quant_min_bound = -6
        self.quant_round_type = 1
        self.is_checkpoint_bf16 = is_checkpoint_bf16

    def name(self) -> str:
        return "modelopt_fp4"

    @classmethod
    def from_config(cls, config: dict) -> "ModelOptNvFp4Config":
        quant_config = config
        quant_method = quant_config.get("quant_algo", "")
        if not quant_method:
            raise ValueError("Missing 'quant_algo' in quantization config")

        # Handle kv_cache_quant_algo with proper type validation
        kv_cache_quant_algo_raw = quant_config.get("kv_cache_quant_algo")
        if kv_cache_quant_algo_raw is None:
            # No KV cache quantization by default
            kv_cache_quant_algo = None
        elif isinstance(kv_cache_quant_algo_raw, str):
            kv_cache_quant_algo = kv_cache_quant_algo_raw
        else:
            raise ValueError(f"kv_cache_quant_algo must be a string, got " f"{type(kv_cache_quant_algo_raw)}")

        # Handle group_size with proper type validation
        group_size_raw = quant_config.get("group_size")
        if group_size_raw is None:
            group_size = 16  # Default value
        elif isinstance(group_size_raw, int):
            group_size = group_size_raw
        else:
            try:
                group_size = int(group_size_raw)
            except (ValueError, TypeError):
                raise ValueError(f"group_size must be an integer, got {type(group_size_raw)}") from None

        # "exclude_modules" is the key in the legacy hf_quant_config.json
        exclude_modules = quant_config.get("exclude_modules", [])
        if not isinstance(exclude_modules, list):
            raise ValueError(f"exclude_modules must be a list, got {type(exclude_modules)}")

        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        # For FP4, these fields are required
        if is_checkpoint_nvfp4_serialized and "quantization" in config:
            # Check if required fields are present in the quantization config
            quant_config = config["quantization"]
            required_fields = ["group_size", "kv_cache_quant_algo", "exclude_modules"]
            missing_fields = [field for field in required_fields if field not in quant_config]
            if missing_fields:
                raise ValueError(
                    f"NVFP4 quantization requires the following fields in " f"hf_quant_config.json: {missing_fields}"
                )
        return cls(
            is_checkpoint_nvfp4_serialized=is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo=kv_cache_quant_algo,
            exclude_modules=exclude_modules,
            group_size=group_size,
        )

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        Get quantization method.
        """
        if isinstance(layer, FusedMoE):
            return ModelOptNvFp4FusedMoE(self)
        else:
            return ModelOptNvFp4LinearMethod(self)


class ModelOptNvFp4LinearMethod(QuantMethodBase):
    """Linear method for Model Optimizer NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    input_scale: paddle.float32, scalar ,
    weight: NVFP4(represented as byte) Shape: [1, X, y/2]
    weight_scale: FP8-E4M3, Shape: [X, Y], aka per block scale,
    weight_scale_2: paddle.float32, scalar,
    Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptNvFp4Config) -> None:
        self.quant_config = quant_config

        self.backend = "none"
        if envs.FD_NVFP4_GEMM_BACKEND is None:
            self.backend = "flashinfer-cutlass"
        elif envs.FD_NVFP4_GEMM_BACKEND.startswith("flashinfer-"):
            self.backend = envs.FD_NVFP4_GEMM_BACKEND

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 GEMM backend found. Please check your platform capability and installtion of Flashinfer."
            )

        logger.info(f"Using {self.backend} for NVFP4 GEMM")

    def create_weights(
        self,
        layer,
        **extra_weight_attrs,
    ):
        # 因为模型存储是列存储的，所以这里需要not一下！
        extra_weight_attrs["output_dim"] = not extra_weight_attrs["output_dim"]
        K = layer.weight_shape[0]
        N = layer.weight_shape[1]
        # 因为模型的存储时候权重是[N,K//2]
        # 所以这里创建的权重是为了契合模型存储的权重！
        weight_shape = [N, K // 2]
        layer.weight_dtype = "uint8"

        input_scale_shape = [1]
        weight_scale_shape = [N, K // self.quant_config.group_size]
        weight_scale_2_shape = [1]

        self._create_main_weight(layer, weight_shape, extra_weight_attrs)
        self._create_input_scale(layer, input_scale_shape)
        self._create_weight_scales(layer, weight_scale_shape, weight_scale_2_shape, extra_weight_attrs)

    def _create_main_weight(self, layer, weight_shape, extra_weight_attrs):
        """创建主权重参数

        参数:
            layer: 当前层对象
            weight_shape: 权重形状
            extra_weight_attrs: 额外权重属性
        """
        layer.weight = layer.create_parameter(
            shape=weight_shape,
            dtype=layer.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        set_weight_attrs(
            layer.weight,
            extra_weight_attrs,
        )

    def _create_input_scale(self, layer, input_scale_shape):
        """创建输入缩放参数

        参数:
            layer: 当前层对象
            input_scale_shape: 输入缩放形状
        """
        layer.input_scale = layer.create_parameter(
            shape=input_scale_shape,
            dtype=paddle.float32,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

    def _create_weight_scales(self, layer, weight_scale_shape, weight_scale_2_shape, extra_weight_attrs):
        """创建权重缩放参数

        参数:
            layer: 当前层对象
            weight_scale_shape: 权重缩放形状
            weight_scale_2_shape: 权重缩放2形状
            extra_weight_attrs: 额外权重属性
        """
        layer.weight_scale = layer.create_parameter(
            shape=weight_scale_shape,
            dtype=paddle.float8_e4m3fn,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        set_weight_attrs(
            layer.weight_scale,
            extra_weight_attrs,
        )

        layer.weight_scale_2 = layer.create_parameter(
            shape=weight_scale_2_shape,
            dtype=paddle.float32,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

    def process_weights_after_loading(self, layer) -> None:

        input_scale_2 = layer.input_scale.max().to(paddle.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(paddle.float32)
        alpha = input_scale_2 * weight_scale_2
        input_scale_inv = (1 / input_scale_2).to(paddle.float32)
        weight_scale_interleaved = _process_scale_interleaved(layer.weight_scale)
        free_tensor(layer.input_scale)
        free_tensor(layer.weight_scale_2)

        layer.weight_scale_2 = layer.create_parameter(
            shape=weight_scale_2.shape,
            dtype=weight_scale_2.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.input_scale = layer.create_parameter(
            shape=input_scale_2.shape,
            dtype=input_scale_2.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.alpha = layer.create_parameter(
            shape=alpha.shape,
            dtype=alpha.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.input_scale_inv = layer.create_parameter(
            shape=input_scale_inv.shape,
            dtype=input_scale_inv.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight_scale_interleaved = layer.create_parameter(
            shape=weight_scale_interleaved.shape,
            dtype=weight_scale_interleaved.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight_scale_2.copy_(weight_scale_2, False)
        layer.input_scale.copy_(input_scale_2, False)
        layer.alpha.copy_(alpha, False)
        layer.input_scale_inv.copy_(input_scale_inv, False)
        layer.weight_scale_interleaved.copy_(weight_scale_interleaved, False)

    def apply(
        self,
        layer,
        x,
    ):
        x_m, _ = x.shape
        w_n, _ = layer.weight.shape
        output_shape = [x_m, w_n]
        output_dtype = x.dtype

        # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
        from flashinfer import fp4_quantize

        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)

        assert x_fp4.dtype == paddle.uint8
        assert layer.weight.dtype == paddle.uint8
        assert layer.weight_scale_interleaved.dtype == paddle.float8_e4m3fn
        assert layer.alpha.dtype == paddle.float32

        if self.backend.startswith("flashinfer-"):
            backend = self.backend[len("flashinfer-") :]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}.")

        # shape 恢复到[K//2,N]
        w = layer.weight.T
        # shape 恢复到[K//group_size, N]
        w_scale_interleaved = layer.weight_scale_interleaved.T

        if backend == "cutlass":
            x_scale_interleaved = x_scale_interleaved.view(paddle.uint8)
            w_scale_interleaved = w_scale_interleaved.view(paddle.uint8)
        from flashinfer import mm_fp4 as fp4_gemm

        out = fp4_gemm(x_fp4, w, x_scale_interleaved, w_scale_interleaved, layer.alpha, output_dtype, backend=backend)
        if layer.with_bias:
            out = paddle.add(out, layer.bias)
        assert out.shape == output_shape
        return out


class ModelOptNvFp4FusedMoE(MoEMethodBase):
    """Fused MoE method for Model Optimizer NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    input_scale: paddle.float32, scalar ,
    weight: NVFP4(represented as byte) Shape: [1, X, y/2]
    weight_scale: FP8-E4M3, Shape: [X, Y], aka per block scale,
    weight_scale_2: paddle.float32, scalar,
    Args:
    quant_config: The ModelOpt quantization config.
    moe_config: The MoE configuration.
    layer: The linear layer.
    """

    def __init__(self, quant_config: ModelOptNvFp4Config):
        self.quant_config = quant_config
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.backend = "none"

        if envs.FD_MOE_BACKEND is None:
            # currently support flashinfer-cutlass,  flashinfer-trtllm will support in the future
            self.backend = "flashinfer-cutlass"
        elif envs.FD_MOE_BACKEND.startswith("flashinfer-"):
            self.backend = envs.FD_MOE_BACKEND

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 flashinfer MoE backend found. Please check your platform capability and installtion of FlashInfer."
            )

        logger.info(f"Using {self.backend} for NVFP4 FusedMoE")

    def create_weights(self, layer, **extra_weight_attrs):
        """
        NVFP4 MoE create weight.
        """
        self.up_gate_proj_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            layer.hidden_size // 2,
        ]
        self.down_proj_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size // 2,
        ]
        self.up_gate_proj_scale_shape = self.up_gate_proj_weight_shape[0:2] + [
            layer.hidden_size // self.quant_config.group_size
        ]
        self.down_proj_scale_shape = self.down_proj_weight_shape[0:2] + [
            layer.moe_intermediate_size // self.quant_config.group_size
        ]

        self.weight_scale_dtype = paddle.float8_e4m3fn
        self.weight_dtype = paddle.uint8
        up_gate_proj_weight_name = self.added_weight_attrs[0]
        down_proj_weight_name = self.added_weight_attrs[1]
        up_gate_proj_scale_name = self.added_scale_attrs[0]
        down_proj_scale_name = self.added_scale_attrs[1]
        setattr(
            layer,
            up_gate_proj_weight_name,
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            down_proj_weight_name,
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        # weight_scale
        setattr(
            layer,
            up_gate_proj_scale_name,
            layer.create_parameter(
                shape=self.up_gate_proj_scale_shape,
                dtype=self.weight_scale_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            down_proj_scale_name,
            layer.create_parameter(
                shape=self.down_proj_scale_shape,
                dtype=self.weight_scale_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        # weight_scale_2
        layer.up_gate_proj_weight_scale_2 = layer.create_parameter(
            shape=[layer.num_local_experts, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.down_proj_weight_scale_2 = layer.create_parameter(
            shape=[layer.num_local_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        # input_scale
        layer.up_gate_proj_input_scale = layer.create_parameter(
            shape=[layer.num_local_experts, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.down_proj_input_scale = layer.create_parameter(
            shape=[layer.num_local_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        set_weight_attrs(
            getattr(layer, up_gate_proj_weight_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )
        set_weight_attrs(
            getattr(layer, up_gate_proj_scale_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )

        set_weight_attrs(
            getattr(layer, down_proj_weight_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )
        set_weight_attrs(
            getattr(layer, down_proj_scale_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )

        set_weight_attrs(layer.up_gate_proj_weight_scale_2, {**extra_weight_attrs, "weight_type": "weight_scale_2"})
        set_weight_attrs(layer.down_proj_weight_scale_2, {**extra_weight_attrs, "weight_type": "weight_scale_2"})
        set_weight_attrs(layer.up_gate_proj_input_scale, {**extra_weight_attrs, "weight_type": "input_scale"})
        set_weight_attrs(layer.down_proj_input_scale, {**extra_weight_attrs, "weight_type": "input_scale"})

    def process_weights_after_loading(self, layer):
        """ """

        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13

        # [a, b] = layer.up_gate_proj_weight.split(2, axis=1)
        # layer.up_gate_proj_weight.set_value(paddle.concat([b, a], axis=1))
        # [a, b] = layer.up_gate_proj_weight_scale.split(2, axis=1)
        # layer.up_gate_proj_weight_scale.set_value(paddle.concat([b, a], axis=1))

        up_gate_proj_weight_scale_2 = layer.up_gate_proj_weight_scale_2[:, 0]
        free_tensor(layer.up_gate_proj_weight_scale_2)
        create_parameter_and_copy(layer, name="up_gate_proj_weight_scale_2", weight=up_gate_proj_weight_scale_2)
        up_gate_proj_input_scale = paddle.max(layer.up_gate_proj_input_scale).cast("float32")
        down_proj_input_scale = paddle.max(layer.down_proj_input_scale).cast("float32")

        # Create shared parameters
        create_parameter_and_copy(
            layer, "g1_alphas", (up_gate_proj_input_scale * up_gate_proj_weight_scale_2).cast("float32")
        )
        create_parameter_and_copy(
            layer, "g2_alphas", (down_proj_input_scale * layer.down_proj_weight_scale_2).cast("float32")
        )
        create_parameter_and_copy(
            layer, "up_gate_proj_input_scale_quant", (1 / up_gate_proj_input_scale).cast("float32")
        )
        create_parameter_and_copy(layer, "down_proj_input_scale_quant", (1 / down_proj_input_scale).cast("float32"))

        for name, weight_scale in [
            ("up_gate", layer.up_gate_proj_weight_scale),
            ("down", layer.down_proj_weight_scale),
        ]:
            assert weight_scale.shape[2] % 16 == 0, f"Expected {name}_weight_scale.dim(2) to be divisible by 16"
            assert (
                weight_scale.dtype == paddle.float8_e4m3fn
            ), f"{name} Weight Blockscale must be represented as FP8-E4M3"

        up_gate_proj_blockscale_swizzled = _process_scale_interleaved(layer.up_gate_proj_weight_scale)
        create_parameter_and_copy(
            layer, name="up_gate_proj_blockscale_swizzled", weight=up_gate_proj_blockscale_swizzled
        )
        free_tensor(layer.up_gate_proj_weight_scale)
        layer.up_gate_proj_weight_scale = None
        down_proj_blockscale_swizzled = _process_scale_interleaved(layer.down_proj_weight_scale)
        create_parameter_and_copy(layer, name="down_proj_blockscale_swizzled", weight=down_proj_blockscale_swizzled)
        free_tensor(layer.down_proj_weight_scale)
        layer.down_proj_weight_scale = None
    
    def _run_cutedsl_grouped_masked(self, layer, hidden_states_3d, masked_m):

        if self.backend != "flashinfer-cutedsl":
            raise NotImplementedError("NVFP4 EP backend only supports CuteDSL implementation.")

        masked_m = masked_m.cast(paddle.int32)
        num_experts = int(layer.num_local_experts)

        def _to_expert_scale_vec(scale: paddle.Tensor, name: str) -> paddle.Tensor:
            scale = scale.cast("float32")
            if len(scale.shape) == 0:
                return paddle.ones([num_experts], dtype="float32") * scale
            if len(scale.shape) == 1:
                if scale.shape[0] == num_experts:
                    return scale
                if scale.shape[0] == 1:
                    return paddle.tile(scale, [num_experts])
                raise ValueError(f"{name} shape mismatch: {scale.shape}, expected ({num_experts},)")
            if len(scale.shape) == 2 and scale.shape[1] == 2:
                return scale.max(axis=1).values.cast("float32")
            raise ValueError(f"{name} rank not supported: shape={scale.shape}")

        w1_alpha = _to_expert_scale_vec(layer.g1_alphas, "g1_alphas")
        w2_alpha = _to_expert_scale_vec(layer.g2_alphas, "g2_alphas")
        input_global_scale = _to_expert_scale_vec(layer.up_gate_proj_input_scale_quant, "up_gate_proj_input_scale_quant")
        a2_global_scale = _to_expert_scale_vec(layer.down_proj_input_scale_quant, "down_proj_input_scale_quant")

        n = layer.down_proj_weight.shape[-1] * 2

        if isinstance(hidden_states_3d, tuple) and hidden_states_3d[1] is not None:
            a_q = hidden_states_3d[0].view(paddle.uint8)
            a_q_sf = hidden_states_3d[1].view(paddle.float8_e4m3fn)
            m, k_by_2, _ = a_q.shape
            k = k_by_2 * 2
        else:
            hidden_states = hidden_states_3d[0] if isinstance(hidden_states_3d, tuple) else hidden_states_3d
            _, m, k = hidden_states.shape
            a_q, a_q_sf = scaled_fp4_grouped_quantize(hidden_states, masked_m, input_global_scale)

        ab_dtype = "float4_e2m1fn"
        sf_dtype = "float8_e4m3fn"
        c_dtype = "bfloat16"
        sf_vec_size = 16

        gateup_output = paddle.empty([num_experts, m, n * 2], dtype=paddle.bfloat16).transpose([1, 2, 0])
        grouped_gemm_nt_masked(
            (a_q, a_q_sf),
            (_perm(layer.up_gate_proj_weight, 1, 2, 0), layer.up_gate_proj_blockscale_swizzled),
            gateup_output,
            masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            alpha=w1_alpha.reshape([1, 1, num_experts]),
            alpha_dtype=_get_cute_dtype(w1_alpha),
        )

        diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
            gateup_output.transpose([2, 0, 1]),
            masked_m,
            a2_global_scale,
        )

        out = paddle.empty([num_experts, m, k], dtype=paddle.bfloat16).transpose([1, 2, 0])
        grouped_gemm_nt_masked(
            (diq, diq_sf),
            (_perm(layer.down_proj_weight, 1, 2, 0), layer.down_proj_blockscale_swizzled),
            out,
            masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            alpha=w2_alpha.reshape([1, 1, num_experts]),
            alpha_dtype=_get_cute_dtype(w2_alpha),
        )

        return out.transpose([2, 0, 1])

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None
    ) -> paddle.Tensor:
        
        # 1. top experts and weights
        gate_out = gate(x.cast("float32"))
        topk_idx, topk_weights = self.ep_prefill_runner.moe_select(layer, gate_out)

        hidden_size = x.shape[1]

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)
        
        event = deep_ep.Buffer.capture()

        # 2. ep dispatch
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = self.ep_prefill_runner.dispatch(
            x,
            topk_idx,
            topk_weights,
            expert_alignment=128,
            previous_event=event,
        )

        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()

        # BF16 dispatch without scale or a tuple (FP8 dispatch)
        if isinstance(recv_x, tuple):
            recv_x_value, recv_x_scale = recv_x
        else:
            recv_x_value = recv_x
            recv_x_scale = None

        # 3. compute ffn
        if self.ep_prefill_runner.num_worst_tokens > 0:
            top_k = layer.top_k
            num_local_experts = layer.num_local_experts

            if top_k in (4, 8):
                token_split_factor = 2 if int(os.getenv("USE_TBO", "0")) == 1 else 1
                max_tokens_per_rank = (
                    layer.fd_config.scheduler_config.max_num_batched_tokens
                    // layer.fd_config.parallel_config.tensor_parallel_size
                    // token_split_factor
                )

                if recv_x_scale is None:
                    recv_x_scale = paddle.zeros(
                        [recv_x_value.shape[0], 1], dtype=paddle.int32
                    )
                # premute_scale 没有被使用的原因是 _run_cutedsl_grouped_masked 会在内部量化
                permute_input, permute_scale, permuted_indice_map, token_nums_per_expert = (
                    call_prefill_permute_to_masked_gemm(
                        x=recv_x_value,
                        scale=recv_x_scale,
                        topk_ids=recv_topk_idx,
                        num_local_experts=num_local_experts,
                        max_token_num=layer.ep_size * max_tokens_per_rank,
                    )
                )

                # token_nums_per_expert: [num_local_experts, 1] -> [num_local_experts]
                ffn_out = self._run_cutedsl_grouped_masked(
                    layer, permute_input, token_nums_per_expert.reshape([-1])
                )

                tmp_ffn_out = call_depermute_prefill_combine(
                    x=ffn_out,
                    indice_map=permuted_indice_map,
                    topk_weights=recv_topk_weights,
                    num_worst_tokens=recv_x_value.shape[0],
                )
            else:
                # eb-45的top-k=6，但是call_prefill_permute_to_masked_gemm只支持topk=4 or 8
                # 因此让 ai 用 python 实现支持topk=6的情况， 用于验证是否接入正确。端到端出字正常。
                expert_token_lists = [[] for _ in range(num_local_experts)]
                recv_n = recv_x_value.shape[0]
                recv_topk_idx_numpy = recv_topk_idx.numpy()  # [N_recv, top_k]
                for ti in range(recv_n):
                    for ki in range(top_k):
                        ei = int(recv_topk_idx_numpy[ti, ki])
                        if ei >= 0:
                            expert_token_lists[ei].append(ti)

                token_counts = [len(lst) for lst in expert_token_lists]
                max_expert_tokens = max(token_counts) if token_counts else 0

                # Build blocked input: [num_local_experts, max_expert_tokens, H]
                H = recv_x_value.shape[1]
                if max_expert_tokens > 0:
                    blocked = paddle.zeros(
                        [num_local_experts, max_expert_tokens, H],
                        dtype=recv_x_value.dtype,
                    )
                    for ei, indices in enumerate(expert_token_lists):
                        if indices:
                            blocked[ei, : len(indices)] = recv_x_value[
                                paddle.to_tensor(indices, dtype=paddle.int64)
                            ]
                    masked_m = paddle.to_tensor(token_counts, dtype=paddle.int32)
                    ffn_out_blocked = self._run_cutedsl_grouped_masked(
                        layer, blocked, masked_m
                    )
                    # ffn_out_blocked: [num_local_experts, max_expert_tokens, H]

                    # De-permute: accumulate weighted expert outputs back to
                    # [N_recv, H] in-place.
                    tmp_ffn_out = paddle.zeros(
                        [recv_n, H], dtype=paddle.float32
                    )
                    recv_topk_weights_np = recv_topk_weights.cast(paddle.float32).numpy()
                    ffn_out_f32 = ffn_out_blocked.cast(paddle.float32)
                    for ti in range(recv_n):
                        for ki in range(top_k):
                            ei = int(recv_topk_idx_numpy[ti, ki])
                            if ei >= 0:
                                slot = expert_token_lists[ei].index(ti)
                                w = float(recv_topk_weights_np[ti, ki])
                                tmp_ffn_out[ti] += w * ffn_out_f32[ei, slot]
                    tmp_ffn_out = tmp_ffn_out.cast(paddle.bfloat16)
                else:
                    tmp_ffn_out = paddle.zeros([recv_n, H], dtype=paddle.bfloat16)

        else:
            tmp_ffn_out = paddle.empty([0, hidden_size], paddle.bfloat16)
            logger.warning("num_worst_tokens is disabled!")

        if shared_experts is not None:
            s_x = shared_experts(x)

        # 4. EP combine
        event = deep_ep.Buffer.capture()
        tmp_ffn_out, event = self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights, event)

        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()

        if shared_experts is not None:
            tmp_ffn_out += s_x

        return tmp_ffn_out

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:
        if layer.fd_config.parallel_config.use_internode_ll_two_stage:
            raise NotImplementedError("NVFP4 CuteDSL EP decode does not support DeepEP two-stage low-latency.")

        gate_out = gate(x.cast("float32"))
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        recv_x, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x,
            topk_idx,
            topk_weights,
            use_fp8=False,
        )

        ffn_out = self._run_cutedsl_grouped_masked(layer, recv_x, token_nums_per_expert)

        if shared_experts is not None:
            s_x = shared_experts(x)
        
        out = self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

        if shared_experts is not None:
            out += s_x

        return out

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        pass

    def apply(
        self,
        layer,
        x,
        gate,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ):
        """
        flashinfer nvfp4 fusedmoe for Model Optimizer
        """
        if self.backend == "flashinfer-cutlass":
            gate_out = gate(x.cast("float32"))
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                layer.top_k,
                True,  # apply_norm_weight,
                False,
            )

            if topk_ids_hookfunc is not None:
                topk_ids_hookfunc(topk_ids)

            output_dtype = x.dtype
            x_sf = None
            output = paddle.empty_like(x)

            # flashinfer cutlass
            from flashinfer.fused_moe import (
                cutlass_fused_moe as flashinfer_cutlass_fused_moe,
            )

            _ = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids.to(paddle.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=getattr(layer, self.added_weight_attrs[0]).view(paddle.long),
                fc2_expert_weights=getattr(layer, self.added_weight_attrs[1]).view(paddle.long),
                output_dtype=output_dtype,
                input_sf=x_sf,
                quant_scales=[
                    layer.up_gate_proj_input_scale_quant,
                    layer.up_gate_proj_blockscale_swizzled.view(paddle.int32),
                    layer.g1_alphas,
                    layer.down_proj_input_scale_quant,
                    layer.down_proj_blockscale_swizzled.view(paddle.int32),
                    layer.g2_alphas,
                ],
                ep_size=layer.ep_size,
                ep_rank=layer.ep_rank,
                tp_size=layer.tp_size,
                tp_rank=layer.tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                output=output,
            )

            return output
        
        elif self.backend == "flashinfer-cutedsl":
            if layer.fd_config.model_config.moe_phase.phase == "prefill":
                return self.apply_ep_prefill(layer, x, gate, topk_ids_hookfunc=topk_ids_hookfunc, shared_experts=shared_experts)
            else:
                return self.apply_ep_decode(layer, x, gate, topk_ids_hookfunc=topk_ids_hookfunc, shared_experts=shared_experts)
        
        # flashinfer-trtllm
        return paddle.empty_like(x)
