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

from typing import Optional

import paddle
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.utils import (
    create_parameter_and_copy,
    free_tensor,
    set_weight_attrs,
)

from .quant_base import QuantConfigBase, QuantMethodBase

paddle.compat.enable_torch_proxy(scope={"flashinfer"})


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


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

        return None


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
        extra_weight_attrs["output_dim"] = not extra_weight_attrs["output_dim"]
        weight_shape = layer.weight_shape[::-1]
        weight_shape[1] = weight_shape[1] // 2
        layer.weight_dtype = "uint8"
        input_scale_shape = [1]
        weight_scale_shape = [layer.weight_shape[::-1][0], layer.weight_shape[::-1][1] // self.quant_config.group_size]
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
        layer.weight_scale_2 = layer.create_parameter(
            shape=weight_scale_2_shape,
            dtype=paddle.float32,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
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

    def process_weights_after_loading(self, layer) -> None:
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
            padded_scales = padded_scales.contiguous().to(paddle.device.get_device())
            padded_scales = (
                padded_scales.reshape(M_padded, K_padded)
                if scale_dim == 2
                else padded_scales.reshape(B, M_padded, K_padded)
            )
            return padded_scales

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

        w = layer.weight.T
        w_scale_interleaved = layer.weight_scale_interleaved.T

        if backend == "cutlass":
            x_scale_interleaved = x_scale_interleaved.view(paddle.uint8)
            w_scale_interleaved = w_scale_interleaved.view(paddle.uint8)
        from flashinfer import mm_fp4 as fp4_gemm

        out = fp4_gemm(x_fp4, w, x_scale_interleaved, w_scale_interleaved, layer.alpha, output_dtype, backend=backend)
        if layer.with_bias:
            out = paddle.add(out, layer.bias)
        return out.view(*output_shape)


class ModelOptNvFp4FusedMoE(QuantMethodBase):
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
        self.quant_config = quant_config
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
        Triton MoE create weight process.
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
        self.up_gate_proj_scale_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            layer.hidden_size // self.quant_config.group_size,
        ]
        self.down_proj_scale_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size // self.quant_config.group_size,
        ]

        self.weight_scale_dtype = paddle.float8_e4m3fn
        self.weight_dtype = paddle.uint8
        self.added_scale_attrs = ["up_gate_proj_weight_scale", "down_proj_weight_scale"]
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

        set_weight_attrs(
            layer.up_gate_proj_weight_scale_2,
            {**extra_weight_attrs, "weight_type": "weight_scale_2"},
        )
        set_weight_attrs(layer.down_proj_weight_scale_2, {**extra_weight_attrs, "weight_type": "weight_scale_2"})
        set_weight_attrs(layer.up_gate_proj_input_scale, {**extra_weight_attrs, "weight_type": "input_scale"})
        set_weight_attrs(layer.down_proj_input_scale, {**extra_weight_attrs, "weight_type": "input_scale"})

    def swizzle_blockscale(self, scale):
        assert scale.dtype == paddle.float8_e4m3fn
        # Pad and blockwise interleave weight_scale
        scale_dim = len(scale.shape)
        if len(scale.shape) == 2:
            scale = scale.unsqueeze(0)
        assert len(scale.shape) == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = paddle.empty([B, M_padded, K_padded], dtype=scale.dtype)
        padded_scale[:B, :M, :K].copy_(scale)
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().to(paddle.device.get_device())
        return (
            swizzled_scale.reshape(M_padded, K_padded)
            if scale_dim == 2
            else swizzled_scale.reshape(B, M_padded, K_padded)
        )

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        # 目前默认给True
        return True

    def process_weights_after_loading(self, layer):
        """ """
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

        up_gate_proj_blockscale_swizzled = self.swizzle_blockscale(layer.up_gate_proj_weight_scale)
        free_tensor(layer.up_gate_proj_weight_scale)
        layer.up_gate_proj_weight_scale = None
        create_parameter_and_copy(
            layer, name="up_gate_proj_blockscale_swizzled", weight=up_gate_proj_blockscale_swizzled
        )
        down_proj_blockscale_swizzled = self.swizzle_blockscale(layer.down_proj_weight_scale)
        free_tensor(layer.down_proj_weight_scale)
        layer.down_proj_weight_scale = None
        create_parameter_and_copy(layer, name="down_proj_blockscale_swizzled", weight=down_proj_blockscale_swizzled)

    def apply(
        self,
        layer,
        x,
        gate,
        topk_ids_hookfunc=None,
    ):
        """
        flashinfer nvfp4 fusedmoe for Model Optimizer
        """
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

        if self.backend == "flashinfer-cutlass":
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

        # flashinfer-trtllm
        return output
