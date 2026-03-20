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

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase
from fastdeploy.model_executor.utils import (
    create_parameter_and_copy,
    free_tensor,
    set_weight_attrs,
)

from .quant_base import QuantConfigBase, QuantMethodBase

paddle.compat.enable_torch_proxy(scope={"flashinfer"})

from fastdeploy.model_executor.layers.moe.flashinfer_cutedsl_moe import (
    flashinfer_cutedsl_moe_masked,
)


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
        logger.info(f"-----{self.backend} -----")

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
        # logger.info(f"self.down_proj_weight_shaoe前面:{self.down_proj_scale_shape}")
        self.up_gate_proj_scale_shape = self.up_gate_proj_weight_shape[0:2] + [
            layer.hidden_size // self.quant_config.group_size
        ]
        self.down_proj_scale_shape = self.down_proj_weight_shape[0:2] + [
            layer.moe_intermediate_size // self.quant_config.group_size
        ]
        # logger.info(f"后面down:{self.down_proj_scale_shape}")

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

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        # 目前默认给True
        return False

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

        # logger.info(f"layer.down_proj.weight_scale:{layer.down_proj_weight_scale}")
        for name, weight_scale in [
            ("up_gate", layer.up_gate_proj_weight_scale),
            ("down", layer.down_proj_weight_scale),
        ]:

            if weight_scale.shape[2] % 4 != 0:
                logger.warning(
                    "NVFP4 %s_weight_scale K' not multiple of 4: shape=%s, group_size=%s",
                    name,
                    tuple(weight_scale.shape),
                    getattr(self.quant_config, "group_size", None),
                )

        # up_gate_proj_blockscale_swizzled = _process_scale_interleaved(layer.up_gate_proj_weight_scale)
        up_gate_proj_blockscale_swizzled = layer.up_gate_proj_weight_scale
        create_parameter_and_copy(
            layer, name="up_gate_proj_blockscale_swizzled", weight=up_gate_proj_blockscale_swizzled
        )
        free_tensor(layer.up_gate_proj_weight_scale)
        layer.up_gate_proj_weight_scale = None

        # down_proj_blockscale_swizzled = _process_scale_interleaved(layer.down_proj_weight_scale)
        down_proj_blockscale_swizzled = layer.down_proj_weight_scale
        create_parameter_and_copy(layer, name="down_proj_blockscale_swizzled", weight=down_proj_blockscale_swizzled)
        free_tensor(layer.down_proj_weight_scale)
        layer.down_proj_weight_scale = None

        logger.info(f"up_gate_proj_input_scale:{up_gate_proj_input_scale}")
        logger.info(f"up_gate_proj_weight_scale_2:{up_gate_proj_weight_scale_2}")
        logger.info(f"down_proj_input_scale:{down_proj_input_scale}")
        logger.info(f"down_proj_weight_scale_2:{layer.down_proj_weight_scale_2}")

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
        input_global_scale = _to_expert_scale_vec(
            layer.up_gate_proj_input_scale_quant, "up_gate_proj_input_scale_quant"
        )
        a2_global_scale = _to_expert_scale_vec(layer.down_proj_input_scale_quant, "down_proj_input_scale_quant")

        if isinstance(hidden_states_3d, tuple) and hidden_states_3d[1] is not None:
            hidden_states = hidden_states_3d
        else:
            hidden_states = (hidden_states_3d[0] if isinstance(hidden_states_3d, tuple) else hidden_states_3d, None)

        return flashinfer_cutedsl_moe_masked(
            hidden_states=hidden_states,
            input_global_scale=input_global_scale,
            w1=layer.up_gate_proj_weight,
            w1_blockscale=layer.up_gate_proj_blockscale_swizzled,
            w1_alpha=w1_alpha,
            w2=layer.down_proj_weight,
            a2_global_scale=a2_global_scale,
            w2_blockscale=layer.down_proj_blockscale_swizzled,
            w2_alpha=w2_alpha,
            masked_m=masked_m,
        )

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:

        logger.info("------apply_ep_prefill nvfp4 flashinfer-cutedsl: fallback to apply_tp-------")
        return self.apply_tp(layer, x, gate, topk_ids_hookfunc=topk_ids_hookfunc)

        # if layer.fd_config.parallel_config.use_internode_ll_two_stage:
        #     raise NotImplementedError("NVFP4 CuteDSL EP prefill does not support DeepEP two-stage low-latency.")

        # from fastdeploy.model_executor.layers.moe.ep import deep_ep

        # # 1. Gate routing
        # gate_out = gate(x.cast("float32"))
        # topk_idx, topk_weights = self.ep_prefill_runner.moe_select(layer, gate_out)

        # if topk_ids_hookfunc is not None:
        #     topk_ids_hookfunc(topk_ids=topk_idx)

        # # 2. DeepEP batch dispatch (BF16, no FP8 pre-quantization).
        # # Returns:
        # #   recv_x:                       [N_recv, H]     unique received tokens
        # #   recv_topk_idx:                [N_recv, top_k] LOCAL expert indices per token
        # #   recv_topk_weights:            [N_recv, top_k] expert weights per token
        # #   recv_num_tokens_per_expert:   padded buffer sizes per local expert (multiples of 128)
        # event = deep_ep.Buffer.capture()
        # (
        #     recv_x,
        #     recv_topk_idx,
        #     recv_topk_weights,
        #     recv_num_tokens_per_expert_list,
        #     handle,
        #     event,
        # ) = self.ep_prefill_runner.dispatch(
        #     x,
        #     topk_idx,
        #     topk_weights,
        #     expert_alignment=128,
        #     previous_event=event,
        # )

        # if self.ep_prefill_runner.ep_engine.async_finish:
        #     event.current_stream_wait()

        # hidden_size = x.shape[-1]
        # num_local_experts = int(layer.num_local_experts)

        # # Unwrap tuple if dispatch returns (x_value, scale)
        # if isinstance(recv_x, tuple):
        #     recv_x = recv_x[0]

        # # Padded per-expert buffer sizes (for max_m computation).
        # padded_counts = [int(c) for c in recv_num_tokens_per_expert_list]
        # max_m = max(padded_counts) if padded_counts else 1
        # N_received = recv_x.shape[0]

        # if N_received > 0:
        #     # 3. Build expert-major hidden_states_3d [E, max_m, H].
        #     #
        #     # recv_topk_idx may be [N, top_k] (each token has multiple assignments)
        #     # or [N] (one assignment per token).  Values are LOCAL expert indices.
        #     if recv_topk_idx.ndim == 2:
        #         top_k_local = recv_topk_idx.shape[1]
        #         expert_flat = recv_topk_idx.reshape([-1]).cast(paddle.int64)       # [N*k]
        #         token_flat  = (
        #             paddle.arange(N_received, dtype=paddle.int64)
        #             .unsqueeze(1)
        #             .expand([N_received, top_k_local])
        #             .reshape([-1])
        #         )                                                                    # [N*k]
        #     else:
        #         expert_flat = recv_topk_idx.cast(paddle.int64)                     # [N]
        #         token_flat  = paddle.arange(N_received, dtype=paddle.int64)        # [N]

        #     # Sort pairs by (expert, token) to get expert-major ordering.
        #     sort_key   = expert_flat * (N_received + 1) + token_flat
        #     sort_order = paddle.argsort(sort_key)
        #     sorted_token = token_flat[sort_order]   # token index per sorted pair

        #     # Actual (unpadded) token count per local expert
        #     real_counts = paddle.bincount(
        #         expert_flat, minlength=num_local_experts
        #     ).cast(paddle.int32)                                                    # [E]

        #     # Fill expert e's rows in hidden_states_3d with its assigned tokens
        #     hidden_states_3d = paddle.zeros(
        #         [num_local_experts, max_m, hidden_size], dtype=recv_x.dtype
        #     )
        #     cum = 0
        #     for e in range(num_local_experts):
        #         cnt = int(real_counts[e])
        #         if cnt > 0:
        #             hidden_states_3d[e, :cnt] = recv_x[sorted_token[cum : cum + cnt]]
        #         cum += cnt

        #     # 4. NVFP4 grouped GEMM: [E, max_m, H] → [E, max_m, H]
        #     ffn_out = self._run_cutedsl_grouped_masked(layer, hidden_states_3d, real_counts)

        #     # 5. Weighted sum across experts → tmp_ffn_out [N_recv, H].
        #     #
        #     # combine() expects tmp_ffn_out.shape[0] == N_received (one row per unique
        #     # received token, with expert contributions already combined via weights).
        #     #
        #     # For each expert e and its assigned tokens (sorted_token[cum:cum+cnt]):
        #     #   weight for expert e on token t = recv_topk_weights[t, k] where
        #     #   recv_topk_idx[t, k] == e.  We select it via masked dot product.
        #     tmp_ffn_out = paddle.zeros([N_received, hidden_size], dtype=ffn_out.dtype)
        #     cum = 0
        #     for e in range(num_local_experts):
        #         cnt = int(real_counts[e])
        #         if cnt > 0:
        #             tokens_e = sorted_token[cum : cum + cnt]   # [cnt] token indices
        #             if recv_topk_idx.ndim == 2:
        #                 # mask-select weight for expert e at each token
        #                 idx_e = recv_topk_idx[tokens_e]        # [cnt, top_k]
        #                 w_e   = recv_topk_weights[tokens_e]    # [cnt, top_k]
        #                 # Exactly one column per row satisfies idx_e[:,k] == e
        #                 w_scalar = (
        #                     (idx_e == e).cast(w_e.dtype) * w_e
        #                 ).sum(axis=1)                          # [cnt]
        #             else:
        #                 w_scalar = recv_topk_weights[tokens_e] # [cnt]
        #             # Scatter-add weighted expert output to each token's slot
        #             tmp_ffn_out[tokens_e] = (
        #                 tmp_ffn_out[tokens_e]
        #                 + w_scalar.unsqueeze(-1) * ffn_out[e, :cnt]
        #             )
        #         cum += cnt
        # else:
        #     tmp_ffn_out = paddle.zeros([N_received, hidden_size], dtype=paddle.bfloat16)

        # # 6. DeepEP combine: all-to-all back to originating ranks.
        # event = deep_ep.Buffer.capture()
        # tmp_ffn_out, event = self.ep_prefill_runner.combine(
        #     tmp_ffn_out, handle, recv_topk_weights, event
        # )

        # if self.ep_prefill_runner.ep_engine.async_finish:
        #     event.current_stream_wait()

        # return tmp_ffn_out

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:

        logger.info("----------apply_ep_decode nvfp4 flashinfer-cutedsl-------")

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
        return self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:

        gate_out = gate(x.cast("float32"))
        topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            True,
            False,
        )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids)

        if self.backend == "flashinfer-cutedsl":
            num_local_experts = layer.num_local_experts
            hidden_dim = x.shape[1]
            flat_topk_ids = topk_ids.reshape([-1])

            masked_m = paddle.zeros([num_local_experts], dtype=paddle.int32)
            for i in range(num_local_experts):
                masked_m[i] = int((flat_topk_ids == i).sum())

            max_m = max(1, int(masked_m.max()))
            hidden_states_3d = paddle.zeros([num_local_experts, max_m, hidden_dim], dtype=x.dtype)
            for i in range(num_local_experts):
                count = int(masked_m[i])
                if count > 0:
                    token_indices = paddle.nonzero(flat_topk_ids == i).squeeze(-1)
                    src_indices = token_indices // layer.top_k
                    hidden_states_3d[i, :count, :] = x[src_indices]

            ffn_out = self._run_cutedsl_grouped_masked(layer, hidden_states_3d, masked_m)

            bs = x.shape[0]
            output = paddle.zeros([bs, hidden_dim], dtype=ffn_out.dtype)
            flat_weights = topk_weights.reshape([-1])
            for i in range(num_local_experts):
                count = int(masked_m[i])
                if count == 0:
                    continue
                token_indices = paddle.nonzero(flat_topk_ids == i).squeeze(-1)
                batch_indices = (token_indices // layer.top_k).tolist()
                weights_list = flat_weights[token_indices].cast(ffn_out.dtype)
                expert_out = ffn_out[i, :count, :]
                for j, b in enumerate(batch_indices):
                    output[b] += expert_out[j] * weights_list[j]
        elif self.backend == "flashinfer-cutlass":
            from flashinfer.fused_moe import (
                cutlass_fused_moe as flashinfer_cutlass_fused_moe,
            )

            output_dtype = x.dtype
            x_sf = None
            output = paddle.empty_like(x)

            output = flashinfer_cutlass_fused_moe(
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
        if layer.ep_size > 1:
            # max_decode_tokens = self.ep_decoder_runner.ep_engine.buffer.num_max_dispatch_tokens_per_rank
            # if x.shape[0] <= max_decode_tokens:
            return self.apply_ep_decode(
                layer,
                x,
                gate,
                topk_ids_hookfunc=topk_ids_hookfunc,
                shared_experts=shared_experts,
            )
        #     else:
        #         logger.info(
        #             f"apply_ep_decode: token count {x.shape[0]} > max_decode_tokens {max_decode_tokens}, "
        #             f"fallback to apply_tp"
        #         )
        #         return self.apply_tp(
        #             layer,
        #             x,
        #             gate,
        #             topk_ids_hookfunc=topk_ids_hookfunc,
        #             shared_experts=shared_experts,
        #         )
        # else:
        #     return self.apply_tp(
        #         layer,
        #         x,
        #         gate,
        #         topk_ids_hookfunc=topk_ids_hookfunc,
        #         shared_experts=shared_experts,
        #     )
