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
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.flashinfer import has_flashinfer
from fastdeploy.model_executor.layers.moe import FusedMoE

from .quant_base import QuantConfigBase, QuantMethodBase

if has_flashinfer():
    from flashinfer import fp4_quantize as scaled_fp4_quant  # need to use vllm version
    from flashinfer import mm_fp4 as fp4_gemm


def swizzle_blockscale(scale: paddle.Tensor) -> paddle.Tensor:
    """
    Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Parameters
    ----------
    scale: paddle.Tensor

    Returns
    -------
    torch.Tensor
        The swizzled tensor with the same logical shape as *scale*.
    """
    assert scale.dtype == paddle.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in " "paddle.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    def _round_up(x: int, m: int) -> int:
        return (x + m - 1) // m * m

    M_padded = _round_up(M, 128)
    K_padded = _round_up(K, 4)

    padded = paddle.zeros((B, M_padded, K_padded), dtype=scale.dtype, device=scale.place)
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


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

    def name(self) -> str:
        return "modelopt_fp4"

    @classmethod
    def from_config(cls, config: dict) -> "ModelOptNvFp4Config":
        if "quantization" in config:
            # Traditional ModelOpt format:
            # {"quantization": {"quant_algo": "..."}}
            quant_config = cls.get_from_keys(config, ["quantization"])
            if not isinstance(quant_config, dict):
                raise ValueError("Expected 'quantization' to be a dictionary in config")

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
        else:
            raise ValueError(
                "Missing 'quantization' section in config. Please make sure your model is exported using FastDeploy."
            )

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
        # skip_layer = self.is_layer_excluded(prefix)
        if isinstance(layer, FusedMoE):
            # if skip_layer:
            #     return None
            return ModelOptNvFp4FusedMoE(self, layer.moe_config, layer)
        else:
            # LinearBase
            # if skip_layer:
            #     return UnquantizedLinearMethod()
            # Check if this is a vision model layer that should not be quantized
            # if "vision_tower" in prefix or "vision_model" in prefix:
            #     return UnquantizedLinearMethod()
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
            if has_flashinfer():
                self.backend = "flashinfer-cutlass"
        elif envs.VLLM_NVFP4_GEMM_BACKEND.startswith("flashinfer-"):
            self.backend = envs.FD_NVFP4_GEMM_BACKEND
            assert has_flashinfer(), f"FlashInfer is required for {self.backend}"

        if self.backend == "none":
            raise ValueError("No valid NVFP4 GEMM backend found. " "Please check your platform capability.")

        logger.info(f"Using {self.backend} for NVFP4 GEMM")

    def create_weights(
        self,
        layer,
        **extra_weight_attrs,
    ):
        return

    def process_weights_after_loading(self, layer) -> None:
        return

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
        x_fp4, x_scale_interleaved = scaled_fp4_quant(x, layer.input_scale_inv)

        assert x_fp4.dtype == paddle.uint8
        assert x_scale_interleaved.dtype == paddle.float8_e4m3fn
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
        out = fp4_gemm(x_fp4, w, x_scale_interleaved, w_scale_interleaved, layer.alpha, output_dtype, backend=backend)

        if layer.with_bias:
            out = paddle.add(out, layer.bias)
        return out.view(*output_shape)


class ModelOptNvFp4FusedMoE:
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

    def __init__(self):
        pass
