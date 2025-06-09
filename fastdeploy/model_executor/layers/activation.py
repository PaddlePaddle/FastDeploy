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

# cipher_token=WjI1fQOvhN  # do not edit this line
from paddle import nn
from paddle.incubate.nn.functional import fused_bias_act

from fastdeploy.config import LLMConfig
from fastdeploy.platforms import current_platform


class SiluAndMul(nn.Layer):
    """
    SiluAndMul Layer
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        bias=None,
        act_method="gelu",
        dequant_scales=None,
        shift=None,
        smooth=None,
        quant_scale=-1,
    ):
        """
        Initialize the activation layer with optional parameters for quantization, bias,
        activation method, and more.

        Args:
            llm_config (Any): Arguments related to inference, including quantization
                settings.
            bias (Optional[Tensor]): Optional bias term to be added to the output.
            act_method (str, optional): Activation method to be applied.
                Defaults to "gelu".
            dequant_scales (Optional[List[float]]): Dequantization scales, used in
                quantization scenarios.
            shift (Optional[float]): Shift factor, used in quantization scenarios.
            smooth (Optional[float]): Smoothing factor, used for specific activation
                functions.
            quant_scale (float, optional): Quantization scale, used in quantization
                scenarios. Defaults to -1, indicating no quantization.

        Raises:
            ValueError: If the default data type is not supported (only float32, float16,
                and bfloat16 are supported).
        """
        super().__init__()

        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError

        self.bias = bias
        if act_method == "silu":
            act_method = "swiglu"

        self.act_method = act_method
        self.dequant_scales = dequant_scales
        self.shift = shift
        self.smooth = smooth
        self.quant_scale = quant_scale
        self.quant_round_type = llm_config.quant_config.quant_round_type
        self.quant_max_bound = llm_config.quant_config.quant_max_bound
        self.quant_min_bound = llm_config.quant_config.quant_min_bound

        self._dtype = self._helper.get_default_dtype()
        if self._dtype == "bfloat16":
            self._fuse_kernel_compute_dtype = "bf16"
        elif self._dtype == "float16":
            self._fuse_kernel_compute_dtype = "fp16"
        elif self._dtype == "float32":
            self._fuse_kernel_compute_dtype = "fp32"
        else:
            raise ValueError(f"Just support float32, float16 and \
                    bfloat16 as default dtype, but received {self._dtype}")

        # fp8 is not support smooth quantization
        if "float8" in llm_config.model_config.act_dtype:
            self.dequant_scales = None
            self.shift = None
            self.smooth = None

    def forward_cuda(self, x):
        """
        Forward propagation of the custom activation layer.

        Args:
            x (Tensor): Input tensor to the activation layer.

        Returns:
            Tensor: Output tensor.
        """
        return fused_bias_act(
            x,
            bias=self.bias,
            act_method=self.act_method,
            compute_dtype=self._fuse_kernel_compute_dtype,
            dequant_scales=self.dequant_scales,
            shift=self.shift,
            smooth=self.smooth,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )
