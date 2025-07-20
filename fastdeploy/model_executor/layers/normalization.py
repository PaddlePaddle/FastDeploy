"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Callable, Dict, Optional

import numpy as np
import paddle
from paddle import nn

from fastdeploy.platforms import current_platform

if current_platform.is_gcu():
    from fastdeploy.model_executor.ops.gcu import fused_add_rms_norm, rms_norm
else:
    from paddle.incubate.nn.functional import fused_layer_norm, fused_rms_norm

from fastdeploy.config import FDConfig

from .utils import get_tensor


class RMSNorm(nn.Layer):
    """
    Normalization layer.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        hidden_size: int,
        eps: float = 1e-5,
        prefix: str = "",
        bias: paddle.Tensor = None,
        quant_scale: float = None,
        begin_norm_axis: int = 1,
    ) -> None:
        """
        Initializes the RMSNormalization layer.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            hidden_size (int) : size of hidden state.
            eps:(float, optional): Small value added to the variance to avoid division by zero. Defaults to 1e-5.
            prefix(str,optional):The name of current layer. Defaults to "".
            bias (paddle.Tensor,optional): Initial bias value for the linear layer (if used). Defaults to None.
            quant_scale(float,optional):Quantization scale, used in quantization scenarios. Defaults to -1, indicating no quantization.
            begin_norm_axis (int, optional): The axis along which to perform normalization. Defaults to 1.

        Raises:
            NotImplementedError: If the specified norm_type is not supported.
        """
        super().__init__()
        self.fd_config = fd_config
        self.prefix: str = prefix
        self.hidden_size: int = hidden_size
        if len(prefix) == 0:
            self.weight_key: Optional[str] = None
        else:
            self.weight_key: Optional[str] = f"{prefix}.weight"
        self.with_weight: bool = self.weight_key is not None
        self.eps: float = eps
        if current_platform.is_gcu():
            self.norm_func: Callable = fused_add_rms_norm
        else:
            self.norm_func: Callable = fused_rms_norm
        self.bias: Optional[paddle.Tensor] = bias
        self.quant_scale: Optional[float] = quant_scale
        self._dtype: str = self._helper.get_default_dtype()
        self._norm_weight_dtype: str = self._dtype
        self.quant_round_type: int = self.fd_config.quant_config.quant_round_type if fd_config.quant_config else 0
        self.quant_max_bound: int = self.fd_config.quant_config.quant_max_bound if fd_config.quant_config else 0
        self.quant_min_bound: int = self.fd_config.quant_config.quant_min_bound if fd_config.quant_config else 0
        self.begin_norm_axis: int = begin_norm_axis

        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """

        self.weight = None
        if self.with_weight:
            self.weight = self.create_parameter(
                shape=[self.hidden_size],
                default_initializer=nn.initializer.Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )

    def load_state_dict(self, state_dict: Dict[str, paddle.Tensor | np.ndarray]):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        # weight
        weight_tensor = paddle.cast(get_tensor(state_dict.pop(self.weight_key)), self._norm_weight_dtype)
        self.weight.set_value(weight_tensor)

    def forward(self, x, residual_input: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """
        Defines the forward computation of the layer.

        Args:
            x (paddle.Tensor): Input tensor to be normalized.
            residual_input (paddle.Tensor, optional): Residual input tensor for residual connection.
                Defaults to None. If provided, the normalization layer will also return the residual
                output for further computation.

        Returns:
            paddle.Tensor or tuple of paddle.Tensor:
                - If `residual_input` is None, returns the normalized output tensor.
                - If `residual_input` is provided, returns a tuple of (normalized_output, residual_output).
                  The `residual_output` is the result of applying the normalization and possibly other
                  operations (like linear transformation) on the `residual_input`.
        """
        if current_platform.is_gcu():
            if residual_input is None:
                return rms_norm(x, self.weight, self.eps)
            norm_out = self.norm_func(x, residual_input, self.weight, self.eps)
        else:
            norm_out = self.norm_func(
                x,
                norm_weight=self.weight,
                norm_bias=None,
                epsilon=self.eps,
                begin_norm_axis=self.begin_norm_axis,
                bias=self.bias,
                residual=residual_input,
                quant_scale=(-1 if self.quant_scale is None else self.quant_scale),
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )
        if residual_input is not None:
            return norm_out[0], norm_out[1]
        else:
            return norm_out[0]


class LayerNorm(nn.Layer):
    """
    Initializes the LayerNormalization layer
    """

    def __init__(
        self,
        fd_config: FDConfig,
        hidden_size: int,
        eps: float = 1e-5,
        prefix="",
        bias: paddle.Tensor = None,
        quant_scale: float = None,
        with_bias: bool = False,
    ):
        """
        Initializes the normalization layer.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            hidden_size (int) : size of hidden state.
            eps:(float, optional): Small value added to the variance to avoid division by zero. Defaults to 1e-5.
            prefix (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            bias (float, optional): Initial bias value for the linear layer (if used). Defaults to None.
            quant_scale(float,optional):Quantization scale, used in quantization scenarios. Defaults to -1, indicating no quantization.
            with_bias (bool):Whether to include bias or not. Defaults to False.
        Raises:
            NotImplementedError: If the specified norm_type is not supported.
        """
        super().__init__()
        self.fd_config = fd_config
        self.prefix: str = prefix
        self.hidden_size: int = hidden_size
        if len(prefix) == 0:
            self.weight_key: Optional[str] = None
        else:
            self.weight_key: Optional[str] = f"{prefix}.weight"
        self.with_weight: bool = self.weight_key is not None
        self.bias_key: str = f"{prefix}.bias"
        self.with_bias: bool = with_bias
        self.eps: float = eps
        self.quant_scale: float = quant_scale
        if current_platform.is_gcu():
            self.norm_func: Callable = paddle.nn.functional.layer_norm
        else:
            self.norm_func: Callable = fused_layer_norm
        self.bias: Optional[paddle.Tensor] = bias
        self._dtype: str = self._helper.get_default_dtype()
        self._norm_weight_dtype: str = "float32"

        self.quant_round_type: int = self.fd_config.quant_config.quant_round_type if fd_config.quant_config else 0
        self.quant_max_bound: int = self.fd_config.quant_config.quant_max_bound if fd_config.quant_config else 0
        self.quant_min_bound: int = self.fd_config.quant_config.quant_min_bound if fd_config.quant_config else 0

        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """

        self.weight = None
        if self.with_weight:
            self.weight = self.create_parameter(
                shape=[self.hidden_size],
                default_initializer=nn.initializer.Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
        self.bias = None
        if self.with_bias:
            self.bias = self.create_parameter(
                shape=[self.hidden_size],
                is_bias=True,
                dtype=self._norm_weight_dtype,
            )

    def load_state_dict(self, state_dict: Dict[str, paddle.Tensor | np.ndarray]):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        # weight
        weight_tensor = paddle.cast(get_tensor(state_dict.pop(self.weight_key)), self._norm_weight_dtype)
        self.weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.cast(
                get_tensor(state_dict.pop(self.bias_key)),
                self._norm_weight_dtype,
            )
            self.bias.set_value(bias_tensor)

    def forward(self, x, residual_input: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """
        Defines the forward computation of the layer.

        Args:
            x (paddle.Tensor): Input tensor to be normalized.
            residual_input (paddle.Tensor, optional): Residual input tensor for residual connection.
                Defaults to None. If provided, the normalization layer will also return the residual
                output for further computation.

        Returns:
            paddle.Tensor or tuple of paddle.Tensor:
                - If `residual_input` is None, returns the normalized output tensor.
                - If `residual_input` is provided, returns a tuple of (normalized_output, residual_output).
                  The `residual_output` is the result of applying the normalization and possibly other
                  operations (like linear transformation) on the `residual_input`.
        """
        if current_platform.is_iluvatar():
            if self.weight is None and self.bias is None:
                out = x
                if self.bias is not None:
                    out += self.bias
                if residual_input is not None:
                    out += residual_input
                    return out, out
                else:
                    return out
            else:
                raise NotImplementedError("Iluvatar does not support yet!")

        if current_platform.is_gcu():
            if residual_input is not None:
                y = x + residual_input
                out = self.norm_func(
                    x=y,
                    normalized_shape=y.shape[1:],
                    weight=self.weight,
                    bias=self.bias,
                    epsilon=self.eps,
                )
                return out, y
            else:
                out = self.norm_func(
                    x=x,
                    normalized_shape=x.shape[1:],
                    weight=self.weight,
                    bias=self.bias,
                    epsilon=self.eps,
                )
                return out
        else:
            norm_out = self.norm_func(
                x,
                norm_weight=self.weight,
                norm_bias=self.bias,
                epsilon=self.eps,
                begin_norm_axis=1,
                bias=self.bias,
                residual=residual_input,
                quant_scale=(-1 if self.quant_scale is None else self.quant_scale),
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )
        if residual_input is not None:
            return norm_out[0], norm_out[1]
        else:
            return norm_out[0]
