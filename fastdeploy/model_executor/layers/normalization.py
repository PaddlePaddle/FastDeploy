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

import paddle
from paddle import nn
from paddle.incubate.nn.functional import fused_layer_norm, fused_rms_norm

from .utils import get_tensor


class RMSNorm(nn.Layer):
    """
    Normalization layer.
    """

    def __init__(
        self,
        llm_config,
        hidden_size,
        eps=1e-5,
        prefix="",
        linear_bias=None,
        quant_scale=None,
    ):
        """
        Initializes the normalization layer.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            hidden_size (int) : size of hidden state.
            eps:(float, optional): Small value added to the variance to avoid division by zero. Defaults to 1e-5.
            weight_key (str): Key name of weight in the pdparams state dict. Defaults to None, means no weight.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            linear_bias (float, optional): Initial bias value for the linear layer (if used). Defaults to None.

        Raises:
            NotImplementedError: If the specified norm_type is not supported.
        """
        super().__init__()
        self.llm_config = llm_config
        self.prefix = prefix
        self.hidden_size = hidden_size
        if len(prefix) == 0:
            self.weight_key = None
        else:
            self.weight_key = f"{prefix}.weight"
        self.with_weight = self.weight_key is not None
        self.eps = eps
        self.norm_func = fused_rms_norm
        self.linear_bias = linear_bias
        self.quant_scale = quant_scale
        self._dtype = self._helper.get_default_dtype()
        self._norm_weight_dtype = self._dtype

        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """

        self.ln_weight = None
        if self.with_weight:
            self.ln_weight = self.create_parameter(
                shape=[self.hidden_size],
                default_initializer=nn.initializer.Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        # weight
        weight_tensor = paddle.cast(
            get_tensor(state_dict.pop(self.weight_key)),
            self._norm_weight_dtype)
        self.ln_weight.set_value(weight_tensor)

    def forward(self, x, residual_input=None):
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
        norm_out = self.norm_func(
            x,
            norm_weight=self.ln_weight,
            norm_bias=None,
            epsilon=self.eps,
            begin_norm_axis=1,
            bias=self.linear_bias,
            residual=residual_input,
            quant_scale=-1 if self.quant_scale is None else self.quant_scale,
            quant_round_type=self.llm_config.quant_config.quant_round_type,
            quant_max_bound=self.llm_config.quant_config.quant_max_bound,
            quant_min_bound=self.llm_config.quant_config.quant_min_bound,
        )
        if residual_input is not None:
            return norm_out[0], norm_out[1]
        else:
            return norm_out[0]


class LayerNorm(nn.Layer):
    """
    Normalization layer.
    """

    def __init__(
        self,
        llm_config,
        hidden_size,
        eps=1e-5,
        prefix="",
        linear_bias=None,
        quant_scale=None,
        with_bias=False,
    ):
        """
        Initializes the normalization layer.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            prefix (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            hidden_size (int) : size of hidden state.
            eps:(float, optional): Small value added to the variance to avoid division by zero. Defaults to 1e-5.
            linear_bias (float, optional): Initial bias value for the linear layer (if used). Defaults to None.
        Raises:
            NotImplementedError: If the specified norm_type is not supported.
        """
        super().__init__()
        self.llm_config = llm_config
        self.prefix = prefix
        self.hidden_size = hidden_size
        if len(prefix) == 0:
            self.weight_key = None
        else:
            self.weight_key = f"{prefix}.weight"
        self.with_weight = self.weight_key is not None
        self.bias_key = f"{prefix}.bias"
        self.with_bias = with_bias
        self.eps = eps

        self.norm_func = fused_layer_norm
        self.linear_bias = linear_bias
        self._dtype = self._helper.get_default_dtype()
        self._norm_weight_dtype = "float32"

        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """

        self.ln_weight = None
        if self.with_weight:
            self.ln_weight = self.create_parameter(
                shape=[self.hidden_size],
                default_initializer=nn.initializer.Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
        self.ln_bias = None
        if self.with_bias:
            self.ln_bias = self.create_parameter(
                shape=[self.hidden_size],
                is_bias=True,
                dtype=self._norm_weight_dtype,
            )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        # weight
        weight_tensor = paddle.cast(
            get_tensor(state_dict.pop(self.weight_key)),
            self._norm_weight_dtype)
        self.ln_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.cast(
                get_tensor(state_dict.pop(self.bias_key)),
                self._norm_weight_dtype)
            self.ln_bias.set_value(bias_tensor)

    def forward(self, x, residual_input=None):
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
        norm_out = self.norm_func(
            x,
            norm_weight=self.ln_weight,
            norm_bias=self.ln_bias,
            epsilon=self.eps,
            begin_norm_axis=1,
            bias=self.linear_bias,
            residual=residual_input,
            quant_scale=-1,
            quant_round_type=self.llm_config.quant_config.quant_round_type,
            quant_max_bound=self.llm_config.quant_config.quant_max_bound,
            quant_min_bound=self.llm_config.quant_config.quant_min_bound,
        )
        if residual_input is not None:
            return norm_out[0], norm_out[1]
        else:
            return norm_out[0]
