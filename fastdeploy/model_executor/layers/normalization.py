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
from fastdeploy.model_executor.forward_meta import ForwardMeta

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
        dtype: str = None,
        layer_id: int = -1,
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

        self._norm_weight_dtype = dtype
        if self._norm_weight_dtype is None:
            self._norm_weight_dtype = self._helper.get_default_dtype()
        else:
            assert dtype in [
                "float32",
                "bfloat16",
                "float16",
            ], f"Unsupported dtype: {dtype}. Must be one of: float32, bfloat16, float16"

        self.quant_round_type: int = self.fd_config.quant_config.quant_round_type if fd_config.quant_config else 0
        self.quant_max_bound: int = self.fd_config.quant_config.quant_max_bound if fd_config.quant_config else 0
        self.quant_min_bound: int = self.fd_config.quant_config.quant_min_bound if fd_config.quant_config else 0
        self.begin_norm_axis: int = begin_norm_axis

        self.layer_id = layer_id
        self.ep_size = self.fd_config.parallel_config.expert_parallel_size
        self.tp_size = self.fd_config.parallel_config.tensor_parallel_size
        self.tp_rank = self.fd_config.parallel_config.tensor_parallel_rank
        self.tp_group = self.fd_config.parallel_config.tp_group
        is_input_norm = prefix.endswith(".input_layernorm")
        is_last_norm = prefix.endswith(".norm")
        self.split_x = (
            self.fd_config.parallel_config.use_sequence_parallel_moe
            and self.layer_id == self.fd_config.model_config.moe_layer_start_index
            and is_input_norm
        )
        self.allgather_out = self.fd_config.parallel_config.use_sequence_parallel_moe and (
            (self.layer_id > self.fd_config.model_config.moe_layer_start_index and is_input_norm) or is_last_norm
        )

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
        weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        self.weight.set_value(weight_tensor.astype(self._norm_weight_dtype))

    def split(self, x):
        """
        Split the input tensor across tensor parallel dimension.

        Args:
            x (paddle.Tensor): Input tensor to be split.

        Returns:
            paddle.Tensor: Splitted tensor.
        """
        token_num = x.shape[0]
        token_num_per_rank = (token_num + self.tp_size - 1) // self.tp_size
        # AllGather will hang when the data shapes on multi-ranks are different!
        start_offset = self.tp_rank * token_num_per_rank
        end_offset = (self.tp_rank + 1) * token_num_per_rank
        if start_offset >= token_num:
            start_offset = token_num
        if end_offset > token_num:
            end_offset = token_num
        part_x = paddle.zeros(shape=[token_num_per_rank, x.shape[1]], dtype=x.dtype)
        part_x[: (end_offset - start_offset), :] = x[start_offset:end_offset, :]
        return part_x

    def allgather(self, out, token_num):
        """
        Gather the output tensor from each tensor parallel rank.

        Args:
            out (paddle.Tensor): Output tensor to be gathered.

        Returns:
            paddle.Tensor: Gathered tensor.
        """
        token_num_per_rank = out.shape[0]
        multi_outs = paddle.zeros([token_num_per_rank * self.tp_size, out.shape[1]], dtype=out.dtype)
        paddle.distributed.all_gather(multi_outs, out, self.tp_group)
        return multi_outs[:token_num, :]

    def forward(
        self,
        x,
        residual_input: Optional[paddle.Tensor] = None,
        forward_meta: Optional[ForwardMeta] = None,
        external_rmsnorm: Optional[Callable] = None,
    ) -> paddle.Tensor:
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
        x_dtype = x.dtype
        x = x.astype(self.weight.dtype)
        if residual_input is not None:
            residual_input_dtype = residual_input.dtype
            residual_input = residual_input.astype(self.weight.dtype)

        if residual_input is None:
            residual_out = x
        if external_rmsnorm is None:
            if current_platform.is_gcu():
                if residual_input is None:
                    norm_out = rms_norm(x, self.weight, self.eps)
                    return norm_out.astype(x_dtype), residual_out
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
        else:
            if residual_input is not None:
                x = x + residual_input
            norm_out = external_rmsnorm(x, self.weight, self.eps), x

        out = norm_out[0].astype(x_dtype)
        if residual_input is not None:
            residual_out = norm_out[1].astype(residual_input_dtype)

        if self.split_x:
            assert residual_out is not None
            residual_out = self.split(residual_out)
        if self.allgather_out:
            assert forward_meta is not None
            out = self.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out, residual_out


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
