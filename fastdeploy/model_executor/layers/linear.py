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

import os

import fastdeploy
from paddlenlp.utils.log import logger

import paddle
from paddle import nn

from fastdeploy.platforms import current_platform

from .utils import _set_var_distributed, divide, get_tensor

import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm


class LinearBase(nn.Layer):
    """
    LinearBase Layer
    """

    def __init__(
        self,
        llm_config,
        prefix: str = "",
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initializes a linear layer and provides additional parameters required for inference and quantization.

        Args:
            llm_config (LLMConfig): Inference-related parameters containing attributes such as
                weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int, optional): Number of input features. Defaults to None.
            output_size (int, optional): Number of output features. Defaults to None.
            weight_key (Any, optional): Key for weights. Defaults to None.
            bias_key (Any, optional): Key for biases. Defaults to None.
            skip_quant (bool, optional): Whether to skip quantization. Defaults to False.

        Raises:
            NotImplementedError: Raised if the current platform is not a CUDA platform.
        """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError

        self.llm_config = llm_config
        self.skip_quant = skip_quant
        self.use_smooth_quant = llm_config.model_config.use_smooth_quant
        self.weight_dtype = llm_config.model_config.weight_dtype
        self.act_dtype = llm_config.model_config.act_dtype
        self.input_size = input_size
        self.output_size = output_size
        self.with_bias = with_bias
        self.add_bias = add_bias
        self.prefix = prefix
        # key
        self.weight_key = f"{prefix}.weight"
        self.bias_key = f"{prefix}.bias"
        self.shift_key = f"{prefix}.shift_bias"
        self.smooth_key = f"{prefix}.smooth_weight"
        self.out_scale_key = f"{prefix}.out_scale"

        self._dtype = self._helper.get_default_dtype()

        if llm_config.quant_config:
            self.quant_method = llm_config.quant_config.get_quant_method(self)
        self.use_offline_quant = llm_config.tmp_config.use_offline_quant
        
    def is_y_transposed(self):
        """
        Returns whether the y tensor should be transposed for inference.
        Args:
            None.

        Returns:
            bool, whether the y tensor should be transposed for inference.
        """
        if self.weight_dtype == "int4":
            return True
        if self.weight_dtype == "int8":
            return True
        if "float8" in self.weight_dtype:
            return True
        # bf16/fp16/fp32 y is not transposed
        return False

    def init_weight_shape(self, trans=False):
        """
        Initialize the weight shape for the first feedforward network layer.

        Args:
            trans (bool, optional): Whether to transpose the weight shape.
                Defaults to False. If True, the shape will be reversed.

        Returns:
            None.
        """
        self.linear_weight_shape = [
            self.input_size,
            self.output_size,
        ]
        if trans:
            self.linear_weight_shape.reverse()
        if self.use_smooth_quant:
            self.linear_shift_shape = [self.output_size]
            self.linear_smooth_shape = [self.output_size]
        if self.weight_dtype == "int4":
            self.linear_weight_shape[0] //= 2

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.output_size],
                dtype=self._dtype,
                is_bias=True,
            )

        # smooth quant
        self.linear_shift = None
        self.linear_smooth = None
        if self.use_smooth_quant:
            self.linear_shift = self.create_parameter(
                shape=self.linear_shift_shape,
                dtype=self._dtype,
                is_bias=False,
            )
            self.linear_smooth = self.create_parameter(
                shape=self.linear_smooth_shape,
                dtype=self._dtype,
                is_bias=False,
            )

    def get_weight_create_dtype(self):
        """
        Get the data type for creating weights based on quantization settings.

        Args:
            self (object): The instance of the class where this method is defined.

        Returns:
            str: The data type for creating weights. It depends on the quantization settings:
                - If `self.skip_quant` is True, returns the original data type `self._dtype`.
                - If `self.weight_dtype` is "int4", returns "int8" to ensure compatibility or optimization.
                - Otherwise, returns the specified weight data type `self.weight_dtype`.
        """
        if self.skip_quant:
            return self._dtype
        if self.weight_dtype == "int4":
            return "int8"
        # TODO(wangzhe24) create_parameter not support FP8
        if "float8" in self.weight_dtype:
            return self._dtype
        return self.weight_dtype


    def load_offline_quant_state_dict(self, quant_weight, quant_scale=None):
        """
        Load offline the checkpoint state dictionary into the layer.
        """
        if quant_scale is None:
            if "float8" in self.weight_dtype:
                self.linear_weight.copy_(quant_weight, False)
            else:
                self.linear_weight.set_value(quant_weight)
        else:
            if self.inference_args.weight_block_size[0] != -1:
                self.linear_weight.copy_(quant_weight.view(paddle.float8_e4m3fn), False)
            else:
                self.linear_weight.set_value(quant_weight)
            self.linear_weight_scale.set_value(quant_scale)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        if self.use_offline_quant:
            self.load_offline_quant_state_dict(
                quant_weight=get_tensor(
                    state_dict.pop(self.weight_key + ".quant_weight")
                ),
                quant_scale=get_tensor(
                    state_dict.pop(self.weight_key + ".quant_scale")
                ),
            )
        else:
            # weight
            assert self.weight_key is not None, 'weight_key should not be None.'
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))

            if self.llm_config.quant_config:
                self.quant_method.process_loaded_weights(self, weight_tensor)
            else:
                self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(get_tensor(state_dict.pop(self.bias_key)))
            self.linear_bias.set_value(bias_tensor)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(self.shift_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                shift_tensor = paddle.zeros(
                    shape=self.linear_shift_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(self.smooth_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                smooth_tensor = paddle.ones(
                    shape=[self.linear_smooth_shape],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)

    def forward_cuda(self, x):
        """
        Forward function for ColumnParallelLinear.

        Args:
            x (Tensor): Input tensor to the ColumnParallelLinear layer.

        Returns:
            Tensor: Output tensor.

        Raises:
            NotImplementedError: If the weight dtype is not float8 or act dtype is not equal to weight dtype.
        """
        if self.llm_config.quant_config:
            linear_out = self.quant_method.apply(self, x)
        else:
            linear_out = paddle.matmul(x, self.linear_weight)

        return linear_out


class ReplicatedLinear(LinearBase):
    """
    ReplicatedLinear Layer
    """

    def __init__(
        self,
        llm_config,
        prefix: str = "",
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            prefix (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            layer_index (int): The index of the linear layer in the model

        """
        super().__init__(llm_config=llm_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)
        self.nranks = llm_config.parallel_config.mp_size
        self.input_size = input_size
        self.init_weight()
        self.quant_method.create_weights(self)

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.output_size],
                dtype=self._dtype,
                is_bias=True,
            )

        # smooth quant
        self.linear_shift = None
        self.linear_smooth = None
        if self.use_smooth_quant:
            self.linear_shift = self.create_parameter(
                shape=self.linear_shift_shape,
                dtype=self._dtype,
                is_bias=False,
            )
            self.linear_smooth = self.create_parameter(
                shape=self.linear_smooth_shape,
                dtype=self._dtype,
                is_bias=False,
            )


class ColumnParallelLinear(LinearBase):
    """
    ColumnParallelLinear Layer
    """

    def __init__(
        self,
        llm_config,
        prefix: str = "",
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            prefix (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            layer_index (int): The index of the linear layer in the model

        """
        super().__init__(llm_config=llm_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)
        self.nranks = llm_config.parallel_config.mp_size
        self.input_size = input_size
        self.output_size = divide(output_size, self.nranks)
        self.init_weight()

        self.quant_method.create_weights(self)

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        if self.nranks > 0:
            # col parallel
            _set_var_distributed(self.linear_weight, split_axis=-1)

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.output_size],
                dtype=self._dtype,
                is_bias=True,
            )
            if self.nranks > 0:
                # col parallel
                _set_var_distributed(self.linear_bias, split_axis=-1)

        # smooth quant
        self.linear_shift = None
        self.linear_smooth = None
        if self.use_smooth_quant:
            self.linear_shift = self.create_parameter(
                shape=self.linear_shift_shape,
                dtype=self._dtype,
                is_bias=False,
            )
            self.linear_smooth = self.create_parameter(
                shape=self.linear_smooth_shape,
                dtype=self._dtype,
                is_bias=False,
            )


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    MergedColumnParallelLinear Layer.
    """

    def __init__(
        self,
        llm_config,
        prefix,
        with_bias=False,
        add_bias=False,
        activation="gelu",
        use_fast_ffn=False,
        skip_quant=False,
    ):
        """Packed linear layers with column parallelism.

        Initialize the fused ffn1 Linear layer with given parameters.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.

            prefix (str): Unique name of the layer, used for naming weights and biases.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            with_bias (bool, optional): Whether to include bias term. Defaults to True.
            activation (str, optional): Activation function to use. Defaults to "gelu".
            use_fast_ffn (bool, optional): Whether to use a faster FFN implementation.
                Defaults to False.
            skip_quant (bool, optional): Whether to skip quantization steps. Defaults to False.
        """
        self.use_fast_ffn = use_fast_ffn
        self.activation = activation
        self.embed_dim = llm_config.model_config.hidden_size
        self.dim_feedforward = llm_config.model_config.ffn_hidden_size
        self.nranks = llm_config.parallel_config.mp_size
        self.dim_feedforward_per_rank = divide(self.dim_feedforward,
                                               self.nranks)
        input_size = self.embed_dim
        output_size = self.dim_feedforward * 2
        super().__init__(llm_config=llm_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, 'weight_key should not be None.'
        if self.weight_key in state_dict.keys():
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        else:
            gate_weight_key = self.weight_key.replace("up_gate_proj",
                                                      "gate_proj")
            up_weight_key = self.weight_key.replace("up_gate_proj", "up_proj")
            gate_tensor = get_tensor(state_dict.pop(gate_weight_key))
            up_tensor = get_tensor(state_dict.pop(up_weight_key))
            weight_tensor = paddle.concat([gate_tensor, up_tensor], axis=-1)

            if self.with_bias:
                gate_bias_key = self.bias_key.replace("up_gate_proj",
                                                      "gate_proj")
                bias_tensor = get_tensor(state_dict.pop(gate_bias_key)).astype(
                    paddle.get_default_dtype())
                converted_bias_tensor = paddle.zeros(shape=list(
                    bias_tensor.shape),
                                                     dtype=bias_tensor.dtype)
                if not self.use_fast_ffn:
                    converted_bias_tensor = paddle.concat(
                        [bias_tensor[::2], bias_tensor[1::2]], axis=0)
                else:
                    converted_bias_tensor = bias_tensor
                state_dict[self.bias_key] = converted_bias_tensor

        if not self.use_fast_ffn:
            converted_weight_tensor = paddle.concat(
                [weight_tensor[:, ::2], weight_tensor[:, 1::2]], axis=1)
        else:
            converted_weight_tensor = weight_tensor

        state_dict[self.weight_key] = converted_weight_tensor

        super().load_state_dict(state_dict)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKVParallelLinear Layer.
    """

    def __init__(self, llm_config, prefix, with_bias=False, add_bias=True):
        """
        Initialize the QKV Linear layer with given parameters.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.

            prefix (str): Unique name of the layer, used for naming weights and biases.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            with_bias (bool, optional): Whether to include bias term. Defaults to True.
            skip_quant (bool, optional): Whether to skip quantization steps. Defaults to False.
        """
        self.num_heads = llm_config.model_config.num_attention_heads
        self.kv_num_heads = llm_config.model_config.num_key_value_heads
        self.embed_dim = llm_config.model_config.hidden_size
        self.head_dim = llm_config.model_config.head_dim
        self.nranks = llm_config.parallel_config.mp_size
        self.num_heads_per_rank = divide(self.num_heads, self.nranks)
        self.kv_num_heads_per_rank = divide(self.kv_num_heads, self.nranks)
        input_size = self.embed_dim
        output_size = (self.num_heads + 2 * self.kv_num_heads) * self.head_dim
        super().__init__(llm_config=llm_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, 'weight_key should not be None.'
        # qkv fused in disk
        if self.weight_key in state_dict.keys():
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        else:
            q_weight_key = self.weight_key.replace("qkv_proj", "q_proj")
            k_weight_key = self.weight_key.replace("qkv_proj", "k_proj")
            v_weight_key = self.weight_key.replace("qkv_proj", "v_proj")
            q_tensor = get_tensor(state_dict.pop(q_weight_key))
            k_tensor = get_tensor(state_dict.pop(k_weight_key))
            v_tensor = get_tensor(state_dict.pop(v_weight_key))
            weight_tensor = paddle.concat([q_tensor, k_tensor, v_tensor],
                                          axis=-1).transpose([1, 0])
            weight_tensor = weight_tensor.reshape([
                (self.num_heads_per_rank + 2 * self.kv_num_heads_per_rank) *
                (self.head_dim),
                self.embed_dim,
            ])
            weight_tensor = paddle.transpose(weight_tensor, perm=[1, 0])

        if self.llm_config.quant_config:
            self.quant_method.process_loaded_weights(self, weight_tensor)
        else:
            self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            if self.bias_key in state_dict.keys():
                bias_tensor = paddle.to_tensor(
                    get_tensor(state_dict.pop(self.bias_key)))
                self.linear_bias.set_value(bias_tensor)
            else:
                q_bias_key = self.bias_key.replace("qkv_proj", "q_proj")
                k_bias_key = self.bias_key.replace("qkv_proj", "k_proj")
                v_bias_key = self.bias_key.replace("qkv_proj", "v_proj")
                q_bias = get_tensor(state_dict.pop(q_bias_key))
                k_bias = get_tensor(state_dict.pop(k_bias_key))
                v_bias = get_tensor(state_dict.pop(v_bias_key))
                qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)
            self.linear_bias.set_value(qkv_bias)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(self.shift_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                shift_tensor = paddle.zeros(
                    shape=self.linear_shift_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(self.smooth_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                smooth_tensor = paddle.ones(
                    shape=[self.linear_smooth_shape],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)


class RowParallelLinear(LinearBase):
    """
    RowParallelLinear Layer
    """

    def __init__(
        self,
        llm_config,
        prefix: str = "",
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            prefix (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            layer_index (int): The index of the linear layer in the model

        """
        super().__init__(llm_config=llm_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)
        self.llm_config = llm_config
        self.skip_quant = False
        self.use_smooth_quant = llm_config.model_config.use_smooth_quant
        self.weight_dtype = llm_config.model_config.weight_dtype
        self.act_dtype = llm_config.model_config.act_dtype
        self.nranks = llm_config.parallel_config.mp_size
        self.embed_dim = llm_config.model_config.hidden_size
        self.head_dim = llm_config.model_config.hidden_size // llm_config.model_config.num_attention_heads
        self.num_heads = llm_config.model_config.num_attention_heads // self.nranks
        self.dim_feedforward = llm_config.model_config.ffn_hidden_size // self.nranks
        self.with_bias = with_bias
        self.prefix = prefix
        self.shift_key = f"{prefix}.shift_bias"
        self.smooth_key = f"{prefix}.smooth_weight"
        self.weight_key = f"{prefix}.weight"
        self.bias_key = f"{prefix}.bias"
        self.weight_only_scale_key = f"{prefix}.weight_only_scale"
        self.out_scale_key = f"{prefix}.out_scale"

        self._dtype = self._helper.get_default_dtype()

        if llm_config.quant_config:
            self.quant_method = llm_config.quant_config.get_quant_method(self)
            self.quant_method.create_weights(self)

        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.embed_dim],
                dtype=self._dtype,
                is_bias=True,
            )

        if self.nranks > 0:
            # row parallel
            _set_var_distributed(self.linear_weight, split_axis=0)

        # smooth quant
        self.linear_shift = None
        self.linear_smooth = None
        if self.use_smooth_quant:
            self.linear_shift = self.create_parameter(
                shape=self.linear_shift_shape,
                dtype=self._dtype,
                is_bias=False,
            )
            self.linear_smooth = self.create_parameter(
                shape=self.linear_smooth_shape,
                dtype=self._dtype,
                is_bias=False,
            )

    def forward_cuda(self, x):
        if self.llm_config.quant_config:
            out = self.quant_method.apply(self, x)
        else:
            out = paddle.matmul(x, self.linear_weight)

        if self.nranks > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(out)

        return out
