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

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication_op import \
    tensor_model_parallel_all_reduce
from fastdeploy.platforms import current_platform

from .utils import _set_var_distributed, divide, get_tensor


class LinearBase(nn.Layer):
    """
    LinearBase Layer.
    """

    def __init__(
        self,
        fd_config: FDConfig,
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
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_size (int): Number of output features. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.

        Raises:
            NotImplementedError: Raised if the current platform is not a CUDA platform.
        """
        super().__init__()
        if current_platform.is_cuda() or current_platform.is_xpu():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError

        self.fd_config = fd_config
        self.skip_quant = skip_quant
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
        self.weight_dtype = self._dtype
        self.linear_weight_shape = [
            self.input_size,
            self.output_size,
        ]
        if fd_config.quant_config:
            self.quant_method = fd_config.quant_config.get_quant_method(self)
        if fd_config.model_config.is_quantized:
            self.weight_key = f"{prefix}.quant_weight"
            self.weight_scale_key = f"{prefix}.weight_scale"
            self.act_scale_key = f"{prefix}.activation_scale"

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        if self.skip_quant:
            self.weight_dtype = self._dtype
        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.weight_dtype,
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

    def load_prequant_weight(self, state_dict: dict):
        """
        Load the prequantized weight from the state dictionary.

        Args:
            state_dict (dict): A dictionary containing the prequantized weights and scales.
        """
        self.quant_method.process_prequanted_weights(self, state_dict)

    def load_weight(self, state_dict: dict):
        """
        Load the weight from the state dictionary.

        Args:
            state_dict (dict): A dictionary containing the weights
        """
        weight_tensor = get_tensor(state_dict.pop(self.weight_key))

        if self.fd_config.quant_config:
            self.quant_method.process_loaded_weights(self, weight_tensor)
        else:
            self.linear_weight.set_value(weight_tensor)

    def load_state_dict(self, state_dict: dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        self.state_dict = state_dict
        assert self.weight_key is not None, 'weight_key should not be None.'
        if self.fd_config.model_config.is_quantized:
            self.load_prequant_weight(state_dict)
        else:
            self.load_weight(state_dict)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(
                get_tensor(state_dict.pop(self.bias_key)))
            self.linear_bias.set_value(bias_tensor)

    def forward_cuda(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward function for Linear.

        Args:
            x (Tensor): Input tensor to the Linear.

        Returns:
            Tensor: Output tensor.

        Raises:
            NotImplementedError: If the weight dtype is not float8 or act dtype is not equal to weight dtype.
        """
        if self.fd_config.quant_config:
            linear_out = self.quant_method.apply(self, x)
        else:
            linear_out = paddle.matmul(x, self.linear_weight)
            if self.with_bias:
                linear_out = paddle.add(linear_out, self.linear_bias)

        return linear_out


class ReplicatedLinear(LinearBase):
    """
    ReplicatedLinear Layer.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initializes a replicated linear layer.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_size (int): Number of output features. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        super().__init__(fd_config=fd_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)

        self.hidden_size = fd_config.model_config.hidden_size
        self.linear_weight_shape = [
            self.input_size,
            self.output_size,
        ]
        if fd_config.quant_config:
            self.quant_method.create_weights(self)
        self.init_weight()


class ColumnParallelLinear(LinearBase):
    """
    ColumnParallelLinear Layer.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(
        self,
        fd_config: FDConfig,
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
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_size (int): Number of output features. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        super().__init__(fd_config=fd_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)
        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.input_size = input_size
        self.output_size = divide(
            output_size,
            self.nranks)  # Split the output_size using TP inference.
        self.hidden_size = fd_config.model_config.hidden_size
        self.linear_weight_shape = [
            self.input_size,
            self.output_size,
        ]
        if fd_config.quant_config:
            self.quant_method.create_weights(self)
        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        if self.skip_quant:
            self.weight_dtype = self._dtype
        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.weight_dtype,
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


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    MergedColumnParallelLinear Layer.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str,
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        activation: str = "gelu",
        use_fast_ffn: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initialize the fused ffn1 Linear layer with given parameters.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_size (int): Number of output features. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            activation (str): Activation function to use. Defaults to "gelu".
            use_fast_ffn (bool): Whether to use a faster FFN implementation.
                Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        self.use_fast_ffn = use_fast_ffn
        self.activation = activation
        self.hidden_size = fd_config.model_config.hidden_size
        self.nranks = fd_config.parallel_config.tensor_parallel_degree

        super().__init__(fd_config=fd_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)

    def load_state_dict(self, state_dict: dict):
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

    def __init__(self, fd_config, prefix, with_bias=False, add_bias=True):
        """
        Initialize the QKV Linear layer with given parameters.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to True.
        """
        self.num_heads = fd_config.model_config.num_attention_heads
        self.kv_num_heads = fd_config.model_config.num_key_value_heads
        self.hidden_size = fd_config.model_config.hidden_size
        self.head_dim = fd_config.model_config.head_dim
        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.num_heads_per_rank = divide(self.num_heads, self.nranks)
        self.kv_num_heads_per_rank = divide(self.kv_num_heads, self.nranks)
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.kv_num_heads) * self.head_dim
        super().__init__(fd_config=fd_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias)

    def load_weight(self, state_dict: dict):
        """
        Load the weight from the state dictionary.

        Args:
            state_dict (dict): A dictionary containing the weights
        """
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
                self.hidden_size,
            ])
            weight_tensor = paddle.transpose(weight_tensor, perm=[1, 0])

        if self.fd_config.quant_config:
            self.quant_method.process_loaded_weights(self, weight_tensor)
        else:
            self.linear_weight.set_value(weight_tensor)

    def load_state_dict(self, state_dict: dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, 'weight_key should not be None.'
        # qkv fused in disk

        if self.fd_config.model_config.is_quantized:
            self.load_prequant_weight(state_dict)
        else:
            self.load_weight(state_dict)

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


class RowParallelLinear(LinearBase):
    """
    RowParallelLinear Layer.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
        input_size: int = None,
        output_size: int = None,
        with_bias: bool = False,
        add_bias: bool = False,
        reduce_results: bool = True,
        skip_quant: bool = False,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_size (int): Number of output features. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        super().__init__(fd_config=fd_config,
                         prefix=prefix,
                         input_size=input_size,
                         output_size=output_size,
                         with_bias=with_bias,
                         add_bias=add_bias,
                         skip_quant=skip_quant)
        self.fd_config = fd_config
        self.skip_quant = False
        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.hidden_size = fd_config.model_config.hidden_size
        self.head_dim = fd_config.model_config.head_dim
        self.num_heads = fd_config.model_config.num_attention_heads // self.nranks

        # Split input_size when using TP inference.
        self.input_size = divide(input_size, self.nranks)
        self.output_size = output_size

        self.linear_weight_shape = [
            self.input_size,
            self.output_size,
        ]
        self._dtype = self._helper.get_default_dtype()

        if fd_config.quant_config:
            self.quant_method = fd_config.quant_config.get_quant_method(self)
            self.quant_method.create_weights(self)

        self.reduce_results = reduce_results
        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        if self.skip_quant:
            self.weight_dtype = self._dtype

        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            dtype=self.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.hidden_size],
                dtype=self._dtype,
                is_bias=True,
            )

        if self.nranks > 0:
            # row parallel
            _set_var_distributed(self.linear_weight, split_axis=0)

        # smooth quant
        self.linear_shift = None
        self.linear_smooth = None

    def forward_cuda(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.fd_config.quant_config:
            out = self.quant_method.apply(self, x)
        else:
            out = paddle.matmul(x, self.linear_weight)

        if self.reduce_results and self.nranks > 1:
            tensor_model_parallel_all_reduce(out)

        return out


class KVBatchLinear(LinearBase):
    """
    KVBatchLinear Layer for handling combined KV projections with bmm.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
        kv_lora_rank: int = None,
        num_attention_heads: int = None,
        qk_nope_head_dim: int = None,
        v_head_dim: int = None,
        with_bias: bool = False,
        skip_quant: bool = False,
    ):
        """
        Initializes a KV batch linear layer that internally splits into K and V projections.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
            kv_lora_rank (int): LoRA rank for KV projection. Defaults to None.
            num_attention_heads (int): Number of attention heads. Defaults to None.
            qk_nope_head_dim (int): Dimension for Q/K projection (nope part). Defaults to None.
            v_head_dim (int): Dimension for V projection. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        self.nranks = fd_config.parallel_config.tensor_parallel_degree
        self.kv_lora_rank = kv_lora_rank
        self.num_attention_heads = num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        # Split num_attention_heads when using TP inference.
        self.num_heads_per_partition = divide(num_attention_heads, self.nranks)

        # Initialize parent with combined dimensions
        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=None,  # Will be determined from weight shape
            output_size=None,  # Will be determined from weight shape
            with_bias=with_bias,
            add_bias=False,
            skip_quant=skip_quant,
        )
        self.weight_dtype = self._dtype

        # Override weight keys to use the combined kv_b_proj
        self.weight_key = f"{prefix}.weight"  # e.g., "kv_b_proj.weight"
        self.k_weight_key = f"{prefix.replace('kv_b_proj', 'k_b_proj')}.weight"
        self.v_weight_key = f"{prefix.replace('kv_b_proj', 'v_b_proj')}.weight"

    def load_state_dict(self, state_dict: dict):
        """
        Load the combined KV weight and split it into K and V projections
        """
        # Get the combined KV weight
        # NOTE(Ryan):Do not pop weight_key here, it will be popped in other class
        kv_weight_tensor = get_tensor(state_dict[self.weight_key])

        # Reshape and split the weight
        w = kv_weight_tensor.reshape([
            self.kv_lora_rank,
            self.num_heads_per_partition,
            -1,
        ]).transpose(perm=[1, 2, 0])

        # Split into K and V weights
        # wk_b: [num_heads, qk_nope_head_dim, kv_lora_rank]
        wk_b = w[:, :self.qk_nope_head_dim, :]

        if self.v_head_dim is None:
            raise ValueError("self.v_head_dim should not be None")
        # wv_b: [num_heads, kv_lora_rank, v_head_dim]
        wv_b = w[:, -self.v_head_dim:, :].transpose(perm=[0, 2, 1])

        # Create K projection weight
        self.k_b_proj_weight = self.create_parameter(
            shape=wk_b.shape,
            dtype=self.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        # Create V projection weight
        self.v_b_proj_weight = self.create_parameter(
            shape=wv_b.shape,
            dtype=self.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.k_b_proj_weight.set_value(wk_b)
        self.v_b_proj_weight.set_value(wv_b)

    def forward_k_b(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass for K_b projection using bmm

        Args:
            x: Input tensor (e.g., query_nope.transpose([1, 0, 2]))

        Returns:
            K_b projection output
        """

        out = paddle.bmm(x, self.k_b_proj_weight)
        return out

    def forward_v_b(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Forward pass for V_b projection using bmm

        Args:
            x: Input tensor (e.g., fmha_out_decode)

        Returns:
            V_b projection output
        """
        out = paddle.bmm(x, self.v_b_proj_weight)
        return out

    def forward_cuda(self,
                     x: paddle.Tensor,
                     proj_type: str = 'k') -> paddle.Tensor:
        """
        Forward function that can handle both K and V projections

        Args:
            x: Input tensor
            proj_type: 'k' or 'v' to select which projection to use

        Returns:
            Projection output
        """
        if proj_type == 'k':
            return self.forward_k_b(x)
        elif proj_type == 'v':
            return self.forward_v_b(x)
        else:
            raise ValueError(f"proj_type must be 'k' or 'v', got {proj_type}")
