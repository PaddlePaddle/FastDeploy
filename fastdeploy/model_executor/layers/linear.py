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

from typing import Optional

import numpy as np
import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce
from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase
from fastdeploy.model_executor.utils import (
    default_weight_loader,
    h2d_copy,
    process_weight_transpose,
    set_weight_attrs,
    slice_fn,
)
from fastdeploy.platforms import current_platform

from .utils import _set_var_distributed, divide, get_tensor


class UnquantizedLinearMethod(QuantMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        extra_weight_attrs is a dictionary that may include parameters like:
        - split_axis: axis along which to split the tensor in a distributed environment
        - output_dim: determines whether the split is applied along the output dimension (rows) or input dimension (columns)
        - weight_loader: a callable or method responsible for loading the weight data
        """
        self.model_format = extra_weight_attrs.get("model_format")
        self.weight_shape = (
            layer.weight_shape[::-1] if extra_weight_attrs.get("model_format") == "torch" else layer.weight_shape
        )

        layer.weight = layer.create_parameter(
            shape=self.weight_shape,
            dtype=layer.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        if self.model_format == "torch" and "output_dim" in extra_weight_attrs:
            extra_weight_attrs["output_dim"] = not extra_weight_attrs["output_dim"]

        set_weight_attrs(
            layer.weight,
            {
                **extra_weight_attrs,
                "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
            },
        )

    def process_weights_after_loading(self, layer):
        if self.model_format == "torch":
            process_weight_transpose(layer, "weight")

    def process_loaded_weights(self, layer, weights) -> None:
        # mlp.gate.weight is precision-sensitive, so we cast it to float32 for computation
        if layer.weight.dtype != weights.dtype:
            weights = weights.cast(layer.weight.dtype)
        layer.weight.set_value(weights)

    def apply(self, layer: nn.Layer, x: paddle.Tensor) -> paddle.Tensor:
        linear_out = paddle.matmul(x, layer.weight)
        if layer.with_bias:
            linear_out = paddle.add(linear_out, layer.bias)
        return linear_out


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
        weight_dtype: str = "",
        weight_key: str = "",
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
        if (
            current_platform.is_cuda()
            or current_platform.is_xpu()
            or current_platform.is_iluvatar()
            or current_platform.is_gcu()
            or current_platform.is_dcu()
            or current_platform.is_maca()
            or current_platform.is_intel_hpu()
        ):
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
        self.is_quantized = fd_config.model_config.is_quantized and not (
            fd_config.quant_config.name() == "mix_quant" and fd_config.quant_config.dense_quant_type is None
        )
        # key
        if weight_key:
            self.weight_key = f"{prefix}.{weight_key}"
        elif self.is_quantized and not skip_quant:
            self.weight_key = f"{prefix}.quant_weight"
            self.weight_scale_key = f"{prefix}.weight_scale"
            self.act_scale_key = f"{prefix}.activation_scale"
        else:
            self.weight_key = f"{prefix}.weight"
        self.bias_key = f"{prefix}.bias"
        self.shift_key = f"{prefix}.shift_bias"
        self.smooth_key = f"{prefix}.smooth_weight"
        self.out_scale_key = f"{prefix}.out_scale"

        self._dtype = self._helper.get_default_dtype()
        if weight_dtype:
            self.weight_dtype = weight_dtype
        elif self.skip_quant:
            self.weight_dtype = self._dtype
        else:
            self.weight_dtype = self._dtype
        self.weight_shape = [
            self.input_size,
            self.output_size,
        ]

        if fd_config.quant_config and not skip_quant and fd_config.quant_config.get_quant_method(self):
            self.quant_method = fd_config.quant_config.get_quant_method(self)
        else:
            self.quant_method: Optional[QuantMethodBase] = UnquantizedLinearMethod()

        self.bias = None
        if self.with_bias:
            self.bias = self.create_parameter(
                shape=[self.output_size],
                dtype=self.weight_dtype,
                is_bias=True,
            )
            setattr(
                self.bias,
                "weight_loader",
                self.weight_loader if hasattr(self, "weight_loader") else default_weight_loader(self.fd_config),
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
        if isinstance(self.quant_method, UnquantizedLinearMethod):
            # for gate
            self.load_weight(state_dict)
        else:
            self.quant_method.process_prequanted_weights(self, state_dict)

    def load_weight(self, state_dict: dict):
        """
        Load the weight from the state dictionary.

        Args:
            state_dict (dict): A dictionary containing the weights
        """
        if "qkv_a_proj_with_mqa" in self.weight_key:
            self.weight_key_q = self.weight_key.replace("qkv_a_proj_with_mqa", "q_a_proj")
            self.weight_key_kv = self.weight_key.replace("qkv_a_proj_with_mqa", "kv_a_proj_with_mqa")
            q_weight_tensor = get_tensor(state_dict.pop(self.weight_key_q))
            kv_weight_tensor = get_tensor(state_dict.pop(self.weight_key_kv))
            weight_tensor = paddle.concat([q_weight_tensor, kv_weight_tensor], axis=-1)
        else:
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        self.quant_method.process_loaded_weights(self, weight_tensor)

    def load_state_dict(self, state_dict: dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        self.state_dict = state_dict
        assert self.weight_key is not None, "weight_key should not be None."
        if self.is_quantized:
            self.load_prequant_weight(state_dict)
        else:
            self.load_weight(state_dict)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(get_tensor(state_dict.pop(self.bias_key)))
            self.bias.set_value(bias_tensor)

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
        linear_out = self.quant_method.apply(self, x)

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
        weight_dtype: str = "",
        weight_key: str = "",
        model_format: Optional[str] = None,
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
        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=input_size,
            output_size=output_size,
            with_bias=with_bias,
            add_bias=add_bias,
            skip_quant=skip_quant,
            weight_dtype=weight_dtype,
            weight_key=weight_key,
        )

        self.hidden_size = fd_config.model_config.hidden_size

        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            weight_loader=(
                self.weight_loader if hasattr(self, "weight_loader") else default_weight_loader(self.fd_config)
            ),
            model_format=fd_config.model_config.model_format if model_format is None else model_format,
        )


class MergedReplicatedLinear(ReplicatedLinear):
    """
    MergedReplicatedLinear linear layer.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
        input_size: int = None,
        output_sizes: list[int] = None,
        with_bias: bool = False,
        add_bias: bool = False,
        skip_quant: bool = False,
        weight_dtype: str = "",
        weight_key: str = "",
    ):
        """
        Initializes a mergedreplicated linear layer.
        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_sizes (list[int]): Number of output features list. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=input_size,
            output_size=sum(output_sizes),
            with_bias=with_bias,
            add_bias=add_bias,
            skip_quant=skip_quant,
            weight_dtype=weight_dtype,
            weight_key=weight_key,
        )
        self.output_sizes = output_sizes

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if weight_need_transpose:
            loaded_weight = get_tensor(loaded_weight).transpose([1, 0])

        assert loaded_shard_id in ["q_a", "kv_a"]
        if not param._is_initialized():
            param.initialize()

        if loaded_shard_id == "q_a":
            param_shard_offset = 0
            param_shard_size = self.output_sizes[0]
        else:
            # loaded_shard_id == "kv_a"
            param_shard_offset = self.output_sizes[0]
            param_shard_size = self.output_sizes[1]

        if hasattr(param, "tensor_track"):
            param.tensor_track.mark(start=param_shard_offset, end=param_shard_offset + param_shard_size)
        param = slice_fn(param, True, start=param_shard_offset, end=param_shard_offset + param_shard_size)
        assert param.shape == loaded_weight.shape, (
            f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
        )
        # Ensure loaded weight dtype matches model param dtype
        if loaded_weight.dtype != param.dtype:
            if loaded_weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
                loaded_weight = loaded_weight.view(param.dtype)
            else:
                loaded_weight = loaded_weight.cast(param.dtype)
        # (bukejiyu) After this fix, the early H2D copy for non-GPU devices is no longer needed and can be safely removed.
        loaded_weight = get_tensor(loaded_weight)
        h2d_copy(param, loaded_weight)


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
        weight_dtype: str = "",
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
        self.fd_config = fd_config
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.input_size = input_size
        self.output_size = divide(output_size, self.tp_size)  # Split the output_size using TP inference.
        self.hidden_size = fd_config.model_config.hidden_size

        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=self.input_size,
            output_size=self.output_size,
            with_bias=with_bias,
            add_bias=add_bias,
            skip_quant=skip_quant,
            weight_dtype=weight_dtype,
        )

        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            output_dim=True,
            weight_loader=(
                self.weight_loader if hasattr(self, "weight_loader") else default_weight_loader(self.fd_config)
            ),
            model_format=fd_config.model_config.model_format,
        )

        if self.tp_size > 0:
            if self.with_bias:
                # col parallel
                _set_var_distributed(self.bias, split_axis=1)
                set_weight_attrs(self.bias, {"output_dim": True})


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
        skip_quant: bool = False,
    ):
        """
        Initialize the fused up_gate_proj Linear layer with given parameters.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int): Number of input features. Defaults to None.
            output_size (int): Number of output features. Defaults to None.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to False.
            activation (str): Activation function to use. Defaults to "gelu".
            skip_quant (bool): Whether to skip quantization. Defaults to False.
        """
        self.activation = activation
        self.hidden_size = fd_config.model_config.hidden_size
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.output_size = output_size
        self.local_rank = fd_config.parallel_config.tensor_parallel_rank

        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=input_size,
            output_size=output_size,
            with_bias=with_bias,
            add_bias=add_bias,
            skip_quant=skip_quant,
        )

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        # for xpu and other backend
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None
        shard_dim = -1 if output_dim else 0
        output_size = param.shape[shard_dim]
        if loaded_shard_id is None:
            if weight_need_transpose:
                loaded_weight = get_tensor(loaded_weight)
                loaded_weight = loaded_weight.transpose([1, 0])
                # Avoid redundant transpose of fused weights when weight_loader is called iteratively
                param.weight_need_transpose = False
            # Loaded weight is already fused on disk.
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("gate", 0, output_size * self.tp_size // 2),
                ("up", output_size * self.tp_size // 2, output_size * self.tp_size // 2),
            ]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = slice_fn(
                    loaded_weight, output_dim, start=shard_offset, end=shard_offset + shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
        else:
            # split gate up
            assert loaded_shard_id in ["gate", "up"]
            if weight_need_transpose:
                loaded_weight = get_tensor(loaded_weight)
                loaded_weight = loaded_weight.transpose([1, 0])
            # Tensor parallelism splits the weight along the output_dim
            if self.tp_size > 1 and output_dim is not None:
                dim = -1 if output_dim else 0
                if isinstance(loaded_weight, (np.ndarray, paddle.Tensor)):
                    size = loaded_weight.shape[dim]
                else:
                    size = loaded_weight.get_shape()[dim]
                block_size = size // self.tp_size
                shard_offset = self.local_rank * block_size
                shard_size = (self.local_rank + 1) * block_size
                loaded_weight = slice_fn(loaded_weight, output_dim, start=shard_offset, end=shard_size)
            if not param._is_initialized():
                param.initialize()
            param_shard_size = output_size // 2
            if loaded_shard_id == "gate":
                param_shard_offset = 0
            else:
                # loaded_shard_id == "up"
                param_shard_offset = param_shard_size
            if hasattr(param, "tensor_track"):
                param.tensor_track.mark(start=param_shard_offset, end=param_shard_offset + param_shard_size)
            param = slice_fn(param, output_dim, start=param_shard_offset, end=param_shard_offset + param_shard_size)
            assert param.shape == loaded_weight.shape, (
                f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
            )
            # Ensure loaded weight dtype matches model param dtype
            if loaded_weight.dtype != param.dtype:
                if loaded_weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
                    loaded_weight = loaded_weight.view(param.dtype)
                else:
                    loaded_weight = loaded_weight.cast(param.dtype)

            h2d_copy(param, loaded_weight)

    def load_state_dict(self, state_dict: dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, "weight_key should not be None."
        if self.weight_key in state_dict.keys():
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        else:
            gate_weight_key = self.weight_key.replace("up_gate_proj", "gate_proj")
            up_weight_key = self.weight_key.replace("up_gate_proj", "up_proj")
            gate_tensor = get_tensor(state_dict.pop(gate_weight_key))
            up_tensor = get_tensor(state_dict.pop(up_weight_key))
            weight_tensor = paddle.concat([gate_tensor, up_tensor], axis=-1)

            if self.with_bias:
                gate_bias_key = self.bias_key.replace("up_gate_proj", "gate_proj")
                bias_tensor = get_tensor(state_dict.pop(gate_bias_key)).astype(paddle.get_default_dtype())

                state_dict[self.bias_key] = bias_tensor

        state_dict[self.weight_key] = weight_tensor

        super().load_state_dict(state_dict)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKVParallelLinear Layer.
    """

    def __init__(
        self,
        fd_config,
        prefix,
        with_bias=False,
        add_bias=True,
        num_heads: Optional[int] = None,
        kv_num_heads: Optional[int] = None,
        hidden_size: Optional[int] = None,
        head_dim: Optional[int] = None,
        skip_quant: bool = False,
        weight_dtype: str = "",
    ):
        """
        Initialize the QKV Linear layer with given parameters.

        Args:
            fd_config (FDConfig): Inference-related parameters.
            prefix (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            with_bias (bool): Whether to include bias or not. Defaults to False.
            add_bias (bool): Whether to add bias in the current layer or in the pre/post layer. Defaults to True.
            num_heads (Optional[int]): Number of attention heads in the model.
            kv_num_heads (Optional[int]): Number of key/value heads, used for multi-query or grouped-query attention.
            hidden_size (Optional[int]): Total hidden layer dimension, typically the embedding size.
            head_dim (Optional[int]): Size of each attention head, usually computed as hidden_size divided by num_heads.
        """
        self.num_heads = fd_config.model_config.num_attention_heads if num_heads is None else num_heads
        self.kv_num_heads = fd_config.model_config.num_key_value_heads if kv_num_heads is None else kv_num_heads
        self.hidden_size = fd_config.model_config.hidden_size if hidden_size is None else hidden_size
        self.head_dim = fd_config.model_config.head_dim if head_dim is None else head_dim
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.local_rank = fd_config.parallel_config.tensor_parallel_rank
        self.num_heads_per_rank = divide(self.num_heads, self.tp_size)
        if self.kv_num_heads < self.tp_size and self.tp_size % self.kv_num_heads == 0:
            self.kv_num_heads_per_rank = 1
            self.num_kv_head_replicas = divide(self.tp_size, self.kv_num_heads)
            output_size = (self.num_heads + 2 * self.tp_size) * self.head_dim
        else:
            self.kv_num_heads_per_rank = divide(self.kv_num_heads, self.tp_size)
            self.num_kv_head_replicas = 1
            output_size = (self.num_heads + 2 * self.kv_num_heads) * self.head_dim
        input_size = self.hidden_size
        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=input_size,
            output_size=output_size,
            with_bias=with_bias,
            add_bias=add_bias,
            skip_quant=skip_quant,
            weight_dtype=weight_dtype,
        )

    def _get_shard_size_mapping(self, loaded_shard_id: str, head_dim: int):
        shard_size_mapping = {
            "q": self.num_heads_per_rank * head_dim,
            "k": self.kv_num_heads_per_rank * head_dim,
            "v": self.kv_num_heads_per_rank * head_dim,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None
        dim = -1 if output_dim else 0
        head_dim = param.shape[dim] // (self.num_heads_per_rank + 2 * self.kv_num_heads_per_rank)
        weight_need_transpose = getattr(param, "weight_need_transpose", False)
        if loaded_shard_id is None:
            if weight_need_transpose:
                loaded_weight = get_tensor(loaded_weight)
                loaded_weight = loaded_weight.transpose([1, 0])
                # Avoid redundant transpose of fused weights when weight_loader is called iteratively
                param.weight_need_transpose = False
            # Loaded weight is already fused on disk
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.num_heads * head_dim),
                ("k", self.num_heads * head_dim, self.kv_num_heads * head_dim),
                ("v", (self.num_heads + self.kv_num_heads) * head_dim, self.kv_num_heads * head_dim),
            ]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = slice_fn(
                    loaded_weight, output_dim, start=shard_offset, end=shard_offset + shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
        else:
            # split q k v
            assert loaded_shard_id in ["q", "k", "v"]
            if weight_need_transpose:
                loaded_weight = get_tensor(loaded_weight)
                loaded_weight = loaded_weight.transpose([1, 0])
            # Tensor parallelism splits the weight along the output_dim
            if self.tp_size > 1 and output_dim is not None:
                block_size = self._get_shard_size_mapping(loaded_shard_id, head_dim)
                shard_id = self.local_rank if loaded_shard_id == "q" else self.local_rank // self.num_kv_head_replicas
                shard_offset = shard_id * block_size
                shard_size = block_size
                loaded_weight = slice_fn(loaded_weight, output_dim, start=shard_offset, end=shard_offset + shard_size)

            if not param._is_initialized():
                param.initialize()

            if loaded_shard_id == "q":

                param_shard_offset = 0
                param_shard_size = self.num_heads_per_rank * head_dim
            elif loaded_shard_id == "k":
                param_shard_offset = self.num_heads_per_rank * head_dim
                param_shard_size = self.kv_num_heads_per_rank * head_dim
            else:
                # loaded_shard_id == "v"
                param_shard_offset = (self.num_heads_per_rank + self.kv_num_heads_per_rank) * head_dim
                param_shard_size = self.kv_num_heads_per_rank * head_dim
            if hasattr(param, "tensor_track"):
                param.tensor_track.mark(start=param_shard_offset, end=param_shard_offset + param_shard_size)

            param = slice_fn(param, output_dim, start=param_shard_offset, end=param_shard_offset + param_shard_size)
            assert param.shape == loaded_weight.shape, (
                f" Attempted to load weight ({loaded_weight.shape}) " f"into parameter ({param.shape})"
            )
            # Ensure loaded weight dtype matches model param dtype
            if loaded_weight.dtype != param.dtype:
                if loaded_weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
                    loaded_weight = loaded_weight.view(param.dtype)
                else:
                    loaded_weight = loaded_weight.cast(param.dtype)
            h2d_copy(param, loaded_weight)

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

            if self.kv_num_heads < self.tp_size:
                sharedkv_index = (
                    self.fd_config.parallel_config.tensor_parallel_rank * self.kv_num_heads
                ) // self.tp_size
                sharedkv_start = sharedkv_index * self.head_dim
                sharedkv_end = sharedkv_start + self.head_dim
                k_tensor = k_tensor[:, sharedkv_start:sharedkv_end]
                v_tensor = v_tensor[:, sharedkv_start:sharedkv_end]
            weight_tensor = paddle.concat([q_tensor, k_tensor, v_tensor], axis=-1).transpose([1, 0])
            weight_tensor = weight_tensor.reshape(
                [
                    (self.num_heads_per_rank + 2 * self.kv_num_heads_per_rank) * (self.head_dim),
                    self.hidden_size,
                ]
            )
            weight_tensor = paddle.transpose(weight_tensor, perm=[1, 0])

        self.quant_method.process_loaded_weights(self, weight_tensor)

    def load_state_dict(self, state_dict: dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, "weight_key should not be None."
        # qkv fused in disk

        if self.is_quantized:
            self.load_prequant_weight(state_dict)
        else:
            self.load_weight(state_dict)

        # bias
        if self.with_bias:
            if self.bias_key in state_dict.keys():
                bias_tensor = paddle.to_tensor(get_tensor(state_dict.pop(self.bias_key)))
                self.bias.set_value(bias_tensor)
            else:
                q_bias_key = self.bias_key.replace("qkv_proj", "q_proj")
                k_bias_key = self.bias_key.replace("qkv_proj", "k_proj")
                v_bias_key = self.bias_key.replace("qkv_proj", "v_proj")
                q_bias = get_tensor(state_dict.pop(q_bias_key))
                k_bias = get_tensor(state_dict.pop(k_bias_key))
                v_bias = get_tensor(state_dict.pop(v_bias_key))
                qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=-1)
                self.bias.set_value(qkv_bias)


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
        weight_dtype: str = "",
        layer_id: int = -1,
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
        self.fd_config = fd_config
        self.ep_size = fd_config.parallel_config.expert_parallel_size
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.tp_group = fd_config.parallel_config.tp_group
        self.hidden_size = fd_config.model_config.hidden_size
        self.head_dim = fd_config.model_config.head_dim
        self.split_token = (
            fd_config.parallel_config.use_sequence_parallel_moe
            and layer_id >= fd_config.model_config.moe_layer_start_index
            and layer_id < fd_config.model_config.num_hidden_layers
        )

        # Split input_size when using TP inference.
        if self.split_token:
            self.input_size = input_size
        else:
            self.input_size = divide(input_size, self.tp_size)
        self.output_size = output_size

        super().__init__(
            fd_config=fd_config,
            prefix=prefix,
            input_size=self.input_size,
            output_size=self.output_size,
            with_bias=with_bias,
            add_bias=add_bias,
            skip_quant=skip_quant,
            weight_dtype=weight_dtype,
        )

        assert self.quant_method is not None
        create_weight_kwargs = dict(
            layer=self,
            output_dim=None if self.split_token else False,
            weight_loader=(
                self.weight_loader if hasattr(self, "weight_loader") else default_weight_loader(self.fd_config)
            ),
            model_format=fd_config.model_config.model_format,
        )
        if self.tp_size > 1:
            create_weight_kwargs["split_axis"] = 0
            create_weight_kwargs["is_distributed"] = True
        self.quant_method.create_weights(**create_weight_kwargs)

        self.reduce_results = reduce_results and not self.split_token

        if add_bias:
            assert with_bias, "with_bias must be True when add_bias is True."
            if self.tp_size > 1 and self.reduce_results:
                set_weight_attrs(self.bias, {"tp_row_bias": True})

    def all2all_transpose(self, x: paddle.Tensor) -> paddle.Tensor:
        token_num = x.shape[0]
        token_num_pad = (token_num + self.tp_size - 1) // self.tp_size * self.tp_size
        if token_num_pad > token_num:
            x_new = paddle.zeros([token_num_pad, x.shape[1]], x.dtype)
            x_new[:token_num, :] = x
            x = x_new
        out = paddle.zeros_like(x)
        paddle.distributed.alltoall(out, x, group=self.tp_group)
        out.reshape_([self.tp_size, -1, x.shape[1]])
        out = paddle.transpose(out, [1, 0, 2])
        out.reshape_([x.shape[0] // self.tp_size, self.input_size])
        return out

    def forward_cuda(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.split_token:
            x = self.all2all_transpose(x)

        out = self.quant_method.apply(self, x)

        if self.reduce_results and self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out, self.tp_group)

        return out


class KVBatchLinear(nn.Layer):
    """
    KVBatchLinear Layer for handling combined KV projections with bmm.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        kv_b_proj: nn.Layer,
        prefix: str = "",
        kv_lora_rank: int = None,
        num_attention_heads: int = None,
        qk_nope_head_dim: int = None,
        v_head_dim: int = None,
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
        """
        super().__init__()
        self.tp_size = fd_config.parallel_config.tensor_parallel_size
        self.kv_lora_rank = kv_lora_rank
        self.num_attention_heads = num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        # Split num_attention_heads when using TP inference.
        self.num_heads_per_partition = divide(num_attention_heads, self.tp_size)
        self.local_rank = fd_config.parallel_config.tensor_parallel_rank
        self.fd_config = fd_config
        self.kv_b_proj = kv_b_proj

        self.weight_dtype = self._helper.get_default_dtype()

        # Override weight keys to use the combined kv_b_proj
        self.weight_key = f"{prefix}.weight"  # e.g., "kv_b_proj.weight"

    def process_weights_after_loading(self):
        if self.fd_config.load_config.dynamic_load_weight:
            return
        w = self.kv_b_proj.weight.reshape(
            [
                self.kv_lora_rank,
                self.num_heads_per_partition,
                -1,
            ]
        ).transpose(perm=[1, 2, 0])
        self.kv_b_proj = None

        if w.dtype != self.weight_dtype:
            w = w.cast(self.weight_dtype)

        # Split into K and V weights
        # wk_b: [num_heads, qk_nope_head_dim, kv_lora_rank]
        wk_b = w[:, : self.qk_nope_head_dim, :]
        if self.v_head_dim is None:
            raise ValueError("self.v_head_dim should not be None")
        # wv_b: [num_heads, kv_lora_rank, v_head_dim]
        wv_b = w[:, -self.v_head_dim :, :].transpose(perm=[0, 2, 1])
        self.k_b_proj_weight = wk_b
        self.v_b_proj_weight = wv_b

    def load_state_dict(self, state_dict: dict):
        """
        Load the combined KV weight and split it into K and V projections
        """
        # Get the combined KV weight
        # NOTE(Ryan):Do not pop weight_key here, it will be popped in other class
        kv_weight_tensor = get_tensor(state_dict[self.weight_key])

        # Reshape and split the weight
        w = kv_weight_tensor.reshape(
            [
                self.kv_lora_rank,
                self.num_heads_per_partition,
                -1,
            ]
        ).transpose(perm=[1, 2, 0])

        # Split into K and V weights
        # wk_b: [num_heads, qk_nope_head_dim, kv_lora_rank]
        wk_b = w[:, : self.qk_nope_head_dim, :]

        if self.v_head_dim is None:
            raise ValueError("self.v_head_dim should not be None")
        # wv_b: [num_heads, kv_lora_rank, v_head_dim]
        wv_b = w[:, -self.v_head_dim :, :].transpose(perm=[0, 2, 1])

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

    def forward(self, x: paddle.Tensor, proj_type: str = "k") -> paddle.Tensor:
        """
        Forward function that can handle both K and V projections

        Args:
            x: Input tensor
            proj_type: 'k' or 'v' to select which projection to use

        Returns:
            Projection output
        """
        if proj_type == "k":
            return self.forward_k_b(x)
        elif proj_type == "v":
            return self.forward_v_b(x)
        else:
            raise ValueError(f"proj_type must be 'k' or 'v', got {proj_type}")
