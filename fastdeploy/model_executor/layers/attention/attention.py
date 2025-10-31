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

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.quantization.kv_cache import (
    KvCacheQuantzationTypes,
)
from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import os

from safetensors import safe_open

from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import default_weight_loader


class Attention(nn.Layer):
    """
    The AttentionLayer.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        v_head_dim: int = -1,
        rope_type: str = "",
        qkv_bias: Optional[paddle.Tensor] = None,
        qkv_scale: Optional[paddle.Tensor] = None,
        prefix: str = "",
        out_scale: float = -1.0,
        linear_shift: paddle.Tensor = None,
        linear_smooth: paddle.Tensor = None,
        use_neox_rotary_style: bool = False,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        with_sinks: bool = False,
    ) -> None:
        """
        Initializes `LMLayer` with the given parameters.

        Args:
            fd_config (dict): The config of LM model.
            layer_id (int): The id of current layer.
            v_head_dim (int, optional): The head dim of value. Defaults to -1.
            rope_type (str, optional): The type of RoPE. Defaults to "".
            qkv_bias (Optional[paddle.Tensor], optional): The bias of QKV. Defaults to None.
            qkv_scale (Optional[paddle.Tensor], optional): The scale of QKV. Defaults to None.
            prefix (str, optional): The name of current layer. Defaults to "".
            linear_shift (Optional[paddle.Tensor], optional): The shift of linear. Defaults to None.
            linear_smooth (Optional[paddle.Tensor], optional): The smooth of linear. Defaults to None.
            use_qk_norm (bool, optional): Whether to apply rmsnorm on QA after rope. Defaults to False.
            rms_norm_eps (float, optional): The epsilon of RMSNorm. Defaults to 1e-6.

        Raises:
            ValueError: If the `v_head_dim` is less than 0.
        """
        super().__init__()
        self.fd_config = fd_config
        self.num_heads: int = (
            fd_config.model_config.num_attention_heads // fd_config.parallel_config.tensor_parallel_size
        )
        self.head_dim: int = fd_config.model_config.head_dim
        self.kv_num_heads: int = max(
            1,
            fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size,
        )
        self.layer_id: int = layer_id
        self.v_head_dim: int = v_head_dim if v_head_dim > 0 else self.head_dim
        self.rope_type: str = rope_type
        self.qk_head_dim: int = self.head_dim
        self.prefix: str = prefix
        # not use
        self.linear_shift: paddle.Tensor | None = linear_shift
        self.linear_smooth: paddle.Tensor | None = linear_smooth
        self.qkv_bias: paddle.Tensor | None = qkv_bias
        self.qkv_scale: paddle.Tensor | None = qkv_scale
        self._dtype = self._helper.get_default_dtype()

        self.out_scale: float = out_scale
        self.use_neox_rotary_style: bool = use_neox_rotary_style

        self.with_sinks: bool = with_sinks

        if fd_config.quant_config and hasattr(fd_config.quant_config, "kv_cache_quant_type"):
            self.quant_method: QuantMethodBase = fd_config.quant_config.get_quant_method(self)

            # set for RL model, as RL do not need load state dict
            if fd_config.quant_config.kv_cache_quant_type == KvCacheQuantzationTypes.BLOCK_WISE_FP8:
                self.cache_quant_type_str = "block_wise_fp8"
                self.quant_max_bound = 448.0
                self.quant_min_bound = -448.0
        else:
            self.quant_method = None

        if self.quant_method is None:
            logger.info(f"Attention is running in cache kv {self._dtype} mode")
        else:
            logger.info(f"Attention is running in cache kv {self.quant_method.cache_quant_config.quant_type} mode")
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps
        if self.use_qk_norm:
            self.q_norm_key = f"{self.prefix}.q_norm"
            self.k_norm_key = f"{self.prefix}.k_norm"
        self.init_weight()

        if self.with_sinks:
            self.sinks = self.create_parameter(
                shape=[self.num_heads],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

        if (
            hasattr(self.fd_config.model_config, "layer_types")
            and self.fd_config.model_config.layer_types[self.layer_id] == "sliding_attention"
        ):
            self.sliding_window = self.fd_config.model_config.sliding_window
        else:
            self.sliding_window = 0

        if (
            fd_config.plas_attention_config is not None
            and fd_config.plas_attention_config.plas_encoder_top_k_left is not None
            and fd_config.plas_attention_config.plas_encoder_top_k_right is not None
            and fd_config.plas_attention_config.plas_decoder_top_k_left is not None
            and fd_config.plas_attention_config.plas_decoder_top_k_right is not None
        ):
            mlp_weight_path = os.path.join(
                fd_config.model_config.model, fd_config.plas_attention_config.mlp_weight_name
            )
            self.plas_use_mlp = mlp_weight_path is not None and os.path.exists(mlp_weight_path)
            plas_block_size = fd_config.plas_attention_config.plas_block_size
            plas_max_seq_length = fd_config.plas_attention_config.plas_max_seq_length
            if self.plas_use_mlp:
                mlp_weight = {}
                with safe_open(mlp_weight_path, framework="np", device="cpu") as f:
                    for key_name in f.keys():
                        weight = f.get_tensor(key_name)
                        weight = paddle.Tensor(weight, zero_copy=True)
                        weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                        mlp_weight[key_name] = weight

                if self.layer_id < fd_config.model_config.num_hidden_layers - 1:
                    self.attn_gate_weight = mlp_weight[
                        f"ernie.layers.{self.layer_id}.self_attn.attn_gate.weight"
                    ].astype(paddle.get_default_dtype())[
                        fd_config.parallel_config.tensor_parallel_rank
                        * self.kv_num_heads : (fd_config.parallel_config.tensor_parallel_rank + 1)
                        * self.kv_num_heads
                    ]
                    assert self.attn_gate_weight.shape[1] % plas_block_size == 0

            self.cache_k_block_means = paddle.zeros(
                [
                    fd_config.scheduler_config.max_num_seqs,
                    plas_max_seq_length // plas_block_size,
                    self.kv_num_heads,
                    self.head_dim,
                ],
                dtype=paddle.get_default_dtype(),
            )

    def init_weight(self):
        if self.quant_method is not None:
            self.quant_method.create_weights(
                self,
                weight_loader=(
                    self.weight_loader if hasattr(self, "weight_loader") else default_weight_loader(self.fd_config)
                ),
            )

        if self.use_qk_norm:
            self.q_norm_weight = self.create_parameter(
                shape=[self.qk_head_dim],
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            self.k_norm_weight = self.create_parameter(
                shape=[self.qk_head_dim],
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

    def load_state_dict(self, state_dict: Dict[str, paddle.Tensor | np.ndarray]):
        """
        Attention only have quant related scales not other parameters.
        """
        if self.quant_method is not None:
            self.quant_method.process_loaded_weights(self, state_dict)
        if self.use_qk_norm:
            q_norm_weight_tensor = paddle.to_tensor(get_tensor(state_dict.pop(self.q_norm_key + ".weight")))
            k_norm_weight_tensor = paddle.to_tensor(get_tensor(state_dict.pop(self.k_norm_key + ".weight")))
            self.q_norm_weight.set_value(q_norm_weight_tensor.astype("float32"))
            self.k_norm_weight.set_value(k_norm_weight_tensor.astype("float32"))

        if self.with_sinks:
            sinks_tensor = paddle.to_tensor(get_tensor(state_dict.pop(f"{self.prefix}.sinks")))
            self.sinks.set_value(sinks_tensor)

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        loaded_weight = get_tensor(loaded_weight).cast(paddle.get_default_dtype())
        if self.quant_method.cache_quant_config.has_zero_point:  # cache_int4_zp
            loaded_weight = 1.0 / loaded_weight
        else:
            loaded_weight = self.quant_method.cache_quant_config.max_bound / loaded_weight

        param.copy_(loaded_weight, False)

    def forward(
        self,
        q: paddle.Tensor = None,
        k: paddle.Tensor = None,
        v: paddle.Tensor = None,
        qkv: paddle.Tensor = None,
        compressed_kv: paddle.Tensor = None,
        k_pe: paddle.Tensor = None,
        forward_meta: ForwardMeta = None,
    ) -> paddle.Tensor:
        """
        The forward function of attention layer.
        args:
            q: the query tensor
            k: the key tensor
            v: the value tensor
            forward_meta: the forward meta data
            compressed_kv: optional compressed key-value cache (for MLA)
            k_pe: optional key positional encoding (for MLA)
        """
        return forward_meta.attn_backend.forward(
            q,
            k,
            v,
            qkv,
            compressed_kv,
            k_pe,
            self,
            forward_meta,
        )
