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

from typing import Dict, Optional

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase
from fastdeploy.worker.forward_meta import ForwardMeta


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

        Raises:
            ValueError: If the `v_head_dim` is less than 0.
        """
        super().__init__()
        self.num_heads: int = fd_config.model_config.num_attention_heads // fd_config.parallel_config.tensor_parallel_degree
        self.head_dim: int = fd_config.model_config.head_dim
        self.kv_num_heads: int = \
            fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_degree
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

        if fd_config.quant_config and hasattr(fd_config.quant_config,
                                              "kv_cache_quant_type"):
            self.kvcache_quant_method: QuantMethodBase = fd_config.quant_config.get_quant_method(
                self)
        else:
            self.kvcache_quant_method = None

        if self.kvcache_quant_method is None:
            logger.info(f"Attention is running in cache kv {self._dtype} mode")
        else:
            logger.info(
                f"Attention is running in cache kv {self.kvcache_quant_method.cache_quant_config.quant_type} mode"
            )

    def load_state_dict(self, state_dict: Dict[str,
                                               paddle.Tensor | np.ndarray]):
        '''
        Attention only have quant related scales not other parameters.
        '''
        if self.kvcache_quant_method is not None:
            self.kvcache_quant_method.create_weights(self, state_dict)

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
