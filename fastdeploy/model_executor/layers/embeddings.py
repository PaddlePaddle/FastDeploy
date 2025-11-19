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

from dataclasses import dataclass
from typing import Dict

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.utils import h2d_copy, set_weight_attrs, slice_fn
from fastdeploy.platforms import current_platform

from .utils import (
    DEFAULT_VOCAB_PADDING_SIZE,
    get_tensor,
    pad_vocab_size,
    vocab_range_from_global_vocab_size,
)


@dataclass
class VocabParallelEmbeddingShardIndices:
    """Indices for a shard of a vocab parallel embedding."""

    padded_org_vocab_start_index: int
    padded_org_vocab_end_index: int
    padded_added_vocab_start_index: int
    padded_added_vocab_end_index: int

    org_vocab_start_index: int
    org_vocab_end_index: int
    added_vocab_start_index: int
    added_vocab_end_index: int

    @property
    def num_org_elements(self) -> int:
        return self.org_vocab_end_index - self.org_vocab_start_index

    @property
    def num_added_elements(self) -> int:
        return self.added_vocab_end_index - self.added_vocab_start_index

    @property
    def num_org_elements_padded(self) -> int:
        return self.padded_org_vocab_end_index - self.padded_org_vocab_start_index

    @property
    def num_added_elements_padded(self) -> int:
        return self.padded_added_vocab_end_index - self.padded_added_vocab_start_index

    @property
    def num_org_vocab_padding(self) -> int:
        return self.num_org_elements_padded - self.num_org_elements

    @property
    def num_added_vocab_padding(self) -> int:
        return self.num_added_elements_padded - self.num_added_elements

    @property
    def num_elements_padded(self) -> int:
        return self.num_org_elements_padded + self.num_added_elements_padded

    def __post_init__(self):
        # sanity checks
        assert self.padded_org_vocab_start_index <= self.padded_org_vocab_end_index
        assert self.padded_added_vocab_start_index <= self.padded_added_vocab_end_index

        assert self.org_vocab_start_index <= self.org_vocab_end_index
        assert self.added_vocab_start_index <= self.added_vocab_end_index

        assert self.org_vocab_start_index <= self.padded_org_vocab_start_index
        assert self.added_vocab_start_index <= self.padded_added_vocab_start_index
        assert self.org_vocab_end_index <= self.padded_org_vocab_end_index
        assert self.added_vocab_end_index <= self.padded_added_vocab_end_index

        assert self.num_org_elements <= self.num_org_elements_padded
        assert self.num_added_elements <= self.num_added_elements_padded


class VocabParallelEmbedding(nn.Layer):
    """
    VocabParallelEmbedding Layer
    """

    def __init__(
        self,
        fd_config: FDConfig,
        num_embeddings: int,
        embedding_dim: int = 768,
        params_dtype: str = "bfloat16",
        prefix="",
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
    ) -> None:
        """
        Initialize the VocabParallelEmbedding layer for the model.

        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            num_embeddings (int)  : vocabulary size.
            embedding_dim (int) : size of hidden state.
            params_dtype  (str) : data type of parameters.
            prefix (str): The name of current layer. Defaults to "".
        """
        super().__init__()
        self.fd_config = fd_config
        hcg = fleet.get_hybrid_communicate_group()
        self.mp_rank: int = hcg.get_model_parallel_rank()
        self.column_cut = False
        self.world_size: int = fd_config.parallel_config.tensor_parallel_size
        self.tensor_parallel_rank = fd_config.parallel_config.tensor_parallel_rank
        self.tp_group = fd_config.parallel_config.tp_group
        self.hidden_dropout_prob: float = fd_config.model_config.hidden_dropout_prob
        self.initializer_range: float = fd_config.model_config.initializer_range
        self.max_position_embeddings: int = fd_config.model_config.max_position_embeddings
        self.tie_word_embeddings: bool = fd_config.model_config.tie_word_embeddings
        self.params_dtype: str = params_dtype
        self.padding_size = padding_size

        self.org_vocab_size = num_embeddings
        self.num_embeddings = num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size

        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size, self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded
        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            self.tensor_parallel_rank,
            self.world_size,
        )

        if num_embeddings % self.world_size != 0:
            self.num_embeddings_padded = pad_vocab_size(num_embeddings, self.padding_size)

        if not self.column_cut:
            self.embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                self.num_embeddings_padded,
                embedding_dim,
                mp_group=self.tp_group,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range),
                ),
            )
            set_weight_attrs(self.embeddings.weight, {"output_dim": False})
            set_weight_attrs(self.embeddings.weight, {"weight_loader": self.weight_loader})
        else:
            # column cut embedding
            self.embeddings = nn.Embedding(
                num_embeddings,
                embedding_dim // self.world_size,
            )

            self.embeddings.weight.is_distributed = True
            self.embeddings.weight.split_axis = 1

            set_weight_attrs(self.embeddings.weight, {"output_dim": True})

        self.prefix = prefix
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def load_state_dict(self, state_dict: Dict[str, paddle.Tensor | np.ndarray]):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        if self.tie_word_embeddings:
            weight_tensor = get_tensor(state_dict[self.prefix + ".weight"]).astype(paddle.get_default_dtype())
        else:
            weight_tensor = get_tensor(state_dict.pop(self.prefix + ".weight")).astype(paddle.get_default_dtype())

        self.embeddings.weight.set_value(weight_tensor)

    @classmethod
    def _get_indices(
        cls,
        vocab_size_paded: int,
        org_vocab_size_padded: int,
        vocab_size: int,
        org_vocab_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> VocabParallelEmbeddingShardIndices:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring, based on the given tp_rank and
        tp_size."""

        num_added_embeddings_padded = vocab_size_paded - org_vocab_size_padded
        padded_org_vocab_start_index, padded_org_vocab_end_index = vocab_range_from_global_vocab_size(
            org_vocab_size_padded, tp_rank, tp_size
        )

        padded_added_vocab_start_index, padded_added_vocab_end_index = vocab_range_from_global_vocab_size(
            num_added_embeddings_padded, tp_rank, tp_size, offset=org_vocab_size
        )
        # remove padding
        org_vocab_start_index = min(padded_org_vocab_start_index, org_vocab_size)
        org_vocab_end_index = min(padded_org_vocab_end_index, org_vocab_size)
        added_vocab_start_index = min(padded_added_vocab_start_index, vocab_size)
        added_vocab_end_index = min(padded_added_vocab_end_index, vocab_size)
        return VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index,
            padded_org_vocab_end_index,
            padded_added_vocab_start_index,
            padded_added_vocab_end_index,
            org_vocab_start_index,
            org_vocab_end_index,
            added_vocab_start_index,
            added_vocab_end_index,
        )

    def weight_loader(self, param, loaded_weight, shard_id=None):
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)

        if not param._is_initialized():
            param.initialize()

        loaded_weight = get_tensor(loaded_weight)
        if param.dtype != loaded_weight.dtype:
            if loaded_weight.dtype == paddle.int8 and param.dtype == paddle.float8_e4m3fn:
                loaded_weight = loaded_weight.cast(param.dtype)
            else:
                loaded_weight = loaded_weight.cast(param.dtype)

        if output_dim is None:
            assert (
                param.shape == loaded_weight.shape
            ), f"Shape mismatch: param {param.shape} vs loaded_weight {loaded_weight.shape}"
            param.copy_(loaded_weight, False)
            return

        start_idx = self.shard_indices.org_vocab_start_index
        end_idx = self.shard_indices.org_vocab_end_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
            packed_factor = getattr(param, "packed_factor", getattr(param, "pack_factor", 1))
            assert loaded_weight.shape[output_dim] == (self.org_vocab_size // packed_factor)
            start_idx = start_idx // packed_factor
            shard_size = shard_size // packed_factor
        else:
            assert loaded_weight.shape[output_dim] == self.org_vocab_size, (
                f"Loaded weight dim {output_dim} size {loaded_weight.shape[output_dim]} "
                f"!= org_vocab_size {self.org_vocab_size}"
            )

        shard_weight = slice_fn(loaded_weight, output_dim, start_idx, end_idx)

        if output_dim == 0:
            h2d_copy(param[: shard_weight.shape[0]], shard_weight)
            if not current_platform.is_maca():
                param[shard_weight.shape[0] :].fill_(0)
        else:
            h2d_copy(param[:, : shard_weight.shape[1]], shard_weight)
            param[:, shard_weight.shape[1] :].fill_(0)

    def forward(self, ids_remove_padding=None) -> paddle.Tensor:
        """
        Defines the forward computation of the layer.

        Args:
            ids_remove_padding (Tensor, optional): Tensor of token IDs, with padding removed.
                If None, no input is provided.

        Returns:
            Tensor: Embedded tensor representation of the input IDs.
        """
        if self.column_cut:
            input_embedings = self.embeddings(ids_remove_padding)
            inputs_embeds_temp = []
            paddle.distributed.all_gather(
                inputs_embeds_temp,
                input_embedings,
                group=self.tp_group,
                sync_op=True,
            )
            input_embedings = paddle.concat(inputs_embeds_temp, -1)
        else:
            input_embedings = self.embeddings(ids_remove_padding)

        return input_embedings
