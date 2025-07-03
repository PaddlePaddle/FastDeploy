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

import abc
import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle

if TYPE_CHECKING:
    from fastdeploy.model_executor.layers.attention import (Attention,
                                                            AttentionBackend)

logger = logging.getLogger(__name__)


class ForwardMode(IntEnum):
    """
    Forward mode used during attention.
    """

    # for prefill and extend
    EXTEND = auto()
    # for generation
    DECODE = auto()

    MIXED = auto()

    def is_prefill(self):
        """Whether it's a prefill forward"""
        return self == ForwardMode.EXTEND

    def is_decode(self):
        """Whether it's a decode forward"""
        return self == ForwardMode.DECODE

    def is_mixed(self):
        """Whether it's a decode forward"""
        return self == ForwardMode.MIXED


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size: int, max_context_len: int):

        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = paddle.zeros((size, max_context_len),
                                         dtype=paddle.int32)
        self.free_slots = list(range(size))

    def write(self, indices, values):
        """Write data into request buffer"""
        self.req_to_token[indices] = values

    def available_size(self):
        """Get number of slots left"""
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        """Allocate `need_size` slots"""
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        """Free slot"""
        if isinstance(free_index, (int, )):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        """Clear all slots"""
        self.free_slots = list(range(self.size))


class KVCache(abc.ABC):
    """Abstract base class representing a key value cache"""

    @abc.abstractmethod
    def get_kv_buffer(self,
                      layer_id: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Return cached keys and values given layer id.
        Args:
        layer_id: int
        Returns:
            tuple: (keys, values)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: 'Attention',
        loc: paddle.Tensor,
        cache_k: paddle.Tensor,
        cache_v: paddle.Tensor,
    ) -> None:
        """
        Set cached keys and values given layer id.
        Args:
        layer: Attention
        loc: paddle.Tensor
        cache_k: paddle.Tensor
        cache_v: paddle.Tensor
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        """Transfer kv_data between devices"""
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer_per_layer(self, indices, flat_data, layer_id):
        """Not used yet"""
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        """Not used yet"""
        self.layer_transfer_counter = layer_transfer_counter


class MHATokenToKVPool(KVCache):
    """Token To Key Value Pool for MultiHeadAttention"""

    def __init__(
        self,
        max_block_num: int,
        block_size: int,
        dtype: paddle.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.max_block_num = max_block_num
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        if dtype in (paddle.int8, paddle.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = paddle.uint8
        else:
            self.store_dtype = dtype

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers()

        k_size, v_size = self.get_kv_size_bytes()
        GB = 1024 * 1024 * 1024
        logger.info(
            f"KV Cache is allocated. #tokens: {self.size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            paddle.zeros(
                (self.max_block_num, self.head_num, self.block_size,
                 self.head_dim),
                dtype=self.store_dtype,
            ) for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            paddle.zeros(
                (self.max_block_num, self.head_num, self.block_size,
                 self.head_dim),
                dtype=self.store_dtype,
            ) for _ in range(self.layer_num)
        ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        """for debugging purpose"""
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * 4
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * 4
        return k_size_bytes, v_size_bytes

    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data for a specific layer from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        self.k_buffer[layer_id][indices] = k_data
        self.v_buffer[layer_id][indices] = v_data

    def get_key_buffer(self, layer_id: int):
        """Return cached keys given layer id."""
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        """Return cached values given layer id."""
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        """Return cached keys and values given layer id."""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: 'Attention',
        loc: paddle.Tensor,
        cache_k: paddle.Tensor,
        cache_v: paddle.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        """Set cached keys and values given layer id."""
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v


@dataclass
class ForwardMeta():
    """
    ForwardMeta is used to store the global meta information of the forward.
    """
    #
    input_ids: paddle.Tensor

    #attention meta
    forward_mode: ForwardMode = ForwardMode.MIXED

    #
    ids_remove_padding: paddle.Tensor = None

    #
    seq_lens_encoder: Optional[paddle.Tensor] = None

    #
    seq_lens_decoder: Optional[paddle.Tensor] = None

    #
    seq_lens_this_time: Optional[paddle.Tensor] = None

    #
    cum_offsets: Optional[paddle.Tensor] = None

    #
    block_tables: Optional[paddle.Tensor] = None

    #
    attn_backend: 'AttentionBackend' = None

    #
    rotary_embs: Optional[paddle.Tensor] = None

    #
    padding_offset: Optional[paddle.Tensor] = None

    #
    cu_seqlens_q: Optional[paddle.Tensor] = None

    #
    cu_seqlens_k: Optional[paddle.Tensor] = None

    #
    caches: Optional[paddle.Tensor] = None

    #
    attn_mask: Optional[paddle.Tensor] = None

    #
    pre_caches_length: int = 0

    # Use cuda graph in this step. Used to avoid run cuda graph when in dummy run or prefill stage.
    step_use_cudagraph: bool = False

    # for attention backend
    decoder_batch_ids: Optional[paddle.Tensor] = None
    # for attention backend
    decoder_tile_ids_per_batch: Optional[paddle.Tensor] = None
    # is_decode_batch or not
    is_decode_batch: bool = False

    @classmethod
    def init_forward_meta(cls, share_inputs: Dict,
                          attn_backend: "AttentionBackend"):
        """ init forward meta """
        # TODO(gongshaotian): delete this func
        ret = cls(
            forward_mode=ForwardMode.MIXED,
            input_ids=share_inputs["input_ids"],
            ids_remove_padding=share_inputs["ids_remove_padding"],
            seq_lens_encoder=share_inputs["seq_lens_encoder"],
            seq_lens_decoder=share_inputs["seq_lens_decoder"],
            seq_lens_this_time=share_inputs["seq_lens_this_time"],
            cum_offsets=share_inputs["cum_offsets"],
            block_tables=share_inputs["block_tables"],
            attn_backend=attn_backend,
            rotary_embs=share_inputs["rope_emb"],
            padding_offset=share_inputs["padding_offset"],
            cu_seqlens_q=share_inputs["cu_seqlens_q"],
            cu_seqlens_k=share_inputs["cu_seqlens_k"],
            caches=share_inputs["caches"],
            decoder_batch_ids=share_inputs.get("decoder_batch_ids", None),
            decoder_tile_ids_per_batch=share_inputs.get(
                "decoder_tile_ids_per_batch", None),
        )
        return ret
    
    def clear_caches(self):
        """safe clear caches"""
        if self.caches:
            del self.caches


@dataclass
class XPUForwardMeta(ForwardMeta):
    """
    XPUForwardMeta is used to store the global meta information of the forward, and some XPU specific meta info.
    """
    #
    encoder_batch_map: Optional[paddle.Tensor] = None

    #
    decoder_batch_map: Optional[paddle.Tensor] = None

    #
    encoder_batch_idx: Optional[paddle.Tensor] = None

    #
    decoder_batch_idx: Optional[paddle.Tensor] = None

    #
    encoder_seq_lod: Optional[paddle.Tensor] = None

    #
    decoder_context_len: Optional[paddle.Tensor] = None

    #
    decoder_context_len_cache: Optional[paddle.Tensor] = None

    #
    encoder_batch_map_cpu: Optional[paddle.Tensor] = None

    #
    decoder_batch_map_cpu: Optional[paddle.Tensor] = None

    #
    encoder_batch_idx_cpu: Optional[paddle.Tensor] = None

    #
    decoder_batch_idx_cpu: Optional[paddle.Tensor] = None

    #
    encoder_seq_lod_cpu: Optional[paddle.Tensor] = None

    #
    decoder_context_len_cpu: Optional[paddle.Tensor] = None

    #
    decoder_context_len_cache_cpu: Optional[paddle.Tensor] = None

    #
    batch_tensor: Optional[paddle.Tensor] = None

    #
    enc_batch: Optional[paddle.Tensor] = None

    #
    dec_batch: Optional[paddle.Tensor] = None

    #
    total_enc_len: Optional[paddle.Tensor] = None
