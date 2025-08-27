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

import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional

import paddle

from fastdeploy.model_executor.layers.attention import AttentionBackend

logger = logging.getLogger(__name__)


class ForwardMode(IntEnum):
    """
    Forward mode used during attention.
    """

    # Prefill and Extend mode
    EXTEND = auto()
    # Decode mode
    DECODE = auto()
    # Mixed mode
    MIXED = auto()
    # Native mode
    NATIVE = auto()

    def is_prefill(self):
        """Is Extend mode"""
        return self == ForwardMode.EXTEND

    def is_decode(self):
        """Is Decode mode"""
        return self == ForwardMode.DECODE

    def is_mixed(self):
        """Is Mixed mode"""
        return self == ForwardMode.MIXED

    def is_native(self):
        """Is Native mode"""
        return self == ForwardMode.NATIVE


@dataclass
class ForwardMeta:
    """
    ForwardMeta is used to store the global meta information of the model forward.
    """

    # Input tokens IDs
    input_ids: paddle.Tensor
    # Input tokens IDs of removed padding
    ids_remove_padding: paddle.Tensor
    # Rotation position embedding
    rotary_embs: Optional[paddle.Tensor] = None

    # Use cuda graph in this step or not. Used to avoid run cuda graph when in dummy run or prefill stage.
    step_use_cudagraph: bool = False

    # Attention backend object
    attn_backend: AttentionBackend = None
    # Forward mode used during attention
    forward_mode: ForwardMode = ForwardMode.MIXED
    # Attention mask
    attn_mask: Optional[paddle.Tensor] = None
    # Attention mask offset
    attn_mask_offsets: Optional[paddle.Tensor] = None
    # Decoder batch id. Used by attention backend.
    decoder_batch_ids: Optional[paddle.Tensor] = None
    # Tile ID for each batch of the decoder. Used by attention backend.
    decoder_tile_ids_per_batch: Optional[paddle.Tensor] = None
    # The number of blocks that attention backend can use in decode stage
    decoder_num_blocks_cpu: Optional[paddle.Tensor] = None
    # Recorded multiple lengths related to prefill or decode
    max_len_tensor_cpu: Optional[paddle.Tensor] = None

    # Sequence length of encoder for ever batch
    seq_lens_encoder: Optional[paddle.Tensor] = None
    # Sequence length of Encoder for ever batch
    seq_lens_decoder: Optional[paddle.Tensor] = None
    # The sequence length processed in the current step
    seq_lens_this_time: Optional[paddle.Tensor] = None

    # batch_id_per_token tensor, used to indicate which token belongs which batch after padding removal to the original input_ids
    batch_id_per_token: Optional[paddle.Tensor] = None
    # Accumulated sequence length of query
    cu_seqlens_q: Optional[paddle.Tensor] = None
    # Accumulated sequence length of key
    cu_seqlens_k: Optional[paddle.Tensor] = None

    # Pre-cache length
    pre_caches_length: int = 0
    # Block tables
    block_tables: Optional[paddle.Tensor] = None
    # KV caches
    caches: Optional[list[paddle.Tensor]] = None

    def clear_caches(self):
        """Safely clean up the caches"""
        if self.caches:
            del self.caches

    def __str__(self) -> str:
        """
        Returns a concise string representation of the ForwardMeta object in a compact format.
        """

        def format_str(obj):
            """
            A helper function to recursively get a concise string representation of objects.
            """
            if obj is None:
                return "None"
            elif isinstance(obj, paddle.Tensor):
                tensor_info = {
                    "data_ptr": obj.data_ptr(),
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "place": str(obj.place),
                }
                return tensor_info
            elif isinstance(obj, (list, tuple)):
                return [format_str(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: format_str(value) for key, value in obj.items()}
            elif not isinstance(obj, (int, float, str, bool)) and hasattr(obj, "__dict__"):
                info = {key: format_str(value) for key, value in obj.__dict__.items() if not key.startswith("_")}
                return f"<{obj.__class__.__name__} object info: {info}>"
            else:
                return str(obj)

        simplified_info = format_str(self.__dict__)
        lines = [f"  {key}: {value}" for key, value in simplified_info.items()]
        return "{\n" + ",\n".join(lines) + "\n}"


@dataclass
class XPUForwardMeta(ForwardMeta):
    """
    XPUForwardMeta is used to store the global meta information of the forward, and some XPU specific meta info.
    """

    # Accumulated offset
    cum_offsets: Optional[paddle.Tensor] = None
    # TODO(wanghaitao): Supplementary notes
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
