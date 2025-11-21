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
from typing import TYPE_CHECKING, Dict, Optional

import paddle

from fastdeploy.model_executor.layers.attention import AttentionBackend

if TYPE_CHECKING:
    from fastdeploy.model_executor.layers.attention import AttentionBackend_HPU
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

    # A common pattern for launching CUDA kernels is to set the kernel's grids.x dimension
    # using a `num_blocks` variable, and then map each thread block to a specific batch and
    # data tile using `batch_ids` and `tile_ids_per_batch`.
    #
    # The variable names below follow this pattern, using a common prefix (e.g., `encoder_`, `decoder_`, `kv_`)
    # for variables that are logically grouped together. The mapping works as follows:
    #
    # Usage: `my_kernel<<<grids, ...>>>(..., batch_ids, tile_ids, ...)`
    #   `grids.x` = `num_blocks_cpu`
    #   `batch_id` = `batch_ids[blockIdx.x]`
    #   `tile_id`  = `tile_ids[blockIdx.x]`

    # Maps the thread block index (blockIdx.x) to the corresponding batch for the decoder stage in multi_query_append_attention_warp1_4_kernel.
    # Decoder batch id. Used by attention backend.
    decoder_batch_ids: Optional[paddle.Tensor] = None
    # Maps the thread block index (blockIdx.x) to the specific data tile being processed within that batch for the decoder stage in multi_query_append_attention_warp1_4_kernel.
    decoder_tile_ids_per_batch: Optional[paddle.Tensor] = None
    # The number of blocks that attention backend can use in decode stage
    decoder_num_blocks_device: Optional[paddle.Tensor] = None
    # The number of CUDA blocks to launch in the x-dimension for the multi_query_append_attention_warp1_4_kernel, defining its grids.x.
    decoder_num_blocks_cpu: Optional[paddle.Tensor] = None
    # A tensor that holds multiple lengths related to prefill or decode stages.
    max_len_tensor_cpu: Optional[paddle.Tensor] = None
    # Maps the thread block index (blockIdx.x) to the corresponding batch for the encoder stage in multi_query_append_attention_kernel.
    encoder_batch_ids: Optional[paddle.Tensor] = None
    # Maps the thread block index (blockIdx.x) to the specific data tile being processed within that batch for the encoder stage in multi_query_append_attention_kernel.
    encoder_tile_ids_per_batch: Optional[paddle.Tensor] = None
    # The number of CUDA blocks to launch in the x-dimension for the multi_query_append_attention_kernel, defining its grids.x.
    encoder_num_blocks_x_cpu: Optional[paddle.Tensor] = None
    # Maps the thread block index (blockIdx.x) to the corresponding batch for the append_write_cache_kv kernel.
    kv_batch_ids: Optional[paddle.Tensor] = None
    # Maps the thread block index (blockIdx.x) to the specific data tile being processed within that batch for the append_write_cache_kv kernel.
    kv_tile_ids_per_batch: Optional[paddle.Tensor] = None
    # The number of CUDA blocks to launch in the x-dimension for the append_write_cache_kv kernel, defining its grids.x.
    kv_num_blocks_x_cpu: Optional[paddle.Tensor] = None

    decoder_chunk_size_device: Optional[paddle.Tensor] = None

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
    # Flag of profile run
    is_dummy_or_profile_run: bool = False

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
                    "content": obj if obj.numel() < 70 else "Too big to show",
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
    # TODO(yinwei): Supplementary notes
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
    decoder_seq_lod: Optional[paddle.Tensor] = None
    #
    encoder_kv_lod: Optional[paddle.Tensor] = None
    #
    prefix_len: Optional[paddle.Tensor] = None
    #
    decoder_context_len: Optional[paddle.Tensor] = None
    #
    decoder_context_len_cache: Optional[paddle.Tensor] = None
    #
    prefix_block_tables: Optional[paddle.Tensor] = None
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
    decoder_seq_lod_cpu: Optional[paddle.Tensor] = None
    #
    encoder_kv_lod_cpu: Optional[paddle.Tensor] = None
    #
    prefix_len_cpu: Optional[paddle.Tensor] = None
    #
    decoder_context_len_cpu: Optional[paddle.Tensor] = None
    #
    decoder_context_len_cache_cpu: Optional[paddle.Tensor] = None
    #
    len_info_cpu: Optional[paddle.Tensor] = None
    #
    batch_tensor: Optional[paddle.Tensor] = None
    #
    enc_batch: Optional[paddle.Tensor] = None
    #
    dec_batch: Optional[paddle.Tensor] = None
    #
    total_enc_len: Optional[paddle.Tensor] = None
    # position embedding type in rope, supports 'NORMAL' or 'HALF_HEAD_DIM'
    pos_emb_type: Optional[str] = "NORMAL"
    # for pd_disaggregation
    kv_signal_sender: Optional[paddle.Tensor] = None


@dataclass
class DCUForwardMeta(ForwardMeta):
    """
    DCUForwardMeta is used to store the global meta information of the forward, and some DCU specific meta info.
    """

    # Accumulated offset
    cum_offsets: Optional[paddle.Tensor] = None


@dataclass
class HPUForwardMeta(ForwardMeta):
    """
    HPUForwardMeta is used to store the global meta information of the forward on intel HPU.
    """

    #
    input_ids: paddle.Tensor = None

    # attention meta
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
    block_groups: Optional[paddle.Tensor] = None

    #
    block_list: Optional[paddle.Tensor] = None

    #
    block_indices: Optional[paddle.Tensor] = None

    #
    block_offsets: Optional[paddle.Tensor] = None

    #
    block_mapping: Optional[paddle.Tensor] = None

    #
    attention_mask: Optional[paddle.Tensor] = None

    #
    block_size: Optional[paddle.Tensor] = None

    #
    batch_ids: Optional[paddle.Tensor] = None

    #
    total_batch: Optional[paddle.Tensor] = None

    #
    is_prompt: Optional[paddle.Tensor] = None

    #
    attn_backend: "AttentionBackend_HPU" = None

    #
    rotary_embs: Optional[paddle.Tensor] = None

    #
    caches: Optional[paddle.Tensor] = None

    #
    attn_mask: Optional[paddle.Tensor] = None

    #
    pre_caches_length: int = 0

    @classmethod
    def init_forward_meta(cls, share_inputs: Dict, attn_backend: "AttentionBackend_HPU"):
        """init forward meta"""
        # TODO(gongshaotian): delete this func
        is_prompt = share_inputs["is_prompt"]
        forward_mode = ForwardMode.DECODE
        if is_prompt:
            forward_mode = ForwardMode.EXTEND
        ret = cls(
            forward_mode=forward_mode,
            input_ids=share_inputs["input_ids"],
            ids_remove_padding=share_inputs["ids_remove_padding"],
            seq_lens_encoder=share_inputs["seq_lens_encoder"],
            seq_lens_decoder=share_inputs["seq_lens_decoder"],
            seq_lens_this_time=share_inputs["seq_lens_this_time"],
            block_tables=share_inputs["block_tables"],
            block_groups=share_inputs["block_groups"],
            block_list=share_inputs["block_list"],
            block_indices=share_inputs["block_indices"],
            block_offsets=share_inputs["block_offsets"],
            block_mapping=share_inputs["block_mapping"],
            attention_mask=share_inputs["block_bias"],
            block_size=share_inputs["block_size"],
            total_batch=share_inputs["total_batch"],
            batch_ids=share_inputs["batch_ids"],
            is_prompt=share_inputs["is_prompt"],
            attn_backend=attn_backend,
            rotary_embs=share_inputs["rotary_embs"],
            caches=share_inputs["caches"],
        )
        return ret

    def clear_caches(self):
        """safe clear caches"""
        if self.caches:
            del self.caches
