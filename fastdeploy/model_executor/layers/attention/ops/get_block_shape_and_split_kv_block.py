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

import paddle

from fastdeploy.platforms import current_platform


def get_block_shape_and_split_kv_block(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    cum_offsets: paddle.Tensor,
    encoder_block_shape_q: int,
    decoder_block_shape_q: int,
    group_size: int,
    block_size: int,
    decoder_step_token_num: int
):
    """
    get_block_shape_and_split_kv_block
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import get_block_shape_and_split_kv_block
        (
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_len_kv,
            set_max_lengths,
        ) = get_block_shape_and_split_kv_block(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            cum_offsets,
            encoder_block_shape_q,
            decoder_block_shape_q,
            group_size,
            block_size,
            decoder_step_token_num
        )
        return (
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            max_len_kv,
            set_max_lengths,
        )
    else:
        raise NotImplementedError()
