# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from fastdeploy.model_executor.ops.xpu import get_infer_param

seq_lens_encoder = paddle.to_tensor([100, 0, 0, 0, 300], dtype="int32")
seq_lens_decoder = paddle.to_tensor([0, 5, 0, 25, 64], dtype="int32")
seq_lens_this_time = paddle.to_tensor([100, 1, 0, 1, 300], dtype="int32")
block_table = paddle.arange(0, 40, dtype="int32")
block_table = block_table.reshape((5, 8))
(
    encoder_batch_map,
    decoder_batch_map,
    encoder_batch_idx,
    decoder_batch_idx,
    encoder_seq_lod,
    decoder_seq_lod,
    encoder_kv_lod,
    prefix_len,
    decoder_context_len,
    decoder_context_len_cache,
    prefix_block_tables,
    encoder_batch_map_cpu,
    decoder_batch_map_cpu,
    encoder_batch_idx_cpu,
    decoder_batch_idx_cpu,
    encoder_seq_lod_cpu,
    decoder_seq_lod_cpu,
    encoder_kv_lod_cpu,
    prefix_len_cpu,
    decoder_context_len_cpu,
    decoder_context_len_cache_cpu,
    len_info_cpu,
) = get_infer_param(
    seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, block_table, 64
)  # block_size

print("block_table", block_table)
print("encoder_batch_map", encoder_batch_map)  # [0, 4, 0, 0, 0]
print("decoder_batch_map", decoder_batch_map)  # [1, 3, 0, 0, 0]
print("encoder_batch_idx", encoder_batch_idx)  # [0, 3, 0, 0, 0]
print("decoder_batch_idx", decoder_batch_idx)  # [1, 2, 0, 0, 0]
print("encoder_seq_lod", encoder_seq_lod)  # [0, 100, 400 ,0 ,0 ,0]
print("decoder_seq_lod", decoder_seq_lod)  # [0, 1,   2   ,0 ,0 ,0]
print("encoder_kv_lod", encoder_kv_lod)  # [0, 100, 464, 0, 0, 0]
print("prefix_len", prefix_len)  # [0, 64, 0, 0, 0]
print("decoder_context_len", decoder_context_len)  # [6, 26, 0, 0, 0]
print("decoder_context_len_cache", decoder_context_len_cache)  # [5, 25, 0, 0, 0]
print("prefix_block_tables", prefix_block_tables)
print("encoder_batch_map_cpu", encoder_batch_map_cpu)  # [0, 4, 0, 0, 0]
print("decoder_batch_map_cpu", decoder_batch_map_cpu)  # [1, 3, 0, 0, 0]
print("encoder_batch_idx_cpu", encoder_batch_idx_cpu)  # [0, 3, 0, 0, 0]
print("decoder_batch_idx_cpu", decoder_batch_idx_cpu)  # [1, 2, 0, 0, 0]
print("encoder_seq_lod_cpu", encoder_seq_lod_cpu)  # [0, 100, 400 ,0 ,0 ,0]
print("decoder_seq_lod_cpu", decoder_seq_lod_cpu)  # [0, 1,   2   ,0 ,0 ,0]
print("encoder_kv_lod_cpu", encoder_kv_lod_cpu)  # [0, 100, 464, 0, 0, 0]
print("prefix_len_cpu", prefix_len_cpu)  # [0, 64, 0, 0, 0]
print("decoder_context_len_cpu", decoder_context_len_cpu)  # [6, 26, 0, 0, 0]
print("decoder_context_len_cache_cpu", decoder_context_len_cache_cpu)  # [5, 25, 0, 0, 0]
print(
    "len_info_cpu", len_info_cpu
)  # {enc_batch, dec_batch, total_enc_len, max_seq_len, max_kv_len, prefix_block_num_per_seq} = [2, 2, 400, 300, 364, 6]

"""
block_table Tensor(shape=[5, 8], dtype=int32, place=Place(xpu:0), stop_gradient=True,
       [[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ],
        [8 , 9 , 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29, 30, 31],
        [32, 33, 34, 35, 36, 37, 38, 39]])

prefix_block_tables Tensor(shape=[5, 8], dtype=int32, place=Place(xpu:0), stop_gradient=True,
       [[ 0,  1, -1, -1, -1, -1, 32, 33],
        [34, 35, 36, 37, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1]])

The size of the prefix_block_tables tensor is same as block_table to avoid problems with InferShape of the prefix_block_tables.
However, the actual size used by prefix_block_tables is [block_bs, prefix_block_num_per_seq], where prefix_block_num_per_seq = ceil(max_kv_len / block_size).
Therefore, do not use the tensor shape of prefix_block_tables. Its shape is obtained through block_table.dims[0] and len_info_cpu[-1]
"""
