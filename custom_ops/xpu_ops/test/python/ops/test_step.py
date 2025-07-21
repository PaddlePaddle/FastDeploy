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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import step_paddle

np.random.seed(2023)

max_bs = 128
bs = max_bs
max_seq_len = 8192
block_size = 64
block_bs = 8
block_ratio = 0.75

stop_flags = np.random.randint(0, 2, [max_bs]).astype("bool")
seq_lens_this_time = np.zeros([bs], "int32")
seq_lens_encoder = np.zeros([max_bs], "int32")
seq_lens_decoder = np.zeros([max_bs], "int32")
step_idx = np.zeros([max_bs], "int64")
for i in range(bs):
    seq_lens_decoder[i] = 2 + i * 2
    seq_lens_this_time[i] = 1
ori_seq_lens_encoder = np.zeros([max_bs], "int32")
ori_seq_lens_encoder[:] = seq_lens_decoder[:] // 2
step_idx = (seq_lens_decoder - ori_seq_lens_encoder).astype("int64")

max_block_num = block_bs * max_seq_len // block_size
free_list_len = int(max_block_num * (1 - block_ratio))
free_list_len = np.full([1], free_list_len, "int32")
free_list = np.arange(max_block_num - 1, max_block_num - free_list_len - 1, -1, dtype="int32")

encoder_block_lens = np.zeros([max_bs], "int32")
used_list_len = np.zeros([max_bs], "int32")
block_tables = np.full([max_bs, 128], -1, "int32")
encoder_block_id = 0
for i in range(bs):
    enc_block_num = (ori_seq_lens_encoder[i] + block_size - 1) // block_size
    encoder_block_lens[i] = enc_block_num
    dec_block_num = (seq_lens_decoder[i] + block_size - 1) // block_size - enc_block_num
    used_list_len[i] = dec_block_num
    block_tables[i, :enc_block_num] = np.arange(encoder_block_id, encoder_block_id + enc_block_num, 1, "int32")
    encoder_block_id += enc_block_num
    if dec_block_num > 0:
        block_tables[i, enc_block_num : enc_block_num + dec_block_num] = free_list[
            free_list_len[0] - 1 - dec_block_num : free_list_len[0] - 1
        ]
        free_list[free_list_len[0] - 1 - dec_block_num : free_list_len[0] - 1] = -1
        free_list_len[0] -= dec_block_num
assert free_list_len[0] >= 0

is_block_step = np.zeros([max_bs], "bool")
is_block_step[:bs] = np.random.randint(0, 2, [bs]).astype("bool")
step_block_list = np.full([max_bs], -1, "int32")
step_lens = np.full([1], 0, "int32")
for i in range(bs):
    if is_block_step[i]:
        step_block_list[step_lens[0]] = i
        step_lens[0] += 1

recover_lens = np.full([1], 0, "int32")
recover_block_list = np.full([max_bs], -1, "int32")

need_block_len = np.full([1], 0, "int32")
need_block_list = np.full([max_bs], -1, "int32")

input_ids = np.random.randint(0, 1000, [max_bs, max_seq_len], "int64")
pre_ids = np.random.randint(0, 1000, [max_bs, max_seq_len], "int64")

next_tokens = np.random.randint(0, 1000, [max_bs], "int64")
encoder_decoder_block_num = 1
first_token_ids = np.random.randint(0, 1000, [max_bs], "int64")

stop_flags = paddle.to_tensor(stop_flags)
seq_lens_this_time = paddle.to_tensor(seq_lens_this_time)
seq_lens_encoder = paddle.to_tensor(seq_lens_encoder)
seq_lens_decoder = paddle.to_tensor(seq_lens_decoder)
ori_seq_lens_encoder = paddle.to_tensor(ori_seq_lens_encoder)
block_tables = paddle.to_tensor(block_tables)
encoder_block_lens = paddle.to_tensor(encoder_block_lens)
is_block_step = paddle.to_tensor(is_block_step)
step_block_list = paddle.to_tensor(step_block_list)
step_lens = paddle.to_tensor(step_lens)
recover_lens = paddle.to_tensor(recover_lens)
recover_block_list = paddle.to_tensor(recover_block_list)
need_block_list = paddle.to_tensor(need_block_list)
need_block_len = paddle.to_tensor(need_block_len)
used_list_len = paddle.to_tensor(used_list_len)
free_list_len = paddle.to_tensor(free_list_len)
free_list = paddle.to_tensor(free_list)
input_ids = paddle.to_tensor(input_ids)
pre_ids = paddle.to_tensor(pre_ids)
step_idx = paddle.to_tensor(step_idx)
next_tokens = paddle.to_tensor(next_tokens)
first_token_ids = paddle.to_tensor(first_token_ids)

# print("-" * 50 + "before step op" + "-" * 50)
# print("stop_flags: ", stop_flags)
# print("seq_lens_this_time: ", seq_lens_this_time)
# print("seq_lens_encoder: ", seq_lens_encoder)
# print("seq_lens_decoder: ", seq_lens_decoder)
# print("ori_seq_lens_encoder: ", ori_seq_lens_encoder)
# print("block_tables: ", block_tables)
# print("encoder_block_lens: ", encoder_block_lens)
# print("is_block_step: ", is_block_step)
# print("step_block_list: ", step_block_list)
# print("step_lens: ", step_lens)
# print("recover_lens: ", recover_lens)
# print("recover_block_list: ", recover_block_list)
# print("need_block_list: ", need_block_list)
# print("need_block_len: ", need_block_len)
# print("used_list_len: ", used_list_len)
# print("free_list_len: ", free_list_len)
# print("free_list: ", free_list)
# print("input_ids: ", input_ids)
# print("pre_ids: ", pre_ids)
# print("step_idx: ", step_idx)
# print("next_tokens: ", next_tokens)

step_paddle(
    stop_flags,
    seq_lens_this_time,
    ori_seq_lens_encoder,
    seq_lens_encoder,
    seq_lens_decoder,
    block_tables,
    encoder_block_lens,
    is_block_step,
    step_block_list,
    step_lens,
    recover_block_list,
    recover_lens,
    need_block_list,
    need_block_len,
    used_list_len,
    free_list,
    free_list_len,
    input_ids,
    pre_ids,
    step_idx,
    next_tokens,
    first_token_ids,
    block_size,
    encoder_decoder_block_num,
)

print("-" * 50 + "after step op" + "-" * 50)
print("stop_flags: ", stop_flags)
print("seq_lens_this_time: ", seq_lens_this_time)
print("seq_lens_encoder: ", seq_lens_encoder)
print("seq_lens_decoder: ", seq_lens_decoder)
print("ori_seq_lens_encoder: ", ori_seq_lens_encoder)
print("block_tables: ", block_tables)
print("encoder_block_lens: ", encoder_block_lens)
print("is_block_step: ", is_block_step)
print("step_block_list: ", step_block_list)
print("step_lens: ", step_lens)
print("recover_lens: ", recover_lens)
print("recover_block_list: ", recover_block_list)
print("need_block_list: ", need_block_list)
print("need_block_len: ", need_block_len)
print("used_list_len: ", used_list_len)
print("free_list_len: ", free_list_len)
print("free_list: ", free_list)
print("input_ids: ", input_ids)
print("pre_ids: ", pre_ids)
print("step_idx: ", step_idx)
print("next_tokens: ", next_tokens)
print("first_token_ids: ", first_token_ids)
