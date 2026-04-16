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

from fastdeploy.model_executor.ops.xpu import get_padding_offset

np.random.seed(2023)

max_len = 10
seq_lens = np.array([4, 3, 6], "int32").reshape(-1, 1)
token_num = int(np.sum(seq_lens))
bs = seq_lens.shape[0]
input_ids = np.zeros([bs, max_len], "int64")
for i in range(bs):
    ids_len = seq_lens[i, 0]
    input_ids[i, 0:ids_len] = np.random.randint(1, 10, seq_lens[i, 0], "int64")

(
    x_remove_padding,
    cum_offsets_out,
    batch_id_per_token,
    cu_seqlens_q,
    cu_seqlens_k,
) = get_padding_offset(
    paddle.to_tensor(input_ids),
    paddle.to_tensor(seq_lens.flatten()),
    token_num,
)

print("input_ids:\n", input_ids)
print("seq_lens:\n", seq_lens.flatten())
print("token_num:\n", token_num)
print("x_remove_padding:\n", x_remove_padding)
print("cum_offsets_out:\n", cum_offsets_out)
print("batch_id_per_token:\n", batch_id_per_token)
print("cu_seqlens_q:\n", cu_seqlens_q)
print("cu_seqlens_k:\n", cu_seqlens_k)

ref_x_remove_padding = np.array([8, 7, 8, 2, 4, 5, 5, 7, 6, 1, 7, 2, 6], "int64")
ref_cum_offsets_out = np.array([0, 6, 13], "int32")
ref_batch_id_per_token = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], "int32")
ref_cu_seqlens_q = np.array([0, 4, 7, 13], "int32")
ref_cu_seqlens_k = np.array([0, 4, 7, 13], "int32")

assert (
    np.sum(np.abs(ref_x_remove_padding - x_remove_padding.numpy())) == 0
), f"Check x_remove_padding failed.\nref: {ref_x_remove_padding}\ngot: {x_remove_padding.numpy()}"
assert (
    np.sum(np.abs(ref_cum_offsets_out - cum_offsets_out.numpy())) == 0
), f"Check cum_offsets_out failed.\nref: {ref_cum_offsets_out}\ngot: {cum_offsets_out.numpy()}"
assert (
    np.sum(np.abs(ref_batch_id_per_token - batch_id_per_token.numpy())) == 0
), f"Check batch_id_per_token failed.\nref: {ref_batch_id_per_token}\ngot: {batch_id_per_token.numpy()}"
assert (
    np.sum(np.abs(ref_cu_seqlens_q - cu_seqlens_q.numpy())) == 0
), f"Check cu_seqlens_q failed.\nref: {ref_cu_seqlens_q}\ngot: {cu_seqlens_q.numpy()}"
assert (
    np.sum(np.abs(ref_cu_seqlens_k - cu_seqlens_k.numpy())) == 0
), f"Check cu_seqlens_k failed.\nref: {ref_cu_seqlens_k}\ngot: {cu_seqlens_k.numpy()}"

print("\nAll checks passed!")
