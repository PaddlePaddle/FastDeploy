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

from fastdeploy.model_executor.ops.xpu import speculate_clear_accept_nums

np.set_printoptions(threshold=np.inf)  # threshold设为无穷大
np.set_printoptions(linewidth=np.inf)  # 确保一行显示完整（可选）


def speculate_clear_accept_nums_np(accept_num, seq_lens_decoder):
    for i in range(len(accept_num)):
        if seq_lens_decoder[i] == 0:
            accept_num[i] = 0
    return accept_num, seq_lens_decoder


max_bs = 1024
accept_num_np = np.random.randint(low=0, high=11, size=[max_bs], dtype="int32")
accept_num_paddle = paddle.to_tensor(accept_num_np)

seq_lens_decoder_np = np.random.randint(low=0, high=2, size=[max_bs], dtype="int32")
seq_lens_decoder_paddle = paddle.to_tensor(seq_lens_decoder_np)

a = accept_num_paddle.numpy()
# print((a - accept_num_np).sum())
assert (a - accept_num_np).sum() == 0, "Check failed."
accept_num_np, seq_lens_decoder_np = speculate_clear_accept_nums_np(accept_num_np, seq_lens_decoder_np)
seq_lens_decoder_paddle = speculate_clear_accept_nums(accept_num_paddle, seq_lens_decoder_paddle)
b = accept_num_paddle.numpy()
# print(b)
# print((accept_num_np - b).sum())
assert (accept_num_np - b).sum() == 0, "Check failed."
