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

"""UT for GPU operator stop_generation_multi_ends"""

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import set_stop_value_multi_ends


def test_set_stop_value_multi_ends_with_stop_seq():
    sampled_token_ids = paddle.to_tensor([[61502], [2]], dtype="int64")
    stop_flags = paddle.to_tensor([[False], [True]], dtype="bool")
    seq_lens_this_time = paddle.to_tensor([[1], [0]], dtype="int32")
    eos_token_id = paddle.to_tensor([2], dtype="int64")
    next_tokens = paddle.to_tensor([[61502], [2]], dtype="int64")

    pre_ids = paddle.full([2, 32768], -1, dtype="int64")
    pre_ids[0, :10] = np.array([21, 22, 23, 24, 25, 26, 27, 28, 8038, 61502])
    step_idx = paddle.to_tensor([[10], [0]], dtype="int64")

    stop_token_ids = paddle.full([2, 5, 8], -1, dtype="int64")
    stop_token_ids[0, 0, :2] = np.array([8038, 61502])

    stop_seqs_len = paddle.full([2, 5], 10, dtype="int32")
    stop_seqs_len[0, 0] = 2

    set_stop_value_multi_ends(
        sampled_token_ids,
        stop_flags,
        seq_lens_this_time,
        eos_token_id,
        next_tokens,
        pre_ids,
        step_idx,
        stop_token_ids,
        stop_seqs_len,
        False,
    )

    assert bool(stop_flags[0, 0]) is True
    assert sampled_token_ids[0, 0] == 2  # eos token id


if __name__ == "__main__":
    test_set_stop_value_multi_ends_with_stop_seq()
