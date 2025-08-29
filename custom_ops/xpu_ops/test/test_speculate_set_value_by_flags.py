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

if paddle.is_compiled_with_xpu():
    from fastdeploy.model_executor.ops.xpu import speculate_set_value_by_flags_and_idx
else:
    from efficientllm.ops.gpu import speculate_set_value_by_flags_and_idx


def test_speculate_set_value_by_flags_and_idx():
    # 将accept_tokens添加到pre_ids的特定位置
    bs = 256
    length = 8192
    max_draft_tokens = 4

    pre_ids_all = paddle.to_tensor(np.full((bs, length), -1), dtype="int64")

    accept_tokens = np.random.randint(100, 200, size=(bs, max_draft_tokens))
    accept_tokens = paddle.to_tensor(accept_tokens, dtype="int64")

    accept_num = np.random.randint(0, max_draft_tokens + 1, size=bs)
    accept_num = paddle.to_tensor(accept_num, dtype="int32")

    stop_flags = np.random.choice([True, False, False, False], size=bs)
    stop_flags = paddle.to_tensor(stop_flags, dtype="bool")

    seq_lens_this_time = paddle.to_tensor(np.full((bs), 1), dtype="int32")
    seq_lens_encoder = paddle.to_tensor(np.full((bs), 0), dtype="int32")
    seq_lens_decoder = paddle.to_tensor(np.full((bs), 2), dtype="int32")

    step_idx = np.random.randint(max_draft_tokens, length, size=bs)
    step_idx = paddle.to_tensor(step_idx, dtype="int64")

    out_xpu = speculate_set_value_by_flags_and_idx(
        pre_ids_all,
        accept_tokens,
        accept_num,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
    )
    out_xpu = out_xpu.numpy()

    out_cpu = paddle.to_tensor(np.full((bs, length), -1), dtype="int64")
    for i in range(bs):
        if stop_flags[i] or (seq_lens_encoder[i] == 0 and seq_lens_decoder[i] == 0):
            continue
        if step_idx[i] >= 0:
            for j in range(accept_num[i]):
                out_cpu[i, step_idx[i] - j] = accept_tokens[i, accept_num[i] - 1 - j]

    # print(f"accept_tokens: {accept_tokens}")
    # print(f"accept_num: {accept_num}")
    # print(f"stop_flags: {stop_flags}")
    # print(f"seq_lens_this_time: {seq_lens_this_time}")
    # print(f"seq_lens_encoder: {seq_lens_encoder}")
    # print(f"seq_lens_decoder: {seq_lens_decoder}")
    # print(f"step_idx: {step_idx}")
    # print(f"out_xpu: {out_xpu}")
    # print(f"out_cpu: {out_cpu}")

    assert np.array_equal(out_xpu, out_cpu), "out_xpu != out_cpu"
    print("test_speculate_set_value_by_flags_and_idx passed!")


if __name__ == "__main__":
    test_speculate_set_value_by_flags_and_idx()
