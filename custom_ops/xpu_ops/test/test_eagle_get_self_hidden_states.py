# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.model_executor.ops.xpu import eagle_get_self_hidden_states


def computeOrder(last_seq_lens_this_time, seq_lens_this_time, step_idx, src_map, bsz):
    in_offset = 0
    out_offset = 0
    for i in range(bsz):
        cur_seq_lens_this_time = seq_lens_this_time[i]
        cur_last_seq_lens_this_time = last_seq_lens_this_time[i]

        # 1. encoder
        if step_idx[i] == 1 and cur_seq_lens_this_time > 0:
            in_offset += 1
            src_map[out_offset] = in_offset - 1
            out_offset += 1
        # 2. decoder
        elif cur_seq_lens_this_time > 0:
            in_offset += cur_last_seq_lens_this_time
            src_map[out_offset] = in_offset - 1
            out_offset += 1
        # 3. stop
        else:
            # first token end
            if step_idx[i] == 1:
                in_offset += 1 if cur_last_seq_lens_this_time > 0 else 0
            # normal end
            else:
                in_offset += cur_last_seq_lens_this_time

    return (out_offset, src_map)


def rebuildSelfHiddenStatesKernel(input, src_map, out, dim_embed, elem_cnt):
    print(f"input.shape {input.shape}")
    print(f"out.shape {out.shape}")
    print(f"elem_cnt {elem_cnt}")
    for elem_id in range(elem_cnt):
        output_token_idx = elem_id // dim_embed
        input_token_idx = src_map[output_token_idx]
        offset = elem_id % dim_embed
        out[output_token_idx * dim_embed + offset] = input[input_token_idx * dim_embed + offset]
    return out


def ref_eagle_get_self_hidden_states(input, last_seq_lens_this_time, seq_lens_this_time, step_idx):
    input_token_num = input.shape[0]
    dim_embed = input.shape[1]
    bsz = seq_lens_this_time.shape[0]
    src_map = np.full(input_token_num, -1, seq_lens_this_time.dtype)

    output_token_num, src_map = computeOrder(last_seq_lens_this_time, seq_lens_this_time, step_idx, src_map, bsz)

    out = np.full([output_token_num * dim_embed], -1, input.dtype)

    elem_cnt = output_token_num * dim_embed

    out = rebuildSelfHiddenStatesKernel(input, src_map, out, dim_embed, elem_cnt)
    out = out.reshape([output_token_num, dim_embed])

    return out


def test_eagle_get_self_hidden_states():
    bs = np.random.randint(1, 8 + 1, dtype=np.int32)
    input_token_num = np.random.randint(2 * 1024, 4 * 1024 + 1, dtype=np.int32)
    dim_embed = np.random.randint(1, 4 * 1024 + 1, dtype=np.int32)

    last_seq_lens_this_time = np.random.randint(0, input_token_num // bs, bs, dtype=np.int32)
    seq_lens_this_time = np.random.randint(0, input_token_num // bs, bs, dtype=np.int32)
    step_idx = np.arange(0, bs, dtype=np.int32)

    last_seq_lens_this_time_tensor = paddle.to_tensor(last_seq_lens_this_time, dtype=paddle.int32)
    seq_lens_this_time_tensor = paddle.to_tensor(seq_lens_this_time, dtype=paddle.int32)
    step_idx_tensor = paddle.to_tensor(step_idx, dtype=paddle.int64)

    # fp32 test
    input = np.random.randint(0, 10, (input_token_num, dim_embed), dtype=np.int32)
    input_tensor = paddle.to_tensor(input, dtype=paddle.float32)
    cpu_out = eagle_get_self_hidden_states(
        input_tensor.cpu(),
        last_seq_lens_this_time_tensor.cpu(),
        seq_lens_this_time_tensor.cpu(),
        step_idx_tensor.cpu(),
    )
    xpu_out = eagle_get_self_hidden_states(
        input_tensor,
        last_seq_lens_this_time_tensor,
        seq_lens_this_time_tensor,
        step_idx_tensor,
    )
    assert np.allclose(cpu_out.numpy(), xpu_out.numpy())

    # bf16/fp16 test
    for dtype in [paddle.bfloat16, paddle.float16]:
        input = np.random.randint(0, 10, (input_token_num, dim_embed), dtype=np.int16)
        input_tensor = paddle.to_tensor(input, dtype=dtype)
        cpu_out = eagle_get_self_hidden_states(
            input_tensor.cpu(),
            last_seq_lens_this_time_tensor.cpu(),
            seq_lens_this_time_tensor.cpu(),
            step_idx_tensor.cpu(),
        )
        xpu_out = eagle_get_self_hidden_states(
            input_tensor,
            last_seq_lens_this_time_tensor,
            seq_lens_this_time_tensor,
            step_idx_tensor,
        )
        assert np.allclose(cpu_out.numpy(), xpu_out.numpy())

    print("All test passed")


if __name__ == "__main__":
    test_eagle_get_self_hidden_states()
