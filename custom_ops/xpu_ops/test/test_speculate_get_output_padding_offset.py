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
    from fastdeploy.model_executor.ops.xpu import speculate_get_output_padding_offset
else:
    from efficientllm.ops.gpu import speculate_get_output_padding_offset


def test_speculate_get_output_padding_offset():
    bsz = 256
    max_seq_len = 8192

    seq_lens_output = np.random.randint(0, 4, size=bsz)
    output_token_num = np.sum(seq_lens_output)

    seq_lens_output = paddle.to_tensor(seq_lens_output, dtype="int32")
    out_token_num = paddle.sum(seq_lens_output)
    output_cum_offsets_tmp = paddle.cumsum(max_seq_len - seq_lens_output, dtype="int32")

    output_padding_offset_xpu, output_cum_offsets_xpu = speculate_get_output_padding_offset(
        output_cum_offsets_tmp, out_token_num, seq_lens_output, max_seq_len
    )

    output_padding_offset_cpu = [-1] * output_token_num
    output_cum_offsets_cpu = [-1] * bsz

    for bi in range(bsz):
        cum_offset = 0 if bi == 0 else output_cum_offsets_tmp[bi - 1]
        output_cum_offsets_cpu[bi] = cum_offset
        for token_i in range(seq_lens_output[bi]):
            output_padding_offset_cpu[bi * max_seq_len - cum_offset + token_i] = cum_offset

    # print(f"seq_lens_output: {seq_lens_output}")
    # print(f"output_cum_offsets_tmp: {output_cum_offsets_tmp}")
    # print(f"output_padding_offset_xpu: {output_padding_offset_xpu}")
    # print(f"output_cum_offsets_xpu: {output_cum_offsets_xpu}")
    # print(f"output_padding_offset_cpu: {output_padding_offset_cpu}")
    # print(f"output_cum_offsets_cpu: {output_cum_offsets_cpu}")

    assert np.array_equal(
        output_padding_offset_xpu, output_padding_offset_cpu
    ), "output_padding_offset_xpu != output_padding_offset_cpu"
    assert np.array_equal(
        output_cum_offsets_xpu, output_cum_offsets_cpu
    ), "output_cum_offsets_xpu != output_cum_offsets_cpu"

    print("test_speculate_get_output_padding_offset passed!")


if __name__ == "__main__":
    test_speculate_get_output_padding_offset()
