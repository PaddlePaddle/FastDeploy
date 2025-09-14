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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_get_output_padding_offset


class TestSpeculateGetOutputPaddingOffset(unittest.TestCase):
    def test_speculate_get_output_padding_offset(self):
        bsz = 256
        max_seq_len = 8192

        seq_lens_output = np.random.randint(0, 4, size=bsz)
        output_token_num = np.sum(seq_lens_output)

        seq_lens_output = paddle.to_tensor(seq_lens_output, dtype="int32")
        out_token_num = paddle.sum(seq_lens_output).astype("int32")
        output_cum_offsets_tmp = paddle.cumsum(max_seq_len - seq_lens_output).astype("int32")

        output_padding_offset_gpu, output_cum_offsets_gpu = speculate_get_output_padding_offset(
            output_cum_offsets_tmp, out_token_num, seq_lens_output, max_seq_len
        )

        output_padding_offset_ref = [-1] * output_token_num
        output_cum_offsets_ref = [-1] * bsz

        for bi in range(bsz):
            cum_offset = 0 if bi == 0 else output_cum_offsets_tmp[bi - 1]
            output_cum_offsets_ref[bi] = cum_offset
            for token_i in range(seq_lens_output[bi]):
                output_padding_offset_ref[bi * max_seq_len - cum_offset + token_i] = cum_offset

        np.testing.assert_allclose(output_padding_offset_gpu, output_padding_offset_ref)
        np.testing.assert_allclose(output_cum_offsets_gpu, output_cum_offsets_ref)


if __name__ == "__main__":
    unittest.main()
