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

from fastdeploy.model_executor.ops.gpu import draft_model_postprocess


def draft_model_postprocess_cpu(
    base_model_draft_tokens,
    base_model_seq_lens_encoder,
    base_model_stop_flags,
):
    bsz = base_model_draft_tokens.shape[0]
    base_model_draft_token_len = base_model_draft_tokens.shape[1]
    base_model_seq_lens_this_time = paddle.ones((bsz), dtype=paddle.int32)
    for tid in range(bsz):
        if (not base_model_stop_flags[tid]) and (base_model_seq_lens_encoder[tid] == 0):
            base_model_draft_tokens_now = base_model_draft_tokens[tid]
            token_num = 0
            for i in range(base_model_draft_token_len):
                if base_model_draft_tokens_now[i] != -1:
                    token_num += 1

            base_model_seq_lens_this_time[tid] = token_num
        elif base_model_stop_flags[tid]:
            base_model_seq_lens_this_time[tid] = 0

    return base_model_seq_lens_this_time


class TestDraftModelPostProcess(unittest.TestCase):
    def _test_draft_model_postprocess(self, batch_size=1, base_model_draft_token_len=8192):
        paddle.seed(66)
        base_model_draft_tokens = paddle.randint(
            low=-1,
            high=1,
            shape=[batch_size, base_model_draft_token_len],
            dtype="int64",
        )
        base_model_seq_lens_encoder = paddle.randint(low=0, high=2, shape=[batch_size], dtype="int32")
        random_floats = paddle.rand(shape=[batch_size])
        base_model_stop_flags = random_floats >= 0.5

        base_model_seq_lens_this_time = draft_model_postprocess_cpu(
            base_model_draft_tokens,
            base_model_seq_lens_encoder,
            base_model_stop_flags,
        )
        base_model_seq_lens_this_time_gpu = paddle.ones((batch_size), dtype=paddle.int32)
        draft_model_postprocess(
            base_model_draft_tokens,
            base_model_seq_lens_this_time_gpu,
            base_model_seq_lens_encoder,
            base_model_stop_flags,
        )
        np.testing.assert_allclose(base_model_seq_lens_this_time.numpy(), base_model_seq_lens_this_time_gpu.numpy())

    def test_enough_cases(self):
        self._test_draft_model_postprocess(100, 1024)
        self._test_draft_model_postprocess(1, 11)
        self._test_draft_model_postprocess(1, 8192)
        self._test_draft_model_postprocess(2, 2048)
        self._test_draft_model_postprocess(3, 1023)
        self._test_draft_model_postprocess(4, 2047)
        self._test_draft_model_postprocess(5, 4095)
        self._test_draft_model_postprocess(10, 9191)


if __name__ == "__main__":
    unittest.main()
