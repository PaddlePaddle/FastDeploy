"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import hybrid_mtp_ngram


class TestNgramMatchMixed(unittest.TestCase):
    def setUp(self):
        self.max_bsz = 2
        self.max_draft_tokens = 5
        self.max_len = 32
        self.max_dec_len = 10
        self.max_ngram_size = 5
        self.min_ngram_size = 2

        # 初始化输入 tensor
        self.input_ids = paddle.full(shape=[self.max_bsz, self.max_len], fill_value=-1, dtype="int64").cpu()
        self.input_ids_len = paddle.full(shape=[self.max_bsz, 1], fill_value=-1, dtype="int64").cpu()
        self.pre_ids = paddle.full(shape=[self.max_bsz, self.max_len], fill_value=-1, dtype="int64").cpu()
        self.step_idx = paddle.full(shape=[self.max_bsz, 1], fill_value=-1, dtype="int64").cpu()
        self.draft_token_num = paddle.full(shape=[self.max_bsz, 1], fill_value=-1, dtype="int32").cpu()
        self.draft_tokens = paddle.full(
            shape=[self.max_bsz, self.max_draft_tokens + 1],
            fill_value=-1,
            dtype="int64",
        ).cpu()
        self.seq_lens_this_time = paddle.full(shape=[self.max_bsz, 1], fill_value=-1, dtype="int32").cpu()
        self.seq_lens_decoder = paddle.full(shape=[self.max_bsz, 1], fill_value=-1, dtype="int32").cpu()
        self.max_dec_len = paddle.full(
            shape=[self.max_bsz, 1],
            fill_value=self.max_dec_len,
            dtype="int64",
        ).cpu()

        # 设置具体数据
        self.input_ids[:, :10] = np.arange(0, 10)
        self.input_ids_len[:] = 10
        pre_ids_np = np.array([10, 9, 8, 7, 6, 10, 9, 8, 7], dtype="int32")
        self.pre_ids[:, : pre_ids_np.shape[0]] = pre_ids_np
        self.step_idx[:] = 8

        self.draft_token_num[:] = 5
        self.draft_tokens[:, :2] = np.array([8, 7])
        self.seq_lens_this_time[:] = 2
        self.seq_lens_decoder[:] = 12
        self.max_dec_len[:] = 512

        # 期望结果
        self.ref_seq_lens_this_time = np.array([[6], [6]], dtype="int32")
        self.ref_draft_tokens = np.array([[8, 7, 6, 10, 9, 8], [8, 7, 6, 10, 9, 8]], dtype="int64")

    def test_ngram_match_mixed(self):
        hybrid_mtp_ngram(
            self.input_ids,
            self.input_ids_len,
            self.pre_ids,
            self.step_idx,
            self.draft_token_num,
            self.draft_tokens,
            self.seq_lens_this_time,
            self.seq_lens_decoder,
            self.max_dec_len,
            self.max_ngram_size,
            self.min_ngram_size,
            self.max_draft_tokens,
        )

        np.testing.assert_allclose(self.seq_lens_this_time.numpy(), self.ref_seq_lens_this_time)
        np.testing.assert_allclose(self.draft_tokens.numpy(), self.ref_draft_tokens)


if __name__ == "__main__":
    unittest.main()
