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

import paddle

from fastdeploy.model_executor.layers.sample.ops.speculate_logprob_utils import (
    speculate_get_accept_tokens_and_logits,
)


class TestSpeculateInsertFirstToken(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 8192

    def test_all_decode(self):
        token_num = 6
        logits = paddle.full(shape=[token_num, self.vocab_size], fill_value=-1, dtype="float32")
        for i in range(token_num):
            logits[i][:] = i

        seq_lens_encoder = paddle.to_tensor([[0], [0], [0]], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([[2], [2], [2]], dtype="int32")
        accept_num = paddle.to_tensor([1, 2, 1], dtype="int32")
        accept_tokens = paddle.to_tensor([[10, -1], [20, 21], [30, -1]], dtype="int64")
        batch_token_num = paddle.where(
            seq_lens_encoder != 0,
            paddle.ones_like(seq_lens_encoder),
            seq_lens_this_time,
        ).squeeze(1)
        cu_seqlens_q_output = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(batch_token_num)]).astype("int32")
        cu_batch_token_offset = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(accept_num)]).astype("int32")
        token_ids = paddle.full(shape=[accept_num.sum()], fill_value=0, dtype="int64")
        target_logits = paddle.empty([accept_num.sum(), logits.shape[1]], dtype=logits.dtype)

        speculate_get_accept_tokens_and_logits(
            token_ids,
            target_logits,
            logits,
            cu_batch_token_offset,
            cu_seqlens_q_output,
            seq_lens_this_time,
            seq_lens_encoder,
            accept_num,
            accept_tokens,
        )

        ref_logits = paddle.full(shape=[4, self.vocab_size], fill_value=-1, dtype="float32")
        ref_logits[0][:] = 0
        ref_logits[1][:] = 2
        ref_logits[2][:] = 3
        ref_logits[3][:] = 4
        ref_token_ids = paddle.to_tensor([10, 20, 21, 30], dtype="int64")

        assert paddle.allclose(target_logits, ref_logits)
        assert paddle.equal_all(token_ids, ref_token_ids)

    def test_partial_decode(self):
        token_num = 5
        logits = paddle.full(shape=[token_num, self.vocab_size], fill_value=-1, dtype="float32")
        for i in range(token_num):
            logits[i][:] = i

        seq_lens_encoder = paddle.to_tensor([[10], [0], [0]], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([[10], [2], [2]], dtype="int32")
        accept_num = paddle.to_tensor([1, 2, 1], dtype="int32")
        accept_tokens = paddle.to_tensor([[10, -1], [20, 21], [30, -1]], dtype="int64")
        batch_token_num = paddle.where(
            seq_lens_encoder != 0,
            paddle.ones_like(seq_lens_encoder),
            seq_lens_this_time,
        ).squeeze(1)
        cu_seqlens_q_output = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(batch_token_num)]).astype("int32")
        cu_batch_token_offset = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(accept_num)]).astype("int32")
        token_ids = paddle.full(shape=[accept_num.sum()], fill_value=0, dtype="int64")
        target_logits = paddle.empty([accept_num.sum(), logits.shape[1]], dtype=logits.dtype)

        speculate_get_accept_tokens_and_logits(
            token_ids,
            target_logits,
            logits,
            cu_batch_token_offset,
            cu_seqlens_q_output,
            seq_lens_this_time,
            seq_lens_encoder,
            accept_num,
            accept_tokens,
        )

        ref_logits = paddle.full(shape=[4, self.vocab_size], fill_value=-1, dtype="float32")
        ref_logits[0][:] = 0
        ref_logits[1][:] = 1
        ref_logits[2][:] = 2
        ref_logits[3][:] = 3
        ref_token_ids = paddle.to_tensor([10, 20, 21, 30], dtype="int64")

        assert paddle.allclose(target_logits, ref_logits)
        assert paddle.equal_all(token_ids, ref_token_ids)

    def test_all_prefill(self):
        token_num = 3
        logits = paddle.full(shape=[token_num, self.vocab_size], fill_value=-1, dtype="float32")
        for i in range(token_num):
            logits[i][:] = i

        seq_lens_encoder = paddle.to_tensor([[10], [10], [10]], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([[10], [10], [10]], dtype="int32")
        accept_num = paddle.to_tensor([1, 1, 1], dtype="int32")
        accept_tokens = paddle.to_tensor([[10, -1], [20, -1], [30, -1]], dtype="int64")
        batch_token_num = paddle.where(
            seq_lens_encoder != 0,
            paddle.ones_like(seq_lens_encoder),
            seq_lens_this_time,
        ).squeeze(1)
        cu_seqlens_q_output = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(batch_token_num)]).astype("int32")
        cu_batch_token_offset = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(accept_num)]).astype("int32")
        token_ids = paddle.full(shape=[accept_num.sum()], fill_value=0, dtype="int64")
        target_logits = paddle.empty([accept_num.sum(), logits.shape[1]], dtype=logits.dtype)

        speculate_get_accept_tokens_and_logits(
            token_ids,
            target_logits,
            logits,
            cu_batch_token_offset,
            cu_seqlens_q_output,
            seq_lens_this_time,
            seq_lens_encoder,
            accept_num,
            accept_tokens,
        )

        ref_logits = paddle.full(shape=[3, self.vocab_size], fill_value=-1, dtype="float32")
        ref_logits[0][:] = 0
        ref_logits[1][:] = 1
        ref_logits[2][:] = 2
        ref_token_ids = paddle.to_tensor([10, 20, 30], dtype="int64")

        assert paddle.allclose(target_logits, ref_logits)
        assert paddle.equal_all(token_ids, ref_token_ids)


if __name__ == "__main__":
    unittest.main()
