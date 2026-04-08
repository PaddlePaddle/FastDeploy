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
    speculate_insert_first_token,
)


class TestSpeculateInsertFirstToken(unittest.TestCase):

    def test_all_decode(self):
        token_num = 5
        accept_tokens = paddle.to_tensor([[1001, 1002], [1003, 1004], [1005, 1006]], dtype="int64")
        next_tokens = paddle.to_tensor([[2001], [2002], [2003], [2004], [2005]], dtype="int64")
        cu_next_token_offset = paddle.to_tensor([0, 2, 3, 5], dtype="int32")
        cu_batch_token_offset = paddle.to_tensor([0, 2, 3, 5], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([[2], [1], [2]], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([[0], [0], [0]], dtype="int32")

        token_id = paddle.empty(token_num, dtype="int64")
        speculate_insert_first_token(
            token_id,
            accept_tokens,
            next_tokens,
            cu_next_token_offset,
            cu_batch_token_offset,
            seq_lens_this_time,
            seq_lens_encoder,
        )

        gold_token_id = paddle.to_tensor([2001, 2002, 2003, 2004, 2005], dtype="int64")

        assert paddle.equal_all(token_id, gold_token_id)

    def test_partial_decode(self):
        token_num = 6
        accept_tokens = paddle.to_tensor([[1001, 1002], [1003, 1004], [1005, 1006]], dtype="int64")
        next_tokens = paddle.to_tensor([[2001], [2002], [2003], [2004], [2005]], dtype="int64")
        cu_next_token_offset = paddle.to_tensor([0, 2, 3, 5], dtype="int32")
        cu_batch_token_offset = paddle.to_tensor([0, 2, 4, 6], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([[2], [10], [2]], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([[0], [10], [0]], dtype="int32")

        token_id = paddle.empty(token_num, dtype="int64")
        speculate_insert_first_token(
            token_id,
            accept_tokens,
            next_tokens,
            cu_next_token_offset,
            cu_batch_token_offset,
            seq_lens_this_time,
            seq_lens_encoder,
        )

        gold_token_id = paddle.to_tensor([2001, 2002, 1003, 2003, 2004, 2005], dtype="int64")

        assert paddle.equal_all(token_id, gold_token_id)

    def test_all_prefill(self):
        token_num = 6
        accept_tokens = paddle.to_tensor([[1001, 1002], [1003, 1004], [1005, 1006]], dtype="int64")
        next_tokens = paddle.to_tensor([[2001], [2002], [2003]], dtype="int64")
        cu_next_token_offset = paddle.to_tensor([0, 1, 2, 3], dtype="int32")
        cu_batch_token_offset = paddle.to_tensor([0, 2, 4, 6], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([[10], [10], [10]], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([[10], [10], [10]], dtype="int32")

        token_id = paddle.empty(token_num, dtype="int64")
        speculate_insert_first_token(
            token_id,
            accept_tokens,
            next_tokens,
            cu_next_token_offset,
            cu_batch_token_offset,
            seq_lens_this_time,
            seq_lens_encoder,
        )

        gold_token_id = paddle.to_tensor([1001, 2001, 1003, 2002, 1005, 2003], dtype="int64")

        assert paddle.equal_all(token_id, gold_token_id)


if __name__ == "__main__":
    unittest.main()
