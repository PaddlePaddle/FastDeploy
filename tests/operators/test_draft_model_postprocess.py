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

from fastdeploy.model_executor.ops.gpu import draft_model_postprocess


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

        base_model_seq_lens_this_time_gpu = paddle.ones((batch_size), dtype=paddle.int32)  # noqa: F841
        draft_model_postprocess(
            base_model_draft_tokens,
            base_model_seq_lens_this_time_gpu,
            base_model_seq_lens_encoder,
            base_model_stop_flags,
        )

    def test_enough_cases(self):
        self._test_draft_model_postprocess(1, 11)
        self._test_draft_model_postprocess(2, 2048)


if __name__ == "__main__":
    unittest.main()
