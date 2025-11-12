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

from fastdeploy.output.stream_transfer_data import DecoderState, StreamTransferData


class TestStreamTransferData(unittest.TestCase):

    def test_dataclass_initialization(self):
        tokens = np.array([1, 2, 3])
        logprobs = np.array([0.1, 0.2, 0.3])
        accept_tokens = np.array([1, 0, 1])
        accept_num = np.array([2])
        pooler_output = np.random.rand(2, 4)

        data = StreamTransferData.__new__(StreamTransferData)
        data.decoder_state = DecoderState.TEXT
        data.batch_id = 42
        data.tokens = tokens
        data.speculaive_decoding = True
        data.logprobs = logprobs
        data.accept_tokens = accept_tokens
        data.accept_num = accept_num
        data.pooler_output = pooler_output

        self.assertEqual(data.decoder_state, DecoderState.TEXT)
        self.assertEqual(data.batch_id, 42)
        self.assertTrue(np.array_equal(data.tokens, tokens))
        self.assertTrue(data.speculaive_decoding)
        self.assertTrue(np.array_equal(data.logprobs, logprobs))
        self.assertTrue(np.array_equal(data.accept_tokens, accept_tokens))
        self.assertTrue(np.array_equal(data.accept_num, accept_num))
        self.assertTrue(np.array_equal(data.pooler_output, pooler_output))

    def test_optional_fields_none(self):
        data = StreamTransferData.__new__(StreamTransferData)
        data.decoder_state = DecoderState.IMAGE
        data.batch_id = 1

        self.assertEqual(data.decoder_state, DecoderState.IMAGE)
        self.assertEqual(data.batch_id, 1)
        self.assertIsNone(getattr(data, "tokens", None))
        self.assertFalse(getattr(data, "speculaive_decoding", False))
        self.assertIsNone(getattr(data, "logprobs", None))
        self.assertIsNone(getattr(data, "accept_tokens", None))
        self.assertIsNone(getattr(data, "accept_num", None))
        self.assertIsNone(getattr(data, "pooler_output", None))


if __name__ == "__main__":
    unittest.main()
