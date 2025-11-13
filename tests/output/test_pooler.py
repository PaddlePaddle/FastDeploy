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

from fastdeploy.output.pooler import PoolerOutput, PoolingSequenceGroupOutput


class TestPoolingSequenceGroupOutput(unittest.TestCase):

    def test_get_data_nbytes_tensor(self):
        tensor = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
        output = PoolingSequenceGroupOutput(data=tensor)
        expected = tensor.numel() * tensor.element_size()
        self.assertEqual(output.get_data_nbytes(), expected)

    def test_get_data_nbytes_numpy(self):
        arr = np.ones((2, 3), dtype=np.float32)
        output = PoolingSequenceGroupOutput(data=arr)
        self.assertEqual(output.get_data_nbytes(), arr.nbytes)

    def test_get_data_nbytes_none(self):
        output = PoolingSequenceGroupOutput(data=None)
        self.assertEqual(output.get_data_nbytes(), 0)

    def test_repr(self):
        output = PoolingSequenceGroupOutput(data=123)
        self.assertIn("PoolingSequenceGroupOutput(data=", repr(output))

    def test_eq_same(self):
        output1 = PoolingSequenceGroupOutput(data=5)
        output2 = PoolingSequenceGroupOutput(data=5)
        self.assertTrue(output1 == output2)

    def test_eq_diff(self):
        output1 = PoolingSequenceGroupOutput(data=5)
        output2 = PoolingSequenceGroupOutput(data=6)
        self.assertFalse(output1 == output2)

    def test_eq_not_implemented(self):
        output = PoolingSequenceGroupOutput(data=5)
        with self.assertRaises(NotImplementedError):
            output == 123


class TestPoolerOutput(unittest.TestCase):

    def test_get_data_nbytes_empty(self):
        pooler = PoolerOutput(outputs=[])
        self.assertEqual(pooler.get_data_nbytes(), 0)

    def test_get_data_nbytes_multiple(self):
        outputs = [
            PoolingSequenceGroupOutput(data=paddle.to_tensor([1, 2])),
            PoolingSequenceGroupOutput(data=np.ones(3, dtype=np.float32)),
        ]
        pooler = PoolerOutput(outputs=outputs)
        expected = outputs[0].get_data_nbytes() + outputs[1].get_data_nbytes()
        self.assertEqual(pooler.get_data_nbytes(), expected)

    def test_len_and_index(self):
        outputs = [PoolingSequenceGroupOutput(data=1), PoolingSequenceGroupOutput(data=2)]
        pooler = PoolerOutput(outputs=outputs)
        self.assertEqual(len(pooler), 2)
        self.assertIs(pooler[0], outputs[0])
        self.assertIs(pooler[1], outputs[1])

    def test_setitem(self):
        outputs = [PoolingSequenceGroupOutput(data=1), PoolingSequenceGroupOutput(data=2)]
        pooler = PoolerOutput(outputs=outputs)
        new_output = PoolingSequenceGroupOutput(data=999)
        pooler[1] = new_output
        self.assertIs(pooler[1], new_output)

    def test_eq_same(self):
        outputs1 = [PoolingSequenceGroupOutput(data=1)]
        outputs2 = [PoolingSequenceGroupOutput(data=1)]
        pooler1 = PoolerOutput(outputs=outputs1)
        pooler2 = PoolerOutput(outputs=outputs2)
        self.assertTrue(pooler1 == pooler2)

    def test_eq_diff(self):
        pooler1 = PoolerOutput(outputs=[PoolingSequenceGroupOutput(data=1)])
        pooler2 = PoolerOutput(outputs=[PoolingSequenceGroupOutput(data=2)])
        self.assertFalse(pooler1 == pooler2)

    def test_eq_type_mismatch(self):
        pooler = PoolerOutput(outputs=[PoolingSequenceGroupOutput(data=1)])
        self.assertFalse(pooler == 123)


if __name__ == "__main__":
    unittest.main()
