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

from fastdeploy.model_executor.entropy_utils import (
    calculate_logits_entropy,
    speculate_calculate_logits_entropy,
)


class TestCalculateLogitsEntropy(unittest.TestCase):

    def test_basic_functionality(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[1], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], []],
            "stop_flags": paddle.to_tensor([[False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3"],
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        self.assertAlmostEqual(share_inputs["entropy_list"][0][0], 0.0024676250759512186, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][2][0], 0.0024676250759512186, places=6)

    def test_temperature_effect(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[1], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], []],
            "stop_flags": paddle.to_tensor([[False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3"],
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0],
            ],
            dtype="float32",
        )
        temperature = paddle.to_tensor([[0.8], [1.0], [0.8]], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        self.assertAlmostEqual(share_inputs["entropy_list"][0][0], 0.0003187173861078918, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][2][0], 0.0003187173861078918, places=6)

    def test_entropy_list_clear(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[1], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], []],
            "stop_flags": paddle.to_tensor([[True], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3"],
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 1.0, 10.0],
            ],
            dtype="float32",
        )
        temperature = paddle.to_tensor([[0.8], [1.0], [0.8]], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        self.assertAlmostEqual(share_inputs["entropy_list"][2][0], 0.0003187173861078918, places=6)

    def test_negative_inf_clip(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[1], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], []],
            "stop_flags": paddle.to_tensor([[False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3"],
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, -float("inf")],
                [1.0, 1.0, -float("inf")],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        self.assertAlmostEqual(share_inputs["entropy_list"][0][0], 0.0017332095885649323, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][2][0], 1.017357349395752, places=6)


class TestSpeculateCalculateLogitsEntropy(unittest.TestCase):

    def test_basic_functionality(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[2], [2], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], [], []],
            "stop_flags": paddle.to_tensor([[False], [False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3", "req_4"],
            "accept_num": paddle.to_tensor([2, 1, 0, 0], dtype="int32"),  # 推理接受数量
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 10.0],
                [1.0, 1.0, 10.0],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 2)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][3]), 0)

        self.assertAlmostEqual(share_inputs["entropy_list"][0][0], 0.0024676250759512186, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][0][1], 0.0024676250759512186, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][1][0], 0.0024676250759512186, places=6)

    def test_temperature_effect(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[2], [2], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], [], []],
            "stop_flags": paddle.to_tensor([[False], [False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3", "req_4"],
            "accept_num": paddle.to_tensor([2, 1, 0, 0], dtype="int32"),  # 推理接受数量
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 10.0],
                [1.0, 1.0, 10.0],
            ],
            dtype="float32",
        )
        temperature = paddle.to_tensor([[0.8], [0.8], [0.8], [0.8]], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 2)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][3]), 0)

        self.assertAlmostEqual(share_inputs["entropy_list"][0][0], 0.0003187173861078918, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][0][1], 0.0003187173861078918, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][1][0], 0.0003187173861078918, places=6)

    def test_entropy_list_clear(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[2], [2], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], [], []],
            "stop_flags": paddle.to_tensor([[True], [False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3", "req_4"],
            "accept_num": paddle.to_tensor([2, 1, 0, 0], dtype="int32"),  # 推理接受数量
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 10.0],
                [1.0, 1.0, 10.0],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        speculate_calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][3]), 0)

        self.assertAlmostEqual(share_inputs["entropy_list"][1][0], 0.0024676250759512186, places=6)

    def test_negative_inf_clip(self):
        share_inputs = {
            "seq_lens_this_time": paddle.to_tensor([[1], [0], [15]], dtype="int32"),
            "seq_lens_encoder": paddle.to_tensor([[0], [0], [15]], dtype="int32"),
            "seq_lens_decoder": paddle.to_tensor([[30], [0], [15]], dtype="int32"),
            "entropy_list": [[], [], []],
            "stop_flags": paddle.to_tensor([[False], [True], [False]], dtype="bool"),
            "req_ids": ["req_1", "req_2", "req_3"],
        }

        logits = paddle.to_tensor(
            [
                [10.0, 1.0, -float("inf")],
                [1.0, 1.0, -float("inf")],
            ],
            dtype="float32",
        )
        temperature = paddle.ones([3], dtype="float32")

        calculate_logits_entropy(logits, share_inputs, temperature)

        self.assertEqual(len(share_inputs["entropy_list"][0]), 1)
        self.assertEqual(len(share_inputs["entropy_list"][1]), 0)
        self.assertEqual(len(share_inputs["entropy_list"][2]), 1)

        self.assertAlmostEqual(share_inputs["entropy_list"][0][0], 0.0017332095885649323, places=6)
        self.assertAlmostEqual(share_inputs["entropy_list"][2][0], 1.017357349395752, places=6)


if __name__ == "__main__":
    unittest.main()
