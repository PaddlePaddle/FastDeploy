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

"""
Unit tests for entropy_utils FD runner path.

Run:  python tests/model_executor/test_entropy_utils_fd_runner_mtp.py
"""

import importlib
import logging
import math
import os
import sys
import types
import unittest

os.environ["EB5_ENABLE_FD_RUNNER"] = "1"

import paddle

# Mock fastdeploy.utils
_mock_fd = types.ModuleType("fastdeploy")
_mock_fd.__path__ = []
_mock_utils = types.ModuleType("fastdeploy.utils")
_mock_utils.data_processor_logger = logging.getLogger("test_entropy")
_mock_fd.utils = _mock_utils
sys.modules["fastdeploy"] = _mock_fd
sys.modules["fastdeploy.utils"] = _mock_utils

# Import module under test
_spec = importlib.util.spec_from_file_location(
    "fastdeploy.model_executor.entropy_utils",
    os.path.join(os.path.dirname(__file__), "../../fastdeploy/model_executor/entropy_utils.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["fastdeploy.model_executor"] = types.ModuleType("fastdeploy.model_executor")
sys.modules["fastdeploy.model_executor.entropy_utils"] = _mod
_spec.loader.exec_module(_mod)

get_entropy = _mod.get_entropy
calculate_logits_entropy_fd = _mod.calculate_logits_entropy_fd
speculate_calculate_logits_entropy_fd = _mod.speculate_calculate_logits_entropy_fd
flush_entropy_on_stop = _mod.flush_entropy_on_stop


def _make_share_inputs(
    bsz, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, stop_flags, req_ids, accept_num=None
):
    d = {
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time, dtype="int32"),
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder, dtype="int32"),
        "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder, dtype="int32"),
        "entropy_list": [[] for _ in range(bsz)],
        "stop_flags": paddle.to_tensor(stop_flags, dtype="bool"),
        "req_ids": req_ids,
    }
    if accept_num is not None:
        d["accept_num"] = paddle.to_tensor(accept_num, dtype="int32")
    return d


class TestGetEntropy(unittest.TestCase):
    def test_uniform_and_deterministic(self):
        logits = paddle.zeros([1, 4], dtype="float32")
        self.assertAlmostEqual(float(get_entropy(logits)[0]), math.log(4), places=5)

        logits = paddle.to_tensor([[100.0, 0.0, 0.0, 0.0]], dtype="float32")
        self.assertAlmostEqual(float(get_entropy(logits)[0]), 0.0, places=5)

    def test_negative_inf(self):
        logits = paddle.to_tensor([[10.0, -float("inf"), -float("inf")]], dtype="float32")
        self.assertGreaterEqual(float(get_entropy(logits)[0]), 0.0)


class TestNonMTP(unittest.TestCase):
    def test_accumulation_and_skip_zero(self):
        si = _make_share_inputs(3, [1, 1, 0], [0, 0, 0], [10, 10, 0], [False, False, False], ["a", "b", "c"])
        logits = paddle.to_tensor([[10.0, 1.0, 1.0], [1.0, 1.0, 10.0], [5.0, 5.0, 5.0]], dtype="float32")
        calculate_logits_entropy_fd(logits, si, paddle.ones([3], dtype="float32"))

        self.assertEqual(len(si["entropy_list"][0]), 1)
        self.assertEqual(len(si["entropy_list"][1]), 1)
        self.assertEqual(len(si["entropy_list"][2]), 0)
        # [10,1,1] and [1,1,10] symmetric => same entropy
        self.assertAlmostEqual(si["entropy_list"][0][0], si["entropy_list"][1][0], places=6)

    def test_stop_flags_and_temperature(self):
        # stop_flags clears entropy
        si = _make_share_inputs(2, [1, 1], [0, 0], [10, 10], [True, False], ["a", "b"])
        logits = paddle.to_tensor([[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]], dtype="float32")
        calculate_logits_entropy_fd(logits, si, paddle.ones([2], dtype="float32"))
        self.assertEqual(si["entropy_list"][0], [])
        self.assertEqual(len(si["entropy_list"][1]), 1)

        # temperature scaling: lower temp => lower entropy
        si = _make_share_inputs(2, [1, 1], [0, 0], [10, 10], [False, False], ["a", "b"])
        logits = paddle.to_tensor([[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]], dtype="float32")
        calculate_logits_entropy_fd(logits, si, paddle.to_tensor([1.0, 0.5], dtype="float32"))
        self.assertGreater(si["entropy_list"][0][0], si["entropy_list"][1][0])


class TestMTP(unittest.TestCase):
    def test_accepted_idx_and_partial(self):
        """accept_num=[2, 0, 1, 2], verifies correct row extraction and per-slot counts."""
        si = _make_share_inputs(
            4,
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [10, 10, 10, 10],
            [False, False, False, False],
            ["a", "b", "c", "d"],
            [2, 0, 1, 2],
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],  # slot 0 (both accepted)
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],  # slot 1 (none accepted)
                [1.0, 1.0, 10.0],
                [5.0, 5.0, 5.0],  # slot 2 (first accepted)
                [10.0, 10.0, 1.0],
                [1.0, 10.0, 10.0],  # slot 3 (both accepted)
            ],
            dtype="float32",
        )
        speculate_calculate_logits_entropy_fd(logits, si, paddle.ones([4], dtype="float32"))

        self.assertEqual(len(si["entropy_list"][0]), 2)
        self.assertEqual(len(si["entropy_list"][1]), 0)
        self.assertEqual(len(si["entropy_list"][2]), 1)
        self.assertEqual(len(si["entropy_list"][3]), 2)
        # [10,1,1] [1,10,1] [1,1,10] all symmetric => same entropy
        self.assertAlmostEqual(si["entropy_list"][0][0], si["entropy_list"][0][1], places=6)
        self.assertAlmostEqual(si["entropy_list"][0][0], si["entropy_list"][2][0], places=6)

    def test_zero_accepted_flushes_stop(self):
        si = _make_share_inputs(2, [1, 1], [0, 0], [10, 10], [True, False], ["a", "b"], [0, 0])
        si["entropy_list"][0] = [1.0, 2.0, 3.0]
        si["entropy_list"][1] = [4.0, 5.0]
        speculate_calculate_logits_entropy_fd(
            paddle.zeros([2, 3], dtype="float32"), si, paddle.ones([2], dtype="float32")
        )
        self.assertEqual(si["entropy_list"][0], [])
        self.assertEqual(si["entropy_list"][1], [4.0, 5.0])

    def test_warmup_skip(self):
        si = _make_share_inputs(
            3, [2, 2, 2], [0, 0, 0], [10, 10, 10], [False, False, False], ["", "real", "  "], [1, 1, 1]
        )
        logits = paddle.to_tensor(
            [
                [10.0, 1.0, 1.0],
                [5.0, 5.0, 5.0],
                [1.0, 10.0, 1.0],
                [5.0, 5.0, 5.0],
                [1.0, 1.0, 10.0],
                [5.0, 5.0, 5.0],
            ],
            dtype="float32",
        )
        speculate_calculate_logits_entropy_fd(logits, si, paddle.ones([3], dtype="float32"))
        self.assertEqual(len(si["entropy_list"][0]), 0)
        self.assertEqual(len(si["entropy_list"][1]), 1)
        self.assertEqual(len(si["entropy_list"][2]), 0)

    def test_multi_step_and_stop(self):
        si = _make_share_inputs(2, [2, 2], [0, 0], [10, 10], [False, False], ["a", "b"], [2, 1])
        logits = paddle.to_tensor(
            [[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0], [5.0, 5.0, 5.0]], dtype="float32"
        )
        speculate_calculate_logits_entropy_fd(logits, si, paddle.ones([2], dtype="float32"))
        self.assertEqual(len(si["entropy_list"][0]), 2)
        self.assertEqual(len(si["entropy_list"][1]), 1)

        # Step 2: slot 1 stops
        si["accept_num"] = paddle.to_tensor([1, 1], dtype="int32")
        si["stop_flags"] = paddle.to_tensor([False, True], dtype="bool")
        logits2 = paddle.to_tensor(
            [[10.0, 1.0, 1.0], [5.0, 5.0, 5.0], [10.0, 1.0, 1.0], [5.0, 5.0, 5.0]], dtype="float32"
        )
        speculate_calculate_logits_entropy_fd(logits2, si, paddle.ones([2], dtype="float32"))
        self.assertEqual(len(si["entropy_list"][0]), 3)
        self.assertEqual(si["entropy_list"][1], [])


class TestFlushEntropyOnStop(unittest.TestCase):
    def test_flush(self):
        si = _make_share_inputs(3, [1, 1, 1], [0, 0, 0], [10, 10, 10], [True, False, True], ["a", "b", "c"])
        si["entropy_list"][0] = [1.0, 2.0]
        si["entropy_list"][2] = [3.0]
        flush_entropy_on_stop(si)
        self.assertEqual(si["entropy_list"][0], [])
        self.assertEqual(si["entropy_list"][1], [])
        self.assertEqual(si["entropy_list"][2], [])


if __name__ == "__main__":
    unittest.main()
