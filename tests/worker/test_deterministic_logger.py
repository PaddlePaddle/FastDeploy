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

import logging
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

# Register fastdeploy as a bare namespace package so that
# ``from fastdeploy.worker.deterministic_logger import ...`` does NOT
# execute fastdeploy/__init__.py (which pulls in paddle, paddleformers, etc.).
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _pkg, _rel_path in [
    ("fastdeploy", "fastdeploy"),
    ("fastdeploy.logger", "fastdeploy/logger"),
    ("fastdeploy.worker", "fastdeploy/worker"),
]:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_project_root, _rel_path)]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

from fastdeploy.logger.deterministic_logger import DeterministicLogger  # noqa: E402


def _make_tensor(array):
    """Create a mock tensor that behaves like a paddle Tensor for testing."""
    arr = np.array(array)
    tensor = Mock()
    tensor.cpu.return_value = tensor
    tensor.numpy.return_value = arr
    tensor.shape = arr.shape
    tensor.__len__ = lambda self: arr.shape[0]
    tensor.__getitem__ = lambda self, idx: _make_tensor(arr[idx])
    return tensor


class TestComputeTensorMd5(unittest.TestCase):
    def test_none_tensor(self):
        result = DeterministicLogger._compute_tensor_md5(None, name="x")
        self.assertEqual(result, "x_md5=None")

    def test_deterministic_hash(self):
        t = _make_tensor([1.0, 2.0, 3.0])
        r1 = DeterministicLogger._compute_tensor_md5(t, name="a")
        r2 = DeterministicLogger._compute_tensor_md5(t, name="a")
        self.assertEqual(r1, r2)
        self.assertIn("a_md5=", r1)

    def test_different_tensors_different_hash(self):
        t1 = _make_tensor([1.0, 2.0])
        t2 = _make_tensor([3.0, 4.0])
        r1 = DeterministicLogger._compute_tensor_md5(t1, name="x")
        r2 = DeterministicLogger._compute_tensor_md5(t2, name="x")
        self.assertNotEqual(r1, r2)

    def test_prefix(self):
        t = _make_tensor([1.0])
        result = DeterministicLogger._compute_tensor_md5(t, name="h", prefix="batch_")
        self.assertTrue(result.startswith("batch_h_md5="))

    def test_md5_truncated_to_16_chars(self):
        t = _make_tensor([1.0, 2.0, 3.0])
        result = DeterministicLogger._compute_tensor_md5(t, name="x")
        md5_value = result.split("=")[1]
        self.assertEqual(len(md5_value), 16)


class TestGetBatchSize(unittest.TestCase):
    def test_returns_first_tensor_batch_size(self):
        t = _make_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = DeterministicLogger._get_batch_size({"a": t})
        self.assertEqual(result, 3)

    def test_skips_none_tensors(self):
        t = _make_tensor([[1.0], [2.0]])
        result = DeterministicLogger._get_batch_size({"a": None, "b": t})
        self.assertEqual(result, 2)

    def test_returns_none_for_empty_dict(self):
        self.assertIsNone(DeterministicLogger._get_batch_size({}))

    def test_returns_none_for_all_none(self):
        self.assertIsNone(DeterministicLogger._get_batch_size({"a": None}))


class TestBuildReqIdStr(unittest.TestCase):
    def test_none_list(self):
        self.assertEqual(DeterministicLogger._build_req_id_str(None), "")

    def test_single_request(self):
        req = Mock(request_id="req-001")
        result = DeterministicLogger._build_req_id_str([req])
        self.assertEqual(result, "[0]req-001")

    def test_multiple_requests_with_none(self):
        r1 = Mock(request_id="r1")
        r2 = Mock(request_id="r2")
        result = DeterministicLogger._build_req_id_str([r1, None, r2])
        self.assertEqual(result, "[0]r1, [2]r2")


class TestGetStageCounts(unittest.TestCase):
    def test_no_seq_lens_encoder(self):
        logger = DeterministicLogger(share_inputs={})
        prefill, decode, enc = logger._get_stage_counts(batch_size=4)
        self.assertEqual(prefill, 0)
        self.assertEqual(decode, 0)
        self.assertIsNone(enc)

    def test_with_seq_lens_encoder(self):
        # seq_lens_encoder: [5, 0, 3, 0] -> 2 prefill, 2 decode
        enc_tensor = _make_tensor([5, 0, 3, 0])
        logger = DeterministicLogger(share_inputs={"seq_lens_encoder": enc_tensor})
        prefill, decode, enc = logger._get_stage_counts(batch_size=4)
        self.assertEqual(prefill, 2)
        self.assertEqual(decode, 2)
        np.testing.assert_array_equal(enc, np.array([5, 0, 3, 0]))

    def test_all_prefill(self):
        enc_tensor = _make_tensor([10, 20])
        logger = DeterministicLogger(share_inputs={"seq_lens_encoder": enc_tensor})
        prefill, decode, _ = logger._get_stage_counts(batch_size=2)
        self.assertEqual(prefill, 2)
        self.assertEqual(decode, 0)

    def test_none_share_inputs(self):
        logger = DeterministicLogger(share_inputs=None)
        prefill, decode, enc = logger._get_stage_counts(batch_size=4)
        self.assertEqual(prefill, 0)
        self.assertEqual(decode, 0)
        self.assertIsNone(enc)


class TestLogTensorMd5s(unittest.TestCase):
    def test_logs_batch_md5(self):
        t = _make_tensor([[1.0, 2.0], [3.0, 4.0]])
        logger = DeterministicLogger(share_inputs={})
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_tensor_md5s({"hidden": t}, stage="test_stage")
        self.assertTrue(any("[DETERMINISM-MD5]" in msg for msg in cm.output))
        self.assertTrue(any("stage=test_stage" in msg for msg in cm.output))

    def test_skips_when_no_valid_tensor(self):
        logger = DeterministicLogger(share_inputs={})
        det_log = logging.getLogger("fastdeploy.deterministic")
        det_log.setLevel(logging.INFO)
        # Should not raise, just silently return
        logger.log_tensor_md5s({"a": None})

    def test_logs_with_request_ids(self):
        t = _make_tensor([[1.0], [2.0]])
        req = Mock(request_id="req-42")
        logger = DeterministicLogger(share_inputs={})
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_tensor_md5s({"x": t}, forward_batch_reqs_list=[req], stage="s")
        self.assertTrue(any("req-42" in msg for msg in cm.output))

    def test_logs_per_request_md5_for_decode(self):
        # 2 requests, both decode (seq_lens_encoder = [0, 0])
        t = _make_tensor([[1.0, 2.0], [3.0, 4.0]])
        enc_tensor = _make_tensor([0, 0])
        r1 = Mock(request_id="r1")
        r2 = Mock(request_id="r2")
        logger = DeterministicLogger(share_inputs={"seq_lens_encoder": enc_tensor})
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_tensor_md5s({"out": t}, forward_batch_reqs_list=[r1, r2], stage="decode")
        req_msgs = [msg for msg in cm.output if "[DETERMINISM-MD5-REQ]" in msg]
        self.assertEqual(len(req_msgs), 2)


class TestLogDeterministicInput(unittest.TestCase):
    def _make_forward_meta(self, ids_list):
        ids_tensor = _make_tensor(ids_list)
        return SimpleNamespace(ids_remove_padding=ids_tensor)

    def test_logs_input_info(self):
        forward_meta = self._make_forward_meta([101, 102, 201])
        share_inputs = {
            "req_ids": ["req-a", "req-b"],
            "seq_lens_this_time": [2, 1],
            "seq_lens_encoder": [2, 0],
            "seq_lens_decoder": [0, 5],
        }
        logger = DeterministicLogger(share_inputs=share_inputs)
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_deterministic_input(forward_meta)
        output = "\n".join(cm.output)
        self.assertIn("batch_size=2", output)
        self.assertIn("req_id=req-a", output)
        self.assertIn("req_id=req-b", output)
        self.assertIn("tokens=[101, 102]", output)
        self.assertIn("tokens=[201]", output)

    def test_no_input_data(self):
        forward_meta = SimpleNamespace(ids_remove_padding=None)
        share_inputs = {
            "req_ids": None,
            "seq_lens_this_time": [],
            "seq_lens_encoder": None,
            "seq_lens_decoder": None,
        }
        logger = DeterministicLogger(share_inputs=share_inputs)
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_deterministic_input(forward_meta)
        self.assertTrue(any("No input data" in msg for msg in cm.output))

    def test_fallback_req_id(self):
        forward_meta = self._make_forward_meta([10, 20])
        share_inputs = {
            "req_ids": None,
            "seq_lens_this_time": [1, 1],
            "seq_lens_encoder": None,
            "seq_lens_decoder": None,
        }
        logger = DeterministicLogger(share_inputs=share_inputs)
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_deterministic_input(forward_meta)
        output = "\n".join(cm.output)
        self.assertIn("req_id=idx_0", output)
        self.assertIn("req_id=idx_1", output)


class TestLogBatchStart(unittest.TestCase):
    def _make_logger(self):
        return DeterministicLogger(share_inputs={})

    def _make_req(self, request_id):
        return Mock(request_id=request_id)

    def test_logs_batch_start(self):
        logger = self._make_logger()
        batch = [self._make_req("prompt_0")]
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_batch_start(batch)
        output = "\n".join(cm.output)
        self.assertIn("[BATCH-START]", output)
        self.assertIn("Run_0", output)
        self.assertIn("Batch_1", output)

    def test_batch_counter_increments(self):
        logger = self._make_logger()
        batch = [self._make_req("prompt_0")]
        with self.assertLogs("fastdeploy.deterministic", level="INFO"):
            logger.log_batch_start(batch)
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_batch_start(batch)
        output = "\n".join(cm.output)
        self.assertIn("Batch_2", output)

    def test_run_id_change_resets_counter(self):
        logger = self._make_logger()
        batch_0 = [self._make_req("prompt_0")]
        batch_1 = [self._make_req("prompt_1")]
        with self.assertLogs("fastdeploy.deterministic", level="INFO"):
            logger.log_batch_start(batch_0)
            logger.log_batch_start(batch_0)  # Batch_2
        # Switch to run_id 1 => counter resets
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_batch_start(batch_1)
        output = "\n".join(cm.output)
        self.assertIn("Run_1", output)
        self.assertIn("Batch_1", output)

    def test_skips_none_requests(self):
        logger = self._make_logger()
        batch = [None, self._make_req("req_5")]
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_batch_start(batch)
        output = "\n".join(cm.output)
        self.assertIn("Run_5", output)

    def test_empty_batch(self):
        logger = self._make_logger()
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_batch_start([])
        output = "\n".join(cm.output)
        self.assertIn("Run_None", output)
        self.assertIn("Batch_1", output)

    def test_none_batch(self):
        logger = self._make_logger()
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_batch_start(None)
        output = "\n".join(cm.output)
        self.assertIn("Batch_1", output)


class TestLogPrefillInput(unittest.TestCase):
    def test_logs_prefill_input(self):
        logger = DeterministicLogger(share_inputs={})
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_prefill_input(
                request_id="req-001",
                idx=0,
                prefill_start_index=0,
                prefill_end_index=5,
                input_ids=[101, 102, 103, 104, 105],
            )
        output = "\n".join(cm.output)
        self.assertIn("[DETERMINISM] Prefill input", output)
        self.assertIn("request_id: req-001", output)
        self.assertIn("idx: 0", output)
        self.assertIn("prefill_start_index: 0", output)
        self.assertIn("prefill_end_index: 5", output)
        self.assertIn("[101, 102, 103, 104, 105]", output)

    def test_logs_with_nonzero_start_index(self):
        logger = DeterministicLogger(share_inputs={})
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_prefill_input(
                request_id="req-002",
                idx=3,
                prefill_start_index=10,
                prefill_end_index=20,
                input_ids=list(range(20)),
            )
        output = "\n".join(cm.output)
        self.assertIn("request_id: req-002", output)
        self.assertIn("idx: 3", output)
        self.assertIn("prefill_start_index: 10", output)
        self.assertIn("prefill_end_index: 20", output)


if __name__ == "__main__":
    unittest.main()
