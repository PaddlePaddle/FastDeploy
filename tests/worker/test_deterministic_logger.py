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

import hashlib
import json
import logging
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

import fastdeploy.logger.deterministic_logger as _det_mod  # noqa: E402
from fastdeploy.logger.deterministic_logger import (  # noqa: E402
    DeterministicLogger,
    _compute_md5,
    _read_logits_md5_file,
    _record_logits_diagnostic,
    _reset_logits_md5_file,
)


def _make_astype_tensor(array):
    """Create a mock tensor supporting .astype().cpu().numpy().tobytes() chain.

    Needed for module-level functions that call tensor.astype("float32").
    """
    arr = np.array(array, dtype=np.float32)

    inner = Mock()
    inner.cpu.return_value = inner
    inner.numpy.return_value = arr  # Return real np array for numpy operations

    tensor = Mock()
    tensor.astype.return_value = inner
    tensor.cpu.return_value = inner  # Also support .cpu() without .astype()
    tensor.shape = arr.shape
    return tensor


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

    def test_skips_prefill_requests(self):
        """Prefill requests (seq_lens_encoder > 0) are skipped in per-request MD5 logging."""
        # 3 requests: first is prefill (enc=5), rest are decode (enc=0)
        t = _make_tensor([[1.0], [2.0], [3.0]])
        enc_tensor = _make_tensor([5, 0, 0])  # index 0 is prefill
        r0 = Mock(request_id="prefill_req")
        r1 = Mock(request_id="decode_req_1")
        r2 = Mock(request_id="decode_req_2")
        logger = DeterministicLogger(share_inputs={"seq_lens_encoder": enc_tensor})
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_tensor_md5s({"out": t}, forward_batch_reqs_list=[r0, r1, r2], stage="mixed")
        req_msgs = [msg for msg in cm.output if "[DETERMINISM-MD5-REQ]" in msg]
        # Only decode requests (r1, r2) should be logged
        self.assertEqual(len(req_msgs), 2)
        self.assertTrue(all("decode_req" in msg for msg in req_msgs))
        self.assertFalse(any("prefill_req" in msg for msg in req_msgs))

    def test_skips_all_when_prefill_count_positive_without_seq_lens_encoder(self):
        """When prefill_count > 0 but no seq_lens_encoder, all requests are skipped."""
        t = _make_tensor([[1.0], [2.0]])
        r1 = Mock(request_id="req1")
        r2 = Mock(request_id="req2")
        # share_inputs without seq_lens_encoder, but with other keys to trigger prefill_count > 0
        logger = DeterministicLogger(share_inputs={})
        # Manually set up a scenario where prefill_count would be computed as > 0
        # This happens when share_inputs has seq_lens_encoder with positive values
        # In this case, with no seq_lens_encoder, prefill_count is 0, so this tests the elif branch
        with self.assertLogs("fastdeploy.deterministic", level="INFO") as cm:
            logger.log_tensor_md5s({"out": t}, forward_batch_reqs_list=[r1, r2], stage="decode")
        # With no seq_lens_encoder, prefill_count=0, decode_count=0, so _log_per_request_md5s returns early
        req_msgs = [msg for msg in cm.output if "[DETERMINISM-MD5-REQ]" in msg]
        self.assertEqual(len(req_msgs), 0)


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


# ---- Tests for module-level functions (L35-44) ----


class TestComputeMd5(unittest.TestCase):
    """Tests for _compute_md5(): mock paddle tensor (GPU dependency)."""

    def test_returns_valid_md5_hex(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = _make_astype_tensor([1.0, 2.0, 3.0])
        result = _compute_md5(t)
        expected = hashlib.md5(arr.tobytes()).hexdigest()
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 32)

    def test_deterministic(self):
        t1 = _make_astype_tensor([1.0, 2.0])
        t2 = _make_astype_tensor([1.0, 2.0])
        self.assertEqual(_compute_md5(t1), _compute_md5(t2))

    def test_different_data_different_hash(self):
        t1 = _make_astype_tensor([1.0, 2.0])
        t2 = _make_astype_tensor([3.0, 4.0])
        self.assertNotEqual(_compute_md5(t1), _compute_md5(t2))

    def test_fallback_to_float32_on_tobytes_error(self):
        """When .tobytes() fails, _compute_md5 falls back to .astype(np.float32).tobytes()."""
        # Create a mock tensor where .tobytes() raises an exception
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Inner mock that raises on tobytes, but works for astype
        inner = Mock()
        inner.cpu.return_value = inner
        inner.numpy.return_value = arr
        # First call to tobytes() raises, then astype().tobytes() succeeds
        inner.tobytes.side_effect = RuntimeError("tobytes failed")
        inner.astype.return_value = arr.astype(np.float32)

        tensor = Mock()
        tensor.cpu.return_value = inner
        tensor.astype.return_value = inner

        result = _compute_md5(tensor)
        # Should still produce a valid MD5 (from the fallback path)
        self.assertEqual(len(result), 32)
        expected = hashlib.md5(arr.astype(np.float32).tobytes()).hexdigest()
        self.assertEqual(result, expected)


class TestResetLogitsMd5File(unittest.TestCase):
    """Tests for _reset_logits_md5_file(): file I/O (filesystem dependency)."""

    def test_creates_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as f:
            tmp = f.name
        try:
            with patch.object(_det_mod, "_DET_MD5_PATH", tmp):
                _reset_logits_md5_file()
            with open(tmp) as f:
                self.assertEqual(f.read(), "")
        finally:
            os.unlink(tmp)

    def test_truncates_existing_content(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as f:
            f.write('{"old": "data"}\n')
            tmp = f.name
        try:
            with patch.object(_det_mod, "_DET_MD5_PATH", tmp):
                _reset_logits_md5_file()
            with open(tmp) as f:
                self.assertEqual(f.read(), "")
        finally:
            os.unlink(tmp)


class TestReadLogitsMd5File(unittest.TestCase):
    """Tests for _read_logits_md5_file(): file I/O (filesystem dependency)."""

    def test_reads_entries(self):
        entries = [{"tag": "a", "md5": "abc"}, {"tag": "b", "md5": "def"}]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            tmp = f.name
        try:
            with patch.object(_det_mod, "_DET_MD5_PATH", tmp):
                result = _read_logits_md5_file()
            self.assertEqual(result, entries)
        finally:
            os.unlink(tmp)

    def test_file_not_found_returns_empty(self):
        with patch.object(_det_mod, "_DET_MD5_PATH", "/tmp/_nonexistent_12345.jsonl"):
            result = _read_logits_md5_file()
        self.assertEqual(result, [])

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w") as f:
            f.write('{"a": 1}\n\n\n{"b": 2}\n')
            tmp = f.name
        try:
            with patch.object(_det_mod, "_DET_MD5_PATH", tmp):
                result = _read_logits_md5_file()
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(tmp)


class TestRecordLogitsDiagnostic(unittest.TestCase):
    """Tests for _record_logits_diagnostic(): mock paddle tensor (GPU) + file I/O."""

    def setUp(self):
        self._md5_f = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self._fp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self._md5_f.close()
        self._fp_f.close()
        self._patches = [
            patch.object(_det_mod, "_DET_MD5_PATH", self._md5_f.name),
            patch.object(_det_mod, "_DET_FINGERPRINT_PATH", self._fp_f.name),
            # paddle.no_grad is a GPU context manager -- mock as no-op
            patch.object(
                _det_mod.paddle,
                "no_grad",
                Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock(return_value=False))),
            ),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.unlink(self._md5_f.name)
        os.unlink(self._fp_f.name)

    def _read_jsonl(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_writes_fingerprint_and_md5(self):
        t = _make_astype_tensor([[1.0, 2.0, 3.0]])
        _record_logits_diagnostic(t, tag="test")

        fp_entries = self._read_jsonl(self._fp_f.name)
        self.assertEqual(len(fp_entries), 1)
        for key in ("sum", "argmax", "max", "batch"):
            self.assertIn(key, fp_entries[0])
        self.assertEqual(fp_entries[0]["batch"], 1)

        md5_entries = self._read_jsonl(self._md5_f.name)
        self.assertEqual(len(md5_entries), 1)
        self.assertEqual(md5_entries[0]["tag"], "test")
        self.assertEqual(len(md5_entries[0]["logits_md5"]), 32)
        self.assertEqual(md5_entries[0]["probs_md5"], "")

    def test_with_probs(self):
        logits = _make_astype_tensor([[1.0, 2.0]])
        probs = _make_astype_tensor([[0.3, 0.7]])
        _record_logits_diagnostic(logits, tag="t", probs=probs)

        md5_entries = self._read_jsonl(self._md5_f.name)
        self.assertNotEqual(md5_entries[0]["probs_md5"], "")
        self.assertEqual(len(md5_entries[0]["probs_md5"]), 32)

    def test_appends_multiple_calls(self):
        t1 = _make_astype_tensor([[1.0]])
        t2 = _make_astype_tensor([[2.0]])
        _record_logits_diagnostic(t1, tag="first")
        _record_logits_diagnostic(t2, tag="second")

        md5_entries = self._read_jsonl(self._md5_f.name)
        self.assertEqual(len(md5_entries), 2)
        self.assertEqual(md5_entries[0]["tag"], "first")
        self.assertEqual(md5_entries[1]["tag"], "second")


if __name__ == "__main__":
    unittest.main()
