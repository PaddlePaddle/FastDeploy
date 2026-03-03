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
Near-zero-overhead deterministic probe for GPU tensor fingerprinting.

During inference: only tensor.clone() (async GPU memcpy, ~10μs) is added
to the CUDA stream — NO reduction kernels, NO host-device sync.
After inference: MD5 hashes are computed from stored clones and written
to disk for cross-run comparison.

This preserves the GPU kernel execution pattern as closely as possible,
minimizing the chance of masking race conditions (Heisenberg bug).

Cross-process design:
  - Worker subprocess: record() stores tensor clones, near-zero overhead.
  - Main process: writes signal file to /tmp/fd_det_probe/run_id before
    each generate().  Worker detects run_id change → flushes previous
    run's data (MD5 computation + disk write).  atexit catches the last run.

Usage:
    FD_DETERMINISTIC_MODE=1 FD_DETERMINISTIC_PROBE=1 python run.py

    # Enable per-layer probing (heavier, second-pass only):
    FD_DETERMINISTIC_PROBE_LAYERS=1
"""

import atexit
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

import paddle

det_logger = logging.getLogger("fastdeploy.deterministic")

# ---------------------------------------------------------------------------
# Global probe instance (created once per worker process)
# ---------------------------------------------------------------------------
_probe: Optional["DeterministicProbe"] = None


def get_probe() -> Optional["DeterministicProbe"]:
    return _probe


def init_probe(log_dir: str = "/tmp/fd_det_probe"):
    global _probe
    _probe = DeterministicProbe(log_dir)
    atexit.register(_probe._atexit_flush)
    return _probe


def signal_run_id(run_id: str, log_dir: str = "/tmp/fd_det_probe"):
    """Called from MAIN process to tell worker which run_id to use."""
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run_id"), "w") as f:
        f.write(run_id)


# ---------------------------------------------------------------------------
# Core probe class
# ---------------------------------------------------------------------------
class DeterministicProbe:
    """Store tensor clones during inference, compute MD5 hashes post-hoc."""

    def __init__(self, log_dir: str = "/tmp/fd_det_probe"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._step_buffer: List[Dict[str, paddle.Tensor]] = []
        self._current_step: Dict[str, paddle.Tensor] = {}
        self._step_count = 0
        self._run_id: Optional[str] = None
        self._signal_path = os.path.join(log_dir, "run_id")

    # -- recording (near-zero overhead) ------------------------------------

    def record(self, tag: str, tensor: paddle.Tensor):
        """Store tensor reference — ZERO GPU overhead.

        No clone(), no computation, no sync.  Just a Python list.append().
        The tensor stays alive on GPU via Python reference counting.

        WARNING: If the tensor is modified in-place later (e.g., sampler
        applies penalties to logits), we see the modified version.  This is
        acceptable for diagnosis: at the first divergence step, penalties
        are identical (same preceding tokens), so modified logits comparison
        is still valid.
        """
        # Check if main process changed run_id → flush previous run
        self._check_run_id_signal()

        self._current_step[tag] = tensor

    def step_done(self):
        """Mark end of one decode/prefill step."""
        self._step_buffer.append(self._current_step)
        self._current_step = {}
        self._step_count += 1

        # Log op call report after first step to verify monkey-patch
        if self._step_count == 1:
            self._log_first_step_op_report()

    @staticmethod
    def _log_first_step_op_report():
        try:
            from fastdeploy.model_executor.layers.batch_invariant_ops import (
                get_op_call_report,
            )

            report = get_op_call_report()
            if report:
                det_logger.info(report)
        except Exception:
            pass

    def _check_run_id_signal(self):
        """Read signal file from main process.  Flush on run_id change."""
        try:
            with open(self._signal_path) as f:
                new_id = f.read().strip()
        except FileNotFoundError:
            return

        if not new_id or new_id == self._run_id:
            return

        # run_id changed → flush previous run's data
        if self._step_buffer:
            self.flush()
        self._run_id = new_id

    # -- flushing (post-inference, sync is OK) -----------------------------

    def flush(self, run_id: Optional[str] = None) -> str:
        """Compute MD5 hashes from stored tensor clones and write to JSON.

        This runs AFTER inference completes, so D2H sync is acceptable.
        """
        rid = run_id or self._run_id or "unknown"
        out_path = os.path.join(self.log_dir, f"probe_{rid}.json")

        if not self._step_buffer:
            _write_json(out_path, {"run_id": rid, "steps": []})
            self._reset()
            return out_path

        steps: List[Dict] = []
        for step_dict in self._step_buffer:
            step_hashes = {}
            for tag, tensor in step_dict.items():
                data = tensor.cast("float32").numpy().tobytes()
                step_hashes[tag] = hashlib.md5(data).hexdigest()
            steps.append(step_hashes)

        result = {"run_id": rid, "steps": steps}
        _write_json(out_path, result)

        det_logger.info(f"[PROBE] flushed {len(self._step_buffer)} steps -> {out_path}")
        self._reset()
        return out_path

    def _atexit_flush(self):
        """Flush remaining data on process exit (catches last run)."""
        if self._step_buffer:
            self.flush()

    def _reset(self):
        self._step_buffer.clear()
        self._current_step.clear()
        self._step_count = 0

    # -- static comparison utility -----------------------------------------

    @staticmethod
    def diff(file_a: str, file_b: str) -> Dict:
        """Compare two probe dump files and return first divergence info."""
        a = _read_json(file_a)
        b = _read_json(file_b)
        steps_a, steps_b = a["steps"], b["steps"]
        n = min(len(steps_a), len(steps_b))

        for i in range(n):
            sa, sb = steps_a[i], steps_b[i]
            all_tags = set(list(sa.keys()) + list(sb.keys()))
            for tag in sorted(all_tags):
                va = sa.get(tag)
                vb = sb.get(tag)
                if va != vb:
                    return {
                        "match": False,
                        "first_diff_step": i,
                        "tag": tag,
                        "a": va,
                        "b": vb,
                        "total_steps": n,
                    }

        if len(steps_a) != len(steps_b):
            return {
                "match": False,
                "first_diff_step": n,
                "tag": "LENGTH",
                "a": len(steps_a),
                "b": len(steps_b),
                "total_steps": n,
            }

        return {"match": True, "total_steps": n}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
