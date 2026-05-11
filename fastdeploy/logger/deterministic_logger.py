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

import hashlib
import json
import logging
import time

import numpy as np
import paddle

_CHOICE_SEPARATOR = "::n::"


def _parse_choice_id(compound_id: str) -> tuple:
    """Parse an internal request ID into (base_request_id, choice_index)."""
    if _CHOICE_SEPARATOR in compound_id:
        base, idx = compound_id.rsplit(_CHOICE_SEPARATOR, 1)
        return base, int(idx)
    return compound_id, None


det_logger = logging.getLogger("fastdeploy.deterministic")

# ---------------------------------------------------------------------------
# File paths for deterministic diagnostics (written under /tmp)
# ---------------------------------------------------------------------------
_DET_MD5_PATH = "/tmp/fd_det_logits_md5.jsonl"
_DET_FINGERPRINT_PATH = "/tmp/fd_det_logits.jsonl"


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------
def _compute_md5(tensor: paddle.Tensor) -> str:
    """Compute MD5 hex-digest of a tensor (after casting to float32 if needed).

    Triggers GPU->CPU sync -- use only in diagnostic paths.
    """
    try:
        data = tensor.cpu().numpy().tobytes()
    except Exception:
        data = tensor.cpu().numpy().astype(np.float32).tobytes()
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# File-based MD5 hash collection (shared across processes via filesystem)
# ---------------------------------------------------------------------------
def _reset_logits_md5_file():
    """Truncate the MD5 hash file (call before each generate run)."""
    with open(_DET_MD5_PATH, "w"):
        pass


def _read_logits_md5_file():
    """Read all MD5 hash entries from the file (call after generate completes)."""
    entries = []
    try:
        with open(_DET_MD5_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        pass
    return entries


def _record_logits_diagnostic(
    logits: paddle.Tensor,
    tag: str = "logits",
    probs: paddle.Tensor = None,
):
    """Record both lightweight fingerprint and bit-exact MD5 in one GPU sync.

    Writes a fingerprint line to ``_DET_FINGERPRINT_PATH`` and an MD5 line to
    ``_DET_MD5_PATH``.  The two files serve different purposes:
      - fingerprint: human-readable summary (sum / argmax / max)
      - MD5: bit-exact cross-run comparison

    WARNING: triggers GPU sync via .cpu().numpy() -- may change CUDA stream
    timing.  Use only for diagnostics.
    """
    with paddle.no_grad():
        fp = logits.astype("float32")
        fp_np = fp.cpu().numpy()

        # --- fingerprint (lightweight summary) ---
        fingerprint = {
            "sum": float(fp_np.sum()),
            "argmax": int(fp_np[0].argmax()),
            "max": float(fp_np[0].max()),
            "batch": logits.shape[0],
        }
        with open(_DET_FINGERPRINT_PATH, "a") as f:
            f.write(json.dumps(fingerprint) + "\n")

        # --- MD5 (bit-exact) ---
        logits_md5 = _compute_md5(fp)
        entry = {"tag": tag, "logits_md5": logits_md5, "probs_md5": ""}
        if probs is not None:
            entry["probs_md5"] = _compute_md5(probs)
        with open(_DET_MD5_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


# Keep old names as aliases so existing callers (sampler.py, tests) still work.
_record_logits_fingerprint = _record_logits_diagnostic
_record_logits_md5 = _record_logits_diagnostic


# ---------------------------------------------------------------------------
# DeterministicLogger -- per-batch / per-request tensor MD5 logging
# ---------------------------------------------------------------------------
class DeterministicLogger:
    """Helper for logging tensor MD5 hashes and input details to assist determinism debugging."""

    def __init__(self, share_inputs):
        self.share_inputs = share_inputs
        self._current_run_id = None
        self._batch_counter = 0

    # ---- batch lifecycle ----

    def log_batch_start(self, model_forward_batch):
        """Log batch start with run_id tracking and batch counting."""
        current_run_id = None
        for req in model_forward_batch or []:
            if req is not None:
                _, index = _parse_choice_id(req.request_id)
                if index is not None:
                    current_run_id = str(index)
                    break
        if current_run_id is not None and current_run_id != self._current_run_id:
            self._current_run_id = current_run_id
            self._batch_counter = 0

        self._batch_counter += 1

        det_logger.info(f"\n{'='*80}")
        det_logger.info(f"[BATCH-START] Run_{self._current_run_id} Batch_{self._batch_counter}")
        det_logger.info(f"{'='*80}\n")

    # ---- tensor MD5 helpers ----

    @staticmethod
    def _compute_tensor_md5(tensor, name="tensor", prefix=""):
        """Compute MD5 hash of tensor for comparison."""
        if tensor is None:
            return f"{name}_md5=None"

        md5_hash = _compute_md5(tensor)
        return f"{prefix}{name}_md5={md5_hash[:16]}"

    def log_tensor_md5s(self, tensor_dict, forward_batch_reqs_list=None, stage="forward"):
        """Log MD5 hash values for multiple tensors, including per-request MD5.

        Args:
            tensor_dict: {name: tensor} dictionary
            forward_batch_reqs_list: request list (may contain None)
            stage: Stage identifier (e.g., "prefill", "decode", "forward")
        """
        batch_size = self._get_batch_size(tensor_dict)
        if batch_size is None:
            return

        prefill_count, decode_count, seq_lens_encoder = self._get_stage_counts(batch_size)

        stage_info = stage
        if prefill_count > 0 or decode_count > 0:
            stage_info += f" (prefill={prefill_count}, decode={decode_count})"

        batch_md5_info = [
            self._compute_tensor_md5(tensor, name, prefix="batch_")
            for name, tensor in tensor_dict.items()
            if tensor is not None
        ]

        req_id_str = self._build_req_id_str(forward_batch_reqs_list)
        det_logger.info(
            f"[DETERMINISM-MD5] stage={stage_info} | batch_size={batch_size} | "
            + (f"requests: {req_id_str} | " if req_id_str else "")
            + " | ".join(batch_md5_info)
        )

        self._log_per_request_md5s(
            tensor_dict, forward_batch_reqs_list, batch_size, prefill_count, decode_count, seq_lens_encoder
        )

    # ---- internal helpers ----

    @staticmethod
    def _get_batch_size(tensor_dict):
        """Get batch size from first tensor with a shape."""
        for name, tensor in tensor_dict.items():
            if tensor is not None and hasattr(tensor, "shape"):
                return tensor.shape[0]
        return None

    def _get_stage_counts(self, batch_size):
        """Get prefill/decode counts and seq_lens_encoder."""
        prefill_count = 0
        decode_count = 0
        seq_lens_encoder = None

        if self.share_inputs is not None and "seq_lens_encoder" in self.share_inputs:
            seq_lens_encoder = self.share_inputs["seq_lens_encoder"].cpu().numpy()
            prefill_count = int((seq_lens_encoder > 0).sum())
            decode_count = int(batch_size - prefill_count)

        return prefill_count, decode_count, seq_lens_encoder

    @staticmethod
    def _build_req_id_str(forward_batch_reqs_list):
        """Build request ID string from forward_batch_reqs_list."""
        if forward_batch_reqs_list is None:
            return ""
        req_info = [f"[{i}]{req.request_id}" for i, req in enumerate(forward_batch_reqs_list) if req is not None]
        return ", ".join(req_info)

    def _log_per_request_md5s(
        self, tensor_dict, forward_batch_reqs_list, batch_size, prefill_count, decode_count, seq_lens_encoder
    ):
        """Log per-request MD5 for decode requests."""
        if decode_count == 0 or forward_batch_reqs_list is None:
            return

        for i, req in enumerate(forward_batch_reqs_list):
            if req is None or i >= batch_size:
                continue

            if seq_lens_encoder is not None:
                if i >= len(seq_lens_encoder) or int(seq_lens_encoder[i]) != 0:
                    continue
            elif prefill_count > 0:
                continue

            req_id = req.request_id
            req_md5_info = [
                self._compute_tensor_md5(tensor[i : i + 1], name)
                for name, tensor in tensor_dict.items()
                if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2
            ]

            if req_md5_info:
                det_logger.info(f"[DETERMINISM-MD5-REQ] {req_id} | decode | " + " | ".join(req_md5_info))

    # ---- input logging ----

    def log_prefill_input(self, request_id, idx, prefill_start_index, prefill_end_index, input_ids):
        """Log prefill input details for determinism verification."""
        det_logger.info(
            f"[DETERMINISM] Prefill input - request_id: {request_id}, "
            f"idx: {idx}, prefill_start_index: {prefill_start_index}, "
            f"prefill_end_index: {prefill_end_index}, "
            f"input_ids: {input_ids}"
        )

    def log_deterministic_input(self, forward_meta):
        """Log determinism inference input information, supports multiple batch requests."""
        ids = forward_meta.ids_remove_padding
        req_ids = self.share_inputs.get("req_ids", None)
        seq_lens_this_time = self.share_inputs.get("seq_lens_this_time", None)
        seq_lens_encoder = self.share_inputs.get("seq_lens_encoder", None)
        seq_lens_decoder = self.share_inputs.get("seq_lens_decoder", None)

        num_requests = len(seq_lens_this_time) if seq_lens_this_time is not None else 0

        det_logger.info(f"[DETERMINISM-INPUT] time={time.time():.6f} | batch_size={num_requests}")

        if num_requests == 0 or ids is None:
            det_logger.info("[DETERMINISM-INPUT] No input data")
            return

        ids_list = ids.cpu().numpy().tolist()
        offset = 0

        for i in range(num_requests):
            req_id = req_ids[i] if req_ids is not None and i < len(req_ids) else f"idx_{i}"
            seq_len = int(seq_lens_this_time[i])
            seq_len_enc = int(seq_lens_encoder[i]) if seq_lens_encoder is not None and i < len(seq_lens_encoder) else 0
            seq_len_dec = int(seq_lens_decoder[i]) if seq_lens_decoder is not None and i < len(seq_lens_decoder) else 0

            request_tokens = ids_list[offset : offset + seq_len] if seq_len > 0 else []
            offset += seq_len

            det_logger.info(
                f"[DETERMINISM-INPUT] req_id={req_id} | tokens={request_tokens} | "
                f"len={seq_len} | seq_len_enc={seq_len_enc} | seq_len_dec={seq_len_dec}"
            )
