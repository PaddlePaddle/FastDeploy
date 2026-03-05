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
import logging
import time

import numpy as np

det_logger = logging.getLogger("fastdeploy.deterministic")


class DeterministicLogger:
    """Helper for logging tensor MD5 hashes and input details to assist determinism debugging."""

    def __init__(self, share_inputs):
        self.share_inputs = share_inputs
        self._current_run_id = None
        self._batch_counter = 0

    def log_batch_start(self, model_forward_batch):
        """Log batch start with run_id tracking and batch counting."""
        current_run_id = None
        for req in model_forward_batch or []:
            if req is not None:
                parts = req.request_id.split("_")
                if len(parts) > 1:
                    current_run_id = parts[-1]
                    break
        if current_run_id is not None and current_run_id != self._current_run_id:
            self._current_run_id = current_run_id
            self._batch_counter = 0

        self._batch_counter += 1

        det_logger.info(f"\n{'='*80}")
        det_logger.info(f"[BATCH-START] Run_{self._current_run_id} Batch_{self._batch_counter}")
        det_logger.info(f"{'='*80}\n")

    @staticmethod
    def _compute_tensor_md5(tensor, name="tensor", prefix=""):
        """Compute MD5 hash of tensor for comparison"""
        if tensor is None:
            return f"{name}_md5=None"

        # Copy tensor to CPU and convert to numpy array
        try:
            tensor_cpu = tensor.cpu().numpy().tobytes()
        except Exception:
            # For data types that don't support direct tobytes (e.g., float16), convert first
            tensor_cpu = tensor.cpu().numpy().astype(np.float32).tobytes()

        md5_hash = hashlib.md5(tensor_cpu).hexdigest()
        return f"{prefix}{name}_md5={md5_hash[:16]}"  # Print only first 16 chars to reduce log length

    def log_tensor_md5s(self, tensor_dict, forward_batch_reqs_list=None, stage="forward"):
        """Log MD5 hash values for multiple tensors, including per-request MD5

        Args:
            tensor_dict: {name: tensor} dictionary
            forward_batch_reqs_list: forward_batch_reqs_list list (may contain None)
            stage: Stage identifier (e.g., "prefill", "decode", "forward")
        """
        # Get batch size from first valid tensor
        batch_size = self._get_batch_size(tensor_dict)
        if batch_size is None:
            return

        # Get prefill/decode counts
        prefill_count, decode_count, seq_lens_encoder = self._get_stage_counts(batch_size)

        # Build stage information
        stage_info = stage
        if prefill_count > 0 or decode_count > 0:
            stage_info += f" (prefill={prefill_count}, decode={decode_count})"

        # Compute and log batch MD5
        batch_md5_info = [
            self._compute_tensor_md5(tensor, name, prefix="batch_")
            for name, tensor in tensor_dict.items()
            if tensor is not None
        ]

        # Log overall batch MD5
        req_id_str = self._build_req_id_str(forward_batch_reqs_list)
        det_logger.info(
            f"[DETERMINISM-MD5] stage={stage_info} | batch_size={batch_size} | "
            + (f"requests: {req_id_str} | " if req_id_str else "")
            + " | ".join(batch_md5_info)
        )

        # Log per-request MD5 for decode requests
        self._log_per_request_md5s(
            tensor_dict, forward_batch_reqs_list, batch_size, prefill_count, decode_count, seq_lens_encoder
        )

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
        """Log per-request MD5 for decode requests.

        In decode phase, tensor shape is [batch_size, hidden_dim] or [batch_size, vocab_size].
        Can split by batch dimension directly.
        """
        if decode_count == 0 or forward_batch_reqs_list is None:
            return

        for i, req in enumerate(forward_batch_reqs_list):
            if req is None or i >= batch_size:
                continue

            # Check if this is a decode request
            if seq_lens_encoder is not None:
                if i >= len(seq_lens_encoder) or int(seq_lens_encoder[i]) != 0:
                    continue  # Skip prefill requests
            elif prefill_count > 0:
                continue  # Mixed batch without seq_lens_encoder, skip all

            req_id = req.request_id
            req_md5_info = [
                self._compute_tensor_md5(tensor[i : i + 1], name)
                for name, tensor in tensor_dict.items()
                if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2
            ]

            if req_md5_info:
                det_logger.info(f"[DETERMINISM-MD5-REQ] {req_id} | decode | " + " | ".join(req_md5_info))

    def log_prefill_input(self, request_id, idx, prefill_start_index, prefill_end_index, input_ids):
        """Log prefill input details for determinism verification."""
        det_logger.info(
            f"[DETERMINISM] Prefill input - request_id: {request_id}, "
            f"idx: {idx}, prefill_start_index: {prefill_start_index}, "
            f"prefill_end_index: {prefill_end_index}, "
            f"input_ids: {input_ids}"
        )

    def log_deterministic_input(self, forward_meta):
        """Log determinism inference input information, supports multiple batch requests"""
        ids = forward_meta.ids_remove_padding
        req_ids = self.share_inputs.get("req_ids", None)
        seq_lens_this_time = self.share_inputs.get("seq_lens_this_time", None)
        seq_lens_encoder = self.share_inputs.get("seq_lens_encoder", None)
        seq_lens_decoder = self.share_inputs.get("seq_lens_decoder", None)

        # Get batch size
        num_requests = len(seq_lens_this_time) if seq_lens_this_time is not None else 0

        det_logger.info(f"[DETERMINISM-INPUT] time={time.time():.6f} | batch_size={num_requests}")

        if num_requests == 0 or ids is None:
            det_logger.info("[DETERMINISM-INPUT] No input data")
            return

        # Split ids for each request
        ids_list = ids.cpu().numpy().tolist()
        offset = 0

        for i in range(num_requests):
            # Get current request information
            req_id = req_ids[i] if req_ids is not None and i < len(req_ids) else f"idx_{i}"
            seq_len = int(seq_lens_this_time[i])
            seq_len_enc = int(seq_lens_encoder[i]) if seq_lens_encoder is not None and i < len(seq_lens_encoder) else 0
            seq_len_dec = int(seq_lens_decoder[i]) if seq_lens_decoder is not None and i < len(seq_lens_decoder) else 0

            # Get current request's tokens
            if seq_len > 0:
                request_tokens = ids_list[offset : offset + seq_len]
            else:
                request_tokens = []

            offset += seq_len

            # Print one line log
            det_logger.info(
                f"[DETERMINISM-INPUT] req_id={req_id} | tokens={request_tokens} | "
                f"len={seq_len} | seq_len_enc={seq_len_enc} | seq_len_dec={seq_len_dec}"
            )
