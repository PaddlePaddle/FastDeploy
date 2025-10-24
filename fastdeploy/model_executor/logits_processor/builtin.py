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

import paddle

from fastdeploy.model_executor.logits_processor.base import LogitsProcessor


class LogitBiasLogitsProcessor(LogitsProcessor):
    """
    Maintains per-request logit biases and applies them to logits.
    """

    def __init__(self):
        self.biases: dict[str, dict[int, float]] = {}  # req_id -> {tok_id -> bias}
        self.device = paddle.device.get_device()
        self.bias_indices = (
            paddle.zeros([], dtype="int32").to(self.device),
            paddle.zeros([], dtype="int32").to(self.device),
        )
        self.bias_tensor: paddle.Tensor = paddle.zeros([]).to(self.device)
        self.skipped = False

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, stop_flags: list[bool], logits_processors_args: list[dict]):
        """
        Build per-step logit-bias state from request slots and move it to device.

        Args:
        stop_flags (list[bool] | None): Per-slot stop indicators for the current
            micro-batch. `False` means the slot is active; `True` means the slot
            is finished and should be ignored. If `None`, the method assumes all
            slots are active.
        logits_processors_args (list[dict]): Per-slot runtime arguments. Each
            item may contain `"logit_bias": dict[int, float]` specifying token
            biases for that slot. Missing or empty maps are treated as no-op.
        """

        batch_ids: list[int] = []
        token_ids: list[int] = []
        biases: list[float] = []

        # Get bias states for each request
        batch_id = 0
        for slot_id, flag in enumerate(stop_flags):
            if not flag:
                tok_id_bias_map = logits_processors_args[slot_id].get("logit_bias") or {}
                batch_ids.extend([batch_id] * len(tok_id_bias_map))
                token_ids.extend(tok_id_bias_map.keys())
                biases.extend(tok_id_bias_map.values())
                batch_id += 1

        # Make bias indices and bias tensor
        self.bias_indices = (
            paddle.tensor(batch_ids, dtype="int32").to(self.device),
            paddle.tensor(token_ids, dtype="int32").to(self.device),
        )
        self.bias_tensor = paddle.tensor(biases, dtype="float32").to(self.device)

    def apply(self, logits: paddle.Tensor) -> paddle.Tensor:
        """Apply logit bias to logits: [batch_size, vocab_size]"""
        logits = logits.clone()
        # NOTE: logits must be cloned before modifying them, otherwise will affect accuracy
        logits[self.bias_indices] += self.bias_tensor
        return logits
