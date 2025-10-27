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

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.logits_processor.base import LogitsProcessor


class LogitBiasLogitsProcessor(LogitsProcessor):
    """
    Maintains per-request logit biases and applies them to logits.
    """

    def __init__(self, fd_config: FDConfig):
        self.biases: dict[str, dict[int, float]] = {}  # req_id -> {tok_id -> bias}
        self.device = paddle.device.get_device()
        self.dtype = fd_config.model_config.dtype
        self.bias_indices = (
            paddle.zeros([], dtype="int32").to(self.device),
            paddle.zeros([], dtype="int32").to(self.device),
        )
        self.bias_tensor: paddle.Tensor = paddle.zeros([]).to(self.device, self.dtype)
        self.skipped = False

    def update_state(self, share_inputs: dict):
        """Build per-step logit-bias state from request slots and move it to device."""

        # Retrive inference states from share_inputs
        stop_flags = share_inputs["stop_flags"]
        logits_processors_args = share_inputs["logits_processors_args"]

        # Get bias states for each request
        batch_ids: list[int] = []
        token_ids: list[int] = []
        biases: list[float] = []
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
        self.bias_tensor = paddle.tensor(biases).to(self.device, self.dtype)

    def apply(self, logits: paddle.Tensor) -> paddle.Tensor:
        """Apply logit bias to logits: [batch_size, vocab_size]"""
        logits[self.bias_indices] += self.bias_tensor
        return logits
