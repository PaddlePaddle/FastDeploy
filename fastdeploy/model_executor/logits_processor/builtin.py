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
        self.device = paddle.device.get_device()
        self.dtype = fd_config.model_config.dtype
        self.batch_ids: list[int] = []
        self.token_ids: list[int] = []
        self.biases: list[float] = []

    def update_state(self, share_inputs: dict):
        """Build per-step logit-bias state from request slots and move it to device."""

        # Retrive inference states from share_inputs
        stop_flags = share_inputs["stop_flags"]
        logits_processors_args = share_inputs["logits_processors_args"]
        logits_processors_args = [a for a, f in zip(logits_processors_args, stop_flags) if not f]

        # Get bias states for each request
        self.batch_ids = []
        self.token_ids: list[int] = []
        self.biases: list[float] = []
        for batch_id, logit_proc_args in enumerate(logits_processors_args):
            tok_id_bias_map = logit_proc_args.get("logit_bias") or {}
            self.batch_ids.extend([batch_id] * len(tok_id_bias_map))
            self.token_ids.extend(tok_id_bias_map.keys())
            self.biases.extend(tok_id_bias_map.values())

        return

    def apply(self, logits: paddle.Tensor) -> paddle.Tensor:
        """Apply logit bias to logits: [batch_size, vocab_size]"""
        # Skip if no bias is applied
        if len(self.biases) == 0:
            return logits

        # Make bias indices and bias tensor
        bias_indices = (
            paddle.tensor(self.batch_ids, dtype="int32").to(self.device),
            paddle.tensor(self.token_ids, dtype="int32").to(self.device),
        )
        bias_tensor = paddle.tensor(self.biases, device=self.device, dtype=self.dtype)
        logits[bias_indices] += bias_tensor
        return logits
