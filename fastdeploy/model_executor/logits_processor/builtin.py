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
from typing import List, Dict

from fastdeploy import SamplingParams
from fastdeploy.model_executor.logits_processor.base import LogitsProcessor
from fastdeploy.config import FDConfig
from fastdeploy.utils import llm_logger
from fastdeploy.engine.request import Request

class LogitBiasLogitsProcessor(LogitsProcessor):
    """
    """

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        self.biases: dict[str, dict[int, float]] = {}    # req_id -> {tok_id -> bias}
        self.device = paddle.device.get_device()
        self.bias_indices = (
            paddle.zeros([], dtype="int32").to(self.device), 
            paddle.zeros([], dtype="int32").to(self.device)
        )
        self.bias_tensor: paddle.Tensor = paddle.zeros([]).to(self.device)
        self.skipped = False

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch: List[Request], share_inputs: Dict):

        if batch is None:
            self.skipped = True
            return
        else:
            self.skipped = False

        need_updates = False
        batch_req_slot_map: dict = {}
        for request in batch:
            # Get request_id (a unique string) and its slot_id in running batch
            request_id: str = request.request_id
            slot_id: int = request.idx
            batch_req_slot_map[request_id] = slot_id

            # Insert bias states for this request
            logit_bias = share_inputs["logit_bias"][slot_id]
            if logit_bias is not None and request_id not in self.biases:  # new request
                self.biases[request_id] = logit_bias.copy()
                need_updates = True
        
        # Remove bias states for requests that are no longer in the batch
        for request_id in list(self.biases):
            if request_id not in batch_req_slot_map:
                self.biases.pop(request_id)
                need_updates = True

        if need_updates:
            # Make bias indices and bias tensor
            slot_ids: list[int] = []
            token_ids: list[int] = []
            biases: list[float] = []
            for request_id, tok_id_bias_map in self.biases.items():
                slot_ids.extend([batch_req_slot_map[request_id]] * len(tok_id_bias_map))
                token_ids.extend(tok_id_bias_map.keys())
                biases.extend(tok_id_bias_map.values())
            llm_logger.debug(f"slot_ids={slot_ids}, token_ids={token_ids}, biases={biases}")

            self.bias_indices = (
                paddle.tensor(slot_ids, dtype="int32").to(self.device),
                paddle.tensor(token_ids, dtype="int32").to(self.device),
            )
            self.bias_tensor = paddle.tensor(biases, dtype="float32").to(self.device)

    def apply(self, logits: paddle.Tensor) -> paddle.Tensor:
        """ Apply logit bias to logits: [batch_size, vocab_size] """
        if not self.skipped:
            logits[self.bias_indices] += self.bias_tensor
        return logits

