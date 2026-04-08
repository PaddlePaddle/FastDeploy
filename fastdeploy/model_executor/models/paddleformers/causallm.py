"""
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

"""Causal LM Mixin for PaddleFormers models.
This mixin provides lm_head and compute_logits functionality.
The forward() method is implemented in PaddleFormersModelBase.
"""

from typing import TYPE_CHECKING

import paddle

from fastdeploy.model_executor.layers.lm_head import ParallelLMHead

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


class CausalLMMixin:
    """Mixin class that provides causal LM functionality for PaddleFormers models.

    This mixin only handles:
    - lm_head initialization
    - compute_logits (hidden_states -> logits)

    The forward() method is inherited from PaddleFormersModelBase which computes
    input_ids -> hidden_states.

    This is a private mixin class and should NOT be instantiated directly.
    Use PaddleFormersForCausalLM instead.
    """

    def __init__(self, fd_config: "FDConfig", **kwargs):

        super().__init__(fd_config, **kwargs)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        with_bias = getattr(self.text_config, "use_bias", False) or getattr(self.text_config, "bias", False)

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=self.text_config.hidden_size,
            num_embeddings=self.text_config.vocab_size,
            prefix="lm_head",
            with_bias=with_bias,
        )

    def compute_logits(self, hidden_state, **kwargs):
        """Compute logits from hidden states using lm_head."""
        logits = self.lm_head(hidden_state)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")

        return logits

    def set_state_dict(self, state_dict):
        self.load_weights(state_dict.items())
