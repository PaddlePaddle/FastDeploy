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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from fastdeploy.distributed.parallel_state import \
    get_tensor_model_parallel_world_size
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.ops import \
    apply_penalty_multi_scores
from fastdeploy.platforms import current_platform


class Sampler(nn.Layer):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        if current_platform.is_cuda():
            self.nranks = get_tensor_model_parallel_world_size()
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError()

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> paddle.Tensor:
        """
        """

        logits = apply_penalty_multi_scores(
            sampling_metadata.prompt_token_ids,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
        )

        probs = F.softmax(logits)

        _, next_tokens = paddle.tensor.top_p_sampling(probs,
                                                      sampling_metadata.top_p)

        if self.nranks > 1:
            paddle.distributed.broadcast(next_tokens, 0)

        return next_tokens
