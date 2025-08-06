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

from dataclasses import dataclass
from typing import Optional

import paddle


@dataclass
class SamplingMetadata:
    """
    metadata for sampling.
    """

    temperature: paddle.Tensor

    pre_token_ids: paddle.Tensor
    eos_token_ids: paddle.Tensor
    frequency_penalties: paddle.Tensor
    presence_penalties: paddle.Tensor
    repetition_penalties: paddle.Tensor

    min_dec_lens: paddle.Tensor

    bad_words_token_ids: paddle.Tensor

    step_idx: paddle.Tensor

    top_p: paddle.Tensor
    top_k: Optional[paddle.Tensor] = None
    min_p: Optional[paddle.Tensor] = None
    seed: Optional[paddle.Tensor] = None
    max_num_logprobs: Optional[int] = None
    enable_early_stop: Optional[int] = False
    stop_flags: Optional[paddle.Tensor] = None
    prompt_ids: Optional[paddle.Tensor] = None
    prompt_lens: Optional[paddle.Tensor] = None
