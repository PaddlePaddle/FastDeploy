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

from .apply_penalty_multi_scores import (
    apply_penalty_multi_scores,
    apply_speculative_penalty_multi_scores,
)
from .speculate_logprob_utils import (
    speculate_get_target_logits,
    speculate_insert_first_token,
)
from .top_k_top_p_sampling import min_p_sampling, top_k_top_p_sampling

__all__ = [
    "apply_penalty_multi_scores",
    "apply_speculative_penalty_multi_scores",
    "top_k_top_p_sampling",
    "min_p_sampling",
    "speculate_get_target_logits",
    "speculate_insert_first_token",
]
