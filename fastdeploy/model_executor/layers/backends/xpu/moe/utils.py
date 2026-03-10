"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.model_executor.ops.xpu import fused_noaux_tc


def get_moe_scores(
    gating_output: paddle.Tensor,
    n_group,
    topk_group,
    top_k,
    routed_scaling_factor,
    e_score_correction_bias,
    renormalize: bool = False,
    expert_id_to_ep_rank_array: paddle.Tensor = None,
    expert_in_rank_num_list: paddle.Tensor = None,
    tokens_per_expert_stats_list: paddle.Tensor = None,
    redundant_ep_rank_num_plus_one: int = 1,
) -> paddle.Tensor:
    """
    compute moe scores using e_score_correction_bias.
    """
    assert e_score_correction_bias is not None, "e_score_correction_bias is none!"
    if expert_id_to_ep_rank_array is None:
        scores, topk_values, topk_idx = fused_noaux_tc(
            gating_output,
            e_score_correction_bias,
            n_group if n_group > 0 else 1,
            topk_group if topk_group > 0 else 1,
            top_k,
            renormalize,
            routed_scaling_factor,
        )
    else:
        raise NotImplementedError("Not support noaux_tc_redundant")
    return scores, topk_values, topk_idx
