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
from paddleformers.utils.log import logger

try:
    from fastdeploy.model_executor.ops.gpu import noaux_tc
except:
    logger.warning("import noaux_tc Failed!")


def get_moe_scores(
    gating_output: paddle.Tensor,
    n_group,
    topk_group,
    top_k,
    routed_scaling_factor,
    e_score_correction_bias,
) -> paddle.Tensor:
    """
    compute moe scores using e_score_correction_bias.
    """
    scores = paddle.nn.functional.sigmoid(gating_output)
    assert e_score_correction_bias is not None, "e_score_correction_bias is none!"
    scores_with_bias = scores + e_score_correction_bias
    scores, topk_values, topk_idx = noaux_tc(
        scores,
        scores_with_bias,
        n_group if n_group > 0 else 1,
        topk_group if topk_group > 0 else 1,
        top_k,
        routed_scaling_factor,
    )
    return scores, topk_values, topk_idx
