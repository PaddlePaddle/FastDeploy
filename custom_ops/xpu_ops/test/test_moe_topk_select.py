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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import f_moe_topk_select

np.random.seed(2025)

token_num = 15
expert_num = 256
moe_topk = 8
apply_norm_weight = True

gating_logits = np.random.random([token_num, expert_num]).astype("float32")
bias = np.random.random([expert_num]).astype("float32")


def ref_moe_topk_select(gating_logits, bias, moe_topk, apply_norm_weight):
    assert apply_norm_weight is True

    def _softmax(x):
        axis = 1
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    softmax_logits = _softmax(gating_logits)
    softmax_logits_with_bias = np.copy(softmax_logits)
    if bias is not None:
        softmax_logits_with_bias += bias.reshape([1, -1])
    sorted_indices = np.argsort(softmax_logits_with_bias, axis=1, kind="stable")[:, ::-1]
    topk_ids = sorted_indices[:, :moe_topk]
    topk_weights = np.take_along_axis(softmax_logits, topk_ids, axis=1)
    topk_weights = topk_weights[:, :moe_topk]
    topk_weights /= np.sum(topk_weights, axis=1, keepdims=True)
    return topk_ids, topk_weights


ref_topk_ids, ref_topk_weights = ref_moe_topk_select(gating_logits, bias, moe_topk, apply_norm_weight)

gating_logits = paddle.to_tensor(gating_logits)
if bias is not None:
    bias = paddle.to_tensor(bias)

topk_ids, topk_weights = f_moe_topk_select(gating_logits, bias, moe_topk, apply_norm_weight)

assert np.array_equal(
    topk_ids.numpy(), ref_topk_ids
), f"\ntopk_ids:\n{topk_ids.numpy()}\nref_topk_ids:\n{ref_topk_ids}"
assert np.allclose(
    topk_weights.numpy(), ref_topk_weights
), f"\ntopk_weights:\n{topk_weights.numpy()}\nref_topk_weights:\n{ref_topk_weights}"

print("Passed all tests.")
