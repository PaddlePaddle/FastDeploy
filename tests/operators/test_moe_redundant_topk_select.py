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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import moe_redundant_topk_select


class TestMoERedundantTopKSelect(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_and_check(
        self,
        gating_shape,
        expert_num,
        moe_topk,
        apply_norm_weight=False,
        enable_softmax_top_k_fused=False,
        use_bias=False,
    ):
        """Helper function to run the operator and check."""
        gating_logits = paddle.to_tensor(np.random.rand(*gating_shape).astype("float32"))
        expert_id_to_ep_rank_array = paddle.to_tensor(
            np.random.randint(0, expert_num, size=(expert_num,)).astype("int32")
        )
        expert_in_rank_num_list = paddle.to_tensor(np.random.randint(1, 4, size=(expert_num,)).astype("int32"))
        tokens_per_expert_stats_list = paddle.zeros([expert_num], dtype="int32")
        bias = None
        if use_bias:
            bias = paddle.to_tensor(np.random.rand(*gating_shape[:-1], expert_num).astype("float32"))

        outputs = moe_redundant_topk_select(
            gating_logits=gating_logits,
            expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
            expert_in_rank_num_list=expert_in_rank_num_list,
            tokens_per_expert_stats_list=tokens_per_expert_stats_list,
            bias=bias,
            moe_topk=moe_topk,
            apply_norm_weight=apply_norm_weight,
            enable_softmax_top_k_fused=enable_softmax_top_k_fused,
            redundant_ep_rank_num_plus_one=2,
        )

        topk_ids, topk_weights = outputs

        # Check shapes are correct
        expected_shape = [int(np.prod(gating_shape[:-1])), moe_topk]
        self.assertEqual(topk_ids.shape, expected_shape)
        self.assertEqual(topk_weights.shape, expected_shape)

        # Check topk_ids are non-negative
        self.assertTrue(np.all(topk_ids.numpy() >= 0))

        # Check topk weights are non-negative
        self.assertTrue(np.all(topk_weights.numpy() >= -1e-6))

        # Check tokens_per_expert_stats_list has valid values
        self.assertEqual(tokens_per_expert_stats_list.shape[0], expert_num)
        self.assertTrue(np.all(tokens_per_expert_stats_list.numpy() >= 0))

    def test_basic_case(self):
        self._run_and_check(gating_shape=(4, 16), expert_num=8, moe_topk=2)

    def test_3d_input_case(self):
        self._run_and_check(gating_shape=(2, 3, 8), expert_num=8, moe_topk=2)

    def test_with_bias(self):
        self._run_and_check(gating_shape=(3, 12), expert_num=4, moe_topk=2, use_bias=True)

    def test_with_norm_weight(self):
        self._run_and_check(gating_shape=(5, 10), expert_num=4, moe_topk=2, apply_norm_weight=True)

    def test_softmax_topk_fused(self):
        self._run_and_check(gating_shape=(6, 8), expert_num=8, moe_topk=2, enable_softmax_top_k_fused=True)


if __name__ == "__main__":
    unittest.main()
