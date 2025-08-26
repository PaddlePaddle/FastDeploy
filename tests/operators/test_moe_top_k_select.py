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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import moe_topk_select


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)
        self.batch_size = 1500
        self.num_experts = 128
        self.top_k = 8

    def moe_topk_select_ref(self, gate_out: paddle.Tensor, bias: paddle.Tensor, top_k: int, apply_norm_weight: bool):
        gate_out_after_softmax = paddle.nn.functional.softmax(gate_out, axis=-1)
        topk_weights_ref, topk_ids_ref = paddle.topk(gate_out_after_softmax, k=top_k, axis=-1)

        if bias is not None:
            gate_out_after_softmax_bias = gate_out_after_softmax + bias
            _, topk_ids_ref = paddle.topk(gate_out_after_softmax_bias, k=top_k, axis=-1)
            batch_indices = paddle.arange(gate_out.shape[0]).unsqueeze(-1).expand_as(topk_ids_ref)
            topk_weights_ref = gate_out_after_softmax.gather_nd(paddle.stack([batch_indices, topk_ids_ref], axis=-1))

        if apply_norm_weight:
            topk_weights_ref = topk_weights_ref / topk_weights_ref.sum(axis=-1, keepdim=True)

        return topk_ids_ref, topk_weights_ref

    def test_moe_topk_select(self):
        """
        Check moe_topk_select.
        """
        gate_out = paddle.rand([self.batch_size, self.num_experts], dtype="float32")
        gate_correction_bias = paddle.rand([1, self.num_experts], dtype="float32")
        gate_correction_bias = gate_correction_bias / 10.0

        for apply_norm_weight in [True, False]:
            for bias in [None, gate_correction_bias]:
                topk_ids_ref, topk_weights_ref = self.moe_topk_select_ref(
                    gate_out, bias, self.top_k, apply_norm_weight
                )
                for fused in [True, False]:
                    topk_ids, topk_weights = moe_topk_select(
                        gate_out,
                        bias,
                        self.top_k,
                        apply_norm_weight,
                        fused,
                    )

                    np.testing.assert_allclose(
                        topk_ids_ref,
                        topk_ids,
                        rtol=1e-05,
                        atol=1e-05,
                    )

                    np.testing.assert_allclose(
                        topk_weights_ref,
                        topk_weights,
                        rtol=1e-05,
                        atol=1e-05,
                    )


if __name__ == "__main__":
    unittest.main()
