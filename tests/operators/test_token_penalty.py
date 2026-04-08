# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""UT for get_token_penalty"""
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import get_token_penalty_once


class TestTokenPenalty(unittest.TestCase):
    def setUp(self):
        paddle.seed(2023)
        self.pre_ids = paddle.randint(0, 10000, (8, 1000))
        self.pre_ids[:, -1] = self.pre_ids[:, -2]
        self.logits = paddle.rand(shape=[8, 10000], dtype="float16")
        self.penalty_scores = np.array([1.2] * 8).astype(np.float16).reshape(-1, 1)
        self.penalty_scores = paddle.to_tensor(self.penalty_scores)

    def test_token_penalty_once(self):
        res = get_token_penalty_once(self.pre_ids, self.logits, self.penalty_scores)

        # 验证结果形状
        self.assertEqual(res.shape, self.logits.shape)

        # 验证惩罚逻辑
        for i in range(8):
            original_values = self.logits[i][self.pre_ids[i]]
            penalized_values = res[i][self.pre_ids[i]]
            # 检查是否应用了惩罚
            for orig, penal in zip(original_values.numpy(), penalized_values.numpy()):
                if orig < 0:
                    self.assertLess(penal, orig, "负值应该乘以惩罚因子")
                else:
                    self.assertLess(penal, orig, "正值应该除以惩罚因子")

    def test_compare_with_naive_implementation(self):
        res = get_token_penalty_once(self.pre_ids, self.logits, self.penalty_scores)

        # 朴素实现
        score = paddle.index_sample(self.logits, self.pre_ids)
        score = paddle.where(score < 0, score * self.penalty_scores, score / self.penalty_scores)

        bsz = paddle.shape(self.logits)[0]
        bsz_range = paddle.arange(start=bsz * 0, end=bsz, step=bsz / bsz, name="bsz_range", dtype="int64").unsqueeze(
            -1
        )
        input_ids = self.pre_ids + bsz_range * self.logits.shape[-1]
        res2 = paddle.scatter(self.logits.flatten(), input_ids.flatten(), score.flatten()).reshape(self.logits.shape)

        # 比较两种实现的结果差异
        max_diff = (res - res2).abs().max().item()
        self.assertLess(max_diff, 1e-5)


if __name__ == "__main__":
    unittest.main()
