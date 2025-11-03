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

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from fastdeploy.distributed.custom_all_reduce import CustomAllreduce


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment,
        including setting random seeds.
        """
        paddle.seed(2025)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        fleet.init(is_collective=True, strategy=strategy)

    def test_case(self):
        """
        Check if the CustomAllreduce function works properly.
        """

        mns = [[1, 2048], [2, 4096], [20, 4096], [128, 4096], [256, 4096], [256, 8192]]

        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        fa = CustomAllreduce(model_parallel_group)

        for m, n in mns:
            data_custom_ar = paddle.rand([m, n], dtype="bfloat16")
            data_paddle = data_custom_ar.clone()
            if fa.should_custom_ar(data_custom_ar):
                data_custom_ar = fa.custom_all_reduce(data_custom_ar)
            dist.all_reduce(data_paddle)
            if dist.get_rank() == 0:
                np.testing.assert_allclose(
                    data_custom_ar.numpy(),
                    data_paddle.numpy(),
                    rtol=1e-04,
                    atol=1e-04,
                )


if __name__ == "__main__":
    unittest.main()
