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

from fastdeploy.worker import eplb


class TestRebalanceExperts(unittest.TestCase):
    def test_rebalance_experts_shapes_param1(self):
        self.run_rebalance_shapes(3, 16, 4, 2, 4, 16)

    def test_rebalance_experts_shapes_param2(self):
        self.run_rebalance_shapes(2, 8, 3, 2, 4, 8)

    def run_rebalance_shapes(self, num_layers, num_experts, num_groups, num_nodes, num_gpus, num_replicas):
        weight = np.random.randint(1, 10, size=(num_layers, num_experts)).astype(np.float32)
        phy2log, log2phy, logcnt = eplb.rebalance_experts(
            weight,
            num_replicas=num_replicas,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
        )

        self.assertEqual(phy2log.shape, (num_layers, num_replicas))
        self.assertEqual(log2phy.shape[0], num_layers)
        self.assertEqual(log2phy.shape[1], num_experts)
        self.assertEqual(logcnt.shape, (num_layers, num_experts))

        self.assertTrue((logcnt >= 1).all())
        self.assertGreaterEqual(phy2log.min(), 0)
        self.assertLess(phy2log.max(), num_experts)

    def test_rebalance_experts_consistency_small(self):
        num_layers = 1
        num_experts = 4
        num_groups = 4
        num_nodes = 2
        num_gpus = 4
        num_replicas = 4

        weight = np.ones((num_layers, num_experts), dtype=np.float32)

        phy2log, log2phy, logcnt = eplb.rebalance_experts(
            weight,
            num_replicas=num_replicas,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
        )

        for layer in range(num_layers):
            for phy in range(num_replicas):
                log_id = phy2log[layer, phy]
                rank = np.where(log2phy[layer, log_id] == phy)[0]
                self.assertEqual(len(rank), 1)


class TestReplicateExperts(unittest.TestCase):
    def test_replicate_experts_edge(self):
        weight = np.ones((1, 4), dtype=np.float32)
        phy2log, rank, logcnt = eplb.replicate_experts(weight, num_phy=4)

        self.assertTrue((logcnt == 1).all())
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(rank.shape, (1, 4))


class TestBalancedPacking(unittest.TestCase):
    def test_balanced_packing_edge(self):
        weight = np.array([[5, 3, 2, 1]], dtype=np.float32)
        pack_index, rank_in_pack = eplb.balanced_packing(weight, num_packs=4)

        self.assertTrue((rank_in_pack == 0).all())
        self.assertTrue((pack_index == np.arange(4)).all())


if __name__ == "__main__":
    unittest.main()
