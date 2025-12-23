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

from fastdeploy.eplb.eplb import (
    balanced_packing,
    rebalance_experts,
    rebalance_experts_hierarchical,
    rebalance_experts_intra_node,
    replicate_experts,
)


class TestEplb(unittest.TestCase):
    """Test cases for eplb.py"""

    def test_balanced_packing_simple(self):
        """Test balanced_packing with simple case"""
        # Test case with 4 items and 2 packs
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_packs = 2

        pack_index, rank_in_pack = balanced_packing(weight, num_packs)

        expected_pack_index = np.array([[0, 1, 1, 0]], dtype=np.int32)
        expected_rank_in_pack = np.array([[1, 1, 0, 0]], dtype=np.int32)

        np.testing.assert_array_equal(pack_index, expected_pack_index)
        np.testing.assert_array_equal(rank_in_pack, expected_rank_in_pack)

    def test_balanced_packing_single_pack(self):
        """Test balanced_packing with single pack"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_packs = 4  # Each pack gets exactly one item

        pack_index, rank_in_pack = balanced_packing(weight, num_packs)

        expected_pack_index = np.array([[0, 1, 2, 3]], dtype=np.int32)
        expected_rank_in_pack = np.array([[0, 0, 0, 0]], dtype=np.int32)

        np.testing.assert_array_equal(pack_index, expected_pack_index)
        np.testing.assert_array_equal(rank_in_pack, expected_rank_in_pack)

    def test_balanced_packing_multiple_layers(self):
        """Test balanced_packing with multiple layers"""
        weight = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32)
        num_packs = 2

        pack_index, rank_in_pack = balanced_packing(weight, num_packs)

        # Verify shape
        self.assertEqual(pack_index.shape, (2, 4))
        self.assertEqual(rank_in_pack.shape, (2, 4))

        # Verify that each pack gets exactly 2 items per layer
        for layer_idx in range(2):
            unique_packs, counts = np.unique(pack_index[layer_idx], return_counts=True)
            np.testing.assert_array_equal(counts, [2, 2])

    def test_replicate_experts_no_redundancy(self):
        """Test replicate_experts with no redundant experts"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_phy = 4  # Same as number of logical experts

        phy2log, rank, logcnt = replicate_experts(weight, num_phy)

        expected_phy2log = np.array([[0, 1, 2, 3]], dtype=np.int32)
        expected_rank = np.array([[0, 0, 0, 0]], dtype=np.int32)
        expected_logcnt = np.array([[1, 1, 1, 1]], dtype=np.int32)

        np.testing.assert_array_equal(phy2log, expected_phy2log)
        np.testing.assert_array_equal(rank, expected_rank)
        np.testing.assert_array_equal(logcnt, expected_logcnt)

    def test_replicate_experts_with_redundancy(self):
        """Test replicate_experts with redundant experts"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_phy = 6  # 2 redundant experts

        phy2log, rank, logcnt = replicate_experts(weight, num_phy)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 6))
        self.assertEqual(rank.shape, (1, 6))
        self.assertEqual(logcnt.shape, (1, 4))

        # Verify that each logical expert has correct count
        expected_logcnt = np.array([[1, 1, 2, 2]], dtype=np.int32)  # Heaviest and lightest get replicated
        np.testing.assert_array_equal(logcnt, expected_logcnt)

    def test_rebalance_experts_intra_node(self):
        """Test rebalance_experts_intra_node function"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_physical_experts = 4
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, phyrank, logcnt = rebalance_experts_intra_node(
            weight, num_physical_experts, num_groups, num_nodes, num_gpus
        )

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(phyrank.shape, (1, 4))
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_hierarchical(self):
        """Test rebalance_experts_hierarchical function"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_physical_experts = 4
        num_groups = 2
        num_nodes = 1
        num_gpus = 1

        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_physical_experts, num_groups, num_nodes, num_gpus
        )

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(phyrank.shape, (1, 4))
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_balance_intra_node(self):
        """Test rebalance_experts with balance_intra_node strategy"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_replicas = 4
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, log2phy, logcnt = rebalance_experts(
            weight, num_replicas, num_groups, num_nodes, num_gpus, "balance_intra_node"
        )

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(log2phy.shape, (1, 4, 1))  # maxlogcnt = 1 when no redundancy
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_hierarchical_strategy(self):
        """Test rebalance_experts with hierarchical strategy"""
        weight = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
        num_replicas = 8
        num_groups = 4  # Divisible by num_nodes
        num_nodes = 2
        num_gpus = 4

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 8))
        self.assertEqual(log2phy.shape, (1, 8, 1))  # maxlogcnt = 1 when no redundancy
        self.assertEqual(logcnt.shape, (1, 8))

    def test_rebalance_experts_global_strategy(self):
        """Test rebalance_experts with global strategy (groups not divisible by nodes)"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_replicas = 4
        num_groups = 3  # Not divisible by num_nodes
        num_nodes = 2
        num_gpus = 2

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(log2phy.shape, (1, 4, 1))
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_with_redundancy(self):
        """Test rebalance_experts with redundant experts"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_replicas = 6  # 2 redundant experts
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 6))
        self.assertEqual(log2phy.shape, (1, 4, 2))  # maxlogcnt = 2 with redundancy
        self.assertEqual(logcnt.shape, (1, 4))

        # Verify that logical expert counts sum to num_replicas
        self.assertEqual(logcnt.sum(), num_replicas)

    def test_edge_cases(self):
        """Test edge cases for rebalance_experts"""
        # Test with all zero weights
        weight = np.zeros((2, 4), dtype=np.float32)
        num_replicas = 4
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Should still produce valid results
        self.assertEqual(phy2log.shape, (2, 4))
        self.assertEqual(log2phy.shape, (2, 4, 1))
        self.assertEqual(logcnt.shape, (2, 4))

    def test_large_scale(self):
        """Test with larger scale parameters"""
        num_layers = 10
        num_experts = 64
        weight = np.random.randint(1, 100, size=(num_layers, num_experts)).astype(np.float32)
        num_replicas = 64
        num_groups = 8
        num_nodes = 4
        num_gpus = 32

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (num_layers, num_replicas))
        self.assertEqual(log2phy.shape[0], num_layers)
        self.assertEqual(log2phy.shape[1], num_experts)
        self.assertEqual(logcnt.shape, (num_layers, num_experts))

        # Verify that logical expert counts sum to num_replicas for each layer
        for layer_idx in range(num_layers):
            self.assertEqual(logcnt[layer_idx].sum(), num_replicas)


if __name__ == "__main__":
    unittest.main()
