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

import os
import tempfile
import unittest

import numpy as np

from fastdeploy.eplb.eplb import (
    balanced_packing,
    rebalance_experts,
    rebalance_experts_hierarchical,
    rebalance_experts_intra_node,
    replicate_experts,
)
from fastdeploy.eplb.utils import RearrangeExpertState, RedundantExpertWorkload


class TestBalancedPacking(unittest.TestCase):
    """Test balanced_packing function"""

    def test_groups_per_pack_1(self):
        # Test when each pack has 1 group (identity mapping)
        weight = np.array([[1, 2, 3]], dtype=np.float32)
        pack_idx, rank = balanced_packing(weight, num_packs=3)
        self.assertTrue(np.array_equal(pack_idx, np.array([[0, 1, 2]])))
        self.assertTrue(np.all(rank == 0))

    def test_balanced_packing_basic(self):
        # Test basic balanced packing with 2 packs
        weight = np.array([[5, 4, 3, 2]], dtype=np.float32)
        pack_idx, rank = balanced_packing(weight, num_packs=2)

        # Each pack gets equal number of items
        counts = [np.sum(pack_idx == i) for i in range(2)]
        self.assertEqual(counts, [2, 2])
        # Ranks are within valid range
        self.assertTrue(np.all(rank >= 0))
        self.assertTrue(np.all(rank <= 1))


class TestReplicateExperts(unittest.TestCase):
    """Test replicate_experts function"""

    def test_no_redundant(self):
        # Test no redundancy (num_phy == num_logical experts)
        weight = np.array([[1, 2, 3]], dtype=np.float32)
        phy2log, rank, logcnt = replicate_experts(weight, num_phy=3)
        self.assertTrue(np.array_equal(phy2log, np.array([[0, 1, 2]])))
        self.assertTrue(np.all(rank == 0))
        self.assertTrue(np.array_equal(logcnt, np.array([[1, 1, 1]])))

    def test_with_redundant(self):
        # Test with redundancy (more physical than logical experts)
        weight = np.array([[3, 1]], dtype=np.float32)
        phy2log, rank, logcnt = replicate_experts(weight, num_phy=4)
        # Higher weight expert gets more replicas
        self.assertGreater(logcnt[0, 0], logcnt[0, 1])
        # Replica count matches physical-to-logical mapping
        for e in range(2):
            reps = logcnt[0, e]
            self.assertEqual(np.sum(phy2log[0] == e), reps)


class TestRebalanceIntraNode(unittest.TestCase):
    """Test rebalance_experts_intra_node function"""

    def test_intra_node_basic(self):
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_nodes = 2
        num_gpus = 4
        num_groups = 2
        num_phy = 4

        phy2log, phyrank, logcnt = rebalance_experts_intra_node(
            weight,
            num_physical_experts=num_phy,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
        )

        # Check output shapes
        self.assertEqual(phy2log.shape, (1, num_phy))
        self.assertEqual(phyrank.shape, (1, num_phy))
        self.assertEqual(logcnt.shape, (1, 4))

        # Replica count matches mapping
        for e in range(4):
            self.assertEqual(np.sum(phy2log[0] == e), logcnt[0, e])


class TestRebalanceHierarchical(unittest.TestCase):
    """Test rebalance_experts_hierarchical function"""

    def test_hierarchical_basic(self):
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_groups = 2
        num_nodes = 1
        num_gpus = 2
        num_phy = 4

        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_physical_experts=num_phy,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
        )

        # Check output shapes
        self.assertEqual(phy2log.shape, (1, num_phy))
        self.assertEqual(phyrank.shape, (1, num_phy))
        self.assertEqual(logcnt.shape, (1, 4))

        # Replica count matches mapping
        for e in range(4):
            self.assertEqual(np.sum(phy2log[0] == e), logcnt[0, e])


class TestRebalanceExpertsEntry(unittest.TestCase):
    """Test rebalance_experts entry function"""

    def test_global_policy(self):
        # Test global rebalance policy (when groups not divisible by nodes)
        weight = np.array([[1, 1, 1, 1]], dtype=np.float32)
        num_groups = 3  # Triggers global policy
        phy2log, log2phy, logcnt = rebalance_experts(
            weight,
            num_replicas=4,
            num_groups=num_groups,
            num_nodes=2,
            num_gpus=2,
            eplb_strategy="",
        )

        # Check output shapes
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(logcnt.shape, (1, 4))

        # Replica count consistency
        for e in range(4):
            self.assertEqual(np.sum(phy2log[0] == e), logcnt[0, e])

    def test_hierarchical_policy(self):
        # Test hierarchical rebalance policy
        weight = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
        phy2log, log2phy, logcnt = rebalance_experts(
            weight,
            num_replicas=6,
            num_groups=3,
            num_nodes=1,
            num_gpus=3,
            eplb_strategy="",
        )
        self.assertEqual(phy2log.shape, (1, 6))
        # Replica count consistency
        for e in range(6):
            self.assertEqual(np.sum(phy2log[0] == e), logcnt[0, e])

    def test_intra_node_strategy(self):
        # Test explicit intra-node rebalance strategy
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        phy2log, log2phy, logcnt = rebalance_experts(
            weight,
            num_replicas=4,
            num_groups=2,
            num_nodes=2,
            num_gpus=2,
            eplb_strategy="balance_intra_node",
        )
        self.assertEqual(phy2log.shape, (1, 4))
        # Replica count consistency
        for e in range(4):
            self.assertEqual(np.sum(phy2log[0] == e), logcnt[0, e])


class TestRedundantExpertWorkload(unittest.TestCase):
    """Test suite for RedundantExpertWorkload class"""

    def setUp(self):
        # Create temporary directory to avoid polluting real /tmp
        self.tmpdir = tempfile.TemporaryDirectory()
        self.meta_dir = self.tmpdir.name

    def tearDown(self):
        # Clean up temporary directory after tests
        self.tmpdir.cleanup()

    def test_dump_and_load(self):
        """Test complete dump -> load workflow"""
        w = RedundantExpertWorkload(redundant_expert_meta_dir=self.meta_dir)

        # Set test fields
        w.tokens_per_expert_stats_list = [1, 2, 3]
        w.ep_rank_to_expert_id_list = [10, 20, 30]
        w.expert_id_to_ep_rank_array = [0, 1, 2]
        w.expert_in_rank_num_list = [2, 2, 2]
        w.cost_milliseconds = 123

        # Execute dump
        msg = w.dump()
        self.assertIn("dump expert workload result", msg)

        # Verify meta file exists
        self.assertTrue(os.path.exists(w.meta_file_name))

        # Load to new instance
        w2 = RedundantExpertWorkload(redundant_expert_meta_dir=self.meta_dir)
        meta, status = w2.load()

        # Verify loaded data matches original
        self.assertEqual(status, "ok")
        self.assertEqual(meta["tokens_per_expert_stats_list"], [1, 2, 3])
        self.assertEqual(meta["ep_rank_to_expert_id_list"], [10, 20, 30])
        self.assertEqual(meta["expert_id_to_ep_rank_array"], [0, 1, 2])
        self.assertEqual(meta["expert_in_rank_num_list"], [2, 2, 2])
        self.assertEqual(meta["cost_milliseconds"], 123)

    def test_load_file_not_exists(self):
        """Test load behavior when meta file doesn't exist"""
        w = RedundantExpertWorkload(redundant_expert_meta_dir=self.meta_dir)

        # Ensure file is deleted
        if os.path.exists(w.meta_file_name):
            os.remove(w.meta_file_name)

        meta, status = w.load()
        self.assertEqual(meta, {})
        self.assertIn("is not exists", status)

    def test_dump_file_error(self):
        """Test dump failure (e.g., directory deleted)"""
        w = RedundantExpertWorkload(redundant_expert_meta_dir=self.meta_dir)

        # Delete directory to cause write failure
        os.rmdir(self.meta_dir)

        msg = w.dump()
        self.assertIn("dump expert workload failed", msg)

    def test_rearrange_expert_state_enum(self):
        """Test RearrangeExpertState enum values exist"""
        self.assertEqual(RearrangeExpertState.free.value, 0)
        self.assertEqual(RearrangeExpertState.doing.value, 1)
        self.assertEqual(RearrangeExpertState.load_succ.value, 2)
        self.assertEqual(RearrangeExpertState.done.value, 3)


if __name__ == "__main__":
    unittest.main()
