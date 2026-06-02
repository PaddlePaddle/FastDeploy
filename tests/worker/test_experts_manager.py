"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.worker.experts_manager import RedundantExpertManger


class TestRedundantExpertMangerInit(unittest.TestCase):
    """Test RedundantExpertManger.__init__."""

    def test_basic_init(self):
        """__init__ sets up all tensors with correct shapes."""
        mgr = RedundantExpertManger(
            n_routed_experts=64,
            num_hidden_layers=2,
            redundant_experts_num=8,
            ep_size=8,
        )
        self.assertEqual(mgr.num_expert, 64)
        self.assertEqual(mgr.redundant_experts_num, 8)
        self.assertEqual(mgr.num_hidden_layers, 2)
        self.assertEqual(mgr.num_replicas, 72)  # 64 + 8
        self.assertEqual(mgr.num_gpus, 8)
        self.assertEqual(mgr.export_per_rank, 9)  # 72 // 8

        # Check tensor shapes
        self.assertEqual(list(mgr.model_ep_rank_to_expert_id_list.shape), [2, 72])
        self.assertEqual(list(mgr.model_expert_id_to_ep_rank_array.shape), [2, 64, 9])
        self.assertEqual(list(mgr.model_expert_in_rank_num_list.shape), [2, 64])
        self.assertEqual(list(mgr.model_tokens_per_expert_stats_list.shape), [2, 64])

    def test_init_with_list_n_routed_experts(self):
        """__init__ handles list input for n_routed_experts (takes first element)."""
        mgr = RedundantExpertManger(
            n_routed_experts=[32, 16],
            num_hidden_layers=1,
            redundant_experts_num=8,
            ep_size=8,
        )
        self.assertEqual(mgr.num_expert, 32)
        self.assertEqual(mgr.num_replicas, 40)  # 32 + 8

    def test_init_assertion_error(self):
        """__init__ raises AssertionError when num_replicas not divisible by ep_size."""
        with self.assertRaises(AssertionError):
            RedundantExpertManger(
                n_routed_experts=10,
                num_hidden_layers=1,
                redundant_experts_num=3,  # 10 + 3 = 13, not divisible by 8
                ep_size=8,
            )


class TestGetEpRankToExpertIdListByLayer(unittest.TestCase):
    """Test RedundantExpertManger.get_ep_rank_to_expert_id_list_by_layer."""

    def test_returns_layer_tensors(self):
        """get_ep_rank_to_expert_id_list_by_layer returns tensors for given layer."""
        mgr = RedundantExpertManger(
            n_routed_experts=16,
            num_hidden_layers=3,
            redundant_experts_num=8,
            ep_size=8,
        )
        result = mgr.get_ep_rank_to_expert_id_list_by_layer(1)

        self.assertEqual(len(result), 4)
        # First tensor: ep_rank_to_expert_id for layer 1
        self.assertEqual(list(result[0].shape), [24])  # 16 + 8
        # Second tensor: expert_id_to_ep_rank for layer 1
        self.assertEqual(list(result[1].shape), [16, 9])  # num_expert, redundant+1
        # Third tensor: expert_in_rank_num for layer 1
        self.assertEqual(list(result[2].shape), [16])
        # Fourth tensor: tokens_per_expert stats for layer 1
        self.assertEqual(list(result[3].shape), [16])


class TestGetEpRankToExpertIdList(unittest.TestCase):
    """Test RedundantExpertManger.get_ep_rank_to_expert_id_list."""

    def test_returns_layer_tensors(self):
        """get_ep_rank_to_expert_id_list returns tensors for given layer."""
        mgr = RedundantExpertManger(
            n_routed_experts=16,
            num_hidden_layers=2,
            redundant_experts_num=8,
            ep_size=8,
        )
        result = mgr.get_ep_rank_to_expert_id_list(0)

        self.assertEqual(len(result), 4)
        self.assertEqual(list(result[0].shape), [24])
        self.assertEqual(list(result[1].shape), [16, 9])
        self.assertEqual(list(result[2].shape), [16])
        self.assertEqual(list(result[3].shape), [16])


class TestGetExpertTokensStats(unittest.TestCase):
    """Test RedundantExpertManger.get_expert_tokens_stats."""

    def _make_mgr(self):
        return RedundantExpertManger(
            n_routed_experts=16,
            num_hidden_layers=2,
            redundant_experts_num=8,
            ep_size=8,
        )

    def test_verbose_false(self):
        """get_expert_tokens_stats with verbose=False returns stats and Nones."""
        mgr = self._make_mgr()
        result = mgr.get_expert_tokens_stats(verbose=False)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])

    def test_verbose_true(self):
        """get_expert_tokens_stats with verbose=True returns all arrays."""
        mgr = self._make_mgr()
        result = mgr.get_expert_tokens_stats(verbose=True)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertIsInstance(result[3], np.ndarray)

    def test_clear_stat(self):
        """get_expert_tokens_stats with clear_stat=True zeros the stats tensor."""
        mgr = self._make_mgr()

        # Stats should be ones initially
        self.assertTrue((mgr.model_tokens_per_expert_stats_list.numpy() == 1).all())

        mgr.get_expert_tokens_stats(clear_stat=True)

        # After clear, should be zeros
        self.assertTrue((mgr.model_tokens_per_expert_stats_list.numpy() == 0).all())

    def test_no_clear_stat(self):
        """get_expert_tokens_stats with clear_stat=False preserves stats."""
        mgr = self._make_mgr()

        mgr.get_expert_tokens_stats(clear_stat=False)

        # Should remain ones
        self.assertTrue((mgr.model_tokens_per_expert_stats_list.numpy() == 1).all())


class TestGetExpertIdToEpRankArray(unittest.TestCase):
    """Test RedundantExpertManger.get_expert_id_to_ep_rank_array."""

    def test_returns_numpy_array(self):
        """get_expert_id_to_ep_rank_array returns numpy array with correct shape."""
        mgr = RedundantExpertManger(
            n_routed_experts=16,
            num_hidden_layers=2,
            redundant_experts_num=8,
            ep_size=8,
        )
        result = mgr.get_expert_id_to_ep_rank_array()

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 16, 9))


class TestUpdateExpertRankTable(unittest.TestCase):
    """Test RedundantExpertManger.update_expert_rank_table."""

    def _make_mgr(self):
        return RedundantExpertManger(
            n_routed_experts=16,
            num_hidden_layers=2,
            redundant_experts_num=8,
            ep_size=8,
        )

    def test_update_with_clear_stat(self):
        """update_expert_rank_table updates tensors and clears stats."""
        mgr = self._make_mgr()

        # Create new mapping data
        num_layers = 2
        num_replicas = 24  # 16 + 8
        rank_expert_list = np.arange(num_replicas, dtype=np.int32).reshape(1, -1).repeat(num_layers, axis=0)
        logical_to_physical_map = np.zeros((num_layers, 16, 2), dtype=np.int32)
        expert_count = np.ones((num_layers, 16), dtype=np.int32)

        mgr.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count, clear_stat=True)

        # Verify stats were cleared
        self.assertTrue((mgr.model_tokens_per_expert_stats_list.numpy() == 0).all())
        # Verify expert_in_rank_num was updated
        self.assertTrue((mgr.model_expert_in_rank_num_list.numpy() == 1).all())

    def test_update_without_clear_stat(self):
        """update_expert_rank_table updates tensors without clearing stats."""
        mgr = self._make_mgr()

        num_layers = 2
        num_replicas = 24
        rank_expert_list = np.arange(num_replicas, dtype=np.int32).reshape(1, -1).repeat(num_layers, axis=0)
        logical_to_physical_map = np.zeros((num_layers, 16, 1), dtype=np.int32)
        expert_count = np.ones((num_layers, 16), dtype=np.int32) * 2

        mgr.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count, clear_stat=False)

        # Stats should remain unchanged (ones from init)
        self.assertTrue((mgr.model_tokens_per_expert_stats_list.numpy() == 1).all())
        # expert_in_rank_num should reflect new count
        self.assertTrue((mgr.model_expert_in_rank_num_list.numpy() == 2).all())


if __name__ == "__main__":
    unittest.main()
