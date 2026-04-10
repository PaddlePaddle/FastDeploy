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
Unit tests for GDNStatePool.

Tests cover:
  1. Pool allocation shapes and dtypes
  2. Slot 0 padding sentinel invariant
  3. Per-layer pool view indexing
  4. reset_slots zeroing across all layers
  5. offset_slot_ids PAD_SLOT_ID mapping
  6. Read/write round-trip per slot
  7. Multi-layer independence
"""

import unittest

import paddle

from fastdeploy.cache_manager.gdn_state_pool import (
    PAD_SLOT_ID,
    GDNSlotAllocator,
    GDNStatePool,
)


class TestGDNStatePool(unittest.TestCase):
    """Tests for GDNStatePool construction and basic operations."""

    # Typical Qwen3.5 config (small scale for testing)
    MAX_NUM_SEQS = 8
    NUM_GDN_LAYERS = 4
    CONV_DIM = 64  # e.g. (key_dim*2 + value_dim) // tp_size
    CONV_KERNEL_SIZE = 4
    NUM_V_HEADS = 4  # TP-local
    HEAD_K_DIM = 16
    HEAD_V_DIM = 16

    def setUp(self):
        self.pool = GDNStatePool(
            max_num_seqs=self.MAX_NUM_SEQS,
            num_gdn_layers=self.NUM_GDN_LAYERS,
            conv_dim=self.CONV_DIM,
            conv_kernel_size=self.CONV_KERNEL_SIZE,
            num_v_heads=self.NUM_V_HEADS,
            head_k_dim=self.HEAD_K_DIM,
            head_v_dim=self.HEAD_V_DIM,
        )

    # ----------------------------------------------------------------
    # 1. Shape and dtype checks
    # ----------------------------------------------------------------
    def test_conv_pool_shape(self):
        """Conv pool shape: [num_gdn_layers, pool_size, conv_dim, conv_kernel_size-1]"""
        pool_size = self.MAX_NUM_SEQS + 1
        expected = [self.NUM_GDN_LAYERS, pool_size, self.CONV_DIM, self.CONV_KERNEL_SIZE - 1]
        self.assertEqual(list(self.pool.conv_pool.shape), expected)

    def test_ssm_pool_shape(self):
        """SSM pool shape: [num_gdn_layers, pool_size, num_v_heads, head_k_dim, head_v_dim]"""
        pool_size = self.MAX_NUM_SEQS + 1
        expected = [self.NUM_GDN_LAYERS, pool_size, self.NUM_V_HEADS, self.HEAD_K_DIM, self.HEAD_V_DIM]
        self.assertEqual(list(self.pool.ssm_pool.shape), expected)

    def test_conv_pool_dtype(self):
        """Conv pool should be bfloat16 by default."""
        self.assertEqual(self.pool.conv_pool.dtype, paddle.bfloat16)

    def test_ssm_pool_dtype(self):
        """SSM pool should be float32 for numerical stability."""
        self.assertEqual(self.pool.ssm_pool.dtype, paddle.float32)

    def test_custom_conv_dtype(self):
        """Conv pool should respect custom dtype."""
        pool = GDNStatePool(
            max_num_seqs=4,
            num_gdn_layers=1,
            conv_dim=16,
            conv_kernel_size=4,
            num_v_heads=2,
            head_k_dim=8,
            head_v_dim=8,
            conv_dtype=paddle.float32,
        )
        self.assertEqual(pool.conv_pool.dtype, paddle.float32)

    # ----------------------------------------------------------------
    # 2. Slot 0 padding sentinel
    # ----------------------------------------------------------------
    def test_slot_zero_is_zero_after_init(self):
        """Slot 0 (padding sentinel) must be all-zeros after construction."""
        for layer_idx in range(self.NUM_GDN_LAYERS):
            conv_slot0 = self.pool.get_layer_conv_pool(layer_idx)[0]
            ssm_slot0 = self.pool.get_layer_ssm_pool(layer_idx)[0]
            self.assertTrue(paddle.all(conv_slot0 == 0).item())
            self.assertTrue(paddle.all(ssm_slot0 == 0).item())

    def test_slot_zero_stays_zero_after_write(self):
        """Writing to slot 0 should work (it's just a safety net for PAD reads)."""
        # Simulate a padded write going to slot 0
        self.pool.conv_pool[0, 0] = 999.0
        # After reset_slots, slot 0 should be zero again
        self.pool.reset_slots([0])
        conv_slot0 = self.pool.get_layer_conv_pool(0)[0]
        self.assertTrue(paddle.all(conv_slot0 == 0).item())

    # ----------------------------------------------------------------
    # 3. Per-layer pool view
    # ----------------------------------------------------------------
    def test_get_layer_conv_pool_shape(self):
        """get_layer_conv_pool returns [pool_size, conv_dim, conv_kernel_size-1]."""
        pool_size = self.MAX_NUM_SEQS + 1
        view = self.pool.get_layer_conv_pool(0)
        expected = [pool_size, self.CONV_DIM, self.CONV_KERNEL_SIZE - 1]
        self.assertEqual(list(view.shape), expected)

    def test_get_layer_ssm_pool_shape(self):
        """get_layer_ssm_pool returns [pool_size, num_v_heads, head_k_dim, head_v_dim]."""
        pool_size = self.MAX_NUM_SEQS + 1
        view = self.pool.get_layer_ssm_pool(0)
        expected = [pool_size, self.NUM_V_HEADS, self.HEAD_K_DIM, self.HEAD_V_DIM]
        self.assertEqual(list(view.shape), expected)

    def test_layer_views_are_independent(self):
        """Writing to layer 0's pool should not affect layer 1's pool."""
        self.pool.get_layer_ssm_pool(0)[1] = 42.0
        ssm_layer1_slot1 = self.pool.get_layer_ssm_pool(1)[1]
        self.assertTrue(paddle.all(ssm_layer1_slot1 == 0).item())

    # ----------------------------------------------------------------
    # 4. reset_slots
    # ----------------------------------------------------------------
    def test_reset_slots_zeros_conv_and_ssm(self):
        """reset_slots should zero out conv and SSM state for given slots across all layers."""
        # Write non-zero data to slots 1, 2, 3
        for layer_idx in range(self.NUM_GDN_LAYERS):
            for slot in [1, 2, 3]:
                self.pool.conv_pool[layer_idx, slot] = float(slot)
                self.pool.ssm_pool[layer_idx, slot] = float(slot)

        # Reset slots 1 and 3
        self.pool.reset_slots([1, 3])

        for layer_idx in range(self.NUM_GDN_LAYERS):
            # Slot 1 and 3 should be zero
            self.assertTrue(paddle.all(self.pool.conv_pool[layer_idx, 1] == 0).item())
            self.assertTrue(paddle.all(self.pool.ssm_pool[layer_idx, 1] == 0).item())
            self.assertTrue(paddle.all(self.pool.conv_pool[layer_idx, 3] == 0).item())
            self.assertTrue(paddle.all(self.pool.ssm_pool[layer_idx, 3] == 0).item())
            # Slot 2 should still have data
            self.assertTrue(paddle.all(self.pool.conv_pool[layer_idx, 2] == 2.0).item())
            self.assertTrue(paddle.all(self.pool.ssm_pool[layer_idx, 2] == 2.0).item())

    def test_reset_slots_empty_list(self):
        """reset_slots with empty list should be a no-op."""
        self.pool.ssm_pool[0, 1] = 99.0
        self.pool.reset_slots([])
        self.assertTrue(paddle.all(self.pool.ssm_pool[0, 1] == 99.0).item())

    # ----------------------------------------------------------------
    # 5. offset_slot_ids
    # ----------------------------------------------------------------
    def test_offset_slot_ids_basic(self):
        """offset_slot_ids: -1->0, 0->1, 1->2, etc."""
        raw = paddle.to_tensor([-1, 0, 1, 5], dtype=paddle.int32)
        offset = GDNStatePool.offset_slot_ids(raw)
        expected = paddle.to_tensor([0, 1, 2, 6], dtype=paddle.int32)
        self.assertTrue(paddle.all(offset == expected).item())

    def test_offset_slot_ids_pad_maps_to_sentinel(self):
        """PAD_SLOT_ID (-1) should map to slot 0 (the zero-filled sentinel)."""
        raw = paddle.to_tensor([PAD_SLOT_ID], dtype=paddle.int32)
        offset = GDNStatePool.offset_slot_ids(raw)
        self.assertEqual(offset[0].item(), 0)

    # ----------------------------------------------------------------
    # 6. Read/write round-trip
    # ----------------------------------------------------------------
    def test_ssm_read_write_roundtrip(self):
        """Write a known state to a slot, read it back, verify equality."""
        layer_idx = 2
        slot_id = 5  # after +1 offset
        state = paddle.randn([self.NUM_V_HEADS, self.HEAD_K_DIM, self.HEAD_V_DIM], dtype=paddle.float32)

        # Write
        self.pool.get_layer_ssm_pool(layer_idx)[slot_id] = state

        # Read back
        read_back = self.pool.get_layer_ssm_pool(layer_idx)[slot_id]
        self.assertTrue(paddle.allclose(read_back, state).item())

    def test_conv_read_write_roundtrip(self):
        """Write a known conv state to a slot, read it back, verify equality."""
        layer_idx = 1
        slot_id = 3
        state = paddle.randn([self.CONV_DIM, self.CONV_KERNEL_SIZE - 1], dtype=paddle.bfloat16)

        self.pool.get_layer_conv_pool(layer_idx)[slot_id] = state
        read_back = self.pool.get_layer_conv_pool(layer_idx)[slot_id]
        self.assertTrue(paddle.allclose(read_back.cast(paddle.float32), state.cast(paddle.float32), atol=1e-2).item())

    # ----------------------------------------------------------------
    # 7. Stored attributes
    # ----------------------------------------------------------------
    def test_stored_attributes(self):
        """Pool should store construction parameters for later inspection."""
        self.assertEqual(self.pool.max_num_seqs, self.MAX_NUM_SEQS)
        self.assertEqual(self.pool.num_gdn_layers, self.NUM_GDN_LAYERS)
        self.assertEqual(self.pool.conv_dim, self.CONV_DIM)
        self.assertEqual(self.pool.conv_kernel_size, self.CONV_KERNEL_SIZE)
        self.assertEqual(self.pool.num_v_heads, self.NUM_V_HEADS)
        self.assertEqual(self.pool.head_k_dim, self.HEAD_K_DIM)
        self.assertEqual(self.pool.head_v_dim, self.HEAD_V_DIM)

    # ----------------------------------------------------------------
    # 8. Pool allocate/free (GPU-side)
    # ----------------------------------------------------------------
    def test_pool_allocate_returns_valid_slots(self):
        """Pool allocate() should return 1-based slot IDs."""
        slots = self.pool.allocate(3)
        self.assertEqual(len(slots), 3)
        for s in slots:
            self.assertGreater(s, 0)
            self.assertLessEqual(s, self.MAX_NUM_SEQS)

    def test_pool_allocate_exhaustion(self):
        """Pool allocate() should raise when exhausted."""
        self.pool.allocate(self.MAX_NUM_SEQS)
        with self.assertRaises(RuntimeError):
            self.pool.allocate(1)

    def test_pool_free_recycles_slots(self):
        """Pool free() should make slots available again and zero state."""
        slots = self.pool.allocate(2)
        # Write data to allocated slots
        for s in slots:
            self.pool.ssm_pool[0, s] = 42.0
        self.pool.free(slots)
        # State should be zeroed
        for s in slots:
            self.assertTrue(paddle.all(self.pool.ssm_pool[0, s] == 0).item())
        # Should be able to allocate again
        new_slots = self.pool.allocate(2)
        self.assertEqual(len(new_slots), 2)

    def test_pool_num_free_slots(self):
        """num_free_slots should track available count."""
        self.assertEqual(self.pool.num_free_slots, self.MAX_NUM_SEQS)
        self.pool.allocate(3)
        self.assertEqual(self.pool.num_free_slots, self.MAX_NUM_SEQS - 3)


class TestGDNSlotAllocator(unittest.TestCase):
    """Tests for the lightweight CPU-only slot allocator."""

    MAX_NUM_SEQS = 8

    def setUp(self):
        self.allocator = GDNSlotAllocator(self.MAX_NUM_SEQS)

    def test_allocate_returns_1_based(self):
        """Allocated slots should be 1-based (slot 0 is sentinel)."""
        slot = self.allocator.allocate()
        self.assertGreater(slot, 0)
        self.assertLessEqual(slot, self.MAX_NUM_SEQS)

    def test_allocate_unique(self):
        """Each allocation should return a unique slot."""
        slots = set()
        for _ in range(self.MAX_NUM_SEQS):
            slots.add(self.allocator.allocate())
        self.assertEqual(len(slots), self.MAX_NUM_SEQS)

    def test_allocate_exhaustion_raises(self):
        """Should raise RuntimeError when no free slots."""
        for _ in range(self.MAX_NUM_SEQS):
            self.allocator.allocate()
        with self.assertRaises(RuntimeError):
            self.allocator.allocate()

    def test_free_makes_slot_available(self):
        """Freed slot should be re-allocatable."""
        slot = self.allocator.allocate()
        self.allocator.free(slot)
        new_slot = self.allocator.allocate()
        self.assertEqual(new_slot, slot)  # LIFO: last freed = next allocated

    def test_free_slot_zero_ignored(self):
        """Freeing slot 0 (sentinel) should be a no-op."""
        initial_free = self.allocator.num_free_slots
        self.allocator.free(0)
        self.assertEqual(self.allocator.num_free_slots, initial_free)

    def test_num_free_slots(self):
        """num_free_slots tracks available count correctly."""
        self.assertEqual(self.allocator.num_free_slots, self.MAX_NUM_SEQS)
        self.allocator.allocate()
        self.assertEqual(self.allocator.num_free_slots, self.MAX_NUM_SEQS - 1)
        self.allocator.allocate()
        self.assertEqual(self.allocator.num_free_slots, self.MAX_NUM_SEQS - 2)

    def test_allocate_free_lifecycle(self):
        """Simulate a full request lifecycle: allocate → use → free → re-allocate."""
        # Allocate all slots
        all_slots = [self.allocator.allocate() for _ in range(self.MAX_NUM_SEQS)]
        self.assertEqual(self.allocator.num_free_slots, 0)

        # Free half
        for s in all_slots[:4]:
            self.allocator.free(s)
        self.assertEqual(self.allocator.num_free_slots, 4)

        # Re-allocate
        new_slots = [self.allocator.allocate() for _ in range(4)]
        self.assertEqual(self.allocator.num_free_slots, 0)
        # All new_slots should be from the freed set
        self.assertEqual(set(new_slots), set(all_slots[:4]))


if __name__ == "__main__":
    unittest.main()
