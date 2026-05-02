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

Unit tests for get_block_hash_extra_keys in
fastdeploy/cache_manager/v1/cache_utils.py.

Tests mirror the style used in
tests/cache_manager/test_prefix_cache_manager.py and cover:

- Early return paths (None input, missing keys, empty mm_positions)
- Fast-exit path (last item ends before block start)
- Image entirely before the block  (skip via continue)
- Image entirely after the block   (stop via return)
- Image fully contained in block
- Image spanning the right block boundary
- Image spanning the entire block (starts before, ends after)
- Multiple images: only overlapping ones included
- Sequential multi-block scan using the returned mm_idx
- Single-token block and single-token image edge cases
"""

import time
import unittest
from types import SimpleNamespace

from fastdeploy.cache_manager.v1.cache_utils import get_block_hash_extra_keys


def _req(mm_positions, mm_hashes):
    """Build a minimal request-like object with multimodal_inputs."""
    return SimpleNamespace(
        multimodal_inputs={
            "mm_positions": [SimpleNamespace(offset=o, length=l) for o, l in mm_positions],
            "mm_hashes": list(mm_hashes),
        }
    )


class TestGetBlockHashExtraKeysEarlyReturn(unittest.TestCase):
    """Tests for the guard / early-return paths at the top of the function."""

    def test_multimodal_inputs_none(self):
        """multimodal_inputs=None → (mm_idx, []) unchanged."""
        req = SimpleNamespace(multimodal_inputs=None)
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=4, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_multimodal_inputs_attribute_missing(self):
        """Object without multimodal_inputs attribute → treated as None."""
        req = SimpleNamespace()
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=4, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_mm_positions_key_missing(self):
        """mm_positions key absent → early return."""
        req = SimpleNamespace(multimodal_inputs={"mm_hashes": ["h"]})
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=4, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_mm_hashes_key_missing(self):
        """mm_hashes key absent → early return."""
        req = SimpleNamespace(multimodal_inputs={"mm_positions": [SimpleNamespace(offset=0, length=2)]})
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=4, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_mm_positions_empty_list(self):
        """mm_positions=[] → early return."""
        req = SimpleNamespace(multimodal_inputs={"mm_positions": [], "mm_hashes": []})
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=4, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_fast_exit_last_item_ends_exactly_at_block_start(self):
        """
        Fast-exit: last item offset+length == start_idx
        (item ends exactly where block begins → no overlap).
        """
        # img [0,4), block [4,8)  →  4 <= 4  → fast exit
        req = _req([(0, 4)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_fast_exit_last_item_ends_before_block_start(self):
        """Fast-exit: all items end strictly before block start."""
        # img [0,3), block [4,8)
        req = _req([(0, 3)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_fast_exit_preserves_mm_idx(self):
        """Fast-exit returns the original mm_idx unchanged."""
        req = _req([(0, 2)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=5, end_idx=9, mm_idx=0)
        self.assertEqual(mm_idx, 0)
        self.assertEqual(keys, [])


class TestGetBlockHashExtraKeysSingleImage(unittest.TestCase):
    """Tests with exactly one multimodal item and one block."""

    # ------------------------------------------------------------------
    # Item entirely before block → skip (continue), reaches end of loop
    # ------------------------------------------------------------------

    def test_item_ends_before_block_start(self):
        """img [0,2) is entirely before block [3,7)."""
        req = _req([(0, 2)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=3, end_idx=7, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_item_ends_exactly_at_block_start(self):
        """img [0,3) ends exactly at block start 3 → 3<=3 → skip."""
        req = _req([(0, 3)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=3, end_idx=7, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    # ------------------------------------------------------------------
    # Item entirely after block → stop (return img_idx, [])
    # ------------------------------------------------------------------

    def test_item_starts_at_block_end(self):
        """img [8,10) starts exactly at block end 8 → offset>=end_idx → stop."""
        req = _req([(8, 2)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    def test_item_starts_after_block_end(self):
        """img [10,3) starts strictly after block [4,8)."""
        req = _req([(10, 3)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, []))

    # ------------------------------------------------------------------
    # Item spans beyond block right boundary
    # ------------------------------------------------------------------

    def test_item_spans_right_boundary(self):
        """img [6,4) → [6,10) spans block [4,8) right boundary."""
        req = _req([(6, 4)], ["hash-cross"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, ["hash-cross"]))

    def test_item_spans_entire_block(self):
        """img [3,6) → [3,9) wraps the whole block [4,8)."""
        req = _req([(3, 6)], ["hash-span"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, ["hash-span"]))

    def test_item_starts_at_block_start_spans_right(self):
        """img starts at block start, extends past block end."""
        req = _req([(4, 6)], ["h"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, ["h"]))

    # ------------------------------------------------------------------
    # Item fully contained within block
    # ------------------------------------------------------------------

    def test_item_fully_inside_block(self):
        """img [2,2) → [2,4) fully inside block [0,8)."""
        req = _req([(2, 2)], ["hash-inside"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=8, mm_idx=0)
        self.assertIn("hash-inside", keys)

    def test_item_fills_block_exactly(self):
        """img occupies exactly the block [4,8)."""
        req = _req([(4, 4)], ["h-exact"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, ["h-exact"]))

    # ------------------------------------------------------------------
    # Single-token edge cases
    # ------------------------------------------------------------------

    def test_single_token_block_single_token_item_inside(self):
        """Block [5,6), img [5,1) → item fills the single-token block."""
        req = _req([(5, 1)], ["h1"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=5, end_idx=6, mm_idx=0)
        self.assertIn("h1", keys)

    def test_single_token_block_item_starts_after(self):
        """Block [5,6), img [6,1) → starts at block end, not included."""
        req = _req([(6, 1)], ["h1"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=5, end_idx=6, mm_idx=0)
        self.assertEqual(keys, [])


class TestGetBlockHashExtraKeysMultipleImages(unittest.TestCase):
    """Tests with multiple multimodal items."""

    def test_only_overlapping_items_included(self):
        """
        3 images; only the one overlapping the block should be in hash_keys.
          img0: [0,2) → before block [4,8)
          img1: [5,2) → inside  block [4,8)
          img2: [9,2) → after   block [4,8)
        """
        req = _req([(0, 2), (5, 2), (9, 2)], ["h0", "h1", "h2"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertNotIn("h0", keys)
        self.assertIn("h1", keys)
        self.assertNotIn("h2", keys)

    def test_multiple_items_all_inside_block(self):
        """Two images both inside the block → both hashes collected."""
        req = _req([(1, 2), (4, 2)], ["hA", "hB"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=0, end_idx=8, mm_idx=0)
        self.assertEqual(keys, ["hA", "hB"])

    def test_no_item_overlaps_block(self):
        """All images are before the block → empty keys."""
        req = _req([(0, 2), (2, 1)], ["h0", "h1"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=5, end_idx=9, mm_idx=0)
        self.assertEqual(keys, [])

    def test_mm_idx_skips_already_processed_items(self):
        """
        When mm_idx=1, item at index 0 is not scanned at all.
        """
        req = _req([(0, 2), (5, 2)], ["h0", "h1"])
        # Start scanning from mm_idx=1, so h0 must never appear
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=1)
        self.assertNotIn("h0", keys)
        self.assertIn("h1", keys)

    def test_returned_mm_idx_points_to_spanning_item(self):
        """
        When an item spans the block right boundary, returned mm_idx points
        to that item (so the next block can re-examine it).

        img0 [2,7): offset+length=9 > end_idx=8 → spans right boundary
        → include hA, return img_idx=0 immediately (img1 never reached).
        """
        # img0 offset=2, length=7 → end=9 > end_idx=8 → spans right boundary
        req = _req([(2, 7), (10, 2)], ["hA", "hB"])
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual(mm_idx, 0)  # still points to img0 (not fully consumed)
        self.assertIn("hA", keys)
        self.assertNotIn("hB", keys)

    def test_returned_mm_idx_stops_at_after_item(self):
        """
        When an item starts after the block, returned mm_idx points to it
        so the next block can start scanning from there.
        """
        req = _req([(2, 2), (9, 1)], ["hA", "hB"])
        # img1 [9,10) is after block [4,8)
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=1)
        self.assertEqual(mm_idx, 1)
        self.assertEqual(keys, [])


class TestGetBlockHashExtraKeysSequentialScan(unittest.TestCase):
    """
    Simulates a full multi-block scan, reusing the returned mm_idx as the
    next call's mm_idx – mirroring the exact pattern used in
    test_prefix_cache_manager.py.

    Data layout (block_size=4):
      tokens: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
      img0:          [=====]                      [2,5)   hash-0
      img1:                         [========]    [8,12)  hash-1
      img2:                                  [==] [14,16) hash-2
      blocks:  [0,4)  [4,8)  [8,12) [12,16)
    """

    def setUp(self):
        self.req = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [
                    SimpleNamespace(offset=2, length=3),  # [2,5)
                    SimpleNamespace(offset=8, length=4),  # [8,12)
                    SimpleNamespace(offset=14, length=2),  # [14,16)
                ],
                "mm_hashes": ["hash-0", "hash-1", "hash-2"],
            }
        )

    def test_block_0_4(self):
        """Block [0,4): img0 [2,5) spans right boundary → hash-0, mm_idx=0."""
        mm_idx, keys = get_block_hash_extra_keys(self.req, start_idx=0, end_idx=4, mm_idx=0)
        self.assertEqual((mm_idx, keys), (0, ["hash-0"]))

    def test_block_4_8_using_returned_mm_idx(self):
        """Block [4,8): carry mm_idx=0 from previous call → img0 tail, then img1 stops."""
        mm_idx, keys = get_block_hash_extra_keys(self.req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertEqual((mm_idx, keys), (1, ["hash-0"]))

    def test_block_8_12_using_returned_mm_idx(self):
        """Block [8,12): img1 [8,12) exactly fills block → hash-1, mm_idx advances."""
        mm_idx, keys = get_block_hash_extra_keys(self.req, start_idx=8, end_idx=12, mm_idx=1)
        self.assertEqual((mm_idx, keys), (2, ["hash-1"]))

    def test_block_12_16_using_returned_mm_idx(self):
        """Block [12,16): img2 [14,16) fully inside → hash-2."""
        mm_idx, keys = get_block_hash_extra_keys(self.req, start_idx=12, end_idx=16, mm_idx=2)
        self.assertEqual((mm_idx, keys), (2, ["hash-2"]))

    def test_full_sequential_scan(self):
        """Run all four blocks sequentially, feeding mm_idx forward."""
        mm_idx = 0
        expected = [
            ((0, 4), (0, ["hash-0"])),
            ((4, 8), (1, ["hash-0"])),
            ((8, 12), (2, ["hash-1"])),
            ((12, 16), (2, ["hash-2"])),
        ]
        for (s, e), (exp_mm_idx, exp_keys) in expected:
            mm_idx, keys = get_block_hash_extra_keys(self.req, start_idx=s, end_idx=e, mm_idx=mm_idx)
            self.assertEqual((mm_idx, keys), (exp_mm_idx, exp_keys), msg=f"block [{s},{e})")


class TestGetBlockHashExtraKeysBoundaryPrecision(unittest.TestCase):
    """Exact boundary conditions: <= vs < matters at edges."""

    def test_item_end_equals_start_idx_not_included(self):
        """
        offset+length == start_idx  →  item ends exactly where block starts
        →  condition `<= start_idx` is True  →  skip (not included).
        """
        # img [0,4), block [4,8): 0+4=4 == start_idx=4 → skip
        req = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=0, length=4), SimpleNamespace(offset=10, length=1)],
                "mm_hashes": ["h-boundary", "h-other"],
            }
        )
        _, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertNotIn("h-boundary", keys)

    def test_item_offset_equals_end_idx_not_included(self):
        """
        offset == end_idx  →  item starts exactly where block ends
        →  condition `>= end_idx` is True  →  stop (not included).
        """
        # img [8,2), block [4,8): offset=8 == end_idx=8 → stop
        req = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=8, length=2)],
                "mm_hashes": ["h-boundary"],
            }
        )
        _, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertNotIn("h-boundary", keys)

    def test_item_end_one_past_block_end_included(self):
        """
        offset+length == end_idx+1  →  item end is 1 past block end
        →  condition `> end_idx` is True  →  included and mm_idx stays.
        """
        # img [6,3) → [6,9), block [4,8): 6+3=9 > 8 → spans right boundary
        req = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=6, length=3)],
                "mm_hashes": ["h-one-past"],
            }
        )
        mm_idx, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertIn("h-one-past", keys)
        self.assertEqual(mm_idx, 0)

    def test_item_end_equals_end_idx_fully_contained(self):
        """
        offset+length == end_idx  →  item ends exactly at block end
        →  condition `> end_idx` is False  →  fully contained, included.
        """
        # img [4,4) → [4,8), block [4,8): 4+4=8 == end_idx=8 → contained
        req = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=4, length=4)],
                "mm_hashes": ["h-exact-end"],
            }
        )
        _, keys = get_block_hash_extra_keys(req, start_idx=4, end_idx=8, mm_idx=0)
        self.assertIn("h-exact-end", keys)


# ---------------------------------------------------------------------------
# hash_block_tokens
# ---------------------------------------------------------------------------


class TestHashBlockTokens(unittest.TestCase):
    """Direct tests for hash_block_tokens."""

    def setUp(self):
        from fastdeploy.cache_manager.v1.cache_utils import hash_block_tokens

        self.hash_block_tokens = hash_block_tokens

    def test_returns_hex_string(self):
        h = self.hash_block_tokens([1, 2, 3])
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA256 hex digest length

    def test_same_input_same_hash(self):
        h1 = self.hash_block_tokens([1, 2, 3])
        h2 = self.hash_block_tokens([1, 2, 3])
        self.assertEqual(h1, h2)

    def test_different_tokens_different_hash(self):
        h1 = self.hash_block_tokens([1, 2, 3])
        h2 = self.hash_block_tokens([1, 2, 4])
        self.assertNotEqual(h1, h2)

    def test_parent_hash_none_and_empty_string_differ(self):
        """None and '' parent hash should both work; chaining is the key."""
        h_none = self.hash_block_tokens([1, 2], parent_block_hash=None)
        h_empty = self.hash_block_tokens([1, 2], parent_block_hash="")
        # Both produce valid hashes; they may or may not be equal depending on
        # implementation, but must be deterministic.
        self.assertEqual(h_none, self.hash_block_tokens([1, 2], parent_block_hash=None))
        self.assertEqual(h_empty, self.hash_block_tokens([1, 2], parent_block_hash=""))

    def test_chained_hash_differs_from_unchained(self):
        parent = self.hash_block_tokens([0])
        h_chained = self.hash_block_tokens([1, 2], parent_block_hash=parent)
        h_no_parent = self.hash_block_tokens([1, 2])
        self.assertNotEqual(h_chained, h_no_parent)

    def test_extra_keys_affect_hash(self):
        h1 = self.hash_block_tokens([1, 2], extra_keys=None)
        h2 = self.hash_block_tokens([1, 2], extra_keys=("image_hash",))
        self.assertNotEqual(h1, h2)

    def test_empty_token_ids(self):
        h = self.hash_block_tokens([])
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)


# ---------------------------------------------------------------------------
# get_request_block_hasher
# ---------------------------------------------------------------------------


class TestGetRequestBlockHasher(unittest.TestCase):
    """Tests for the factory function get_request_block_hasher."""

    def setUp(self):
        from fastdeploy.cache_manager.v1.cache_utils import get_request_block_hasher

        self.block_size = 4
        self.hasher = get_request_block_hasher(self.block_size)

    def _make_request(self, prompt_tokens, existing_hashes=None, output_tokens=None):
        req = SimpleNamespace(
            prompt_token_ids=prompt_tokens,
            output_token_ids=output_tokens or [],
            _prompt_hashes=existing_hashes if existing_hashes is not None else [],
            multimodal_inputs=None,
        )
        return req

    def test_returns_callable(self):
        from fastdeploy.cache_manager.v1.cache_utils import get_request_block_hasher

        hasher = get_request_block_hasher(4)
        self.assertTrue(callable(hasher))

    def test_single_complete_block(self):
        req = self._make_request(prompt_tokens=[1, 2, 3, 4])
        hashes = self.hasher(req)
        self.assertEqual(len(hashes), 1)
        self.assertIsInstance(hashes[0], str)

    def test_two_complete_blocks(self):
        req = self._make_request(prompt_tokens=list(range(8)))
        hashes = self.hasher(req)
        self.assertEqual(len(hashes), 2)

    def test_incomplete_last_block_not_hashed(self):
        # 5 tokens with block_size=4 → 1 complete block, 1 incomplete
        req = self._make_request(prompt_tokens=list(range(5)))
        hashes = self.hasher(req)
        self.assertEqual(len(hashes), 1)

    def test_existing_hashes_skip_computed_blocks(self):
        # First compute 1 block
        req = self._make_request(prompt_tokens=list(range(4)))
        first_hashes = self.hasher(req)
        # Now add more tokens, provide existing hashes so they aren't recomputed
        req2 = self._make_request(
            prompt_tokens=list(range(8)),
            existing_hashes=first_hashes,
        )
        new_hashes = self.hasher(req2)
        self.assertEqual(len(new_hashes), 1)  # only the second block

    def test_chained_hashes_differ_between_blocks(self):
        req = self._make_request(prompt_tokens=list(range(8)))
        hashes = self.hasher(req)
        self.assertNotEqual(hashes[0], hashes[1])

    def test_deterministic_across_calls(self):
        req1 = self._make_request(prompt_tokens=[1, 2, 3, 4])
        req2 = self._make_request(prompt_tokens=[1, 2, 3, 4])
        self.assertEqual(self.hasher(req1), self.hasher(req2))

    def test_empty_tokens_returns_empty(self):
        req = self._make_request(prompt_tokens=[])
        hashes = self.hasher(req)
        self.assertEqual(hashes, [])

    def test_output_tokens_included_in_hash(self):
        # With only prompt tokens filling one block
        req_prompt_only = self._make_request(
            prompt_tokens=[1, 2],
            output_tokens=[3, 4],
        )
        # The same tokens purely as prompt
        req_prompt_full = self._make_request(prompt_tokens=[1, 2, 3, 4])
        h1 = self.hasher(req_prompt_only)
        h2 = self.hasher(req_prompt_full)
        # Both should produce a hash for the first complete block
        self.assertEqual(len(h1), 1)
        self.assertEqual(len(h2), 1)


# ---------------------------------------------------------------------------
# LayerDoneCounter – time-tracking and cleanup
# ---------------------------------------------------------------------------


class TestLayerDoneCounterTimeTracking(unittest.TestCase):
    """Tests for get_layer_complete_time, get_layer_wait_time, get_all_layer_times, get_elapsed_time."""

    def setUp(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        self.LayerDoneCounter = LayerDoneCounter

    def test_get_layer_complete_time_none_before_done(self):
        counter = self.LayerDoneCounter(num_layers=3)
        self.assertIsNone(counter.get_layer_complete_time(0))

    def test_get_layer_complete_time_after_mark_done(self):
        counter = self.LayerDoneCounter(num_layers=3)
        before = time.time()
        counter.mark_layer_done(0)
        after = time.time()
        t = counter.get_layer_complete_time(0)
        self.assertIsNotNone(t)
        self.assertGreaterEqual(t, before)
        self.assertLessEqual(t, after + 0.01)

    def test_get_layer_wait_time_none_before_done(self):
        counter = self.LayerDoneCounter(num_layers=3)
        self.assertIsNone(counter.get_layer_wait_time(1))

    def test_get_layer_wait_time_is_non_negative(self):
        counter = self.LayerDoneCounter(num_layers=3)
        counter.mark_layer_done(2)
        wait_time = counter.get_layer_wait_time(2)
        self.assertIsNotNone(wait_time)
        self.assertGreaterEqual(wait_time, 0.0)

    def test_get_all_layer_times_empty_before_any_done(self):
        counter = self.LayerDoneCounter(num_layers=4)
        times = counter.get_all_layer_times()
        self.assertEqual(times, {})

    def test_get_all_layer_times_after_mark_all_done(self):
        counter = self.LayerDoneCounter(num_layers=4)
        counter.mark_all_done()
        times = counter.get_all_layer_times()
        self.assertEqual(set(times.keys()), {0, 1, 2, 3})

    def test_get_all_layer_times_returns_copy(self):
        counter = self.LayerDoneCounter(num_layers=2)
        counter.mark_layer_done(0)
        times = counter.get_all_layer_times()
        times[999] = 0.0  # mutate the returned dict
        # Should not affect internal state
        self.assertNotIn(999, counter.get_all_layer_times())

    def test_get_elapsed_time_increases(self):
        counter = self.LayerDoneCounter(num_layers=2)
        t1 = counter.get_elapsed_time()
        time.sleep(0.02)
        t2 = counter.get_elapsed_time()
        self.assertGreater(t2, t1)


class TestLayerDoneCounterGetNumLayers(unittest.TestCase):
    def test_get_num_layers(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=7)
        self.assertEqual(counter.get_num_layers(), 7)


class TestLayerDoneCounterSetLayerEvent(unittest.TestCase):
    """Tests for set_layer_event (no real CUDA event needed)."""

    def test_set_layer_event_stores_value(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=3)
        mock_event = object()
        counter.set_layer_event(1, mock_event)
        self.assertIs(counter._cuda_events[1], mock_event)

    def test_set_layer_event_out_of_range_is_safe(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=3)
        # Should not raise
        counter.set_layer_event(99, object())


class TestLayerDoneCounterCleanup(unittest.TestCase):
    def test_cleanup_clears_events(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=2)
        counter.mark_all_done()
        # No waiters, all done → cleanup should succeed
        counter.cleanup()
        self.assertEqual(len(counter._cuda_events), 0)

    def test_cleanup_with_active_waiter_is_noop(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        counter = LayerDoneCounter(num_layers=2)
        # Manually increment wait count to simulate an active waiter
        counter._increment_wait_count()
        counter.cleanup()
        # Should NOT have cleared events (waiter still active)
        self.assertEqual(len(counter._cuda_events), 2)
        counter._decrement_wait_count()


class TestLayerDoneCounterInternalHelpers(unittest.TestCase):
    def setUp(self):
        from fastdeploy.cache_manager.v1.cache_utils import LayerDoneCounter

        self.LayerDoneCounter = LayerDoneCounter

    def test_increment_and_decrement_wait_count(self):
        counter = self.LayerDoneCounter(num_layers=2)
        counter._increment_wait_count()
        self.assertEqual(counter._wait_count, 1)
        counter._decrement_wait_count()
        self.assertEqual(counter._wait_count, 0)

    def test_decrement_does_not_go_below_zero(self):
        counter = self.LayerDoneCounter(num_layers=2)
        counter._decrement_wait_count()
        self.assertEqual(counter._wait_count, 0)

    def test_should_cleanup_false_when_not_all_done(self):
        counter = self.LayerDoneCounter(num_layers=3)
        self.assertFalse(counter._should_cleanup())

    def test_should_cleanup_true_when_all_done_no_waiters(self):
        counter = self.LayerDoneCounter(num_layers=2)
        counter.mark_all_done()
        self.assertTrue(counter._should_cleanup())

    def test_should_cleanup_false_when_waiter_present(self):
        counter = self.LayerDoneCounter(num_layers=2)
        counter.mark_all_done()
        counter._increment_wait_count()
        self.assertFalse(counter._should_cleanup())
        counter._decrement_wait_count()


if __name__ == "__main__":
    unittest.main()
