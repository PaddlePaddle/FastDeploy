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
Unit tests for prefix cache + multimodal image feature slicing fix.

Tests cover:
  1. _calc_image_feature_range() correctness across various boundary cases
  2. Multi-request batch scenario where feature_slice_info could be overwritten

Usage:
  pytest tests/worker/test_prefix_cache_mm_slice.py -v
"""

import unittest
from dataclasses import dataclass


# Replicate ImagePosition locally to avoid heavy imports
@dataclass
class ImagePosition:
    offset: int = 0
    length: int = 0


class CalcImageFeatureRangeHelper:
    """
    Standalone reimplementation of GPUModelRunner._calc_image_feature_range
    so we can test it without instantiating the full model runner.
    """

    @staticmethod
    def calc(mm_inputs: dict, prefill_start: int, prefill_end: int) -> tuple:
        if mm_inputs is None or "mm_positions" not in mm_inputs:
            return (0, 0)

        mm_positions = mm_inputs.get("mm_positions", [])
        if not mm_positions:
            return (0, 0)

        if prefill_start == 0:
            total_image_tokens = sum(pos.length for pos in mm_positions)
            return (0, total_image_tokens)

        feature_start = -1
        feature_end = 0
        cumulative_tokens = 0

        for pos in mm_positions:
            img_token_start = pos.offset
            img_token_end = pos.offset + pos.length

            overlap_start = max(img_token_start, prefill_start)
            overlap_end = min(img_token_end, prefill_end)

            if overlap_start < overlap_end:
                local_start = overlap_start - img_token_start
                local_end = overlap_end - img_token_start

                if feature_start == -1:
                    feature_start = cumulative_tokens + local_start
                feature_end = cumulative_tokens + local_end

            cumulative_tokens += pos.length

        if feature_start == -1:
            feature_start = 0
        return (feature_start, feature_end)


calc = CalcImageFeatureRangeHelper.calc


# ──────────────────────────────────────────────────────────────────────
# Test Suite 1: _calc_image_feature_range correctness
# ──────────────────────────────────────────────────────────────────────


class TestCalcImageFeatureRange(unittest.TestCase):
    """Test _calc_image_feature_range with various boundary conditions."""

    # ── Edge cases: empty / None inputs ────────────────────────────

    def test_none_inputs(self):
        self.assertEqual(calc(None, 0, 100), (0, 0))

    def test_no_mm_positions_key(self):
        self.assertEqual(calc({}, 0, 100), (0, 0))

    def test_empty_mm_positions(self):
        self.assertEqual(calc({"mm_positions": []}, 0, 100), (0, 0))

    # ── Full prefill (prefill_start == 0) ──────────────────────────

    def test_full_prefill_single_image(self):
        """prefill_start=0 should return all image features."""
        mm = {"mm_positions": [ImagePosition(offset=10, length=3577)]}
        self.assertEqual(calc(mm, 0, 4000), (0, 3577))

    def test_full_prefill_multiple_images(self):
        """prefill_start=0, multiple images, return total features."""
        mm = {"mm_positions": [
            ImagePosition(offset=10, length=1000),
            ImagePosition(offset=1200, length=2000),
            ImagePosition(offset=3500, length=500),
        ]}
        self.assertEqual(calc(mm, 0, 5000), (0, 3500))

    # ── Single image, prefix cache hit ─────────────────────────────

    def test_single_image_tail_overlap(self):
        """
        Typical scenario: one image at offset=10, length=3577.
        Prefix cache covers tokens [0, 3563), prefill range is [3563, 3587).
        Image spans [10, 3587). Overlap = [3563, 3587) = 24 tokens.
        Local range = [3553, 3577).
        """
        mm = {"mm_positions": [ImagePosition(offset=10, length=3577)]}
        start, end = calc(mm, 3563, 3587)
        self.assertEqual(end - start, 24)
        self.assertEqual(start, 3553)
        self.assertEqual(end, 3577)

    def test_single_image_fully_cached(self):
        """
        Image entirely in cached prefix → no overlap with prefill window.
        Image at [10, 110), prefix cache covers [0, 200), prefill is [200, 300).
        """
        mm = {"mm_positions": [ImagePosition(offset=10, length=100)]}
        start, end = calc(mm, 200, 300)
        # No overlap → feature_start stays -1, then becomes 0; feature_end stays 0
        self.assertEqual(start, 0)
        self.assertEqual(end, 0)

    def test_single_image_fully_in_prefill(self):
        """
        Image entirely within the prefill window (no prefix cache hit on image).
        """
        mm = {"mm_positions": [ImagePosition(offset=100, length=500)]}
        start, end = calc(mm, 50, 700)
        self.assertEqual(start, 0)
        self.assertEqual(end, 500)

    def test_single_image_head_overlap(self):
        """
        Prefill window covers only the first part of the image.
        Image at [100, 600), prefill is [50, 300).
        Overlap = [100, 300) → 200 tokens, local [0, 200).
        """
        mm = {"mm_positions": [ImagePosition(offset=100, length=500)]}
        start, end = calc(mm, 50, 300)
        self.assertEqual(start, 0)
        self.assertEqual(end, 200)

    def test_single_image_middle_overlap(self):
        """
        Prefill window is strictly inside the image.
        Image at [100, 600), prefill is [200, 400).
        Overlap = [200, 400) → local [100, 300).
        """
        mm = {"mm_positions": [ImagePosition(offset=100, length=500)]}
        start, end = calc(mm, 200, 400)
        self.assertEqual(start, 100)
        self.assertEqual(end, 300)

    # ── Multiple images, prefix cache hit ──────────────────────────

    def test_multi_image_first_fully_cached(self):
        """
        Two images. First is fully cached, second partially overlaps.
        Image1: [10, 110) len=100
        Image2: [200, 700) len=500
        Prefill: [150, 400)
        Image1: no overlap (10..110 vs 150..400 → no)
        Image2: overlap [200, 400) → local [0, 200), cumulative offset=100
        """
        mm = {"mm_positions": [
            ImagePosition(offset=10, length=100),
            ImagePosition(offset=200, length=500),
        ]}
        start, end = calc(mm, 150, 400)
        self.assertEqual(start, 100)  # 100 (cum from img1) + 0 (local_start)
        self.assertEqual(end, 300)    # 100 + 200

    def test_multi_image_both_overlap(self):
        """
        Two images, both partially overlap with prefill.
        Image1: [100, 300) len=200
        Image2: [300, 600) len=300
        Prefill: [250, 450)
        Image1 overlap: [250, 300) → local [150, 200), cum=0
        Image2 overlap: [300, 450) → local [0, 150), cum=200
        feature_start = 0 + 150 = 150
        feature_end   = 200 + 150 = 350
        """
        mm = {"mm_positions": [
            ImagePosition(offset=100, length=200),
            ImagePosition(offset=300, length=300),
        ]}
        start, end = calc(mm, 250, 450)
        self.assertEqual(start, 150)
        self.assertEqual(end, 350)

    def test_multi_image_middle_only(self):
        """
        Three images, only the middle one overlaps.
        Image1: [10, 100)  len=90
        Image2: [200, 500) len=300
        Image3: [600, 900) len=300
        Prefill: [250, 450)
        Image2 overlap: [250, 450) → local [50, 250), cum=90
        """
        mm = {"mm_positions": [
            ImagePosition(offset=10, length=90),
            ImagePosition(offset=200, length=300),
            ImagePosition(offset=600, length=300),
        ]}
        start, end = calc(mm, 250, 450)
        self.assertEqual(start, 90 + 50)   # 140
        self.assertEqual(end, 90 + 250)     # 340

    def test_multi_image_all_cached(self):
        """All images in cached prefix, none overlap."""
        mm = {"mm_positions": [
            ImagePosition(offset=10, length=100),
            ImagePosition(offset=200, length=100),
        ]}
        start, end = calc(mm, 500, 600)
        self.assertEqual(start, 0)
        self.assertEqual(end, 0)

    # ── Consistency: sliced count == image tokens in prefill ───────

    def test_slice_count_matches_image_tokens_in_prefill(self):
        """
        The number of sliced features must equal the number of image tokens
        in the prefill window. This is the invariant that
        get_input_embeddings() relies on.
        """
        mm_positions = [
            ImagePosition(offset=50, length=3577),
        ]
        mm = {"mm_positions": mm_positions}

        prefill_start = 3600
        prefill_end = 3627  # 3627 = 50 + 3577 = end of image

        # Image tokens in prefill: overlap = [3600, 3627) → 27 tokens
        start, end = calc(mm, prefill_start, prefill_end)
        self.assertEqual(end - start, 27)

    def test_slice_count_exact_boundary(self):
        """Prefill starts exactly at image start."""
        mm = {"mm_positions": [ImagePosition(offset=100, length=500)]}
        start, end = calc(mm, 100, 600)
        self.assertEqual(end - start, 500)
        self.assertEqual(start, 0)
        self.assertEqual(end, 500)


# ──────────────────────────────────────────────────────────────────────
# Test Suite 2: Multi-request batch — feature_slice_info overwrite bug
# ──────────────────────────────────────────────────────────────────────


class TestMultiRequestBatchSlicing(unittest.TestCase):
    """
    Tests for per-request feature slicing in multi mm-request batches.

    Verifies:
      - The old (single slice_info) approach was broken for multi-request batches
      - The new (per-request feature_slice_infos) approach is correct
      - Single-request batches still work correctly
    """

    def _simulate_old_batch_logic(self, requests):
        """
        Simulate the OLD (buggy) batch logic:
        - feature_slice_info stores only the LAST request's slice info
        - After extract_vision_features, apply a single global slice
        """
        total_features = 0
        feature_slice_info = None

        for req in requests:
            mm_inputs = req["mm_inputs"]
            prefill_start = req["prefill_start"]
            prefill_end = req["prefill_end"]
            req_feature_count = sum(p.length for p in mm_inputs["mm_positions"])
            total_features += req_feature_count

            if prefill_start > 0:
                s, e = calc(mm_inputs, prefill_start, prefill_end)
                feature_slice_info = (s, e)

        if feature_slice_info is not None:
            sliced_count = feature_slice_info[1] - feature_slice_info[0]
        else:
            sliced_count = total_features

        return total_features, sliced_count, feature_slice_info

    def _simulate_new_batch_logic(self, requests):
        """
        Simulate the NEW (fixed) batch logic matching insert_tasks_v1:
        - Track per-request slice info with global offset (_feature_offset)
        - After extract_vision_features, slice per-request and concat
        """
        total_features = 0
        feature_slice_infos = []
        global_offset = 0

        for req in requests:
            mm_inputs = req["mm_inputs"]
            prefill_start = req["prefill_start"]
            prefill_end = req["prefill_end"]
            req_feature_count = sum(p.length for p in mm_inputs["mm_positions"])

            if prefill_start > 0:
                s, e = calc(mm_inputs, prefill_start, prefill_end)
                feature_slice_infos.append((global_offset + s, global_offset + e))
            else:
                feature_slice_infos.append((global_offset, global_offset + req_feature_count))

            global_offset += req_feature_count
            total_features += req_feature_count

        sliced_count = sum(e - s for s, e in feature_slice_infos)
        return total_features, sliced_count, feature_slice_infos

    def _count_image_tokens_in_prefill(self, mm_inputs, prefill_start, prefill_end):
        """Count how many image tokens fall in [prefill_start, prefill_end)."""
        count = 0
        for pos in mm_inputs["mm_positions"]:
            overlap_start = max(pos.offset, prefill_start)
            overlap_end = min(pos.offset + pos.length, prefill_end)
            if overlap_start < overlap_end:
                count += overlap_end - overlap_start
        return count

    def test_two_requests_both_cache_hit_old_logic_is_wrong(self):
        """
        Two requests in same batch, both with prefix cache hit.
        Old logic kept only the last request's slice_info → wrong.
        New logic uses per-request feature_slice_infos → correct.

        Request A: image at [10, 3587), prefill [3563, 3600)
          → 24 image tokens in prefill
        Request B: image at [10, 2010), prefill [1990, 2100)
          → 20 image tokens in prefill
        Expected total sliced features: 24 + 20 = 44
        """
        req_a = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=3577)]},
            "prefill_start": 3563,
            "prefill_end": 3600,
        }
        req_b = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=2000)]},
            "prefill_start": 1990,
            "prefill_end": 2100,
        }

        expected_a = self._count_image_tokens_in_prefill(
            req_a["mm_inputs"], req_a["prefill_start"], req_a["prefill_end"]
        )
        expected_b = self._count_image_tokens_in_prefill(
            req_b["mm_inputs"], req_b["prefill_start"], req_b["prefill_end"]
        )
        expected_total = expected_a + expected_b
        self.assertEqual(expected_a, 24)
        self.assertEqual(expected_b, 20)
        self.assertEqual(expected_total, 44)

        # Old (buggy) logic
        total_feat, buggy_sliced, _ = self._simulate_old_batch_logic([req_a, req_b])
        self.assertEqual(total_feat, 3577 + 2000)
        self.assertNotEqual(buggy_sliced, expected_total,
                            "Old logic should NOT produce correct result for multi-request batch")

        # New (fixed) logic
        _, correct_sliced, _ = self._simulate_new_batch_logic([req_a, req_b])
        self.assertEqual(correct_sliced, expected_total,
                         "New per-request slicing should produce 44 features")

    def test_two_requests_one_cache_hit_one_cold(self):
        """
        Request A: cold (prefill_start=0), full image features needed
        Request B: cache hit, partial image features needed

        Current logic overwrites with B's slice → discards A's features.
        """
        req_a = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=1000)]},
            "prefill_start": 0,
            "prefill_end": 1100,
        }
        req_b = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=2000)]},
            "prefill_start": 1500,
            "prefill_end": 2100,
        }

        expected_a = 1000  # cold → all features
        expected_b = self._count_image_tokens_in_prefill(
            req_b["mm_inputs"], req_b["prefill_start"], req_b["prefill_end"]
        )
        expected_total = expected_a + expected_b

        # Old logic: req_a has prefill_start=0 so no slice_info set,
        # req_b sets slice_info. The global slice cuts the concatenated
        # tensor with B's local indices → A's 1000 features are lost.
        _, buggy_sliced, buggy_info = self._simulate_old_batch_logic([req_a, req_b])
        self.assertNotEqual(buggy_sliced, expected_total)

        # New logic
        _, correct_sliced, _ = self._simulate_new_batch_logic([req_a, req_b])
        self.assertEqual(correct_sliced, expected_total)

    def test_single_request_cache_hit_is_correct(self):
        """
        When there's only one mm-request in the batch (the common case),
        the current logic is correct.
        Image [10, 3587), prefill [3563, 3587) → overlap 24 tokens.
        """
        req = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=3577)]},
            "prefill_start": 3563,
            "prefill_end": 3587,
        }
        expected = self._count_image_tokens_in_prefill(
            req["mm_inputs"], req["prefill_start"], req["prefill_end"]
        )
        self.assertEqual(expected, 24)

        _, old_sliced, _ = self._simulate_old_batch_logic([req])
        self.assertEqual(old_sliced, expected,
                         "Single request case should be correct even with old logic")

        _, new_sliced, _ = self._simulate_new_batch_logic([req])
        self.assertEqual(new_sliced, expected)

    def test_correct_logic_global_offsets(self):
        """
        Verify the correct logic produces proper global offsets
        into the concatenated feature tensor.
        """
        req_a = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=1000)]},
            "prefill_start": 500,
            "prefill_end": 1010,
        }
        req_b = {
            "mm_inputs": {"mm_positions": [ImagePosition(offset=10, length=2000)]},
            "prefill_start": 1000,
            "prefill_end": 2010,
        }

        _, _, slices = self._simulate_new_batch_logic([req_a, req_b])

        # req_a: image [10, 1010), prefill [500, 1010)
        #   overlap [500, 1010) → local [490, 1000), global offset=0
        #   slice = (490, 1000)
        self.assertEqual(slices[0], (490, 1000))

        # req_b: image [10, 2010), prefill [1000, 2010)
        #   overlap [1000, 2010) → local [990, 2000), global offset=1000
        #   slice = (1990, 3000)
        self.assertEqual(slices[1], (1990, 3000))

        # Total sliced = 510 + 1010 = 1520
        total = sum(e - s for s, e in slices)
        self.assertEqual(total, 510 + 1010)


if __name__ == "__main__":
    unittest.main()
