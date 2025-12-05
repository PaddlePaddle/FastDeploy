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
from dataclasses import dataclass
from unittest.mock import Mock

from fastdeploy.engine.request import ImagePosition
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


@dataclass
class TestRequest:
    multimodal_inputs: dict = None


class TestFeaturePositions(unittest.TestCase):

    def setUp(self):
        # Create a mock GPUModelRunner instance for testing
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.enable_mm = True
        self.mock_fd_config.model_config = self.mock_model_config

        # Mock other necessary configurations
        self.mock_fd_config.scheduler_config = Mock()
        self.mock_fd_config.scheduler_config.max_num_seqs = 10
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config

    def test_completely_within_range(self):
        """Test positions that are completely within the prefill range"""
        mm_positions = [
            ImagePosition(offset=10, length=5),  # [10, 14]
            ImagePosition(offset=15, length=5),  # [15, 19]
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].offset, 0)
        self.assertEqual(result[0].length, 5)
        self.assertEqual(result[1].offset, 0)
        self.assertEqual(result[1].length, 5)

    def test_completely_outside_range(self):
        """Test positions that are completely outside the prefill range"""
        mm_positions = [
            ImagePosition(offset=5, length=3),  # [5, 7] - before range
            ImagePosition(offset=25, length=5),  # [25, 29] - after range
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 0)

    def test_partial_overlap_start(self):
        """Test positions that partially overlap at the start of the range"""
        mm_positions = [
            ImagePosition(offset=8, length=5),  # [8, 12] overlaps with [10, 20]
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].offset, 2)  # Adjusted to start at prefill_start_index
        self.assertEqual(result[0].length, 3)  # Length reduced to fit within range

    def test_partial_overlap_end(self):
        """Test positions that partially overlap at the end of the range"""
        mm_positions = [
            ImagePosition(offset=8, length=50),  # [8, 58] overlaps with [10, 20]
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].offset, 2)  # Offset remains the same
        self.assertEqual(result[0].length, 10)  # Length reduced to fit within range

    def test_exact_range_boundary(self):
        """Test positions that exactly match the range boundaries"""
        mm_positions = [
            ImagePosition(offset=10, length=10),  # Exactly matches [10, 20]
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].offset, 0)
        self.assertEqual(result[0].length, 10)

    def test_edge_overlap(self):
        """Test positions that exactly touch the range boundaries"""
        mm_positions = [
            ImagePosition(offset=20, length=5),  # Starts exactly at end boundary but should be excluded
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 0)  # Should be excluded - ends at boundary means outside

    def test_multiple_overlapping_positions(self):
        """Test mixed positions with different overlap scenarios"""
        mm_positions = [
            ImagePosition(offset=5, length=3),  # [5, 8] - before range
            ImagePosition(offset=8, length=5),  # [8, 13] - overlaps start
            ImagePosition(offset=13, length=6),  # [13, 19] - completely within
            ImagePosition(offset=19, length=5),  # [19, 24] - overlaps end
            ImagePosition(offset=24, length=3),  # [24, 27] - after range
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)
        self.assertEqual(len(result), 3)

        # First position (overlapping start)
        self.assertEqual(result[0].offset, 2)
        self.assertEqual(result[0].length, 3)

        # Second position (completely within)
        self.assertEqual(result[1].offset, 0)
        self.assertEqual(result[1].length, 6)

        # Third position (overlapping end)
        self.assertEqual(result[2].offset, 0)
        self.assertEqual(result[2].length, 1)

    def test_zero_length_range(self):
        """Test with zero-length prefill range"""
        mm_positions = [
            ImagePosition(offset=10, length=5),
        ]
        prefill_start_index = 15
        prefill_end_index = 15  # Zero-length range

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 0)

    def test_empty_positions_list(self):
        """Test with an empty positions list"""
        mm_positions = []
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 0)

    def test_identical_positions_copy(self):
        """Test that positions within range are correctly deep copied"""
        mm_positions = [
            ImagePosition(offset=12, length=5),
        ]
        prefill_start_index = 10
        prefill_end_index = 20

        result = self.runner._get_feature_positions(mm_positions, prefill_start_index, prefill_end_index)

        self.assertEqual(len(result), 1)
        # Verify it's a copy, not the same object
        self.assertIsNot(result[0], mm_positions[0])
        # But has the same values
        self.assertEqual(result[0].offset, 0)
        self.assertEqual(result[0].length, 5)


if __name__ == "__main__":
    unittest.main()
