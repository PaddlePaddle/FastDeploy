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

import numpy as np
import paddle

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


class TestProcessMMFeatures(unittest.TestCase):

    def setUp(self):
        # Create a mock GPUModelRunner instance for testing
        self.mock_fd_config = Mock()
        self.mock_model_config = Mock()
        self.mock_model_config.enable_mm = True
        self.mock_model_config.model_type = "qwen"
        self.mock_fd_config.model_config = self.mock_model_config

        # Mock other necessary configurations
        self.mock_fd_config.scheduler_config = Mock()
        self.mock_fd_config.scheduler_config.max_num_seqs = 10
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1

        self.runner = GPUModelRunner.__new__(GPUModelRunner)
        self.runner.fd_config = self.mock_fd_config
        self.runner.model_config = self.mock_model_config
        self.runner.enable_mm = True
        self.runner.is_pooling_model = False
        self.runner.encoder_cache = {}
        self.runner.share_inputs = {
            "image_features": None,
            "rope_emb": paddle.full(shape=[2, 1], fill_value=0, dtype="float32"),
        }
        self.runner.extract_vision_features = Mock()
        self.runner.prepare_rope3d = Mock()

    def _create_mock_request(self, with_image=False, task_type_value=0, **kwargs):
        """Helper method to create mock requests"""
        request = Mock()
        request.task_type.value = task_type_value
        request.idx = kwargs.get("idx", 0)
        request.request_id = kwargs.get("request_id", "test_req")
        request.with_image = with_image
        request.prefill_start_index = kwargs.get("prefill_start_index", 0)
        request.prefill_end_index = kwargs.get("prefill_end_index", 10)
        request.num_image_start = kwargs.get("num_image_start", 0)
        request.num_image_end = kwargs.get("num_image_end", 0)
        request.image_start = kwargs.get("image_start", 0)
        request.image_end = kwargs.get("image_end", 0)

        # Setup multimodal_inputs
        request.multimodal_inputs = {
            "position_ids": kwargs.get("position_ids", np.array([[1, 2, 3]])),
        }

        if with_image:
            request.multimodal_inputs.update(
                {
                    "images": kwargs.get("images", []),
                    "grid_thw": kwargs.get("grid_thw", []),
                    "mm_positions": kwargs.get("mm_positions", []),
                    "mm_hashes": kwargs.get("mm_hashes", []),
                    "vit_seqlen": kwargs.get("vit_seqlen", []),
                    "vit_position_ids": kwargs.get("vit_position_ids", []),
                    "mm_num_token_func": lambda **kwargs: 123,
                }
            )

        # Add get method for evict_mm_hashes
        request.get = Mock(side_effect=lambda key, default=None: kwargs.get(key, default))

        return request

    def test_process_mm_features_no_mm_enabled(self):
        """Test when multimodal is not enabled"""
        self.runner.enable_mm = False
        request_list = [self._create_mock_request()]

        self.runner._process_mm_features(request_list)

        # Should return early without processing
        self.assertIsNone(self.runner.share_inputs["image_features"])

    def test_process_mm_features_no_prefill_requests(self):
        """Test when there are no prefill requests"""
        request_list = [
            self._create_mock_request(task_type_value=1),  # Not prefill
            self._create_mock_request(task_type_value=2),  # Not prefill
        ]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]
        self.runner._process_mm_features(request_list)

        # Should not process any requests
        self.assertIsNone(self.runner.share_inputs["image_features"])

    def test_process_mm_features_evict_cache(self):
        """Test eviction of multimodal cache"""
        # Pre-populate cache
        self.runner.encoder_cache["hash1"] = "cached_feature1"
        self.runner.encoder_cache["hash2"] = "cached_feature2"

        request_list = [self._create_mock_request(task_type_value=0, evict_mm_hashes=["hash1"])]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]
        self.runner._process_mm_features(request_list)

        # Check that hash1 was evicted but hash2 remains
        self.assertNotIn("hash1", self.runner.encoder_cache)
        self.assertIn("hash2", self.runner.encoder_cache)

    def test_process_mm_features_with_image_no_cache(self):
        """Test processing images without cache"""
        # Mock image features output
        self.runner.extract_vision_features.return_value = paddle.full(shape=[2, 1], fill_value=0, dtype="float32")

        # Setup grid_thw to return a value for paddle.prod
        grid_thw = [np.array([1, 4, 4])]  # prod will be 16, //4 = 4

        request_list = [
            self._create_mock_request(
                task_type_value=0,
                with_image=True,
                idx=0,
                num_image_start=0,
                num_image_end=1,
                grid_thw=grid_thw,
                mm_hashes=["new_hash"],
                mm_positions=[Mock(offset=0, length=4)],
                images=[1] * 16,  # 16 image tokens
                vit_seqlen=[4],
                vit_position_ids=[[0, 1, 2, 3]],
            )
        ]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]
        self.runner._process_mm_features(request_list)

        # Verify extract_vision_features was called
        self.runner.extract_vision_features.assert_called_once()

        # Verify cache was populated
        self.assertIn("new_hash", self.runner.encoder_cache)

        # Verify image features were set
        self.assertIsNotNone(self.runner.share_inputs["image_features"])

    def test_process_mm_features_with_cache_hit(self):
        """Test processing images with cache hit"""
        import numpy as np

        # Pre-populate cache
        cached_feature = Mock()
        cached_feature.cuda = paddle.full(shape=[2, 1], fill_value=0, dtype="float32")
        self.runner.encoder_cache["cached_hash"] = cached_feature

        # Mock image features output (should not be used due to cache hit)
        mock_features = Mock()
        self.runner.extract_vision_features.return_value = mock_features

        grid_thw = [np.array([1, 4, 4])]

        request_list = [
            self._create_mock_request(
                task_type_value=0,
                with_image=True,
                idx=0,
                num_image_start=0,
                num_image_end=1,
                grid_thw=grid_thw,
                mm_hashes=["cached_hash"],
                mm_positions=[Mock(offset=0, length=4)],
                images=[1] * 16,
                vit_seqlen=[4],
                vit_position_ids=[[0, 1, 2, 3]],
            )
        ]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]
        self.runner._process_mm_features(request_list)

        # Verify extract_vision_features was NOT called (cache hit)
        self.runner.extract_vision_features.assert_not_called()

        # Verify image features were set using cached feature
        self.assertIsNotNone(self.runner.share_inputs["image_features"])

    def test_process_mm_features_mixed_cache(self):
        """Test processing with mixed cache hit and miss"""
        import numpy as np

        # Pre-populate one cache entry
        cached_feature = Mock()
        cached_feature.cuda = paddle.full(shape=[2, 1], fill_value=0, dtype="float32")
        self.runner.encoder_cache["hash1"] = cached_feature

        self.runner.extract_vision_features.return_value = paddle.full(shape=[2, 1], fill_value=0, dtype="float32")
        grid_thw = [np.array([1, 4, 4]), np.array([1, 4, 4])]

        request_list = [
            self._create_mock_request(
                task_type_value=0,
                with_image=True,
                idx=0,
                num_image_start=0,
                num_image_end=2,
                grid_thw=grid_thw,
                mm_hashes=["hash1", "hash2"],  # hash1 in cache, hash2 not
                mm_positions=[Mock(offset=0, length=4), Mock(offset=4, length=4)],
                images=[1] * 32,  # 2 images, 16 tokens each
                vit_seqlen=[4, 4],
                vit_position_ids=[[0, 1, 2, 3], [4, 5, 6, 7]],
            )
        ]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]
        self.runner._process_mm_features(request_list)

        # Verify extract_vision_features was called (for hash2)
        self.runner.extract_vision_features.assert_called_once()

        # Verify both hashes are now in cache
        self.assertIn("hash1", self.runner.encoder_cache)
        self.assertIn("hash2", self.runner.encoder_cache)

        # Verify image features were set
        self.assertIsNotNone(self.runner.share_inputs["image_features"])

    def test_process_mm_features_no_encoder_cache(self):
        """Test processing without encoder cache"""
        import numpy as np

        self.runner.encoder_cache = None

        # Mock image features output
        self.runner.extract_vision_features.return_value = paddle.full(shape=[2, 1], fill_value=0, dtype="float32")
        grid_thw = [np.array([1, 4, 4])]

        request_list = [
            self._create_mock_request(
                task_type_value=0,
                with_image=True,
                idx=0,
                image_start=0,
                image_end=16,
                num_image_start=0,
                num_image_end=1,
                grid_thw=grid_thw,
                mm_positions=[Mock(offset=0, length=4)],
                images=[1] * 16,
                vit_seqlen=[4],
                vit_position_ids=[[0, 1, 2, 3]],
            )
        ]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]
        self.runner._process_mm_features(request_list)

        # Verify extract_vision_features was called
        self.runner.extract_vision_features.assert_called_once()

        # Verify image features were set
        self.assertIsNotNone(self.runner.share_inputs["image_features"])

    def test_process_mm_features_rope_3d_position_ids(self):
        """Test 3D position IDs processing"""
        request_list = [
            self._create_mock_request(
                task_type_value=0,
                idx=0,
                position_ids=np.array([[1, 2, 3]]),
                max_tokens=2048,
            ),
            self._create_mock_request(
                task_type_value=0,
                idx=1,
                position_ids=np.array([[4, 5, 6]]),
                max_tokens=1024,
            ),
        ]

        # Mock prepare_rope3d to return list of rope embeddings
        self.runner.prepare_rope3d.return_value = [1, 2]

        self.runner._process_mm_features(request_list)

        # Verify prepare_rope3d was called with correct parameters
        self.runner.prepare_rope3d.assert_called_once()

        # Verify rope embeddings were set in share_inputs
        self.assertEqual(self.runner.share_inputs["rope_emb"][0], paddle.Tensor([1]))
        self.assertEqual(self.runner.share_inputs["rope_emb"][1], paddle.Tensor([2]))

    def test_process_mm_features_pooling_model(self):
        """Test processing with pooling model"""
        self.runner.is_pooling_model = True

        request_list = [
            self._create_mock_request(
                task_type_value=0,
                idx=0,
                position_ids=np.array([[1, 2, 3]]),
            ),
        ]

        self.runner.prepare_rope3d.return_value = [1]

        self.runner._process_mm_features(request_list)

        # Verify max_tokens_lst contains 0 for pooling model
        call_args = self.runner.prepare_rope3d.call_args
        self.assertEqual(call_args[0][2], [0, 1])  # max_tokens_lst


if __name__ == "__main__":
    unittest.main()
