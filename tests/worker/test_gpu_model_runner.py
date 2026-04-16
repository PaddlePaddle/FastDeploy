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
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import paddle

from fastdeploy.engine.request import ImagePosition
from fastdeploy.spec_decode import SpecMethod
from fastdeploy.worker.gpu_model_runner import GPUModelRunner
from fastdeploy.worker.input_batch import InputBatch


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
        self.runner.scheduler_config = self.mock_fd_config.scheduler_config

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
        self.runner.scheduler_config = self.mock_fd_config.scheduler_config
        self.runner.enable_mm = True
        self.runner.is_pooling_model = False
        self.runner.encoder_cache = {}
        self.runner.share_inputs = InputBatch(self.mock_fd_config)
        self.runner.share_inputs.image_features = None
        self.runner.share_inputs.image_features_list = None
        self.runner.share_inputs.rope_emb = paddle.full(shape=[2, 1], fill_value=0, dtype="float32")
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

        self.assertIsNone(self.runner.share_inputs["image_features_list"])

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
        self.assertFalse(
            any(isinstance(t, paddle.Tensor) for t in self.runner.share_inputs["image_features_list"]),
        )

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
        self.assertTrue(
            any(isinstance(t, paddle.Tensor) for t in self.runner.share_inputs["image_features_list"]),
        )

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
        self.assertTrue(
            any(isinstance(t, paddle.Tensor) for t in self.runner.share_inputs["image_features_list"]),
        )

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
        self.assertTrue(
            any(isinstance(t, paddle.Tensor) for t in self.runner.share_inputs["image_features_list"]),
        )

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
        self.assertTrue(
            any(isinstance(t, paddle.Tensor) for t in self.runner.share_inputs["image_features_list"]),
        )


class TestSleepWakeupBehavior(unittest.TestCase):
    def _make_runner(self):
        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.is_weight_sleeping = False
        runner.is_kvcache_sleeping = False
        runner.use_cudagraph = False
        runner.spec_method = None
        runner.local_rank = 0
        runner.device_id = 1
        runner.num_gpu_blocks = 8
        runner.model = Mock(clear_graph_opt_backend=Mock())
        runner.clear_cache = Mock()
        runner.initialize_kv_cache = Mock()
        runner.capture_model = Mock()
        runner.share_inputs = Mock(reset_share_inputs=Mock())
        runner.dynamic_weight_manager = Mock(
            clear_deepep_buffer=Mock(),
            clear_model_weight=Mock(),
            clear_communication_group=Mock(),
            restart_communication_group=Mock(),
            recreate_deepep_buffer=Mock(),
            reload_model_weights=Mock(),
        )
        runner.fd_config = Mock()
        runner.fd_config.parallel_config = Mock(
            enable_expert_parallel=False,
            shutdown_comm_group_if_worker_idle=False,
        )
        runner.proposer = Mock(
            clear_mtp_cache=Mock(),
            initialize_kv_cache=Mock(),
            model_inputs=Mock(reset_model_inputs=Mock()),
        )
        return runner

    @patch("fastdeploy.worker.gpu_model_runner.print_gpu_memory_use")
    @patch("paddle.device.cuda.empty_cache")
    def test_sleep_offloads_weight_and_cache(self, mock_empty_cache, mock_print_memory):
        runner = self._make_runner()
        runner.use_cudagraph = True
        runner.spec_method = SpecMethod.MTP
        runner.fd_config.parallel_config.enable_expert_parallel = True
        runner.fd_config.parallel_config.shutdown_comm_group_if_worker_idle = True

        runner.sleep("weight,kv_cache")

        runner.model.clear_graph_opt_backend.assert_called_once()
        runner.dynamic_weight_manager.clear_deepep_buffer.assert_called_once()
        runner.dynamic_weight_manager.clear_model_weight.assert_called_once()
        runner.dynamic_weight_manager.clear_communication_group.assert_called_once()
        runner.proposer.clear_mtp_cache.assert_called_once()
        runner.clear_cache.assert_called_once()
        self.assertTrue(runner.is_weight_sleeping)
        self.assertTrue(runner.is_kvcache_sleeping)
        mock_empty_cache.assert_called_once()
        mock_print_memory.assert_called_once()

    @patch("fastdeploy.worker.gpu_model_runner.print_gpu_memory_use")
    @patch("paddle.device.cuda.empty_cache")
    def test_sleep_weight_is_idempotent(self, mock_empty_cache, mock_print_memory):
        runner = self._make_runner()
        runner.is_weight_sleeping = True

        runner.sleep("weight")

        runner.dynamic_weight_manager.clear_model_weight.assert_not_called()
        runner.clear_cache.assert_not_called()
        mock_empty_cache.assert_not_called()
        mock_print_memory.assert_not_called()

    def test_wakeup_rejects_weight_only_when_cudagraph_requires_kvcache(self):
        runner = self._make_runner()
        runner.use_cudagraph = True
        runner.is_kvcache_sleeping = True

        with self.assertRaises(RuntimeError):
            runner.wakeup("weight")

    @patch("fastdeploy.worker.gpu_model_runner.print_gpu_memory_use")
    def test_wakeup_restores_weight_and_cache(self, mock_print_memory):
        runner = self._make_runner()
        runner.use_cudagraph = True
        runner.spec_method = SpecMethod.MTP
        runner.is_weight_sleeping = True
        runner.is_kvcache_sleeping = True
        runner.fd_config.parallel_config.enable_expert_parallel = True
        runner.fd_config.parallel_config.shutdown_comm_group_if_worker_idle = True

        runner.wakeup("weight,kv_cache")

        runner.proposer.model_inputs.reset_model_inputs.assert_called_once()
        runner.share_inputs.reset_share_inputs.assert_called_once()
        runner.proposer.initialize_kv_cache.assert_called_once_with(main_model_num_blocks=runner.num_gpu_blocks)
        runner.initialize_kv_cache.assert_called_once()
        runner.dynamic_weight_manager.restart_communication_group.assert_called_once()
        runner.dynamic_weight_manager.recreate_deepep_buffer.assert_called_once()
        runner.dynamic_weight_manager.reload_model_weights.assert_called_once()
        runner.capture_model.assert_called_once()
        self.assertFalse(runner.is_weight_sleeping)
        self.assertFalse(runner.is_kvcache_sleeping)
        mock_print_memory.assert_called_once()

    @patch("fastdeploy.worker.gpu_model_runner.print_gpu_memory_use")
    def test_wakeup_kvcache_is_idempotent(self, mock_print_memory):
        runner = self._make_runner()
        runner.is_kvcache_sleeping = False

        runner.wakeup("kv_cache")

        runner.initialize_kv_cache.assert_not_called()
        runner.dynamic_weight_manager.reload_model_weights.assert_not_called()
        mock_print_memory.assert_not_called()


def _sync_async_set_value(tgt, src):
    """Synchronous stand-in for async_set_value used in tests (no CUDA required).

    Writes to real numpy arrays; silently skips Mock objects (untracked share_inputs
    fields whose values we do not assert on).
    """
    from unittest.mock import MagicMock

    import numpy as np

    if isinstance(tgt, MagicMock):
        return  # untracked field — nothing to write
    if isinstance(src, (int, float, bool)):
        tgt[:] = src
    elif isinstance(src, (list, np.ndarray)):
        tgt[:] = np.array(src).reshape(tgt.shape)
    elif hasattr(src, "numpy"):
        tgt[:] = src.numpy()
    else:
        tgt[:] = src


class TestInsertTasksV1SplitwiseSuffix(unittest.TestCase):
    """Tests for insert_tasks_v1 splitwise_role=\'decode\' + SpecMethod.SUFFIX branch."""

    def _make_share_inputs(self, bsz=4, max_draft=6):
        """Mock-backed share_inputs; only keys we assert on hold real numpy arrays."""
        import numpy as np

        # Keys whose values we want to inspect after the call
        tracked = {
            "seq_lens_encoder": np.zeros((bsz, 1), dtype=np.int32),
            "draft_tokens": np.zeros((bsz, max_draft), dtype=np.int64),
            "seq_lens_this_time_buffer": np.zeros((bsz, 1), dtype=np.int32),
            "req_ids": [""] * bsz,
            "preempted_idx": np.zeros((bsz, 1), dtype=np.int32),
            "num_running_requests": 0,
            "running_requests_ids": [],
        }

        class _SI:
            def get_index_by_batch_id(self, batch_id):
                return batch_id

            def __getitem__(self, key):
                # Return real array for tracked keys; Mock for everything else
                if key in tracked:
                    return tracked[key]
                return MagicMock()

            def __setitem__(self, key, value):
                tracked[key] = value

        return _SI()

    def _make_runner(self, bsz=4, num_spec_tokens=3):
        from unittest.mock import Mock

        from fastdeploy.spec_decode import SpecMethod
        from fastdeploy.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.enable_mm = False
        runner.is_pooling_model = False
        runner.speculative_decoding = True
        runner.spec_method = SpecMethod.SUFFIX
        runner.speculative_config = Mock(num_speculative_tokens=num_spec_tokens)
        runner.deterministic_logger = None
        runner.routing_replay_manager = Mock()
        runner.prompt_logprobs_reqs = {}
        runner.in_progress_prompt_logprobs = {}
        runner.forward_batch_reqs_list = [None] * bsz
        runner._cached_launch_token_num = -1
        runner._cached_real_bsz = 0
        runner.exist_prefill_flag = True
        runner.proposer = Mock()
        runner.sampler = Mock()
        runner.model_config = Mock(eos_tokens_lens=1)
        runner.share_inputs = self._make_share_inputs(bsz=bsz, max_draft=num_spec_tokens + 2)

        fd_config = Mock()
        fd_config.scheduler_config.splitwise_role = "decode"
        fd_config.routing_replay_config.enable_routing_replay = False
        runner.fd_config = fd_config
        runner.scheduler_config = fd_config.scheduler_config
        return runner

    def _make_prefill_request(self, idx, draft_token_ids):
        from unittest.mock import Mock

        from fastdeploy.engine.request import RequestType

        req = Mock()
        req.task_type = Mock(value=RequestType.PREFILL.value)
        req.idx = idx
        req.request_id = f"req_{idx}"
        req.prompt_token_ids = [10, 20, 30]
        req.output_token_ids = [99]
        req.draft_token_ids = draft_token_ids
        req.pooling_params = None
        req.guided_json = None
        req.guided_regex = None
        req.structural_tag = None
        req.guided_grammar = None
        req.prefill_start_index = 0
        req.prefill_end_index = 3
        req.multimodal_inputs = None
        req.get = Mock(return_value=None)
        req.eos_token_ids = [2]
        req.block_tables = []
        return req

    @patch("fastdeploy.worker.gpu_model_runner.async_set_value", side_effect=_sync_async_set_value)
    def test_draft_tokens_and_seq_lens_written(self, _mock_asv):
        """draft_tokens[0:2] and seq_lens_this_time_buffer=2 are written."""
        runner = self._make_runner(num_spec_tokens=3)
        req = self._make_prefill_request(idx=0, draft_token_ids=[101, 202, 303])
        runner.insert_tasks_v1([req], num_running_requests=1)

        self.assertEqual(runner.share_inputs["draft_tokens"][0, 0], 101)
        self.assertEqual(runner.share_inputs["draft_tokens"][0, 1], 202)
        self.assertEqual(runner.share_inputs["seq_lens_this_time_buffer"][0, 0], 2)

    @patch("fastdeploy.worker.gpu_model_runner.async_set_value", side_effect=_sync_async_set_value)
    def test_exist_prefill_flag_cleared(self, _mock_asv):
        runner = self._make_runner()
        req = self._make_prefill_request(idx=0, draft_token_ids=[1, 2])
        runner.insert_tasks_v1([req], num_running_requests=1)
        self.assertFalse(runner.exist_prefill_flag)

    @patch("fastdeploy.worker.gpu_model_runner.async_set_value", side_effect=_sync_async_set_value)
    def test_cached_launch_token_num_incremented(self, _mock_asv):
        runner = self._make_runner(num_spec_tokens=3)
        runner._cached_launch_token_num = 10
        runner._cached_real_bsz = 2
        req = self._make_prefill_request(idx=0, draft_token_ids=[1, 2])
        runner.insert_tasks_v1([req], num_running_requests=1)
        # token_num_one_step = num_speculative_tokens + 1 = 4
        self.assertEqual(runner._cached_launch_token_num, 14)
        self.assertEqual(runner._cached_real_bsz, 3)

    @patch("fastdeploy.worker.gpu_model_runner.async_set_value", side_effect=_sync_async_set_value)
    def test_cached_launch_token_num_skipped_when_negative_one(self, _mock_asv):
        runner = self._make_runner(num_spec_tokens=3)
        runner._cached_launch_token_num = -1
        req = self._make_prefill_request(idx=0, draft_token_ids=[1, 2])
        runner.insert_tasks_v1([req], num_running_requests=1)
        self.assertEqual(runner._cached_launch_token_num, -1)

    @patch("fastdeploy.worker.gpu_model_runner.async_set_value", side_effect=_sync_async_set_value)
    def test_raises_when_fewer_than_two_draft_tokens(self, _mock_asv):
        runner = self._make_runner()
        req = self._make_prefill_request(idx=0, draft_token_ids=[42])
        with self.assertRaises(ValueError):
            runner.insert_tasks_v1([req], num_running_requests=1)


if __name__ == "__main__":
    unittest.main()
