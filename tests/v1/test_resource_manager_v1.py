import concurrent.futures
import pickle
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.config import CacheConfig, FDConfig, ParallelConfig, SchedulerConfig
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request
from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


class TestResourceManagerV1(unittest.TestCase):
    def setUp(self):
        max_num_seqs = 2
        engine_args = EngineArgs(
            max_num_seqs=max_num_seqs,
            num_gpu_blocks_override=102,
            max_num_batched_tokens=3200,
        )
        args = asdict(engine_args)

        cache_cfg = CacheConfig(args)
        model_cfg = SimpleNamespace(enable_mm=True)  # Enable multimodal for feature testing
        speculative_cfg = SimpleNamespace(method=None)
        model_cfg.print = print
        model_cfg.max_model_len = 5120
        cache_cfg.bytes_per_layer_per_block = 1
        parallel_cfg = ParallelConfig(args)
        scheduler_cfg = SchedulerConfig(args)
        graph_opt_cfg = engine_args.create_graph_optimization_config()

        fd_config = FDConfig(
            model_config=model_cfg,
            cache_config=cache_cfg,
            parallel_config=parallel_cfg,
            graph_opt_config=graph_opt_cfg,
            speculative_config=speculative_cfg,
            scheduler_config=scheduler_cfg,
        )
        self.manager = ResourceManagerV1(
            max_num_seqs=max_num_seqs, config=fd_config, tensor_parallel_size=8, splitwise_role="mixed"
        )
        req_dict = {
            "request_id": "test_request",
            "multimodal_inputs": {},
        }
        self.request = Request.from_dict(req_dict)
        self.request.async_process_futures = []
        self.request.multimodal_inputs = {}

    def test_waiting_async_process_no_futures(self):
        """Test when there are no async process futures"""
        result = self.manager._waiting_async_process(self.request)
        self.assertFalse(result)

    def test_waiting_async_process_future_done_no_error(self):
        """Test when future is done with no error"""
        future = concurrent.futures.Future()
        future.set_result(True)
        self.request.async_process_futures = [future]

        result = self.manager._waiting_async_process(self.request)
        self.assertFalse(result)
        self.assertEqual(len(self.request.async_process_futures), 0)

    def test_waiting_async_process_future_done_with_error(self):
        """Test when future is done with error"""
        future = concurrent.futures.Future()
        future.set_result(True)
        self.request.async_process_futures = [future]
        self.request.error_message = "Download failed"

        result = self.manager._waiting_async_process(self.request)
        self.assertIsNone(result)

    def test_waiting_async_process_future_not_done(self):
        """Test when future is not done"""
        future = concurrent.futures.Future()
        self.request.async_process_futures = [future]

        result = self.manager._waiting_async_process(self.request)
        self.assertTrue(result)
        self.assertEqual(len(self.request.async_process_futures), 1)

    def test_apply_async_preprocess(self):
        """Test applying async preprocess"""
        with patch.object(self.manager.async_preprocess_pool, "submit") as mock_submit:
            mock_submit.return_value = "mock_future"
            self.manager._apply_async_preprocess(self.request)

            mock_submit.assert_called_once_with(self.manager._download_features, self.request)
            self.assertEqual(len(self.request.async_process_futures), 1)
            self.assertEqual(self.request.async_process_futures[0], "mock_future")

    @patch("fastdeploy.utils.init_bos_client")
    @patch("fastdeploy.utils.download_from_bos")
    def test_download_features_no_features(self, mock_download, mock_init):
        """Test when no features to download"""
        self.request.multimodal_inputs = {}
        result = self.manager._download_features(self.request)
        self.assertIsNone(result)
        mock_download.assert_not_called()
        mock_init.assert_not_called()

    def test_download_features_video_success(self):
        """Test successful video feature download"""
        mock_client = MagicMock()
        mock_client.get_object_as_string.return_value = pickle.dumps(np.array([[1, 2, 3]], dtype=np.float32))

        self.request.multimodal_inputs = {"video_feature_urls": ["bos://bucket-name/path/to/object1"]}

        self.manager.bos_client = mock_client
        result = self.manager._download_features(self.request)
        self.assertIsNone(result)
        self.assertIn("video_features", self.request.multimodal_inputs)
        self.assertIsInstance(self.request.multimodal_inputs["video_features"][0], np.ndarray)

    def test_download_features_image_error(self):
        """Test image feature download with error"""
        mock_client = MagicMock()
        mock_client.get_object_as_string.side_effect = Exception("network error")

        self.request.multimodal_inputs = {"image_feature_urls": ["bos://bucket-name/path/to/object1"]}

        self.manager.bos_client = mock_client
        result = self.manager._download_features(self.request)
        self.assertIsNone(result)
        self.assertIn(
            "request test_request download features error",
            self.request.error_message,
        )
        self.assertEqual(self.request.error_code, 530)

    def test_download_features_audio_mixed(self):
        """Test mixed success/error in audio feature download"""
        mock_client = MagicMock()
        mock_client.get_object_as_string.side_effect = [
            pickle.dumps(np.array([[1, 2, 3]], dtype=np.float32)),
            Exception("timeout"),
        ]

        self.request.multimodal_inputs = {
            "audio_feature_urls": ["bos://bucket-name/path/to/object1", "bos://bucket-name/path/to/object2"]
        }

        self.manager.bos_client = mock_client
        result = self.manager._download_features(self.request)
        self.assertIsNone(result)
        self.assertIn(
            "request test_request download features error",
            self.request.error_message,
        )
        self.assertEqual(self.request.error_code, 530)

    def test_download_features_retry(self):
        """Test image feature download with error"""
        mock_client = MagicMock()
        mock_client.get_object_as_string.side_effect = Exception(
            "Your request rate is too high. We have put limits on your bucket."
        )

        self.request.multimodal_inputs = {"image_feature_urls": ["bos://bucket-name/path/to/object1"]}

        self.manager.bos_client = mock_client
        result = self.manager._download_features(self.request)
        self.assertIsNone(result)
        self.assertIn("Failed after 1 retries for bos://bucket-name/path/to/object1", self.request.error_message)
        self.assertEqual(self.request.error_code, 530)


if __name__ == "__main__":
    unittest.main()
