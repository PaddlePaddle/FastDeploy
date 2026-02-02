# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
from unittest.mock import Mock, MagicMock, patch

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request, RequestStatus, RequestType
from fastdeploy.engine.sched.resource_manager_v1 import (
    ResourceManagerV1, 
    SignalConsumer,
    ScheduledDecodeTask,
    ScheduledPreemptTask,
    ScheduledExtendBlocksTask
)

MODEL_NAME = os.getenv("MODEL_PATH", "/path/to/models") + "/ERNIE-4.5-0.3B-Paddle"


class TestSignalConsumer(unittest.TestCase):
    def test_signal_consumer(self):
        signal = 10
        limit = 2
        consumer = SignalConsumer(signal, limit)
        self.assertEqual(consumer.watch(), 10)
        self.assertEqual(consumer.consume(), 10)
        self.assertEqual(consumer.watch(), 10)
        self.assertEqual(consumer.consume(), 10)
        self.assertEqual(consumer.watch(), 0)
        self.assertEqual(consumer.consume(), 0)


class TestResourceManagerV1(unittest.TestCase):
    """Test cases for ResourceManagerV1."""

    def setUp(self):
        """Set up test fixtures."""
        engine_args = EngineArgs(
            model=MODEL_NAME,
            max_model_len=8192,
            tensor_parallel_size=1,
            engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778")),
            cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT", "6779")),
        )
        # Create and start the engine service
        mock_config = engine_args.create_engine_config()
        
        # Configure mock config with necessary attributes
        mock_config.cache_config.block_size = 16
        mock_config.cache_config.max_block_num_per_seq = 100
        mock_config.cache_config.enable_prefix_caching = False
        mock_config.cache_config.enable_output_caching = False
        mock_config.cache_config.enc_dec_block_num = 1
        mock_config.speculative_config.method = None
        mock_config.scheduler_config.max_num_batched_tokens = 2048
        mock_config.scheduler_config.splitwise_role = "mixed"
        mock_config.model_config.enable_mm = False
        mock_config.model_config.architectures = ["LlamaForCausalLM"]
        # Default cache manager config for preallocate tests
        mock_config.cache_config.prealloc_dec_block_slot_num_threshold = 0

        self.manager = ResourceManagerV1(
            max_num_seqs=4,
            config=mock_config,
            tensor_parallel_size=1,
            splitwise_role="mixed",
            local_data_parallel_id=0,
        )

        # Mock cache manager
        self.manager.cache_manager = Mock()
        self.manager.cache_manager.free_blocks = Mock()
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[1, 2])
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.recycle_gpu_blocks = Mock()
        self.manager.cache_manager.request_match_blocks = Mock(return_value=([], 0, {}))
        self.manager.cache_manager.get_required_block_num = Mock(return_value=0)
        self.manager.cache_manager.write_cache_to_storage = Mock()
        self.manager.cache_manager.num_cpu_blocks = 0
        
        # Mock metrics
        self.manager.update_metrics = Mock()

    def tearDown(self) -> None:
        self.manager.need_block_num_signal.clear()

    def test_allocated_slots(self):
        req = Mock(spec=Request)
        req.block_tables = [1, 2, 3]
        # block_size is 16
        self.assertEqual(self.manager.allocated_slots(req), 48)

    def test_get_new_block_nums(self):
        req = Mock(spec=Request)
        req.num_computed_tokens = 10
        req.block_tables = [1] # 1 block allocated (capacity 16)
        # new tokens: 20. Total needed: 30.
        # (10 + 20 + 15) // 16 = 45 // 16 = 2 blocks total needed.
        # 2 - 1 = 1 new block.
        self.assertEqual(self.manager.get_new_block_nums(req, 20), 1)
        
        # Test max block num cap
        self.manager.config.speculative_config.method = "mtp"
        # block_num = min(block_num + 1, max_block_num_per_seq)
        self.assertEqual(self.manager.get_new_block_nums(req, 20), 2)

    def test_reschedule_preempt_task(self):
        req = Mock(spec=Request)
        req.request_id = "req1"
        self.manager.requests["req1"] = req
        self.manager.to_be_rescheduled_request_id_set.add("req1")
        
        process_func = Mock()
        self.manager.reschedule_preempt_task("req1", process_func)
        
        process_func.assert_called_with(req)
        self.assertIn(req, self.manager.waiting)
        self.assertNotIn("req1", self.manager.to_be_rescheduled_request_id_set)

    def test_preempted_all_with_no_running_requests(self):
        """Test preempted_all with no running requests."""
        self.assertEqual(len(self.manager.running), 0)
        preempted_reqs = self.manager.preempted_all()
        self.assertEqual(len(preempted_reqs), 0)

    def test_preempted_all_with_normal_requests(self):
        """Test preempted_all with normal running requests."""
        # Add mock running requests
        req1 = Mock(spec=Request)
        req1.request_id = "req1"
        req1.use_extend_tables = False
        req1.status = RequestStatus.RUNNING
        req1.block_tables = [1, 2, 3]
        req1.num_cached_blocks = 0
        req1.idx = 0

        req2 = Mock(spec=Request)
        req2.request_id = "req2"
        req2.use_extend_tables = False
        req2.status = RequestStatus.RUNNING
        req2.block_tables = [4, 5]
        req2.num_cached_blocks = 0
        req2.idx = 1

        self.manager.running = [req1, req2]
        self.manager.requests = {"req1": req1, "req2": req2}

        preempted_reqs = self.manager.preempted_all()

        # Verify
        self.assertEqual(len(preempted_reqs), 2)
        # LIFO
        self.assertEqual(preempted_reqs[0].request_id, "req2")
        self.assertEqual(preempted_reqs[1].request_id, "req1")

        # Verify request status changed
        self.assertEqual(req1.status, RequestStatus.PREEMPTED)
        self.assertEqual(req2.status, RequestStatus.PREEMPTED)

        # Verify added to to_be_rescheduled_request_id_set
        self.assertIn("req1", self.manager.to_be_rescheduled_request_id_set)
        self.assertIn("req2", self.manager.to_be_rescheduled_request_id_set)

        self.assertEqual(len(self.manager.running), 0)

    def test_trigger_preempt(self):
        req1 = Mock(spec=Request)
        req1.request_id = "req1"
        req1.use_extend_tables = False
        req1.status = RequestStatus.RUNNING
        req1.block_tables = [1]
        req1.idx = 0
        req1.num_cached_blocks = 0

        self.manager.running = [req1]
        self.manager.requests = {"req1": req1}
        
        # Mock cache_manager to fail allocation initially
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(side_effect=[False, True])
        
        preempted_reqs = []
        scheduled_reqs = []
        
        # Try to preempt to free blocks for a new request
        target_req = Mock(spec=Request)
        can_schedule = self.manager._trigger_preempt(target_req, 10, preempted_reqs, scheduled_reqs)
        
        self.assertTrue(can_schedule)
        self.assertEqual(len(preempted_reqs), 1)
        self.assertEqual(preempted_reqs[0], req1)
        self.assertEqual(len(self.manager.running), 0)
        self.assertEqual(req1.status, RequestStatus.PREEMPTED)

    def test_schedule_prefill(self):
        req = Mock(spec=Request)
        req.request_id = "req_prefill"
        req.status = RequestStatus.WAITING
        req.need_prefill_tokens = 10
        req.num_computed_tokens = 0
        req.multimodal_inputs = {}
        req.block_tables = []
        req.async_process_futures = []
        req.get = Mock(return_value=False) # for skip_allocate
        
        self.manager.add_request(req)
        
        # Ensure resources available
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[1])
        
        scheduled, errors = self.manager.schedule()
        
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(len(errors), 0)
        self.assertEqual(scheduled[0].task_type, RequestType.PREFILL)
        self.assertEqual(len(self.manager.running), 1)
        self.assertEqual(self.manager.running[0], req)
        self.assertEqual(req.status, RequestStatus.RUNNING)

    def test_schedule_decode(self):
        req = Mock(spec=Request)
        req.request_id = "req_decode"
        req.idx = 0
        req.status = RequestStatus.RUNNING
        req.need_prefill_tokens = 10
        req.num_computed_tokens = 10
        req.num_total_tokens = 20
        req.block_tables = [1]
        req.use_extend_tables = False
        
        self.manager.running = [req]
        self.manager.requests = {"req_decode": req}
        
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[2])
        
        # Mock signal for need_block_num
        self.manager.need_block_num_signal.value[0] = 0
        
        scheduled, errors = self.manager.schedule()
        
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0].task_type, RequestType.DECODE)
        self.assertIn(2, req.block_tables)

    def test_add_request_async(self):
        req = Mock(spec=Request)
        req.request_id = "req_async"
        req.status = RequestStatus.WAITING
        req.async_process_futures = []
        req.multimodal_inputs = {}
        
        # Test add_request applies async preprocess
        self.manager.add_request(req)
        self.assertEqual(len(self.manager.waiting), 1)
        self.assertEqual(len(req.async_process_futures), 1)

    def test_finish_requests(self):
        req = Mock(spec=Request)
        req.request_id = "req_finish"
        req.idx = 0
        req.status = RequestStatus.RUNNING
        req.block_tables = [1]
        req.extend_block_tables = []
        
        self.manager.running = [req]
        self.manager.requests = {"req_finish": req}
        self.manager.tasks_list[0] = req
        self.manager.stop_flags[0] = False
        
        self.manager.finish_requests("req_finish")
        
        self.assertNotIn("req_finish", self.manager.requests)
        self.assertEqual(len(self.manager.running), 0)
        self.assertTrue(self.manager.stop_flags[0])
        self.manager.cache_manager.recycle_gpu_blocks.assert_called()

    def test_prefix_cache_hit(self):
        self.manager.config.cache_config.enable_prefix_caching = True
        
        req = Mock(spec=Request)
        req.request_id = "req_cache"
        req.status = RequestStatus.WAITING
        req.need_prefill_tokens = 32
        req.num_computed_tokens = 0
        req.multimodal_inputs = {}
        req.async_process_futures = []
        req.get = Mock(return_value=False)
        req.block_tables = []
        req.metrics = Mock()
        
        self.manager.waiting.append(req)
        self.manager.requests["req_cache"] = req
        
        # Match success
        self.manager.cache_manager.request_match_blocks = Mock(return_value=([1], 16, {
            "match_gpu_block_ids": [1],
            "gpu_match_token_num": 16,
            "cpu_match_token_num": 0,
            "storage_match_token_num": 0,
            "gpu_recv_block_ids": [],
            "match_storage_block_ids": [],
            "cpu_cache_prepare_time": 0,
            "storage_cache_prepare_time": 0
        }))
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[2])
        
        scheduled, errors = self.manager.schedule()
        
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(req.num_cached_tokens, 16)
        # 32 total needed, 16 cached. Need 16 more.
        
    def test_preallocate_resource_in_p(self):
        self.manager.config.scheduler_config.splitwise_role = "prefill"
        req = Mock(spec=Request)
        req.request_id = "req_p"
        req.prompt_token_ids = [1] * 32
        req.block_tables = []
        
        # Success case
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[1, 2])
        
        result = self.manager.preallocate_resource_in_p(req)
        self.assertTrue(result)
        self.assertIn("req_p", self.manager.requests)
        self.assertIn(1, req.block_tables)

    def test_preallocate_resource_in_d(self):
        self.manager.config.scheduler_config.splitwise_role = "decode"
        req = Mock(spec=Request)
        req.request_id = "req_d"
        req.prompt_token_ids = [1] * 32
        req.disaggregate_info = {}
        req.reasoning_max_tokens = None
        
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[1, 2])
        
        result = self.manager.preallocate_resource_in_d(req)
        self.assertTrue(result)
        self.assertIn("req_d", self.manager.requests)
        self.assertIn(1, req.block_tables)

    def test_add_prefilled_request(self):
        self.manager.config.scheduler_config.splitwise_role = "decode"
        req = Mock(spec=Request)
        req.request_id = "req_filled"
        req.prompt_token_ids = [1] * 10
        req.output_token_ids = []
        req.metrics = Mock()
        
        self.manager.requests["req_filled"] = req
        
        req_output = Mock()
        req_output.request_id = "req_filled"
        req_output.outputs.token_ids = [100]
        req_output.num_cached_tokens = 0
        req_output.metrics = Mock()
        
        self.manager.add_prefilled_request(req_output)
        
        self.assertIn(req, self.manager.running)
        self.assertEqual(req.output_token_ids, [100])

    def test_prepare_tasks(self):
        req = Mock(spec=Request)
        req.idx = 1
        req.request_id = "req1"
        req.block_tables = [1, 2]
        req.num_computed_tokens = 10
        
        # Test _prepare_prefill_task
        req = self.manager._prepare_prefill_task(req, 5)
        self.assertEqual(req.prefill_start_index, 10)
        self.assertEqual(req.prefill_end_index, 15)
        self.assertEqual(req.task_type, RequestType.PREFILL)
        
        # Test _prepare_decode_task
        task = self.manager._prepare_decode_task(req)
        self.assertEqual(task.idx, 1)
        self.assertEqual(task.request_id, "req1")
        self.assertEqual(task.block_tables, [1, 2])
        self.assertEqual(task.task_type, RequestType.DECODE)

        # Test _prepare_preempt_task
        task = self.manager._prepare_preempt_task(req)
        self.assertEqual(task.idx, 1)
        self.assertEqual(task.request_id, "req1")
        self.assertEqual(task.task_type, RequestType.PREEMPTED)

    def test_info_each_block(self):
        req = Mock(spec=Request)
        req.idx = 1
        req.block_tables = [1]
        req.extend_block_tables = []
        self.manager.running.append(req)
        # Just ensure it doesn't crash
        self.manager._info_each_block()
        
    def test_can_preempt(self):
        req1 = Mock(spec=Request)
        req1.use_extend_tables = False
        self.manager.running.append(req1)
        self.assertTrue(self.manager._can_preempt())
        
        req1.use_extend_tables = True
        self.assertFalse(self.manager._can_preempt())
        
    @patch("time.sleep")
    def test_wait_worker_inflight_requests_finish(self, mock_sleep):
        # Case 1: no running requests
        self.manager.running = []
        self.manager.to_be_rescheduled_request_id_set = set()
        self.manager.wait_worker_inflight_requests_finish()
        
        # Case 2: running requests eventually finish
        self.manager.running = [Mock()]
        
        def side_effect():
            if mock_sleep.call_count >= 2:
                self.manager.running = []
                
        mock_sleep.side_effect = lambda x: side_effect()
        self.manager.wait_worker_inflight_requests_finish(timeout=1)
        self.assertEqual(len(self.manager.running), 0)

    def test_is_mm_request(self):
        req = Mock(spec=Request)
        req.multimodal_inputs = None
        self.assertFalse(self.manager._is_mm_request(req))
        
        req.multimodal_inputs = {"video_feature_urls": ["url"]}
        self.assertTrue(self.manager._is_mm_request(req))
        
        req.multimodal_inputs = {"images": "img", "image_patch_id": 1, "grid_thw": [[1, 1, 1]]}
        self.assertTrue(self.manager._is_mm_request(req))

    def test_revert_chunked_mm_input(self):
        # Setup mm_inputs with mm_positions
        # Position 1: offset 16, length 32
        pos1 = Mock()
        pos1.offset = 16
        pos1.length = 32
        
        mm_inputs = {"mm_positions": [pos1]}
        
        # Case 1: matched_token_num inside the chunk
        # matched = 20 (inside 16-48). block_size = 16.
        # revert to (16 // 16) * 16 = 16.
        self.assertEqual(self.manager.revert_chunked_mm_input(mm_inputs, 20), 16)
        
        # Case 2: matched_token_num before chunk
        self.assertEqual(self.manager.revert_chunked_mm_input(mm_inputs, 10), 10)
        
        # Case 3: matched_token_num after chunk
        self.assertEqual(self.manager.revert_chunked_mm_input(mm_inputs, 50), 50)

    def test_exist_methods(self):
        req_prefill = Mock(spec=Request)
        req_prefill.task_type = RequestType.PREFILL
        req_prefill.multimodal_inputs = {"images": "a"}
        
        req_decode = Mock(spec=Request)
        req_decode.task_type = RequestType.DECODE
        req_decode.multimodal_inputs = {}
        
        # exist_prefill
        self.assertTrue(self.manager.exist_prefill([req_prefill]))
        self.assertFalse(self.manager.exist_prefill([req_decode]))
        
        # exist_mm_prefill
        # Need to mock _is_mm_request or set up req correctly
        # Since _is_mm_request checks keys, I set images/patch/grid
        req_prefill.multimodal_inputs = {
            "images": [1], 
            "image_patch_id": 1, 
            "grid_thw": [[1, 1, 1]]
        }
        self.assertTrue(self.manager.exist_mm_prefill([req_prefill]))
        
        req_prefill.multimodal_inputs = {}
        self.assertFalse(self.manager.exist_mm_prefill([req_prefill]))

    def test_cache_output_tokens(self):
        self.manager.config.cache_config.enable_prefix_caching = True
        self.manager.config.cache_config.enable_output_caching = True
        
        req = Mock(spec=Request)
        req.num_computed_tokens = 20
        req.need_prefill_tokens = 10
        
        self.manager.cache_output_tokens(req)
        self.manager.cache_manager.cache_output_blocks.assert_called_with(req, 16)

    def test_waiting_async_process(self):
        req = Mock(spec=Request)
        future = Mock()
        req.async_process_futures = [future]
        req.get = Mock(return_value=None) # no error
        
        # Not done
        future.done.return_value = False
        self.assertTrue(self.manager.waiting_async_process(req))
        
        # Done, no error
        future.done.return_value = True
        self.assertFalse(self.manager.waiting_async_process(req))
        self.assertEqual(len(req.async_process_futures), 0)
        
        # Done, with error
        req.async_process_futures = [future]
        req.get = Mock(return_value="error")
        self.assertIsNone(self.manager.waiting_async_process(req))

    @patch("fastdeploy.engine.sched.resource_manager_v1.init_bos_client")
    @patch("fastdeploy.engine.sched.resource_manager_v1.download_from_bos")
    def test_download_features(self, mock_download, mock_init_bos):
        req = Mock(spec=Request)
        req.request_id = "req_dl"
        req.multimodal_inputs = {"video_feature_urls": ["url"]}
        req.error_message = None
        
        # Success
        mock_init_bos.return_value = "client"
        mock_download.return_value = [(True, "feature")]
        
        self.manager._download_features(req)
        self.assertEqual(req.multimodal_inputs["video_features"], ["feature"])
        
        # Failure
        req.multimodal_inputs = {"image_feature_urls": ["url"]}
        mock_download.return_value = [(False, "error")]
        self.manager._download_features(req)
        self.assertEqual(req.error_message, "request req_dl download features error: error")
        self.assertEqual(req.error_code, 530)
        
        # Client Init Failure
        self.manager.bos_client = None
        mock_init_bos.side_effect = Exception("init error")
        self.manager._download_features(req)
        self.assertIn("init bos client error", req.error_message)

    def test_get_available_position(self):
        self.manager.stop_flags = [False, True, False, False]
        # Position 1 is True (stopped/free)
        self.assertEqual(self.manager.get_available_position(), 1)
        
        # All used
        self.manager.stop_flags = [False] * 4
        with self.assertRaises(RuntimeError):
            self.manager.get_available_position()

    def test_get_real_bsz(self):
        self.manager.max_num_seqs = 4
        self.manager.stop_flags = [False, False, True, True] 
        # 0 and 1 are False (running). 2 and 3 are True (free).
        # Should return 2 (index 1 + 1)
        self.assertEqual(self.manager.get_real_bsz(), 2)

    def test_pre_recycle_resource(self):
        req = Mock(spec=Request)
        req.idx = 0
        req.request_id = "req_pre_rec"
        req.block_tables = [1]
        req.extend_block_tables = []
        
        self.manager.requests["req_pre_rec"] = req
        self.manager.tasks_list[0] = req
        
        self.manager.pre_recycle_resource("req_pre_rec")
        
        self.assertIsNone(self.manager.tasks_list[0])
        self.assertTrue(self.manager.stop_flags[0])
        self.assertNotIn("req_pre_rec", self.manager.requests)

    def test_add_request_in_p(self):
        req = Mock(spec=Request)
        self.manager.add_request_in_p([req])
        self.assertIn(req, self.manager.running)

    def test_has_resource_for_prefilled_req(self):
        self.manager.config.scheduler_config.splitwise_role = "decode"
        req = Mock(spec=Request)
        req.disaggregate_info = {"block_tables": [1, 2]}
        self.manager.preallocated_reqs["req_check"] = req
        
        # Case 1: Has resource
        self.manager.stop_flags = [True] # available batch > 0
        self.manager.cache_manager.can_allocate_gpu_blocks.return_value = True
        self.assertTrue(self.manager.has_resource_for_prefilled_req("req_check"))
        
        # Case 2: No batch
        self.manager.stop_flags = [False] * 4
        self.assertFalse(self.manager.has_resource_for_prefilled_req("req_check"))
        
    def test_clear_data(self):
        self.manager.waiting.append(Mock())
        self.manager.to_be_rescheduled_request_id_set.add("req")
        self.manager.clear_data()
        self.assertEqual(len(self.manager.waiting), 0)
        self.assertEqual(len(self.manager.to_be_rescheduled_request_id_set), 0)

    def test_log_status(self):
        # Just ensure it doesn't crash
        self.manager.log_status()

    def test_schedule_with_extend_blocks(self):
        # Setup a running request that wants to extend
        req = Mock(spec=Request)
        req.request_id = "req_extend"
        req.idx = 0
        req.status = RequestStatus.RUNNING
        req.need_prefill_tokens = 10
        req.num_computed_tokens = 20 # > need_prefill => decoding
        req.num_total_tokens = 20
        req.block_tables = [1]
        req.use_extend_tables = True
        
        self.manager.running = [req]
        self.manager.requests = {"req_extend": req}
        self.manager.need_block_num_signal.value[0] = 5 # Signal needs 5 blocks
        self.manager.cache_manager.can_allocate_gpu_blocks.return_value = True
        self.manager.cache_manager.allocate_gpu_blocks.return_value = [2]
        
        scheduled, errors = self.manager.schedule()
        
        # Let's verify extend task presence
        extend_tasks = [t for t in scheduled if isinstance(t, ScheduledExtendBlocksTask)]
        self.assertTrue(len(extend_tasks) >= 1)
        self.assertIn("req_extend", self.manager.using_extend_tables_req_id)

    def test_get_num_new_tokens(self):
        req = Mock(spec=Request)
        req.need_prefill_tokens = 100
        req.num_computed_tokens = 10
        req.multimodal_inputs = {}
        req.with_image = False
        req.prompt_token_ids = [0] * 10
        req.output_token_ids = []
        self.manager.config.model_config.enable_mm = False

        # Case 1: Simple text, no budget limit
        tokens = self.manager._get_num_new_tokens(req, 200)
        self.assertEqual(tokens, 90)

        # Case 2: Budget limit
        tokens = self.manager._get_num_new_tokens(req, 50)
        self.assertEqual(tokens, 50)

        # Case 3: MM enabled but no MM inputs
        self.manager.config.model_config.enable_mm = True
        tokens = self.manager._get_num_new_tokens(req, 200)
        self.assertEqual(tokens, 90)

        # Case 4: MM enabled with patch_idx/patch_map
        req.prompt_token_ids = [0] * 10
        req.multimodal_inputs = {
            "patch_idx": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "patch_map": [
                {"modal_id": 0, "end_idx": 2, "image_num": 0, "video_num": 0}, # text
                {"modal_id": 1, "end_idx": 4, "image_num": 0, "video_num": 0}, # image
                {"modal_id": 2, "end_idx": 6, "image_num": 1, "video_num": 0}, # video
                {"modal_id": 3, "end_idx": 8, "image_num": 1, "video_num": 1}, # audio
                {"modal_id": 0, "end_idx": 10, "image_num": 1, "video_num": 1}, # text
            ],
            "image_patch_id": 100,
            "video_patch_id": 101,
            "audio_patch_id": 102,
            "image_end_id": 200,
            "video_end_id": 201,
            "audio_end_id": 202,
            "tts": False
        }
        
        # Test range covering image
        # computed=2, budget=2 (covers indices 2, 3 -> patch 1 (image))
        req.num_computed_tokens = 2
        tokens = self.manager._get_num_new_tokens(req, 2)
        self.assertEqual(tokens, 2)
        # Expected updates on req
        self.assertEqual(req.image_start, 0) 
        self.assertEqual(req.image_end, 1)

    def test_update_mm_hashes(self):
        req = Mock(spec=Request)
        req.multimodal_inputs = None
        
        # Case 1: No multimodal inputs
        self.manager._update_mm_hashes(req) # Should not crash
        
        # Case 2: Basic image inputs
        req.multimodal_inputs = {
            "images": [1], 
            "image_patch_id": 1, 
            "grid_thw": [[1, 1, 1]],
            "mm_positions": [Mock()],
            "mm_hashes": ["hash"]
        }
        self.manager._update_mm_hashes(req)
        # Should stay same for t=1
        self.assertEqual(len(req.multimodal_inputs["grid_thw"]), 1)
        
        # Case 3: Video inputs (t > 1)
        # t=2, h=1, w=1
        pos_mock = Mock()
        pos_mock.offset = 0
        req.multimodal_inputs = {
            "images": list(range(100)), # mock data
            "image_patch_id": 1,
            "grid_thw": [[2, 1, 1]], # t=2, split into 2 frames
            "mm_positions": [pos_mock],
            "mm_hashes": ["hash1"],
            "mm_num_token_func": Mock(return_value=10)
        }
        
        # Mock MultimodalHasher
        with patch("fastdeploy.multimodal.hasher.MultimodalHasher.hash_features", return_value="new_hash"):
             self.manager._update_mm_hashes(req)
             
        # t=2 => split into 2 entries of [2, 1, 1] ? 
        # Code says: grid_thw.extend([[2, h, w]] * (t // 2))
        # if t=2, t//2 = 1. extends 1 entry.
        self.assertEqual(len(req.multimodal_inputs["grid_thw"]), 1)
        self.assertEqual(len(req.multimodal_inputs["mm_positions"]), 1)
        self.assertEqual(len(req.multimodal_inputs["mm_hashes"]), 1)

    def test_finish_requests_async(self):
        req_ids = ["req1"]
        self.manager.finish_requests = Mock()
        future = self.manager.finish_requests_async(req_ids)
        # It returns a future from ThreadPoolExecutor
        self.assertTrue(hasattr(future, "done"))

    def test_get_can_schedule_prefill_threshold_block(self):
        req = Mock(spec=Request)
        req.need_prefill_tokens = 100
        # block_size = 16
        
        # Case 1: Relaxed strategy
        self.manager.can_relax_prefill_strategy = True
        self.assertEqual(self.manager._get_can_schedule_prefill_threshold_block(req, 5), 5)
        
        # Case 2: Strict strategy
        self.manager.can_relax_prefill_strategy = False
        self.manager.running = [Mock(), Mock()] # 2 running
        self.manager.current_reserve_output_block_num = 2
        
        # threshold = (100 + 16 - 1) // 16 + 2 * 2 = 7 + 4 = 11
        self.assertEqual(self.manager._get_can_schedule_prefill_threshold_block(req, 5), 11)
        
        # Case 3: Speculative method
        self.manager.config.speculative_config.method = "mtp"
        # min(11 + 1, max_block_num_per_seq)
        self.assertEqual(self.manager._get_can_schedule_prefill_threshold_block(req, 5), 12)

    @patch("fastdeploy.engine.sched.resource_manager_v1.current_platform")
    def test_schedule_intel_hpu_logic(self, mock_platform):
        mock_platform.is_intel_hpu.return_value = True
        
        req = Mock(spec=Request)
        req.status = RequestStatus.WAITING
        req.need_prefill_tokens = 32
        req.num_computed_tokens = 0
        req.multimodal_inputs = {}
        req.async_process_futures = []
        req.get = Mock(return_value=False)
        req.block_tables = []
        
        self.manager.add_request(req)
        
        # Budget < block_size (16). 
        # need_prefill - computed = 32 >= 16.
        # token_budget = 10
        self.manager.config.scheduler_config.max_num_batched_tokens = 10
        
        scheduled, errors = self.manager.schedule()
        # Should continue (skip)
        self.assertEqual(len(scheduled), 0)
        self.assertEqual(len(self.manager.running), 0)

    @patch("fastdeploy.engine.sched.resource_manager_v1.ErnieArchitectures")
    def test_schedule_ernie5_arch(self, mock_ernie):
        mock_ernie.is_ernie5_arch.return_value = True
        self.manager.config.model_config.architectures = ["Ernie5"]
        
        req = Mock(spec=Request)
        req.status = RequestStatus.WAITING
        req.multimodal_inputs = {"images": "img"} # Make it MM
        req.async_process_futures = []
        req.get = Mock(return_value=False)
        req.task_type = RequestType.PREFILL # Should be PREFILL for exist_mm_prefill check
        
        self.manager.add_request(req)
        
        # To trigger 'get_enough_request' returning True, we need another request already scheduled that is MM PREFILL.
        
        # Let's add two requests.
        req1 = Mock(spec=Request)
        req1.request_id = "req1"
        req1.status = RequestStatus.WAITING
        req1.multimodal_inputs = {"images": "img"}
        req1.task_type = RequestType.PREFILL
        req1.async_process_futures = []
        req1.get = Mock(return_value=False)
        req1.block_tables = []
        req1.need_prefill_tokens = 10
        req1.num_computed_tokens = 0
        
        req2 = Mock(spec=Request)
        req2.request_id = "req2"
        req2.status = RequestStatus.WAITING
        req2.multimodal_inputs = {"images": "img"}
        req2.task_type = RequestType.PREFILL
        req2.async_process_futures = []
        req2.get = Mock(return_value=False)
        req2.block_tables = []
        req2.need_prefill_tokens = 10
        req2.num_computed_tokens = 0
        
        self.manager.waiting.clear()
        self.manager.waiting.append(req1)
        self.manager.waiting.append(req2)
        
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[1])
        
        # When scheduling, first one is scheduled.
        # Second one: get_enough_request should return True (because req1 is in scheduled_reqs and is MM PREFILL).
        # So second one should be skipped.
        
        scheduled, errors = self.manager.schedule()
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0].request_id, "req1")
        # req2 remains in waiting? No, logic says 'break'.
        # "if get_enough_request(request, scheduled_reqs): break"
        # So loop terminates.
        self.assertEqual(len(self.manager.waiting), 1) 
        self.assertEqual(self.manager.waiting[0].request_id, "req2")

    def test_get_prefix_cached_blocks_revert(self):
        self.manager.config.cache_config.enable_prefix_caching = True
        
        req = Mock(spec=Request)
        req.request_id = "req_revert"
        req.status = RequestStatus.WAITING
        req.need_prefill_tokens = 32
        req.num_computed_tokens = 0
        req.multimodal_inputs = {}
        req.async_process_futures = []
        req.get = Mock(return_value=False)
        req.metrics = Mock()
        
        # Mock match returning more tokens than computed (due to mm chunking logic or otherwise)
        # block_size=16. 2 blocks.
        self.manager.cache_manager.request_match_blocks = Mock(return_value=([1, 2], 32, {
            "match_gpu_block_ids": [1, 2],
            "gpu_match_token_num": 32,
            "cpu_match_token_num": 0,
            "storage_match_token_num": 0,
            "gpu_recv_block_ids": [],
            "match_storage_block_ids": [],
            "cpu_cache_prepare_time": 0,
            "storage_cache_prepare_time": 0
        }))
        
        # Simulate revert_chunked_mm_input reducing computed tokens
        with patch.object(self.manager, "revert_chunked_mm_input", return_value=16):
            success = self.manager.get_prefix_cached_blocks(req)
            
        self.assertTrue(success)
        self.assertEqual(req.num_cached_tokens, 32)
        self.assertEqual(req.num_computed_tokens, 16)
        # revert_tokens_num = 16. 1 block.
        # Should have updated metrics to reduce gpu_match_token_num by 16.
        self.assertEqual(req.metrics.gpu_cache_token_num, 16)

    def test_schedule_waiting_preempted(self):
        req = Mock(spec=Request)
        req.request_id = "req_preempted"
        req.status = RequestStatus.PREEMPTED
        req.need_prefill_tokens = 10
        req.num_total_tokens = 10
        req.num_computed_tokens = 0
        req.multimodal_inputs = {}
        req.async_process_futures = []
        req.get = Mock(return_value=False)
        req.block_tables = []
        
        self.manager.waiting.append(req)
        self.manager.cache_manager.can_allocate_gpu_blocks = Mock(return_value=True)
        self.manager.cache_manager.allocate_gpu_blocks = Mock(return_value=[1])
        
        scheduled, errors = self.manager.schedule()
        
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0].request_id, "req_preempted")
        self.assertEqual(req.status, RequestStatus.RUNNING)

    @patch("fastdeploy.engine.sched.resource_manager_v1.current_platform")
    def test_get_num_new_tokens_hpu(self, mock_platform):
        mock_platform.is_intel_hpu.return_value = True
        
        req = Mock(spec=Request)
        req.need_prefill_tokens = 100
        req.num_computed_tokens = 10
        req.multimodal_inputs = {}
        req.with_image = False
        self.manager.config.model_config.enable_mm = False
        
        # token_budget = 50. diff = 90. 90 > 50.
        # budget > block_size (16).
        # num_new_tokens = 50 // 16 * 16 = 3 * 16 = 48.
        
        tokens = self.manager._get_num_new_tokens(req, 50)
        self.assertEqual(tokens, 48)

    def test_schedule_splitwise_prefill_decoding(self):
        self.manager.config.scheduler_config.splitwise_role = "prefill"
        
        req = Mock(spec=Request)
        req.request_id = "req_dec"
        req.status = RequestStatus.RUNNING
        req.need_prefill_tokens = 10
        req.num_computed_tokens = 10 # Decoding
        req.block_tables = [1]
        req.use_extend_tables = False
        self.manager.need_block_num_signal.value[0] = 0
        
        self.manager.running.append(req)
        
        scheduled, errors = self.manager.schedule()
        
        self.assertEqual(len(scheduled), 0)
        # Should remain in running but not be scheduled for decode step
        self.assertEqual(len(self.manager.running), 1)


if __name__ == "__main__":
    unittest.main()
