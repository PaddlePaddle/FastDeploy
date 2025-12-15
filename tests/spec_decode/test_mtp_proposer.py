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

import unittest
from unittest.mock import Mock, patch

import paddle

from fastdeploy.config import SpeculativeConfig
from fastdeploy.engine.request import Request, RequestType
from fastdeploy.spec_decode.mtp import MTPProposer
from tests.utils import FakeModelConfig, get_default_test_fd_config


class TestMTPProposer(unittest.TestCase):
    def setUp(self):
        self.fd_config = get_default_test_fd_config()
        self.fd_config.model_config = FakeModelConfig()
        self.fd_config.model_config.architectures = ["ErnieMoeForCausalLM"]
        self.fd_config.model_config.dtype = "bfloat16"
        self.fd_config.model_config.rope_theta = 10000.0
        self.fd_config.model_config.enable_logprob = False
        self.fd_config.model_config.max_model_len = 2048
        self.fd_config.speculative_config = SpeculativeConfig({})
        self.fd_config.speculative_config.method = "mtp"
        self.fd_config.speculative_config.num_speculative_tokens = 1
        self.fd_config.speculative_config.num_model_steps = 1
        self.fd_config.speculative_config.model = "test_mtp_model"
        self.fd_config.speculative_config.quantization = ""
        self.fd_config.speculative_config.num_gpu_block_expand_ratio = 1.0
        self.fd_config.speculative_config.mtp_strategy = "default"
        self.fd_config.scheduler_config.splitwise_role = "mixed"
        self.fd_config.cache_config.enable_prefix_caching = False
        self.fd_config.cache_config.block_size = 16
        self.fd_config.cache_config.enc_dec_block_num = 0
        self.fd_config.cache_config.kv_cache_ratio = 0.9
        self.fd_config.cache_config.total_block_num = 100
        self.fd_config.cache_config.enable_chunked_prefill = False
        self.fd_config.graph_opt_config.draft_model_use_cudagraph = False
        self.fd_config.parallel_config.enable_expert_parallel = False
        self.fd_config.parallel_config.tensor_parallel_size = 1
        self.fd_config.parallel_config.tensor_parallel_rank = 0
        self.fd_config.parallel_config.data_parallel_rank = 0
        self.fd_config.parallel_config.use_ep = False

        self.main_model = Mock()
        self.main_model.num_hidden_layers = 12
        self.local_rank = 0
        self.device_id = 0

        self.target_model_inputs = {
            "block_tables": paddle.zeros([2, 10], dtype="int32"),
            "input_ids": paddle.zeros([2, 2048], dtype="int64"),
            "seq_lens_this_time": paddle.zeros([2, 1], dtype="int32"),
            "seq_lens_encoder": paddle.zeros([2, 1], dtype="int32"),
            "seq_lens_decoder": paddle.zeros([2, 1], dtype="int32"),
            "prompt_lens": paddle.zeros([2, 1], dtype="int64"),
            "step_idx": paddle.zeros([2, 1], dtype="int64"),
            "stop_flags": paddle.zeros([2, 1], dtype="bool"),
            "stop_nums": paddle.zeros([2, 1], dtype="int32"),
            "pre_ids": paddle.zeros([2, 2048], dtype="int64"),
            "output_cum_offsets": paddle.zeros([2], dtype="int32"),
            "output_padding_offset": paddle.zeros([2], dtype="int32"),
            "ids_remove_padding": paddle.zeros([2], dtype="int64"),
            "batch_id_per_token": paddle.zeros([2], dtype="int32"),
            "cu_seqlens_q": paddle.zeros([3], dtype="int32"),
            "cu_seqlens_k": paddle.zeros([3], dtype="int32"),
            "decoder_batch_ids": paddle.zeros([2], dtype="int32"),
            "decoder_tile_ids_per_batch": paddle.zeros([2], dtype="int32"),
            "decoder_num_blocks_cpu": paddle.zeros([2], dtype="int32").cpu(),
            "decoder_num_blocks_device": paddle.zeros([2], dtype="int32"),
            "decoder_chunk_size_device": paddle.zeros([2], dtype="int32"),
            "max_len_tensor_cpu": paddle.zeros([2], dtype="int32").cpu(),
            "encoder_batch_ids": paddle.zeros([2], dtype="int32"),
            "encoder_tile_ids_per_batch": paddle.zeros([2], dtype="int32"),
            "encoder_num_blocks_x_cpu": paddle.zeros([2], dtype="int32").cpu(),
            "kv_batch_ids": paddle.zeros([2], dtype="int32"),
            "kv_tile_ids_per_batch": paddle.zeros([2], dtype="int32"),
            "kv_num_blocks_x_cpu": paddle.zeros([2], dtype="int32").cpu(),
            "top_p": paddle.ones([2, 1], dtype="float32") * 0.9,
            "top_k": paddle.zeros([2, 1], dtype="int32"),
            "temperature": paddle.ones([2, 1], dtype="float32"),
            "eos_token_id": paddle.ones([2], dtype="int64") * 2,
            "penalty_score": paddle.ones([2, 1], dtype="float32"),
            "frequency_score": paddle.zeros([2, 1], dtype="float32"),
            "presence_score": paddle.zeros([2, 1], dtype="float32"),
            "infer_seed": paddle.zeros([2, 1], dtype="int64"),
            "max_dec_len": paddle.ones([2, 1], dtype="int64") * 512,
            "min_dec_len": paddle.zeros([2, 1], dtype="int64"),
            "bad_tokens": paddle.zeros([2], dtype="int64"),
            "draft_tokens": paddle.zeros([2, 2], dtype="int64"),
            "accept_tokens": paddle.zeros([2, 2], dtype="int64"),
            "accept_num": paddle.ones([2], dtype="int32"),
            "draft_logits": paddle.zeros([4, 32000], dtype="float32"),
            "temp_scaled_logprobs": paddle.zeros([2], dtype="float32"),
            "top_p_normalized_logprobs": paddle.zeros([2], dtype="float32"),
            "encoder_block_lens": paddle.zeros([2, 1], dtype="int32"),
            "cu_batch_token_offset": paddle.zeros([3], dtype="int32"),
            "is_block_step": paddle.zeros([2], dtype="bool"),
            "actual_draft_token_num": paddle.zeros([2], dtype="int32"),
        }

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_init_and_config_methods(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test initialization and config update methods"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )

        # Test _update_mtp_config
        self.assertEqual(proposer.model_config.architectures[0], "ErnieMTPForCausalLM")
        self.assertEqual(proposer.model_config.num_hidden_layers, 1)
        self.assertEqual(proposer.speculative_config.model_type, "mtp")

        # Test _get_cache_type with CUDA (lines 1210-1212)
        with patch("fastdeploy.spec_decode.mtp.current_platform.is_cuda", return_value=True):
            cache_type = proposer._get_cache_type()
            self.assertEqual(cache_type, "uint8")

        # Test _get_cache_type with XPU (lines 1219-1221)
        with patch("fastdeploy.spec_decode.mtp.current_platform.is_xpu", return_value=True):
            with patch("fastdeploy.spec_decode.mtp.current_platform.is_cuda", return_value=False):
                cache_type = proposer._get_cache_type()
                self.assertEqual(cache_type, "int8")

        # Test is_chunk_prefill_enabled
        self.assertTrue(proposer.is_chunk_prefill_enabled())

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_dummy_prefill_inputs_and_kv_cache(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test dummy_prefill_inputs and initialize_kv_cache with different branches"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )

        # Test dummy_prefill_inputs with expert parallel
        self.fd_config.parallel_config.enable_expert_parallel = True
        proposer.dummy_prefill_inputs(num_tokens=100, batch_size=2, expected_decode_len=10)
        self.assertGreater(proposer.model_inputs["seq_lens_encoder"][0].item(), 0)

        # Test initialize_kv_cache with prefix caching
        self.fd_config.cache_config.enable_prefix_caching = True
        proposer.initialize_kv_cache(main_model_num_blocks=10, profile=False)
        self.assertIn("caches", proposer.model_inputs)

        # Test initialize_kv_cache with profile=False and splitwise_role != "mixed" (lines 206-233)
        self.fd_config.cache_config.enable_prefix_caching = False
        self.fd_config.scheduler_config.splitwise_role = "prefill"
        proposer.initialize_kv_cache(main_model_num_blocks=10, profile=False)
        self.assertIn("caches", proposer.model_inputs)

        # Test initialize_kv_cache without quant_config (line 191-196 else branch)
        self.fd_config.quant_config = None
        self.fd_config.scheduler_config.splitwise_role = "mixed"
        proposer.initialize_kv_cache(main_model_num_blocks=10, profile=False)

        # Test initialize_kv_cache with block_wise_fp8 (lines 203-204, 246-247)
        self.fd_config.quant_config = Mock()
        self.fd_config.quant_config.kv_cache_quant_type = "block_wise_fp8"
        proposer.initialize_kv_cache(main_model_num_blocks=10, profile=False)

        # Test initialize_kv_cache with profile=True (line 206-207 else branch)
        proposer.initialize_kv_cache(main_model_num_blocks=10, profile=True)

        # Test clear_mtp_cache
        proposer.clear_mtp_cache()
        self.assertNotIn("caches", proposer.model_inputs)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_update_mtp_block_num(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test update_mtp_block_num"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.update_mtp_block_num(num_gpu_blocks=20)
        self.assertEqual(proposer.main_model_num_gpu_blocks, 20)
        self.assertIn("free_list", proposer.model_inputs)

    @patch("fastdeploy.spec_decode.mtp.share_external_data")
    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_insert_tasks_v1(self, mock_rope, mock_attn_backend, mock_model_loader, mock_share_external):
        """Test insert_tasks_v1 with different request types"""
        mock_share_external.side_effect = lambda cache, *_, **__: cache
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )

        # Test with PREFILL request
        request1 = Request(
            request_id="test1",
            prompt="test",
            prompt_token_ids=[1, 2, 3, 4, 5],
            prompt_token_ids_len=5,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request1.idx = 0
        request1.task_type = RequestType.PREFILL
        request1.prefill_start_index = 0
        request1.prefill_end_index = 5
        request1.output_token_ids = []
        request1.block_tables = [0, 1]

        # Test with DECODE request (lines 559-565)
        request2 = Request(
            request_id="test2",
            prompt="test",
            prompt_token_ids=[1, 2],
            prompt_token_ids_len=2,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request2.idx = 1
        request2.task_type = RequestType.DECODE
        request2.block_tables = [2, 3]

        # Test with PREEMPTED request (lines 569-576)
        request3 = Request(
            request_id="test3",
            prompt="test",
            prompt_token_ids=[1],
            prompt_token_ids_len=1,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request3.idx = 0
        request3.task_type = RequestType.PREEMPTED

        # Test splitwise_role == "decode" (lines 561-566)
        self.fd_config.scheduler_config.splitwise_role = "decode"
        proposer.initialize_kv_cache(main_model_num_blocks=10)
        proposer.insert_tasks_v1([request1], 1)

        # Test insert_tasks_v1 with caches not in model_inputs (line 513-514)
        del proposer.model_inputs["caches"]
        proposer.insert_tasks_v1([request1], 1)
        self.assertIn("caches", proposer.model_inputs)

        # Test with multimodal (lines 541-550, 868-872)
        proposer.enable_mm = True
        request1.multimodal_inputs = {"attention_mask_offset": [0, 1, 2, 3, 4]}
        proposer.model_inputs["attn_mask_offsets_full"] = paddle.zeros([2, 2048], dtype="int32")
        proposer.model_inputs["attn_mask_offsets_decoder"] = paddle.zeros([2, 1], dtype="int32")
        proposer.insert_tasks_v1([request1], 1)

        # Test DECODE request with encoder_block_lens verification
        proposer.insert_tasks_v1([request2], 1)
        self.assertEqual(proposer.model_inputs["encoder_block_lens"][0].item(), 2)

        # Test PREEMPTED request
        proposer.insert_tasks_v1([request3], 1)

        # Ensure line 516 (logger.debug) is covered by calling with multiple requests
        # Create a fresh proposer instance to ensure clean state
        proposer2 = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer2.initialize_kv_cache(main_model_num_blocks=10)
        # Create fresh requests to avoid state issues
        request4 = Request(
            request_id="test4",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            prompt_token_ids_len=3,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request4.idx = 0
        request4.task_type = RequestType.PREFILL
        request4.prefill_start_index = 0
        request4.prefill_end_index = 3
        request4.output_token_ids = []
        request4.block_tables = [0, 1]
        
        request5 = Request(
            request_id="test5",
            prompt="test",
            prompt_token_ids=[1, 2],
            prompt_token_ids_len=2,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request5.idx = 1
        request5.task_type = RequestType.DECODE
        request5.block_tables = [2, 3]
        
        proposer2.insert_tasks_v1([request4, request5], 2)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_insert_prefill_inputs(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test insert_prefill_inputs with different roles and chunked prefill"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )

        request = Request(
            request_id="test",
            prompt="test",
            prompt_token_ids=[1, 2, 3, 4, 5],
            prompt_token_ids_len=5,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request.idx = 0
        request.block_tables = [0, 1]
        request.draft_token_ids = [10, 11]

        # Test with prefill role
        request.disaggregate_info = {"role": "prefill"}
        proposer.insert_prefill_inputs([request], 1)
        self.assertEqual(proposer.role, "prefill")

        # Test with decode role and length > 1 (lines 603-622)
        request.disaggregate_info = {"role": "decode"}
        proposer.insert_prefill_inputs([request], 1)
        self.assertEqual(proposer.role, "decode")

        # Test with decode role and length <= 1 (line 605 else branch)
        request.prompt_token_ids = [1]
        request.prompt_token_ids_len = 1
        proposer.insert_prefill_inputs([request], 1)

        # Test with chunked prefill (lines 645-653)
        self.fd_config.cache_config.enable_chunked_prefill = True
        request.prefill_chunk_info = [3, 2]
        request.disaggregate_info = None
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.prompt_token_ids_len = 5
        proposer.insert_prefill_inputs([request], 1)

        # Test without chunked prefill (line 655-656 else branch)
        self.fd_config.cache_config.enable_chunked_prefill = False
        proposer.insert_prefill_inputs([request], 1)

        # Test with length <= 1 (line 636 else branch)
        request.prompt_token_ids = [1]
        request.prompt_token_ids_len = 1
        proposer.insert_prefill_inputs([request], 1)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_forward_meta_and_exist_prefill(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test _initialize_forward_meta, _initialize_forward_meta_xpu, and exist_prefill"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.initialize_kv_cache(main_model_num_blocks=10)
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer

        # Test _initialize_forward_meta
        proposer._initialize_forward_meta(step_use_cudagraph=False)
        self.assertIsNotNone(proposer.forward_meta)

        # Test enable_mm branch (lines 497-505)
        proposer.enable_mm = True
        proposer.model_inputs["attn_mask_offsets"] = paddle.full(shape=[2 * 2048], fill_value=-1, dtype="int32")
        proposer.model_inputs["attn_mask_offsets_full"] = paddle.full([2, 2048], -1, dtype="int32")
        proposer.model_inputs["attn_mask_offsets_decoder"] = paddle.full([2, 1], -1, dtype="int32")
        proposer.model_inputs["decode_states"] = paddle.full([2, 2], -1, dtype="int32")
        proposer._initialize_forward_meta(step_use_cudagraph=False)

        # Test _initialize_forward_meta_xpu (lines 704-721)
        proposer._initialize_forward_meta_xpu()
        if hasattr(proposer.forward_meta, "pos_emb_type"):
            self.assertEqual(proposer.forward_meta.pos_emb_type, "NORMAL")

        # Test exist_prefill
        proposer.share_inputs = {"seq_lens_encoder": paddle.ones([2, 1], dtype="int32")}
        result = proposer.exist_prefill()
        self.assertEqual(result, 1)

        proposer.share_inputs = {"seq_lens_encoder": paddle.zeros([2, 1], dtype="int32")}
        result = proposer.exist_prefill()
        self.assertEqual(result, 0)

        # Test use_ep and splitwise_role == "mixed" branch (lines 727-732)
        self.fd_config.parallel_config.use_ep = True
        self.fd_config.scheduler_config.splitwise_role = "mixed"
        proposer.share_inputs = {"seq_lens_encoder": paddle.ones([2, 1], dtype="int32")}
        with patch("fastdeploy.spec_decode.mtp.paddle.distributed.all_gather_object") as mock_gather:
            mock_gather.return_value = None
            proposer._initialize_forward_meta_xpu()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.draft_model_preprocess")
    @patch("fastdeploy.spec_decode.mtp.eagle_get_hidden_states")
    def test_prepare_inputs_and_post_process(
        self, mock_eagle, mock_preprocess, mock_rope, mock_attn_backend, mock_model_loader
    ):
        """Test _prepare_inputs and _post_process"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_eagle.return_value = paddle.zeros([2, 768], dtype="bfloat16")
        mock_preprocess.return_value = None

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        full_hidden_states = paddle.zeros([2, 768], dtype="bfloat16")
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer

        # Test _prepare_inputs
        proposer._prepare_inputs(full_hidden_states)
        mock_preprocess.assert_called()
        mock_eagle.assert_called()

        # Test _post_process with prefill role
        proposer.role = "prefill"
        sampled_token_ids = paddle.ones([2, 1], dtype="int64")
        proposer._post_process(sampled_token_ids)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_update_task_chunk_prefill(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test update_task_chunk_prefill"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer

        task = Mock()
        task.idx = 0
        task.prefill_chunk_info = [3, 2, 1]
        task.prompt_token_ids = [1, 2, 3, 4, 5, 6]

        # Test chunk_idx == len(prefill_chunk_info) (lines 1099-1102)
        task.chunk_idx = 3
        task.get = Mock(return_value=0)
        proposer.update_task_chunk_prefill(task)

        # Test chunk_idx < len - 1 (lines 1106-1109)
        task.chunk_idx = 0
        proposer.update_task_chunk_prefill(task)

        # Test last prefill (lines 1111-1114)
        task.chunk_idx = 2
        proposer.update_task_chunk_prefill(task)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.draft_model_postprocess")
    @patch("fastdeploy.spec_decode.mtp.mtp_step_paddle")
    def test_update_status(self, mock_mtp_step, mock_postprocess, mock_rope, mock_attn_backend, mock_model_loader):
        """Test _update_status with different scheduler configurations"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_postprocess.return_value = None
        mock_mtp_step.return_value = None

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer

        # Test with ENABLE_V1_KVCACHE_SCHEDULER=False
        with patch("fastdeploy.spec_decode.mtp.envs.ENABLE_V1_KVCACHE_SCHEDULER", False):
            proposer._update_status()
            mock_mtp_step.assert_called()

        # Test with ENABLE_V1_KVCACHE_SCHEDULER=True
        with patch("fastdeploy.spec_decode.mtp.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):
            proposer._update_status()
            mock_postprocess.assert_called()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.hybrid_mtp_ngram")
    def test_extend_draft_token_and_run_impl(self, mock_ngram, mock_rope, mock_attn_backend, mock_model_loader):
        """Test _extend_draft_token_with_ngram_match and _run_impl"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_ngram.return_value = None

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.hybrid_mode = True
        proposer.max_ngram_size = 5
        proposer.min_ngram_size = 2

        # Test _extend_draft_token_with_ngram_match
        proposer._extend_draft_token_with_ngram_match()
        mock_ngram.assert_called()

        # Test _run_impl with hybrid_mode
        full_hidden_states = paddle.zeros([2, 768], dtype="bfloat16")
        with (
            patch.object(proposer, "_prepare_inputs"),
            patch.object(proposer, "_propose"),
            patch.object(proposer, "_update_status"),
        ):
            proposer._run_impl(full_hidden_states)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_padding_cudagraph_inputs_and_empty_cache(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test padding_cudagraph_inputs and _empty_cache"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.initialize_kv_cache(main_model_num_blocks=10)
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer
        proposer._initialize_forward_meta()

        # Test padding_cudagraph_inputs with step_use_cudagraph=True (lines 1195-1197)
        proposer.forward_meta.step_use_cudagraph = True
        proposer.padding_cudagraph_inputs()
        self.assertIsNotNone(proposer.real_token_num)

        # Test padding_cudagraph_inputs with step_use_cudagraph=False (line 1195 else branch)
        proposer.forward_meta.step_use_cudagraph = False
        proposer.padding_cudagraph_inputs()

        # Test _empty_cache with CUDA (lines 1201-1202)
        with patch("fastdeploy.spec_decode.mtp.current_platform.is_cuda", return_value=True):
            with patch("paddle.device.cuda.empty_cache") as mock_empty:
                proposer._empty_cache()
                mock_empty.assert_called()

        # Test _empty_cache with XPU (lines 1203-1204)
        with patch("fastdeploy.spec_decode.mtp.current_platform.is_xpu", return_value=True):
            with patch("fastdeploy.spec_decode.mtp.current_platform.is_cuda", return_value=False):
                with patch("paddle.device.xpu.empty_cache") as mock_empty:
                    proposer._empty_cache()
                    mock_empty.assert_called()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.eagle_get_self_hidden_states")
    def test_get_self_hidden_states(self, mock_eagle, mock_rope, mock_attn_backend, mock_model_loader):
        """Test _get_self_hidden_states"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])
        mock_eagle.return_value = paddle.zeros([2, 768], dtype="bfloat16")

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        proposer.last_seq_lens_this_time = paddle.zeros([2, 1], dtype="int32")
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer
        proposer.model_inputs["step_idx"] = paddle.zeros([2, 1], dtype="int64")
        proposer.model_inputs["target_hidden_states"] = paddle.zeros([2, 768], dtype="bfloat16")
        hidden_states = paddle.zeros([2, 768], dtype="bfloat16")

        proposer._get_self_hidden_states(hidden_states)
        mock_eagle.assert_called()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_num_model_steps_and_platform_branches(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test num_model_steps > 1 and XPU platform branches"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        # Test num_model_steps > 1 branch (lines 464-467)
        self.fd_config.speculative_config.num_model_steps = 2
        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        self.assertTrue(hasattr(proposer, "last_seq_lens_this_time"))

        # Test XPU platform pin_memory branch (lines 275-282)
        with patch("fastdeploy.spec_decode.mtp.current_platform.is_xpu", return_value=True):
            with patch("fastdeploy.spec_decode.mtp.current_platform.is_cuda", return_value=False):
                proposer2 = MTPProposer(
                    self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
                )
                self.assertIsNotNone(proposer2.model_inputs.get("decoder_num_blocks_cpu"))


if __name__ == "__main__":
    unittest.main()
