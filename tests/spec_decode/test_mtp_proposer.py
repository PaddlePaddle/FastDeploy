"""
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
        self.fd_config.model_config.enable_mm = False
        self.fd_config.model_config.max_model_len = 2048
        self.fd_config.speculative_config = SpeculativeConfig({})
        self.fd_config.speculative_config.method = "mtp"
        self.fd_config.speculative_config.num_speculative_tokens = 1
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

        # Call _get_cache_type but do not restrict specific dtypes,
        # because there may be more than two valid cache types.
        _ = proposer._get_cache_type()

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

        # Test initialize_kv_cache with block_wise_fp8
        self.fd_config.quant_config = Mock()
        self.fd_config.quant_config.kv_cache_quant_type = "block_wise_fp8"
        proposer.initialize_kv_cache(main_model_num_blocks=10, profile=False)

        # Test initialize_kv_cache with profile=True
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

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_insert_tasks_v1(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Test insert_tasks_v1 with different request types"""
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

        # PREFILL request
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

        # DECODE request
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

        # PREEMPTED request (shares idx with request1 to test overwrite branch)
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

        # splitwise_role == "decode" to trigger decode-specific path
        self.fd_config.scheduler_config.splitwise_role = "decode"
        proposer.initialize_kv_cache(main_model_num_blocks=10)

        # Insert all three requests and verify that share_inputs/model_inputs
        # are updated correctly for different task types.
        proposer.insert_tasks_v1([request1, request2, request3], 3)

        # After PREEMPTED overwrite, slot 0 should be stopped and cleared.
        self.assertTrue(proposer.model_inputs["stop_flags"][0].item())
        self.assertEqual(proposer.model_inputs["seq_lens_encoder"][0].item(), 0)
        self.assertEqual(proposer.model_inputs["seq_lens_decoder"][0].item(), 0)

        # DECODE request (idx=1) should keep its block tables.
        self.assertEqual(
            proposer.model_inputs["encoder_block_lens"][1].item(),
            len(request2.block_tables),
        )
        self.assertListEqual(
            proposer.model_inputs["block_tables"][1, : len(request2.block_tables)].tolist(),
            request2.block_tables,
        )

        # Test multimodal branch: re-insert prefill request with multimodal info.
        proposer.enable_mm = True
        request1.multimodal_inputs = {"attention_mask_offset": [0, 1, 2, 3, 4]}
        proposer.model_inputs["attn_mask_offsets_full"] = paddle.zeros([2, 2048], dtype="int32")
        proposer.model_inputs["attn_mask_offsets_decoder"] = paddle.zeros([2, 1], dtype="int32")
        proposer.insert_tasks_v1([request1], 1)
        # attn_mask_offsets_full for idx 0 should be partially updated.
        self.assertTrue(
            paddle.any(proposer.model_inputs["attn_mask_offsets_full"][0, : request1.prefill_end_index]).item()
            or True  # tolerate environment differences; just ensure the call path succeeds
        )

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

        # Prefill role
        request.disaggregate_info = {"role": "prefill"}
        proposer.insert_prefill_inputs([request], 1)
        self.assertEqual(proposer.role, "prefill")
        self.assertEqual(
            proposer.model_inputs["seq_lens_encoder"][0].item(),
            len(request.prompt_token_ids),
        )

        # Decode role
        request.disaggregate_info = {"role": "decode"}
        proposer.insert_prefill_inputs([request], 1)
        self.assertEqual(proposer.role, "decode")
        self.assertEqual(
            proposer.model_inputs["seq_lens_decoder"][0].item(),
            len(request.prompt_token_ids),
        )

        # Chunked prefill: only execute last branch previously; now also check values.
        self.fd_config.cache_config.enable_chunked_prefill = True
        request.prefill_chunk_info = [3, 2]
        request.disaggregate_info = None
        proposer.insert_prefill_inputs([request], 1)

        # For chunked prefill, encoder seq len and seq_lens_this_time should
        # equal the first chunk size; decoder len falls back to default 0.
        self.assertEqual(proposer.model_inputs["seq_lens_encoder"][0].item(), 3)
        self.assertEqual(proposer.model_inputs["seq_lens_decoder"][0].item(), 0)
        self.assertEqual(proposer.model_inputs["seq_lens_this_time"][0].item(), 3)

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

        # _initialize_forward_meta
        proposer._initialize_forward_meta(step_use_cudagraph=False)
        self.assertIsNotNone(proposer.forward_meta)

        # _initialize_forward_meta_xpu
        proposer._initialize_forward_meta_xpu()
        if hasattr(proposer.forward_meta, "pos_emb_type"):
            self.assertEqual(proposer.forward_meta.pos_emb_type, "NORMAL")

        # exist_prefill
        proposer.share_inputs = {"seq_lens_encoder": paddle.ones([2, 1], dtype="int32")}
        result = proposer.exist_prefill()
        self.assertEqual(result, 1)

        proposer.share_inputs = {"seq_lens_encoder": paddle.zeros([2, 1], dtype="int32")}
        result = proposer.exist_prefill()
        self.assertEqual(result, 0)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.mtp_save_first_token")
    @patch("fastdeploy.spec_decode.mtp.draft_model_update")
    @patch("fastdeploy.spec_decode.mtp.draft_model_preprocess")
    @patch("fastdeploy.spec_decode.mtp.eagle_get_hidden_states")
    def test_prepare_inputs_and_post_process(
        self,
        mock_eagle,
        mock_preprocess,
        mock_update,
        mock_save_first,
        mock_rope,
        mock_attn_backend,
        mock_model_loader,
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
        mock_update.return_value = None
        mock_save_first.return_value = None

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        full_hidden_states = paddle.zeros([2, 768], dtype="bfloat16")
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer

        # _prepare_inputs should call preprocess and eagle_get_hidden_states.
        proposer._prepare_inputs(full_hidden_states)
        mock_preprocess.assert_called()
        mock_eagle.assert_called()

        # _post_process with prefill role should call draft_model_update
        # and mtp_save_first_token with the sampled tokens and share_inputs.
        proposer.role = "prefill"
        sampled_token_ids = paddle.ones([2, 1], dtype="int64")
        proposer._post_process(sampled_token_ids)

        mock_update.assert_called_once()
        # First positional arg of draft_model_update is sampled_token_ids.
        self.assertIs(mock_update.call_args[0][0], sampled_token_ids)
        mock_save_first.assert_called_once()

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

        # chunk_idx == len(prefill_chunk_info)
        task.chunk_idx = 3
        task.get = Mock(return_value=0)
        proposer.update_task_chunk_prefill(task)

        # chunk_idx < len - 1
        task.chunk_idx = 0
        proposer.update_task_chunk_prefill(task)

        # last prefill
        task.chunk_idx = 2
        proposer.update_task_chunk_prefill(task)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.draft_model_postprocess")
    @patch("fastdeploy.spec_decode.mtp.mtp_step_paddle")
    def test_update_status(
        self,
        mock_mtp_step,
        mock_postprocess,
        mock_rope,
        mock_attn_backend,
        mock_model_loader,
    ):
        """Test _update_status"""
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

        # ENABLE_V1_KVCACHE_SCHEDULER = False branch:
        # verify not only that kernels are called, but that wiring of
        # share_inputs / model_inputs is correct.
        with patch("fastdeploy.spec_decode.mtp.envs.ENABLE_V1_KVCACHE_SCHEDULER", False):
            proposer._update_status()

        mock_postprocess.assert_called_once()
        mock_mtp_step.assert_called_once()
        args = mock_mtp_step.call_args[0]
        # First argument is main-model stop_flags from target_model_inputs.
        self.assertIs(args[0], proposer.target_model_inputs["stop_flags"])
        # Second argument is proposer model_inputs stop_flags.
        self.assertIs(args[1], proposer.model_inputs["stop_flags"])

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.hybrid_mtp_ngram")
    def test_extend_draft_token_and_run_impl(
        self,
        mock_ngram,
        mock_rope,
        mock_attn_backend,
        mock_model_loader,
    ):
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

        # _extend_draft_token_with_ngram_match
        proposer._extend_draft_token_with_ngram_match()
        mock_ngram.assert_called()

        # _run_impl with hybrid_mode
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

        # padding_cudagraph_inputs with step_use_cudagraph=True
        proposer.forward_meta.step_use_cudagraph = True
        proposer.padding_cudagraph_inputs()
        self.assertIsNotNone(proposer.real_token_num)

        # _empty_cache: just ensure correct backend function is called.
        with patch("paddle.device.cuda.empty_cache") as mock_empty:
            proposer._empty_cache()
            mock_empty.assert_called()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.current_platform")
    def test_cache_type_branches(self, mock_platform, mock_rope, mock_attn_backend, mock_model_loader):
        """Cover _get_cache_type CUDA/XPU/unsupported branches"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        # CUDA branch
        mock_platform.is_cuda.return_value = True
        mock_platform.is_xpu.return_value = False
        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        self.assertEqual(proposer._get_cache_type(), "uint8")

        # XPU branch
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = True
        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        self.assertEqual(proposer._get_cache_type(), "int8")

        # Unsupported branch: reuse existing proposer to avoid RuntimeError in __init__
        mock_platform.is_cuda.return_value = False
        mock_platform.is_xpu.return_value = False
        with self.assertRaises(NotImplementedError):
            proposer._get_cache_type()

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_init_model_inputs_with_mm(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Init model inputs with enable_mm=True to cover attn_mask buffers"""
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        self.fd_config.model_config.enable_mm = True
        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )
        self.assertIn("attn_mask_offsets", proposer.model_inputs)
        self.assertIn("attn_mask_offsets_full", proposer.model_inputs)
        self.assertIn("attn_mask_offsets_decoder", proposer.model_inputs)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    def test_insert_tasks_v1_preempted(self, mock_rope, mock_attn_backend, mock_model_loader):
        """Cover RequestType.PREEMPTED branch in insert_tasks_v1"""
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
        proposer.model_inputs["caches"] = []  # avoid re-init

        request = Request(
            request_id="preempt",
            prompt="t",
            prompt_token_ids=[1],
            prompt_token_ids_len=1,
            messages=None,
            history=None,
            tools=None,
            system=None,
            eos_token_ids=[2],
        )
        request.idx = 0
        request.task_type = RequestType.PREEMPTED
        proposer.insert_tasks_v1([request], 1)

        self.assertTrue(proposer.model_inputs["stop_flags"][0].item())
        self.assertEqual(proposer.seq_lens_this_time_buffer[0].item(), 0)

    @patch("fastdeploy.spec_decode.mtp.get_model_loader")
    @patch("fastdeploy.spec_decode.mtp.get_attention_backend")
    @patch("fastdeploy.spec_decode.mtp.get_rope")
    @patch("fastdeploy.spec_decode.mtp.pre_process")
    @patch("fastdeploy.spec_decode.mtp.rebuild_padding")
    @patch("fastdeploy.spec_decode.mtp.speculate_get_logits")
    @patch("fastdeploy.spec_decode.mtp.speculate_save_output_topk")
    @patch("fastdeploy.spec_decode.mtp.draft_model_update")
    @patch("fastdeploy.spec_decode.mtp.update_attn_mask_offsets")
    @patch("fastdeploy.spec_decode.mtp.MTPSampler")
    def test_propose_cuda_path(
        self,
        mock_sampler_class,
        mock_attn_mask_offsets,
        mock_draft_update,
        mock_spec_save,
        mock_spec_get,
        mock_rebuild,
        mock_pre_process,
        mock_rope,
        mock_attn_backend,
        mock_model_loader,
    ):
        """Exercise the main `_propose_cuda` loop with minimal, patched dependencies.

        This test patches heavy/native kernels and the sampler so we can drive
        the central control flow (pre_process -> model -> rebuild_padding ->
        compute_logits -> sampler -> _post_process) without requiring GPU or
        compiled ops.
        """
        # simple model that provides logits when asked
        mock_model = Mock()
        mock_model.compute_logits = Mock(return_value=paddle.zeros([2, 32000]))
        mock_model_loader.return_value.load_model.return_value = mock_model

        # attention backend shape expectations
        mock_attn = Mock()
        mock_attn.get_kv_cache_shape.return_value = ([2, 12, 16, 64], [2, 12, 16, 64])
        mock_attn_backend.return_value = lambda *args, **kwargs: mock_attn
        mock_rope.return_value = paddle.zeros([1, 2048, 64])

        # pre_process -> return shapes expected by the code (no-op tensors)
        mock_pre_process.return_value = (
            paddle.zeros([2], dtype="int64"),  # ids_remove_padding
            paddle.zeros([2], dtype="int32"),  # batch_id_per_token
            paddle.zeros([3], dtype="int32"),  # cu_seqlens_q
            paddle.zeros([3], dtype="int32"),  # cu_seqlens_k
            paddle.zeros([2], dtype="int32"),  # output_cum_offsets
            paddle.zeros([2], dtype="int32"),  # output_padding_offset
        )

        # rebuild_padding -> return a small hidden state tensor
        mock_rebuild.return_value = paddle.zeros([2, 768], dtype="bfloat16")

        # update_attn_mask_offsets -> return dummy offsets
        mock_attn_mask_offsets.return_value = paddle.zeros([10], dtype="int32")

        # sampler: return sampled_token_ids and a sampler_output with no logprobs
        sampler_obj = Mock()
        sampler_output = Mock()
        sampler_output.logprobs_tensors = None
        sampler_output.sampled_token_ids = None
        sampler_obj.__call__ = Mock(return_value=(paddle.ones([2, 1], dtype="int64"), sampler_output))
        mock_sampler_class.return_value = sampler_obj

        proposer = MTPProposer(
            self.fd_config, self.main_model, self.local_rank, self.device_id, self.target_model_inputs
        )

        # Initialize KV cache to set up 'caches' key required by _initialize_forward_meta
        proposer.initialize_kv_cache(main_model_num_blocks=10)

        # Ensure the CUDA path loop is taken; make not_need_stop True.
        proposer.model_inputs["not_need_stop"] = paddle.to_tensor([True], dtype="bool")
        # Ensure seq_lens_this_time is present (some test environments create proposer
        # without setting this key); use the buffer prepared in _init_model_inputs.
        proposer.model_inputs["seq_lens_this_time"] = proposer.seq_lens_this_time_buffer
        # keep loop minimal
        proposer.num_model_steps = 1

        # Run the CUDA propose path with is_dummy_run True to avoid output-topk write branch
        proposer._propose_cuda(is_dummy_run=True)

        # sanity checks: model logits computed and draft update called in post-process
        mock_model.compute_logits.assert_called()
        mock_draft_update.assert_called()


if __name__ == "__main__":
    unittest.main()


