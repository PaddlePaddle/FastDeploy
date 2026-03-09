"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.engine.engine import LLMEngine

# ═══════════════════ Module-level constants ═══════════════════

MB = "fastdeploy.engine.engine"


# ═══════════════════ Helpers ═══════════════════


def _make_cfg(**overrides):
    """Build a minimal cfg-like object matching LLMEngine expectations."""
    model_cfg = SimpleNamespace(
        model="/fake/model",
        model_type="ernie",
        max_model_len=2048,
        num_hidden_layers=2,
        quantization="{}",
        runner="default",
        convert=None,
        override_pooler_config=None,
        logprobs_mode="none",
        max_logprobs=0,
        enable_logprob=False,
        lm_head_fp32=False,
        moe_gate_fp32=False,
        enable_entropy=False,
        model_impl="default",
    )
    parallel_cfg = SimpleNamespace(
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        device_ids="0",
        data_parallel_size=1,
        expert_parallel_size=1,
        chunked_moe_size=0,
        engine_worker_queue_port=[6778],
        enable_expert_parallel=False,
        enable_chunked_moe=False,
        disable_custom_all_reduce=False,
        use_internode_ll_two_stage=False,
        disable_sequence_parallel_moe=False,
        shutdown_comm_group_if_worker_idle=False,
    )
    scheduler_cfg = SimpleNamespace(
        max_num_seqs=256,
        max_num_batched_tokens=4096,
        splitwise_role="mixed",
        name="local",
        enable_overlap_schedule=False,
    )
    cache_cfg = SimpleNamespace(
        num_gpu_blocks_override=None,
        gpu_memory_utilization=0.9,
        block_size=16,
        enc_dec_block_num=0,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        kv_cache_ratio=1.0,
        kvcache_storage_backend=None,
        num_cpu_blocks=0,
        max_encoder_cache=0,
        cache_transfer_protocol="tcp",
        total_block_num=100,
    )
    load_cfg = SimpleNamespace(
        load_strategy="auto",
        rsync_config={},
        dynamic_load_weight=False,
        load_choices="auto",
    )
    speculative_cfg = SimpleNamespace(
        model_type="main",
        to_json_string=lambda: "{}",
    )
    graph_opt_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    structured_outputs_cfg = SimpleNamespace(
        guided_decoding_backend=None,
        logits_processors=None,
        reasoning_parser="none",
        disable_any_whitespace=False,
    )
    early_stop_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    eplb_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    routing_replay_cfg = SimpleNamespace(to_json_string=lambda: "{}")
    plas_attention_cfg = SimpleNamespace(to_json_string=lambda: "{}")

    cfg = SimpleNamespace(
        model_config=model_cfg,
        parallel_config=parallel_cfg,
        scheduler_config=scheduler_cfg,
        cache_config=cache_cfg,
        load_config=load_cfg,
        speculative_config=speculative_cfg,
        graph_opt_config=graph_opt_cfg,
        structured_outputs_config=structured_outputs_cfg,
        early_stop_config=early_stop_cfg,
        eplb_config=eplb_cfg,
        routing_replay_config=routing_replay_cfg,
        plas_attention_config=plas_attention_cfg,
        worker_num_per_node=1,
        master_ip="127.0.0.1",
        host_ip="127.0.0.1",
        ips=None,
        nnode=1,
        register_info=None,
        node_rank=0,
        print=lambda: None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_engine_no_init(**cfg_overrides):
    """Create an LLMEngine instance without running __init__."""
    engine = object.__new__(LLMEngine)
    engine.cfg = _make_cfg(**cfg_overrides)
    engine.running = True
    engine.is_started = False
    engine.do_profile = 0
    engine.engine = MagicMock()
    engine.guided_decoding_checker = None
    engine.ipc_signal_suffix = 6778
    return engine


# ═══════════════════ Tests: _has_guided_input ═══════════════════


class TestHasGuidedInput(unittest.TestCase):
    """Tests for _has_guided_input() — pure request field checking."""

    def setUp(self):
        self.engine = _make_engine_no_init()

    def test_no_guided_fields(self):
        request = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        self.assertFalse(self.engine._has_guided_input(request))

    def test_guided_json(self):
        request = SimpleNamespace(
            guided_json='{"type": "object"}',
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        self.assertTrue(self.engine._has_guided_input(request))

    def test_guided_regex(self):
        request = SimpleNamespace(
            guided_json=None,
            guided_regex=r"\d+",
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        self.assertTrue(self.engine._has_guided_input(request))

    def test_guided_choice(self):
        request = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=["yes", "no"],
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=None,
        )
        self.assertTrue(self.engine._has_guided_input(request))

    def test_structural_tag(self):
        request = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag="<json>",
            guided_grammar=None,
            guided_json_object=None,
        )
        self.assertTrue(self.engine._has_guided_input(request))

    def test_guided_grammar(self):
        request = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar="expr = number | expr '+' expr",
            guided_json_object=None,
        )
        self.assertTrue(self.engine._has_guided_input(request))

    def test_guided_json_object(self):
        request = SimpleNamespace(
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            structural_tag=None,
            guided_grammar=None,
            guided_json_object=True,
        )
        self.assertTrue(self.engine._has_guided_input(request))


# ═══════════════════ Tests: _setting_environ_variables ═══════════════════


class TestSettingEnvironVariables(unittest.TestCase):
    """Tests for _setting_environ_variables() env var string builder."""

    def setUp(self):
        self.engine = _make_engine_no_init()

    def test_returns_string(self):
        result = self.engine._setting_environ_variables()
        self.assertIsInstance(result, str)

    def test_contains_critical_vars(self):
        result = self.engine._setting_environ_variables()
        self.assertIn("OMP_NUM_THREADS=", result)
        self.assertIn("NCCL_ALGO=Ring", result)
        self.assertIn("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python", result)

    def test_splitwise_prefill_adds_flag(self):
        self.engine.cfg.scheduler_config.splitwise_role = "prefill"
        result = self.engine._setting_environ_variables()
        # When splitwise_role is not "mixed", disaggregation flags should be set
        self.assertTrue(
            "FLAGS_use_pd_disaggregation=1" in result or "FLAGS_use_pd_disaggregation_per_chunk=1" in result
        )

    def test_splitwise_decode_adds_flag(self):
        self.engine.cfg.scheduler_config.splitwise_role = "decode"
        result = self.engine._setting_environ_variables()
        self.assertTrue(
            "FLAGS_use_pd_disaggregation=1" in result or "FLAGS_use_pd_disaggregation_per_chunk=1" in result
        )

    def test_mixed_role_no_disagg_flag(self):
        self.engine.cfg.scheduler_config.splitwise_role = "mixed"
        result = self.engine._setting_environ_variables()
        self.assertNotIn("FLAGS_use_pd_disaggregation=1", result)

    def test_dy2st_vars_present(self):
        result = self.engine._setting_environ_variables()
        self.assertIn("SOT_LOG_LEVEL=", result)
        self.assertIn("SOT_UNSAFE_CACHE_FASTPATH=", result)


# ═══════════════════ Tests: _worker_processes_ready ═══════════════════


class TestWorkerProcessesReady(unittest.TestCase):
    """Tests for _worker_processes_ready() signal polling."""

    def setUp(self):
        self.engine = _make_engine_no_init()

    def test_all_ready(self):
        signal_mock = MagicMock()
        signal_mock.value = np.ones(1, dtype=np.int32)
        self.engine.worker_ready_signal = signal_mock
        self.assertTrue(self.engine._worker_processes_ready())

    def test_not_ready(self):
        signal_mock = MagicMock()
        signal_mock.value = np.zeros(1, dtype=np.int32)
        self.engine.worker_ready_signal = signal_mock
        self.assertFalse(self.engine._worker_processes_ready())

    def test_partially_ready_multi_worker(self):
        self.engine.cfg.worker_num_per_node = 4
        signal_mock = MagicMock()
        signal_mock.value = np.array([1, 1, 0, 1], dtype=np.int32)
        self.engine.worker_ready_signal = signal_mock
        self.assertFalse(self.engine._worker_processes_ready())

    def test_all_ready_multi_worker(self):
        self.engine.cfg.worker_num_per_node = 3
        signal_mock = MagicMock()
        signal_mock.value = np.array([1, 1, 1], dtype=np.int32)
        self.engine.worker_ready_signal = signal_mock
        self.assertTrue(self.engine._worker_processes_ready())


# ═══════════════════ Tests: check_health ═══════════════════


class TestCheckHealth(unittest.TestCase):
    """Tests for check_health() worker liveness check."""

    def setUp(self):
        self.engine = _make_engine_no_init()
        self.engine.engine = MagicMock()

    def test_healthy_when_signal_zero(self):
        signal_mock = MagicMock()
        signal_mock.value = np.array([0], dtype=np.float64)
        self.engine.engine.worker_healthy_live_signal = signal_mock

        healthy, msg = self.engine.check_health()
        self.assertTrue(healthy)
        self.assertEqual(msg, "")

    def test_healthy_when_recent(self):
        signal_mock = MagicMock()
        signal_mock.value = np.array([time.time()], dtype=np.float64)
        self.engine.engine.worker_healthy_live_signal = signal_mock

        healthy, msg = self.engine.check_health()
        self.assertTrue(healthy)

    def test_unhealthy_when_stale(self):
        signal_mock = MagicMock()
        # Simulate a stale heartbeat (old timestamp)
        signal_mock.value = np.array([time.time() - 60], dtype=np.float64)
        self.engine.engine.worker_healthy_live_signal = signal_mock

        healthy, msg = self.engine.check_health(time_interval_threashold=30)
        self.assertFalse(healthy)
        self.assertIn("Not Healthy", msg)


# ═══════════════════ Tests: _format_and_add_data ═══════════════════


class TestFormatAndAddData(unittest.TestCase):
    """Tests for _format_and_add_data() request preprocessing."""

    def setUp(self):
        self.engine = _make_engine_no_init()
        # Mock add_requests to capture calls
        self.engine.add_requests = MagicMock()

    def test_generates_request_id_when_missing(self):
        prompts = {"prompt": "Hello"}
        req_id = self.engine._format_and_add_data(prompts)
        self.assertIsNotNone(req_id)
        # Should be a valid UUID
        uuid.UUID(req_id)
        self.assertEqual(prompts["request_id"], req_id)

    def test_preserves_existing_request_id(self):
        existing_id = "my-custom-id-123"
        prompts = {"prompt": "Hello", "request_id": existing_id}
        req_id = self.engine._format_and_add_data(prompts)
        self.assertEqual(req_id, existing_id)

    def test_sets_max_tokens_default(self):
        prompts = {"prompt": "Hello"}
        self.engine._format_and_add_data(prompts)
        self.assertEqual(prompts["max_tokens"], self.engine.cfg.model_config.max_model_len)

    def test_preserves_existing_max_tokens(self):
        prompts = {"prompt": "Hello", "max_tokens": 100}
        self.engine._format_and_add_data(prompts)
        self.assertEqual(prompts["max_tokens"], 100)

    def test_context_extraction(self):
        prompts = {
            "context": [
                {"role": "system", "utterance": "You are helpful"},
                {"role": "user", "utterance": "Hi"},
                {"role": "assistant", "utterance": "Hello!"},
                {"role": "user", "utterance": "Bye"},
            ]
        }
        self.engine._format_and_add_data(prompts)
        self.assertEqual(prompts["system"], "You are helpful")
        self.assertEqual(prompts["prompt"], ["Hi", "Hello!", "Bye"])

    def test_calls_add_requests(self):
        prompts = {"prompt": "test"}
        self.engine._format_and_add_data(prompts)
        self.engine.add_requests.assert_called_once_with(prompts)


# ═══════════════════ Tests: _init_worker_signals ═══════════════════


class TestInitWorkerSignals(unittest.TestCase):
    """Tests for _init_worker_signals() IPC signal creation."""

    def setUp(self):
        self.engine = _make_engine_no_init()

    @patch(f"{MB}.IPCSignal")
    def test_creates_worker_ready_signal(self, mock_ipc):
        mock_ipc.return_value = MagicMock()
        self.engine._init_worker_signals()
        self.assertTrue(hasattr(self.engine, "worker_ready_signal"))

    @patch(f"{MB}.IPCSignal")
    def test_creates_loaded_model_signal(self, mock_ipc):
        mock_ipc.return_value = MagicMock()
        self.engine._init_worker_signals()
        self.assertTrue(hasattr(self.engine, "loaded_model_signal"))

    @patch(f"{MB}.IPCSignal")
    def test_creates_profile_signal_when_profiling(self, mock_ipc):
        mock_ipc.return_value = MagicMock()
        self.engine.do_profile = 1
        self.engine._init_worker_signals()
        self.assertTrue(hasattr(self.engine, "get_profile_block_num_signal"))

    @patch(f"{MB}.IPCSignal")
    def test_no_profile_signal_when_not_profiling(self, mock_ipc):
        mock_ipc.return_value = MagicMock()
        self.engine.do_profile = 0
        self.engine._init_worker_signals()
        self.assertFalse(hasattr(self.engine, "get_profile_block_num_signal"))

    @patch(f"{MB}.IPCSignal")
    def test_creates_cache_manager_signal_with_prefix_caching(self, mock_ipc):
        mock_ipc.return_value = MagicMock()
        self.engine.cfg.cache_config.enable_prefix_caching = True
        self.engine._init_worker_signals()
        self.assertTrue(hasattr(self.engine, "launched_cache_manager_signal"))

    @patch(f"{MB}.IPCSignal")
    def test_creates_dp_signal_when_dp_gt_1(self, mock_ipc):
        mock_ipc.return_value = MagicMock()
        self.engine.cfg.parallel_config.data_parallel_size = 2
        with patch(f"{MB}.envs") as mock_envs:
            mock_envs.FD_ENABLE_MULTI_API_SERVER = False
            self.engine._init_worker_signals()
            self.assertTrue(hasattr(self.engine, "launched_expert_service_signal"))


# ═══════════════════ Tests: _exit_sub_services ═══════════════════


class TestExitSubServices(unittest.TestCase):
    """Tests for _exit_sub_services() cleanup."""

    def setUp(self):
        self.engine = _make_engine_no_init()

    @patch(f"{MB}.os.getpgid", return_value=12345)
    @patch(f"{MB}.os.killpg")
    def test_sets_running_false(self, mock_killpg, mock_getpgid):
        self.engine.worker_ready_signal = MagicMock()
        self.engine.loaded_model_signal = MagicMock()
        self.engine._exit_sub_services()
        self.assertFalse(self.engine.running)

    @patch(f"{MB}.os.getpgid", return_value=12345)
    @patch(f"{MB}.os.killpg")
    def test_clears_signals(self, mock_killpg, mock_getpgid):
        self.engine.worker_ready_signal = MagicMock()
        self.engine.loaded_model_signal = MagicMock()
        self.engine._exit_sub_services()
        self.engine.worker_ready_signal.clear.assert_called_once()
        self.engine.loaded_model_signal.clear.assert_called_once()

    @patch(f"{MB}.os.getpgid", return_value=12345)
    @patch(f"{MB}.os.killpg")
    def test_kills_worker_proc(self, mock_killpg, mock_getpgid):
        self.engine.worker_ready_signal = MagicMock()
        self.engine.loaded_model_signal = MagicMock()
        worker = MagicMock()
        worker.pid = 99999
        self.engine.worker_proc = worker
        self.engine._exit_sub_services()
        mock_killpg.assert_called()

    @patch(f"{MB}.os.getpgid", return_value=12345)
    @patch(f"{MB}.os.killpg")
    def test_clears_profile_signal_if_exists(self, mock_killpg, mock_getpgid):
        self.engine.worker_ready_signal = MagicMock()
        self.engine.loaded_model_signal = MagicMock()
        self.engine.get_profile_block_num_signal = MagicMock()
        self.engine._exit_sub_services()
        self.engine.get_profile_block_num_signal.clear.assert_called_once()

    @patch(f"{MB}.os.getpgid", return_value=12345)
    @patch(f"{MB}.os.killpg")
    def test_closes_zmq_server(self, mock_killpg, mock_getpgid):
        self.engine.worker_ready_signal = MagicMock()
        self.engine.loaded_model_signal = MagicMock()
        zmq = MagicMock()
        self.engine.zmq_server = zmq
        self.engine._exit_sub_services()
        zmq.close.assert_called_once()


# ═══════════════════ Tests: _stop_profile ═══════════════════


class TestStopProfile(unittest.TestCase):
    """Tests for _stop_profile() profiling completion handler."""

    def setUp(self):
        self.engine = _make_engine_no_init()
        self.engine.do_profile = 1

    def test_resets_profile_flag(self):
        signal = MagicMock()
        signal.value = np.array([100], dtype=np.int32)
        self.engine.get_profile_block_num_signal = signal
        self.engine.engine.resource_manager = MagicMock()
        self.engine.cfg.cache_config = MagicMock()

        self.engine._stop_profile()
        self.assertEqual(self.engine.do_profile, 0)

    def test_resets_cache_config(self):
        signal = MagicMock()
        signal.value = np.array([50], dtype=np.int32)
        self.engine.get_profile_block_num_signal = signal
        self.engine.engine.resource_manager = MagicMock()
        self.engine.cfg.cache_config = MagicMock()

        self.engine._stop_profile()
        self.engine.cfg.cache_config.reset.assert_called_once_with(50)
        self.engine.engine.resource_manager.reset_cache_config.assert_called_once()


# ═══════════════════ Tests: _get_generated_result ═══════════════════


class TestGetGeneratedResult(unittest.TestCase):
    """Tests for _get_generated_result() scheduler result retrieval."""

    def test_delegates_to_scheduler(self):
        engine = _make_engine_no_init()
        engine.engine.scheduler = MagicMock()
        engine.engine.scheduler.get_results.return_value = ["result1"]

        result = engine._get_generated_result()
        self.assertEqual(result, ["result1"])
        engine.engine.scheduler.get_results.assert_called_once()


# ═══════════════════ Tests: from_engine_args ═══════════════════


class TestFromEngineArgs(unittest.TestCase):
    """Tests for from_engine_args() factory method."""

    @patch(f"{MB}.EngineService")
    @patch(f"{MB}.main_process_metrics")
    @patch(f"{MB}.tracing")
    def test_creates_engine_from_args(self, mock_trace, mock_metrics, mock_svc):
        """Test that from_engine_args correctly creates an engine."""
        mock_args = MagicMock(spec=["create_engine_config"])
        mock_cfg = _make_cfg()
        mock_args.create_engine_config.return_value = mock_cfg

        engine = LLMEngine.from_engine_args(mock_args)

        self.assertIsInstance(engine, LLMEngine)
        mock_args.create_engine_config.assert_called_once()

    @patch(f"{MB}.EngineService")
    @patch(f"{MB}.main_process_metrics")
    @patch(f"{MB}.tracing")
    def test_profile_flag_when_no_override(self, mock_trace, mock_metrics, mock_svc):
        """Test do_profile is 1 when num_gpu_blocks_override is None."""
        mock_args = MagicMock(spec=["create_engine_config"])
        cfg = _make_cfg()
        cfg.cache_config.num_gpu_blocks_override = None
        mock_args.create_engine_config.return_value = cfg

        engine = LLMEngine.from_engine_args(mock_args)
        self.assertEqual(engine.do_profile, 1)

    @patch(f"{MB}.EngineService")
    @patch(f"{MB}.main_process_metrics")
    @patch(f"{MB}.tracing")
    def test_no_profile_when_override_set(self, mock_trace, mock_metrics, mock_svc):
        """Test do_profile is 0 when num_gpu_blocks_override is set."""
        mock_args = MagicMock(spec=["create_engine_config"])
        cfg = _make_cfg()
        cfg.cache_config.num_gpu_blocks_override = 100
        mock_args.create_engine_config.return_value = cfg

        engine = LLMEngine.from_engine_args(mock_args)
        self.assertEqual(engine.do_profile, 0)


# ═══════════════════ Tests: launch_components ═══════════════════


class TestLaunchComponents(unittest.TestCase):
    """Tests for launch_components() service startup."""

    def setUp(self):
        self.engine = _make_engine_no_init()

    def test_splitwise_starts_receiver(self):
        self.engine.cfg.scheduler_config.splitwise_role = "prefill"
        self.engine.cfg.scheduler_config.name = "splitwise"
        self.engine.engine.split_connector = MagicMock()
        self.engine.engine.scheduler = MagicMock()

        self.engine.launch_components()

        self.assertTrue(hasattr(self.engine, "splitwise_receive_thread"))
        self.engine.engine.scheduler.start.assert_called_once()

    def test_local_scheduler_no_splitwise(self):
        self.engine.cfg.scheduler_config.splitwise_role = "mixed"
        self.engine.cfg.scheduler_config.name = "local"
        self.engine.engine.scheduler = MagicMock()

        self.engine.launch_components()

        self.engine.engine.scheduler.start.assert_not_called()


# ═══════════════════ Tests: add_requests validation ═══════════════════


class TestAddRequestsValidation(unittest.TestCase):
    """Tests for add_requests() input validation paths."""

    def setUp(self):
        self.engine = _make_engine_no_init()
        self.engine.engine.data_processor = MagicMock()
        self.engine.engine.scheduler = MagicMock()

    def _make_mock_request(self, input_len=10, max_tokens=100):
        """Create a mock Request-like object."""
        req = MagicMock()
        req.prompt_token_ids = list(range(input_len))
        req.prompt_token_ids_len = input_len
        req.need_prefill_tokens = input_len
        req.metrics = SimpleNamespace(
            scheduler_recv_req_time=0.0,
            preprocess_start_time=0.0,
            preprocess_end_time=0.0,
        )

        def get_side_effect(key):
            defaults = {
                "max_tokens": max_tokens,
                "min_tokens": 0,
                "request_id": "test-req",
                "stop_seqs_len": None,
            }
            return defaults.get(key, None)

        req.get = MagicMock(side_effect=get_side_effect)
        req.set = MagicMock()
        req.guided_json = None
        req.guided_regex = None
        req.guided_choice = None
        req.structural_tag = None
        req.guided_grammar = None
        req.guided_json_object = None
        return req

    @patch(f"{MB}.Request")
    def test_input_too_long_raises(self, mock_request_cls):
        from fastdeploy.utils import EngineError

        req = self._make_mock_request(input_len=3000)
        mock_request_cls.from_dict.return_value = req
        self.engine.engine.data_processor.process_request.return_value = req
        self.engine.cfg.model_config.max_model_len = 2048

        with self.assertRaises(EngineError):
            self.engine.add_requests({"prompt": "x" * 3000})

    @patch(f"{MB}.Request")
    def test_min_tokens_too_large_raises(self, mock_request_cls):
        from fastdeploy.utils import EngineError

        req = self._make_mock_request(input_len=2000)

        def get_with_min_tokens(key):
            if key == "min_tokens":
                return 100  # 2000 + 100 >= 2048
            if key == "max_tokens":
                return 48
            if key == "stop_seqs_len":
                return None
            return None

        req.get = MagicMock(side_effect=get_with_min_tokens)
        mock_request_cls.from_dict.return_value = req
        self.engine.engine.data_processor.process_request.return_value = req
        self.engine.cfg.model_config.max_model_len = 2048

        with self.assertRaises(EngineError):
            self.engine.add_requests({"prompt": "test"})


if __name__ == "__main__":
    unittest.main()
