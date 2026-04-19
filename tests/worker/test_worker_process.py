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

import logging
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import ControlRequest
from fastdeploy.worker.worker_process import PaddleDisWorkerProc

_WP = "fastdeploy.worker.worker_process"


# fmt: off
def _make_config(**overrides):
    """Build a lightweight FDConfig-compatible namespace."""
    c = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            local_engine_worker_queue_port=9999, tensor_parallel_size=1,
            tensor_parallel_rank=0, data_parallel_size=1, data_parallel_rank=0,
            local_data_parallel_id=0, engine_pid=12345, pod_ip="127.0.0.1",
            use_ep=False, expert_parallel_size=1, tp_group=None,
            shutdown_comm_group_if_worker_idle=False, do_profile=False),
        cache_config=types.SimpleNamespace(num_cpu_blocks=0, total_block_num=100),
        scheduler_config=types.SimpleNamespace(enable_overlap_schedule=False, splitwise_role="mixed"),
        speculative_config=types.SimpleNamespace(method=None),
        eplb_config=types.SimpleNamespace(enable_eplb=False),
        load_config=types.SimpleNamespace(dynamic_load_weight=False),
        model_config=types.SimpleNamespace(
            enable_mm=False, enable_logprob=False, architectures=["LlamaForCausalLM"],
            tensor_model_parallel_size=1, vision_config=types.SimpleNamespace(dtype=None)),
        enable_mm_runtime=False,
        nnode=1)
    for k, v in overrides.items():
        parts = k.split(".")
        obj = c
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return c
# fmt: on


def _create_worker(ranks=1, local_rank=0, **cfg_kw):
    """Construct PaddleDisWorkerProc with mocked platform and get_worker."""
    from fastdeploy.worker.worker_process import PaddleDisWorkerProc

    return PaddleDisWorkerProc(_make_config(**cfg_kw), ranks=ranks, local_rank=local_rank)


_FD_PATCH_NAMES = [
    "v1_loader_support",
    "parse_quant_config",
    "update_fd_config_for_mm",
    "current_platform",
    "paddle",
    "ModelConfig",
    "DeviceConfig",
    "SpeculativeConfig",
    "ParallelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "EPLBConfig",
    "LoadConfig",
    "GraphOptimizationConfig",
    "PlasAttentionConfig",
    "EarlyStopConfig",
    "StructuredOutputsConfig",
    "RoutingReplayConfig",
    "FDConfig",
]


def _setup_fd_env():
    """Prepare (args, ExitStack, mocks) for initialize_fd_config tests."""
    from fastdeploy.worker.worker_process import parse_args

    with patch.object(sys, "argv", ["prog", "-m", "/tmp/m", "--dtype", "float16"]):
        args = parse_args()
    stack = ExitStack()
    m = {n: stack.enter_context(patch(f"{_WP}.{n}")) for n in _FD_PATCH_NAMES}
    for a in ("is_cuda", "is_xpu", "is_maca", "is_iluvatar", "is_intel_hpu"):
        getattr(m["current_platform"], a).return_value = a == "is_cuda"
    mc = m["ModelConfig"].return_value
    mc.num_hidden_layers, mc.architectures = 2, ["LlamaForCausalLM"]
    mc.is_quantized, mc.quantization_config, mc.head_dim = False, None, 128
    mc.pretrained_config = types.SimpleNamespace()
    pc = m["ParallelConfig"].return_value
    pc.tensor_parallel_size = pc.data_parallel_size = pc.expert_parallel_size = 1
    pc.use_ep = False
    lc = m["LoadConfig"].return_value
    lc.dynamic_load_weight, lc.load_strategy, lc.rsync_config = False, "ipc_snapshot", None
    lc.load_choices = "default_v1"
    m["parse_quant_config"].return_value = None
    return args, stack, m


class TestLoggerIntercept(unittest.TestCase):
    """Test cases for the intercept_paddle_loggers context manager."""

    def test_intercept_paddle_loggers_with_paddle_prefix(self):
        """Test intercept_paddle_loggers configures paddle loggers correctly"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        test_logger_name = "paddle.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()
        test_logger.addHandler(handler1)
        test_logger.addHandler(handler2)
        self.assertEqual(len(test_logger.handlers), 2)

        with intercept_paddle_loggers():
            intercepted_logger = logging.getLogger(test_logger_name)
            self.assertEqual(len(intercepted_logger.handlers), 1)
            self.assertIsInstance(intercepted_logger.handlers[0], logging.StreamHandler)
            self.assertEqual(intercepted_logger.level, logging.INFO)
            self.assertFalse(intercepted_logger.propagate)

        test_logger.handlers = []

    def test_intercept_paddle_loggers_restores_original(self):
        from fastdeploy.logger.logger import intercept_paddle_loggers

        original_getLogger = logging.getLogger

        with intercept_paddle_loggers():
            self.assertNotEqual(logging.getLogger, original_getLogger)

        self.assertEqual(logging.getLogger, original_getLogger)

    def test_intercept_paddle_loggers_non_paddle_logger_unchanged(self):
        """Test non-paddle loggers are not affected by intercept_paddle_loggers"""
        from fastdeploy.logger.logger import intercept_paddle_loggers

        test_logger_name = "other.test.logger"
        test_logger = logging.getLogger(test_logger_name)

        original_handler = logging.StreamHandler()
        test_logger.addHandler(original_handler)
        original_handler_count = len(test_logger.handlers)

        with intercept_paddle_loggers():
            result_logger = logging.getLogger(test_logger_name)
            self.assertEqual(len(result_logger.handlers), original_handler_count)
            self.assertEqual(result_logger.handlers[0], original_handler)

        test_logger.handlers = []

    def test_intercept_paddle_loggers_exception_safety(self):
        from fastdeploy.logger.logger import intercept_paddle_loggers

        original_getLogger = logging.getLogger
        try:
            with intercept_paddle_loggers():
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertEqual(logging.getLogger, original_getLogger)


class TestWorkerUtilFunctions(unittest.TestCase):
    """Worker utility functions with real logic and minimal mocking."""

    def test_parse_args(self):
        from fastdeploy.worker.worker_process import parse_args

        with patch.object(sys, "argv", ["prog"]):
            a = parse_args()
        self.assertEqual(a.model, "./output")
        self.assertEqual(a.dtype, "bfloat16")
        argv = [
            "prog",
            "-m",
            "/tmp/m",
            "--dtype",
            "float16",
            "--tensor_parallel_size",
            "4",
            "--speculative_config",
            '{"method":"eagle"}',
            "--eplb_config",
            '{"enable_eplb":true}',
        ]
        with patch.object(sys, "argv", argv):
            a = parse_args()
        self.assertEqual(a.model, "/tmp/m")
        self.assertEqual(a.tensor_parallel_size, 4)
        self.assertEqual(a.speculative_config["method"], "eagle")
        self.assertTrue(a.eplb_config["enable_eplb"])

    def test_get_worker_dispatch(self):
        from fastdeploy.worker.worker_process import get_worker

        all_plats = ("is_dcu", "is_cuda", "is_xpu", "is_iluvatar", "is_gcu", "is_maca", "is_intel_hpu")
        cases = [
            ("is_dcu", "fastdeploy.worker.dcu_worker", "DcuWorker"),
            ("is_cuda", "fastdeploy.worker.gpu_worker", "GpuWorker"),
            ("is_xpu", "fastdeploy.worker.xpu_worker", "XpuWorker"),
            ("is_iluvatar", "fastdeploy.worker.iluvatar_worker", "IluvatarWorker"),
            ("is_gcu", "fastdeploy.worker.gcu_worker", "GcuWorker"),
            ("is_maca", "fastdeploy.worker.metax_worker", "MetaxWorker"),
            ("is_intel_hpu", "fastdeploy.worker.hpu_worker", "HpuWorker"),
        ]
        for plat_name, mod_path, cls_name in cases:
            with self.subTest(platform=plat_name), patch(f"{_WP}.current_platform") as plat:
                for a in all_plats:
                    getattr(plat, a).return_value = a == plat_name
                sentinel = object()
                with patch.dict(
                    "sys.modules", {mod_path: types.SimpleNamespace(**{cls_name: lambda *a, **kw: sentinel})}
                ):
                    self.assertIs(get_worker(_make_config(), local_rank=0, rank=1), sentinel)
        with patch(f"{_WP}.current_platform") as plat:
            for a in all_plats:
                getattr(plat, a).return_value = False
            with self.assertRaises(NotImplementedError):
                get_worker(_make_config(**{"model_config.enable_logprob": True}), local_rank=0, rank=0)

    def test_update_mm(self):
        from fastdeploy.config import ErnieArchitectures
        from fastdeploy.worker.worker_process import update_fd_config_for_mm

        fd = _make_config(
            **{
                "enable_mm_runtime": True,
                "model_config.architectures": ["Ernie4_5ForCausalLM"],
                "parallel_config.tensor_parallel_size": 4,
                "parallel_config.tensor_parallel_rank": 2,
                "model_config.dtype": "float16",
            }
        )
        with patch.object(ErnieArchitectures, "contains_ernie_arch", return_value=True):
            update_fd_config_for_mm(fd)
        self.assertEqual(fd.model_config.tensor_model_parallel_size, 4)
        fd2 = _make_config(**{"enable_mm_runtime": False})
        orig = fd2.model_config.tensor_model_parallel_size
        update_fd_config_for_mm(fd2)
        self.assertEqual(fd2.model_config.tensor_model_parallel_size, orig)

    def test_init_distributed(self):
        from fastdeploy.worker.worker_process import init_distributed_environment

        with patch(f"{_WP}.dist") as dist, patch(f"{_WP}.fleet") as fleet:
            dist.get_world_size.return_value = 2
            fleet.worker_index.return_value = 1
            self.assertEqual(init_distributed_environment(seed=42), (2, 1))
            fleet.init.assert_called_once()
        with patch(f"{_WP}.dist") as dist, patch(f"{_WP}.fleet"):
            dist.get_world_size.return_value = 0
            self.assertEqual(init_distributed_environment(), (0, 0))

    def test_initialize_fd_config(self):
        from fastdeploy.worker.worker_process import initialize_fd_config

        args, stack, m = _setup_fd_env()
        with stack:
            initialize_fd_config(args, ranks=1, local_rank=0)
            m["FDConfig"].assert_called_once()
            m["update_fd_config_for_mm"].assert_called_once()
        args2, stack2, m2 = _setup_fd_env()
        with stack2:
            m2["ParallelConfig"].return_value.expert_parallel_size = 2
            m2["ModelConfig"].return_value.moe_num_experts = 8
            m2["ModelConfig"].return_value.num_local_experts = None
            m2["EPLBConfig"].return_value.redundant_experts_num = 0
            initialize_fd_config(args2, ranks=1, local_rank=0)
        args3, stack3, m3 = _setup_fd_env()
        with stack3:
            m3["ModelConfig"].return_value.num_hidden_layers = None
            with self.assertRaises(ValueError):
                initialize_fd_config(args3, ranks=1, local_rank=0)


class TestPaddleDisWorkerProc(unittest.TestCase):
    """PaddleDisWorkerProc: constructor, health, cache, control, event loop."""

    def setUp(self):
        self._patcher_plat = patch(f"{_WP}.current_platform")
        self._patcher_gw = patch(f"{_WP}.get_worker")
        self.plat = self._patcher_plat.start()
        self.gw = self._patcher_gw.start()
        self.plat.is_iluvatar.return_value = False
        self.plat.is_cuda.return_value = False
        self.plat.is_xpu.return_value = False

    def tearDown(self):
        self._patcher_gw.stop()
        self._patcher_plat.stop()

    def test_init_and_control(self):
        p = _create_worker(ranks=2, local_rank=1)
        self.assertEqual(p.ranks, 2)
        self.assertEqual(p.local_rank, 1)
        self.assertEqual(p.max_chips_per_node, 8)
        self.gw.assert_called_once()
        self.plat.is_iluvatar.return_value = True
        self.gw.reset_mock()
        self.assertEqual(_create_worker().max_chips_per_node, 16)
        self.plat.is_iluvatar.return_value = False
        with patch(f"{_WP}.FMQ") as fmq:
            p4 = _create_worker(**{"parallel_config.local_engine_worker_queue_port": 5555})
            p4.local_rank = 3
            p4.init_control()
            fmq.return_value.queue.assert_called_once_with("ctrl_w2e_rank3_5555", "producer")

    def test_health_and_task_queue(self):
        with patch(f"{_WP}.IPCSignal"), patch(f"{_WP}.time") as t:
            t.time.return_value = 1000.0
            _create_worker(**{"parallel_config.data_parallel_size": 1}).init_health_status()
        with patch(f"{_WP}.envs") as env, patch(f"{_WP}.TaskQueue") as tq:
            env.FD_ENGINE_TASK_QUEUE_WITH_SHM = True
            _create_worker(**{"parallel_config.local_engine_worker_queue_port": 7777}).start_task_queue_service()
            self.assertIn("fd_task_queue_7777", tq.call_args[1]["address"])
        with patch(f"{_WP}.envs") as env, patch(f"{_WP}.TaskQueue") as tq:
            env.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
            _create_worker(
                **{"parallel_config.pod_ip": "10.0.0.1", "parallel_config.local_engine_worker_queue_port": 8888}
            ).start_task_queue_service()
            self.assertEqual(tq.call_args[1]["address"], ("10.0.0.1", 8888))

    def test_load_model_and_graph(self):
        with patch(f"{_WP}.IPCSignal") as ipc:
            ipc.return_value = types.SimpleNamespace(value=np.zeros([1], dtype=np.int32))
            p = _create_worker()
            p.load_model()
            p.worker.load_model.assert_called_once()
            self.assertEqual(p.loaded_model_signal.value[0], 1)
        with patch(f"{_WP}.IPCSignal") as ipc, patch(f"{_WP}.paddle") as pdl:
            ipc.return_value = types.SimpleNamespace(value=np.zeros([1], dtype=np.int32))
            _create_worker(ranks=2).load_model()
            pdl.distributed.barrier.assert_called_once()
        _create_worker().init_device()
        with patch(f"{_WP}.envs") as env:
            env.ENABLE_V1_KVCACHE_SCHEDULER = True
            p2 = _create_worker()
            p2.graph_optimize_and_warm_up_model()
            p2.worker.graph_optimize_and_warm_up_model.assert_called_once()

    def test_kv_cache(self):
        # Static block num path
        p = _create_worker(**{"cache_config.total_block_num": 42})
        p.initialize_kv_cache()
        p.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=42)
        # Profiled path -- normal calculation
        self.gw.return_value.reset_mock()
        with patch(f"{_WP}.IPCSignal"), patch(f"{_WP}.dist"):
            p2 = _create_worker(**{"parallel_config.do_profile": True})
            p2.worker.determine_available_memory.return_value = 1024**3
            p2.worker.cal_theortical_kvcache.return_value = 1024**2
            p2.initialize_kv_cache()
            p2.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=1024)
        # Profiled path -- zero available -> ValueError
        with patch(f"{_WP}.IPCSignal"), patch(f"{_WP}.dist"):
            p3 = _create_worker(**{"parallel_config.do_profile": True})
            p3.worker.determine_available_memory.return_value = 0
            p3.worker.cal_theortical_kvcache.return_value = 1024
            with self.assertRaises(ValueError):
                p3.initialize_kv_cache()
        # Profiled path -- large available memory (no cap since #7241)
        self.gw.return_value.reset_mock()
        with patch(f"{_WP}.IPCSignal"), patch(f"{_WP}.dist"):
            p4 = _create_worker(**{"parallel_config.do_profile": True})
            p4.worker.determine_available_memory.return_value = 100 * 1024**3
            p4.worker.cal_theortical_kvcache.return_value = 1024
            p4.initialize_kv_cache()
            p4.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=100 * 1024**3 // 1024)

    def test_eplb(self):
        p = _create_worker(**{"eplb_config.enable_eplb": False})
        p._init_eplb_signal()
        self.assertFalse(hasattr(p, "experts_manager"))
        p._run_eplb(tp_rank=0)
        with patch(f"{_WP}.RedundantExpertManager") as rem, patch(f"{_WP}.IPCSignal"), patch(f"{_WP}.create_mmap"):
            p2 = _create_worker(
                **{
                    "eplb_config.enable_eplb": True,
                    "model_config.num_hidden_layers": 4,
                    "model_config.moe_num_experts": 8,
                }
            )
            p2._init_eplb_signal()
            rem.assert_called_once()

    def test_barrier_broadcast(self):
        with patch(f"{_WP}.paddle") as pdl:
            p = _create_worker()
            p.enable_overlap_schedule = False
            p._tp_barrier_wait()
            pdl.distributed.barrier.assert_called_once()
        self.plat.is_xpu.return_value = True
        p2 = _create_worker()
        waited = []
        p2.task_queue = types.SimpleNamespace(
            worker_process_tp_barrier=types.SimpleNamespace(wait=lambda: waited.append(1))
        )
        p2._tp_barrier_wait()
        self.assertEqual(len(waited), 1)
        self.plat.is_xpu.return_value = False
        with patch(f"{_WP}.paddle") as pdl:
            p3 = _create_worker(ranks=2)
            p3.model_weights_signal = np.array([42], dtype=np.int32)
            pdl.full.return_value = types.SimpleNamespace(numpy=lambda: np.array([42]))
            self.assertEqual(p3._broadcast_model_weights_signal(src=0, group=None), 42)

    def test_run_worker_proc(self):
        from fastdeploy.worker.worker_process import run_worker_proc

        with (
            patch(f"{_WP}.parse_args"),
            patch(f"{_WP}.init_distributed_environment", return_value=(1, 0)),
            patch(f"{_WP}.initialize_fd_config"),
            patch(f"{_WP}.current_platform") as plat,
            patch(f"{_WP}.PaddleDisWorkerProc") as cls,
            patch(f"{_WP}.envs") as env,
        ):
            plat.is_iluvatar.return_value = False
            env.FD_DETERMINISTIC_MODE = False
            run_worker_proc()
            cls.assert_called_once()
            cls.return_value.event_loop_normal.assert_called_once()


class TestWorkerProcessControlMethod(unittest.TestCase):
    """Test cases for PaddleDisWorkerProc control method handling."""

    def setUp(self):
        self.mock_fd_config = Mock(spec=FDConfig)
        self.mock_fd_config.parallel_config = Mock()
        self.mock_fd_config.parallel_config.use_ep = False
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1
        self.mock_fd_config.load_config = Mock()
        self.mock_fd_config.load_config.dynamic_load_weight = False

        self.process = PaddleDisWorkerProc.__new__(PaddleDisWorkerProc)
        self.process.fd_config = self.mock_fd_config
        self.process.parallel_config = self.mock_fd_config.parallel_config
        self.process.local_rank = 0
        self.process.eplb_config = types.SimpleNamespace(enable_eplb=False)

        self.process.worker = Mock(spec=[])

        self.mock_queue = Mock()
        self.mock_queue.put = AsyncMock()
        self.process._ctrl_output = self.mock_queue

    def test_run_control_method_unknown_handler(self):
        request = ControlRequest(request_id="test_id", method="unknown_method", args={})
        self.process.run_control_method(request)
        self.mock_queue.put.assert_called_once()
        call_args = self.mock_queue.put.call_args[0][0]
        self.assertEqual(call_args.request_id, "test_id")
        self.assertEqual(call_args.error_code, 400)

    def test_run_control_method_non_callable_handler(self):
        self.process.worker.some_method = "not_callable"
        request = ControlRequest(request_id="test_id", method="some_method", args={})
        self.process.run_control_method(request)
        self.mock_queue.put.assert_called_once()
        call_args = self.mock_queue.put.call_args[0][0]
        self.assertEqual(call_args.error_code, 400)

    def test_run_control_method_success(self):
        mock_result = {"result": "success"}
        self.process.worker.test_method = Mock(return_value=mock_result)
        request = ControlRequest(request_id="test_id", method="test_method", args={"param": "value"})
        self.process.run_control_method(request)
        self.process.worker.test_method.assert_called_once_with(param="value")
        self.mock_queue.put.assert_called_once()
        call_args = self.mock_queue.put.call_args[0][0]
        self.assertEqual(call_args.request_id, "test_id")
        self.assertEqual(call_args.error_code, 200)

    def test_run_control_method_exception(self):
        def failing_method(**kwargs):
            raise ValueError("Test error")

        self.process.worker.test_method = failing_method
        request = ControlRequest(request_id="test_id", method="test_method", args={})
        with patch("fastdeploy.worker.worker_process.traceback") as mock_traceback:
            mock_traceback.format_exc.return_value = "Traceback..."
            self.process.run_control_method(request)
            self.mock_queue.put.assert_called_once()
            call_args = self.mock_queue.put.call_args[0][0]
            self.assertEqual(call_args.request_id, "test_id")
            self.assertEqual(call_args.error_code, 500)

    def test_run_control_directly_when_not_use_ep(self):
        self.process.parallel_config.use_ep = False
        self.process.worker.test_method = Mock(return_value={"result": "ok"})
        control_req = ControlRequest(request_id="test_id", method="test_method", args={})
        self.process.run_control_method(control_req)
        self.process.worker.test_method.assert_called_once()
        self.mock_queue.put.assert_called_once()

    @pytest.mark.skip("This case might hang in ci environment, to be fixed in the future")
    def test_event_loop_caches_ep_control_requests_before_collective_run(self):
        self.process.parallel_config.use_ep = True
        self.process.parallel_config.ep_group = Mock(world_size=1)
        self.process.cached_control_reqs = []
        self.process._run_eplb = Mock()
        self.process._tp_barrier_wait = Mock()
        self.process.run_control_method = Mock()
        self.process.worker_healthy_live_signal = Mock(value=[0])
        self.process.max_chips_per_node = 8
        self.process.nnode = 1
        self.process.ranks = 1
        self.process.task_queue = Mock()
        self.process.task_queue.exist_tasks.return_value = False
        self.process.task_queue.read_finish_flag = types.SimpleNamespace(get=Mock(return_value=1))
        control_req = ControlRequest(request_id="ep-ctrl", method="pause", args={})
        self.process.task_queue.get_tasks.return_value = ([([control_req], 1)], False)
        self.process.exist_task_signal = types.SimpleNamespace(value=[1])
        self.process.worker = types.SimpleNamespace(
            preprocess_new_task=Mock(),
            model_runner=types.SimpleNamespace(),
            execute_model=Mock(),
            exist_prefill=Mock(return_value=False),
        )
        with (
            patch("fastdeploy.utils.all_gather_values", side_effect=SystemExit),
            patch("fastdeploy.worker.worker_process.all_gather_values", side_effect=SystemExit),
        ):
            with self.assertRaises(SystemExit):
                self.process.event_loop_normal()

        self.assertEqual(self.process.cached_control_reqs, [control_req])
        self.process.run_control_method.assert_not_called()

    def test_event_loop_skips_execute_model_when_runner_is_sleeping(self):
        self.process.parallel_config.use_ep = False
        self.process.parallel_config.tensor_parallel_size = 2
        self.process.fd_config.load_config.dynamic_load_weight = True
        self.process.cached_control_reqs = []
        self.process._run_eplb = Mock()
        self.process._tp_barrier_wait = Mock(side_effect=SystemExit)
        self.process.worker_healthy_live_signal = Mock(value=[0])
        self.process.max_chips_per_node = 8
        self.process.nnode = 1
        self.process.ranks = 1
        self.process.local_rank = 0
        self.process.task_queue = Mock()
        self.process.task_queue.exist_tasks.return_value = False
        self.process.task_queue.read_finish_flag = types.SimpleNamespace(get=Mock(return_value=0))
        self.process.exist_task_signal = types.SimpleNamespace(value=[0])
        self.process.worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(is_sleeping=True),
            execute_model=Mock(),
            exist_prefill=Mock(return_value=False),
        )

        with patch("fastdeploy.worker.worker_process.envs.FD_ENABLE_V1_UPDATE_WEIGHTS", "1"):
            with self.assertRaises(SystemExit):
                self.process.event_loop_normal()

        self.process.worker.execute_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
