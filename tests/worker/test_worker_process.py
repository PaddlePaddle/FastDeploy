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
from unittest.mock import patch

import numpy as np

WP = "fastdeploy.worker.worker_process"


# fmt: off
def _cfg(**overrides):
    """Minimal FDConfig-like namespace."""
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
        nnode=1)
    for k, v in overrides.items():
        parts = k.split(".")
        obj = c
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return c
# fmt: on


def _make(ranks=1, local_rank=0, **cfg_kw):
    """Create PaddleDisWorkerProc with platform + get_worker mocked in setUp."""
    from fastdeploy.worker.worker_process import PaddleDisWorkerProc

    return PaddleDisWorkerProc(_cfg(**cfg_kw), ranks=ranks, local_rank=local_rank)


_FD_NAMES = [
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


def _fd_env():
    """Return (args, ExitStack, mocks_dict) for initialize_fd_config tests."""
    from fastdeploy.worker.worker_process import parse_args

    with patch.object(sys, "argv", ["prog", "-m", "/tmp/m", "--dtype", "float16"]):
        args = parse_args()
    stack = ExitStack()
    m = {n: stack.enter_context(patch(f"{WP}.{n}")) for n in _FD_NAMES}
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


class TestWorkerUtils(unittest.TestCase):
    """Utility functions — real logic, minimal mocking."""

    def test_intercept_paddle_loggers(self):
        from fastdeploy.logger.logger import intercept_paddle_loggers

        original = logging.getLogger
        lg = logging.getLogger("paddle.test.check")
        lg.addHandler(logging.StreamHandler())
        lg.addHandler(logging.StreamHandler())
        self.assertEqual(len(lg.handlers), 2)
        with intercept_paddle_loggers():
            self.assertNotEqual(logging.getLogger, original)
            il = logging.getLogger("paddle.test.check")
            self.assertEqual(len(il.handlers), 1)
            self.assertEqual(il.level, logging.INFO)
            self.assertFalse(il.propagate)
        self.assertEqual(logging.getLogger, original)
        lg.handlers = []
        try:
            with intercept_paddle_loggers():
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertEqual(logging.getLogger, original)

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
            with self.subTest(platform=plat_name), patch(f"{WP}.current_platform") as plat:
                for a in all_plats:
                    getattr(plat, a).return_value = a == plat_name
                sentinel = object()
                with patch.dict(
                    "sys.modules", {mod_path: types.SimpleNamespace(**{cls_name: lambda *a, **kw: sentinel})}
                ):
                    self.assertIs(get_worker(_cfg(), local_rank=0, rank=1), sentinel)
        with patch(f"{WP}.current_platform") as plat:
            for a in all_plats:
                getattr(plat, a).return_value = False
            with self.assertRaises(NotImplementedError):
                get_worker(_cfg(**{"model_config.enable_logprob": True}), local_rank=0, rank=0)

    def test_update_mm(self):
        from fastdeploy.config import ErnieArchitectures
        from fastdeploy.worker.worker_process import update_fd_config_for_mm

        fd = _cfg(
            **{
                "model_config.enable_mm": True,
                "model_config.architectures": ["Ernie4_5ForCausalLM"],
                "parallel_config.tensor_parallel_size": 4,
                "parallel_config.tensor_parallel_rank": 2,
                "model_config.dtype": "float16",
            }
        )
        with patch.object(ErnieArchitectures, "contains_ernie_arch", return_value=True):
            update_fd_config_for_mm(fd)
        self.assertEqual(fd.model_config.tensor_model_parallel_size, 4)
        fd2 = _cfg(**{"model_config.enable_mm": False})
        orig = fd2.model_config.tensor_model_parallel_size
        update_fd_config_for_mm(fd2)
        self.assertEqual(fd2.model_config.tensor_model_parallel_size, orig)

    def test_init_distributed(self):
        from fastdeploy.worker.worker_process import init_distributed_environment

        with patch(f"{WP}.dist") as dist, patch(f"{WP}.fleet") as fleet:
            dist.get_world_size.return_value = 2
            fleet.worker_index.return_value = 1
            self.assertEqual(init_distributed_environment(seed=42), (2, 1))
            fleet.init.assert_called_once()
        with patch(f"{WP}.dist") as dist, patch(f"{WP}.fleet"):
            dist.get_world_size.return_value = 0
            self.assertEqual(init_distributed_environment(), (0, 0))

    def test_initialize_fd_config(self):
        from fastdeploy.worker.worker_process import initialize_fd_config

        args, stack, m = _fd_env()
        with stack:
            initialize_fd_config(args, ranks=1, local_rank=0)
            m["FDConfig"].assert_called_once()
            m["update_fd_config_for_mm"].assert_called_once()
        args2, stack2, m2 = _fd_env()
        with stack2:
            m2["ParallelConfig"].return_value.expert_parallel_size = 2
            m2["ModelConfig"].return_value.moe_num_experts = 8
            m2["ModelConfig"].return_value.num_local_experts = None
            m2["EPLBConfig"].return_value.redundant_experts_num = 0
            initialize_fd_config(args2, ranks=1, local_rank=0)
        args3, stack3, m3 = _fd_env()
        with stack3:
            m3["ModelConfig"].return_value.num_hidden_layers = None
            with self.assertRaises(ValueError):
                initialize_fd_config(args3, ranks=1, local_rank=0)


class TestPaddleDisWorkerProc(unittest.TestCase):
    """PaddleDisWorkerProc: constructor, health, cache, control, event loop."""

    def setUp(self):
        self._patcher_plat = patch(f"{WP}.current_platform")
        self._patcher_gw = patch(f"{WP}.get_worker")
        self.plat = self._patcher_plat.start()
        self.gw = self._patcher_gw.start()
        self.plat.is_iluvatar.return_value = False
        self.plat.is_cuda.return_value = False
        self.plat.is_xpu.return_value = False

    def tearDown(self):
        self._patcher_gw.stop()
        self._patcher_plat.stop()

    def test_init_and_control(self):
        p = _make(ranks=2, local_rank=1)
        self.assertEqual(p.ranks, 2)
        self.assertEqual(p.local_rank, 1)
        self.assertEqual(p.max_chips_per_node, 8)
        self.gw.assert_called_once()
        self.plat.is_iluvatar.return_value = True
        self.gw.reset_mock()
        self.assertEqual(_make().max_chips_per_node, 16)
        self.plat.is_iluvatar.return_value = False
        with patch(f"{WP}.FMQ") as fmq:
            p4 = _make(**{"parallel_config.local_engine_worker_queue_port": 5555})
            p4.local_rank = 3
            p4.init_control()
            fmq.return_value.queue.assert_called_once_with("ctrl_w2e_rank3_5555", "producer")

    def test_health_and_task_queue(self):
        with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.IPCLock"), patch(f"{WP}.time") as t:
            t.time.return_value = 1000.0
            _make(**{"parallel_config.data_parallel_size": 1}).init_health_status()
            self.assertGreater(ipc.call_count, 0)
        with patch(f"{WP}.envs") as env, patch(f"{WP}.TaskQueue") as tq:
            env.FD_ENGINE_TASK_QUEUE_WITH_SHM = True
            _make(**{"parallel_config.local_engine_worker_queue_port": 7777}).start_task_queue_service()
            self.assertIn("fd_task_queue_7777", tq.call_args[1]["address"])
        with patch(f"{WP}.envs") as env, patch(f"{WP}.TaskQueue") as tq:
            env.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
            _make(
                **{"parallel_config.pod_ip": "10.0.0.1", "parallel_config.local_engine_worker_queue_port": 8888}
            ).start_task_queue_service()
            self.assertEqual(tq.call_args[1]["address"], ("10.0.0.1", 8888))

    def test_load_model_and_graph(self):
        with patch(f"{WP}.IPCSignal") as ipc:
            ipc.return_value = types.SimpleNamespace(value=np.zeros([1], dtype=np.int32))
            p = _make()
            p.load_model()
            p.worker.load_model.assert_called_once()
            self.assertEqual(p.loaded_model_signal.value[0], 1)
        with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.paddle") as pdl:
            ipc.return_value = types.SimpleNamespace(value=np.zeros([1], dtype=np.int32))
            _make(ranks=2).load_model()
            pdl.distributed.barrier.assert_called_once()
        _make().init_device()
        with patch(f"{WP}.envs") as env:
            env.ENABLE_V1_KVCACHE_SCHEDULER = True
            p2 = _make()
            p2.graph_optimize_and_warm_up_model()
            p2.worker.graph_optimize_and_warm_up_model.assert_called_once()

    def test_kv_cache(self):
        p = _make(**{"cache_config.total_block_num": 42})
        p.initialize_kv_cache()
        p.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=42)
        self.gw.return_value.reset_mock()
        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
            p2 = _make(**{"parallel_config.do_profile": True})
            p2.worker.determine_available_memory.return_value = 1024**3
            p2.worker.cal_theortical_kvcache.return_value = 1024**2
            p2.initialize_kv_cache()
            p2.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=1024)
        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
            p3 = _make(**{"parallel_config.do_profile": True})
            p3.worker.determine_available_memory.return_value = 0
            p3.worker.cal_theortical_kvcache.return_value = 1024
            with self.assertRaises(ValueError):
                p3.initialize_kv_cache()
        self.gw.return_value.reset_mock()
        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
            p4 = _make(**{"parallel_config.do_profile": True})
            p4.worker.determine_available_memory.return_value = 100 * 1024**3
            p4.worker.cal_theortical_kvcache.return_value = 1024
            p4.initialize_kv_cache()
            p4.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=40000)

    def test_eplb(self):
        p = _make(**{"eplb_config.enable_eplb": False})
        p._init_eplb_signal()
        self.assertFalse(hasattr(p, "experts_manager"))
        p._run_eplb(tp_rank=0)
        with patch(f"{WP}.RedundantExpertManager") as rem, patch(f"{WP}.IPCSignal"), patch(f"{WP}.create_mmap"):
            p2 = _make(
                **{
                    "eplb_config.enable_eplb": True,
                    "model_config.num_hidden_layers": 4,
                    "model_config.moe_num_experts": 8,
                }
            )
            p2._init_eplb_signal()
            rem.assert_called_once()

    def test_control_method(self):
        p = _make()
        responses = []

        async def _capture(resp, **kwargs):
            responses.append(resp)

        p._ctrl_output = types.SimpleNamespace(put=_capture)

        # unknown/non-callable → 400
        req = types.SimpleNamespace(request_id="r1", method="bad", args={})
        p.worker.bad = None
        p.run_control_method(req)
        self.assertEqual(responses[-1].error_code, 400)

        # success → 200
        p.worker.do_it.return_value = {"ok": True}
        req.method, req.args = "do_it", {"x": 1}
        p.run_control_method(req)
        p.worker.do_it.assert_called_once_with(x=1)
        self.assertEqual(responses[-1].error_code, 200)

        # exception → 500
        p.worker.fail.side_effect = RuntimeError("boom")
        req.method, req.args = "fail", {}
        p.run_control_method(req)
        self.assertEqual(responses[-1].error_code, 500)

    def test_barrier_broadcast(self):
        with patch(f"{WP}.paddle") as pdl:
            p = _make()
            p.enable_overlap_schedule = False
            p._tp_barrier_wait()
            pdl.distributed.barrier.assert_called_once()
        self.plat.is_xpu.return_value = True
        p2 = _make()
        waited = []
        p2.task_queue = types.SimpleNamespace(
            worker_process_tp_barrier=types.SimpleNamespace(wait=lambda: waited.append(1))
        )
        p2._tp_barrier_wait()
        self.assertEqual(len(waited), 1)
        self.plat.is_xpu.return_value = False
        with patch(f"{WP}.paddle") as pdl:
            p3 = _make(ranks=2)
            p3.model_weights_signal = np.array([42], dtype=np.int32)
            pdl.full.return_value = types.SimpleNamespace(numpy=lambda: np.array([42]))
            self.assertEqual(p3._broadcast_model_weights_signal(src=0, group=None), 42)

    def test_kvcache_lock(self):
        with patch(f"{WP}.envs") as env:
            env.FD_USE_KVCACHE_LOCK = True
            p = _make()
            p.gpu_cache_lock = types.SimpleNamespace(acquire=lambda: None, release=lambda: None)
            p._acquire_kvcache_lock(0)
            p._release_kvcache_lock(0)

    def test_run_worker_proc(self):
        from fastdeploy.worker.worker_process import run_worker_proc

        with (
            patch(f"{WP}.parse_args"),
            patch(f"{WP}.init_distributed_environment", return_value=(1, 0)),
            patch(f"{WP}.initialize_fd_config"),
            patch(f"{WP}.current_platform") as plat,
            patch(f"{WP}.PaddleDisWorkerProc") as cls,
            patch(f"{WP}.envs") as env,
        ):
            plat.is_iluvatar.return_value = False
            env.FD_DETERMINISTIC_MODE = False
            run_worker_proc()
            cls.assert_called_once()
            cls.return_value.event_loop_normal.assert_called_once()


if __name__ == "__main__":
    unittest.main()
