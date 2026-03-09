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
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

WP = "fastdeploy.worker.worker_process"


# -- helpers -------------------------------------------------------------------


def _cfg(**overrides):
    """Minimal FDConfig-like MagicMock."""
    c = MagicMock()
    c.parallel_config.local_engine_worker_queue_port = 9999
    c.parallel_config.tensor_parallel_size = 1
    c.parallel_config.tensor_parallel_rank = 0
    c.parallel_config.data_parallel_size = 1
    c.parallel_config.data_parallel_rank = 0
    c.parallel_config.local_data_parallel_id = 0
    c.parallel_config.engine_pid = 12345
    c.parallel_config.pod_ip = "127.0.0.1"
    c.parallel_config.use_ep = False
    c.parallel_config.expert_parallel_size = 1
    c.parallel_config.tp_group = MagicMock()
    c.parallel_config.shutdown_comm_group_if_worker_idle = False
    c.cache_config.num_cpu_blocks = 0
    c.cache_config.total_block_num = 100
    c.scheduler_config.enable_overlap_schedule = False
    c.scheduler_config.splitwise_role = "mixed"
    c.speculative_config.method = None
    c.eplb_config.enable_eplb = False
    c.load_config.dynamic_load_weight = False
    c.model_config.enable_mm = False
    c.model_config.enable_logprob = False
    c.model_config.architectures = ["LlamaForCausalLM"]
    c.nnode = 1
    for k, v in overrides.items():
        parts = k.split(".")
        obj = c
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return c


@pytest.fixture
def pw():
    """Patches current_platform and get_worker at module level."""
    with patch(f"{WP}.current_platform") as plat, patch(f"{WP}.get_worker") as gw:
        plat.is_iluvatar.return_value = False
        plat.is_cuda.return_value = False
        plat.is_xpu.return_value = False
        yield plat, gw


def _make(pw, ranks=1, local_rank=0, **cfg_kw):
    """Create PaddleDisWorkerProc with heavy deps mocked."""
    from fastdeploy.worker.worker_process import PaddleDisWorkerProc

    return PaddleDisWorkerProc(_cfg(**cfg_kw), ranks=ranks, local_rank=local_rank)


# -- intercept_paddle_loggers -------------------------------------------------


def test_intercept_paddle_loggers():
    from fastdeploy.logger.logger import intercept_paddle_loggers

    lg = logging.getLogger("paddle.test.logger")
    lg.addHandler(logging.StreamHandler())
    lg.addHandler(logging.StreamHandler())
    with intercept_paddle_loggers():
        assert len(logging.getLogger("paddle.test.logger").handlers) == 1
    lg.handlers = []
    orig = logging.getLogger
    with intercept_paddle_loggers():
        assert logging.getLogger != orig
    assert logging.getLogger is orig


# -- get_worker ----------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,module_path,class_name",
    [
        ("is_dcu", "fastdeploy.worker.dcu_worker", "DcuWorker"),
        ("is_cuda", "fastdeploy.worker.gpu_worker", "GpuWorker"),
        ("is_xpu", "fastdeploy.worker.xpu_worker", "XpuWorker"),
        ("is_iluvatar", "fastdeploy.worker.iluvatar_worker", "IluvatarWorker"),
        ("is_gcu", "fastdeploy.worker.gcu_worker", "GcuWorker"),
        ("is_maca", "fastdeploy.worker.metax_worker", "MetaxWorker"),
        ("is_intel_hpu", "fastdeploy.worker.hpu_worker", "HpuWorker"),
    ],
)
def test_get_worker(platform, module_path, class_name):
    from fastdeploy.worker.worker_process import get_worker

    with patch(f"{WP}.current_platform") as plat:
        for a in ("is_dcu", "is_cuda", "is_xpu", "is_iluvatar", "is_gcu", "is_maca", "is_intel_hpu"):
            getattr(plat, a).return_value = False
        getattr(plat, platform).return_value = True
        mock_mod = MagicMock()
        sentinel = MagicMock()
        setattr(mock_mod, class_name, MagicMock(return_value=sentinel))
        with patch.dict("sys.modules", {module_path: mock_mod}):
            assert get_worker(_cfg(), local_rank=0, rank=1) is sentinel


# -- Utility functions ---------------------------------------------------------


def test_update_mm():
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
    assert fd.model_config.tensor_model_parallel_size == 4
    fd2 = _cfg(**{"model_config.enable_mm": False})
    orig = fd2.model_config.tensor_model_parallel_size
    update_fd_config_for_mm(fd2)
    assert fd2.model_config.tensor_model_parallel_size is orig


def test_parse_args():
    from fastdeploy.worker.worker_process import parse_args

    with patch.object(sys, "argv", ["prog"]):
        a = parse_args()
    assert a.model == "./output" and a.dtype == "bfloat16"
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
    assert a.model == "/tmp/m" and a.tensor_parallel_size == 4
    assert a.speculative_config["method"] == "eagle" and a.eplb_config["enable_eplb"]


def test_init_distributed():
    from fastdeploy.worker.worker_process import init_distributed_environment

    with patch(f"{WP}.dist") as dist, patch(f"{WP}.fleet") as fleet:
        dist.get_world_size.return_value = 2
        fleet.worker_index.return_value = 1
        assert init_distributed_environment(seed=42) == (2, 1)
        fleet.init.assert_called_once()
    with patch(f"{WP}.dist") as dist, patch(f"{WP}.fleet"):
        dist.get_world_size.return_value = 0
        assert init_distributed_environment() == (0, 0)


# -- initialize_fd_config -----------------------------------------------------


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


def _fd_config_env():
    """Return (args, ExitStack, mocks_dict) for initialize_fd_config tests."""
    from fastdeploy.worker.worker_process import parse_args

    with patch.object(sys, "argv", ["prog", "-m", "/tmp/m", "--dtype", "float16"]):
        args = parse_args()
    stack = ExitStack()
    m = {n: stack.enter_context(patch(f"{WP}.{n}")) for n in _FD_PATCH_NAMES}
    for a in ("is_cuda", "is_xpu", "is_maca", "is_iluvatar", "is_intel_hpu"):
        getattr(m["current_platform"], a).return_value = a == "is_cuda"
    mc = m["ModelConfig"].return_value
    mc.num_hidden_layers = 2
    mc.architectures = ["LlamaForCausalLM"]
    mc.is_quantized = False
    mc.quantization_config = None
    mc.head_dim = 128
    mc.pretrained_config = MagicMock()
    pc = m["ParallelConfig"].return_value
    pc.tensor_parallel_size = 1
    pc.data_parallel_size = 1
    pc.expert_parallel_size = 1
    pc.use_ep = False
    lc = m["LoadConfig"].return_value
    lc.dynamic_load_weight = False
    lc.load_strategy = "ipc_snapshot"
    lc.rsync_config = None
    lc.load_choices = "default_v1"
    m["parse_quant_config"].return_value = None
    return args, stack, m


def test_initialize_fd_config():
    """Basic path + EP/quant branches."""
    from fastdeploy.worker.worker_process import initialize_fd_config

    args, stack, m = _fd_config_env()
    with stack:
        initialize_fd_config(args, ranks=1, local_rank=0)
        m["FDConfig"].assert_called_once()
        m["update_fd_config_for_mm"].assert_called_once()
    # EP + quant path
    args2, stack2, m2 = _fd_config_env()
    with stack2:
        m2["ParallelConfig"].return_value.data_parallel_size = 2
        m2["ParallelConfig"].return_value.expert_parallel_size = 2
        m2["ModelConfig"].return_value.moe_num_experts = [8]
        m2["EPLBConfig"].return_value.redundant_experts_num = 0
        m2["parse_quant_config"].return_value = MagicMock()
        m2["ModelConfig"].return_value.is_quantized = True
        initialize_fd_config(args2, ranks=2, local_rank=0)
        m2["FDConfig"].assert_called_once()


# -- PaddleDisWorkerProc -------------------------------------------------------


def test_proc_init_and_control(pw):
    """Constructor fields + init_control."""
    plat, gw = pw
    p = _make(pw, ranks=2, local_rank=1)
    assert p.ranks == 2 and p.local_rank == 1 and p.max_chips_per_node == 8
    gw.assert_called_once()
    # Iluvatar → 16 chips
    plat.is_iluvatar.return_value = True
    gw.reset_mock()
    assert _make(pw).max_chips_per_node == 16
    plat.is_iluvatar.return_value = False
    # Speculative + overlap
    plat.is_cuda.return_value = True
    gw.reset_mock()
    p3 = _make(
        pw,
        **{
            "speculative_config.method": "eagle",
            "scheduler_config.enable_overlap_schedule": True,
        },
    )
    assert p3.speculative_decoding and not p3.enable_overlap_schedule
    plat.is_cuda.return_value = False
    # init_control
    with patch(f"{WP}.FMQ") as fmq:
        p4 = _make(pw, **{"parallel_config.local_engine_worker_queue_port": 5555})
        p4.local_rank = 3
        p4.init_control()
        fmq.return_value.queue.assert_called_once_with("ctrl_w2e_rank3_5555", "producer")


def test_health_and_task_queue(pw):
    """init_health_status + start_task_queue_service."""
    with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.time") as t:
        t.time.return_value = 1000.0
        _make(pw, **{"parallel_config.data_parallel_size": 1}).init_health_status()
        assert ipc.call_count > 0
    # SHM task queue
    with patch(f"{WP}.envs") as env, patch(f"{WP}.TaskQueue") as tq:
        env.FD_ENGINE_TASK_QUEUE_WITH_SHM = True
        _make(pw, **{"parallel_config.local_engine_worker_queue_port": 7777}).start_task_queue_service()
        assert "fd_task_queue_7777" in tq.call_args[1]["address"]
    # TCP task queue
    with patch(f"{WP}.envs") as env, patch(f"{WP}.TaskQueue") as tq:
        env.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        _make(
            pw,
            **{
                "parallel_config.pod_ip": "10.0.0.1",
                "parallel_config.local_engine_worker_queue_port": 8888,
            },
        ).start_task_queue_service()
        assert tq.call_args[1]["address"] == ("10.0.0.1", 8888)


def test_load_model_and_graph(pw):
    """load_model, init_device, graph_optimize_and_warm_up_model."""
    with patch(f"{WP}.IPCSignal") as ipc:
        sig = MagicMock()
        sig.value = np.zeros([1], dtype=np.int32)
        ipc.return_value = sig
        p = _make(pw)
        p.load_model()
        p.worker.load_model.assert_called_once()
        assert p.loaded_model_signal.value[0] == 1
    # Distributed load
    with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.paddle") as pdl:
        sig = MagicMock()
        sig.value = np.zeros([1], dtype=np.int32)
        ipc.return_value = sig
        _make(pw, ranks=2).load_model()
        pdl.distributed.barrier.assert_called_once()
    # init_device + graph optimize
    _make(pw).init_device()
    with patch(f"{WP}.envs") as env:
        env.ENABLE_V1_KVCACHE_SCHEDULER = True
        p2 = _make(pw)
        p2.graph_optimize_and_warm_up_model()
        p2.worker.graph_optimize_and_warm_up_model.assert_called_once()
    # Splitwise prefill path
    with patch(f"{WP}.envs") as env, patch(f"{WP}.IPCSignal"):
        env.ENABLE_V1_KVCACHE_SCHEDULER = False
        p3 = _make(pw, **{"scheduler_config.splitwise_role": "prefill"})
        p3.worker.model_runner.device_id = 0
        p3.graph_optimize_and_warm_up_model()


def test_kv_cache(pw):
    """initialize_kv_cache: no-profile, profile, zero-memory, cap at 40000."""
    # No profile
    p = _make(pw, **{"parallel_config.do_profile": False, "cache_config.total_block_num": 42})
    p.initialize_kv_cache()
    p.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=42)
    # Profile: normal (reset worker mock to avoid cross-contamination)
    _, gw = pw
    gw.return_value = MagicMock()
    with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
        p2 = _make(pw, **{"parallel_config.do_profile": True})
        p2.worker.determine_available_memory.return_value = 1024**3
        p2.worker.cal_theortical_kvcache.return_value = 1024**2
        p2.initialize_kv_cache()
        p2.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=1024)
    # Zero memory → ValueError
    with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
        p3 = _make(pw, **{"parallel_config.do_profile": True})
        p3.worker.determine_available_memory.return_value = 0
        p3.worker.cal_theortical_kvcache.return_value = 1024
        with pytest.raises(ValueError):
            p3.initialize_kv_cache()
    # Capped at MAX_BLOCK_NUM (40000)
    with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
        p4 = _make(pw, **{"parallel_config.do_profile": True})
        p4.worker.initialize_cache.reset_mock()
        p4.worker.determine_available_memory.return_value = 100 * 1024**3
        p4.worker.cal_theortical_kvcache.return_value = 1024
        p4.initialize_kv_cache()
        p4.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=40000)


def test_control_method(pw):
    """run_control_method: bad method, success, exception."""
    p = _make(pw)
    p._ctrl_output = MagicMock()
    p._ctrl_output.put = AsyncMock()
    req = MagicMock()
    req.request_id, req.method, req.args = "r1", "bad", {}
    p.worker.bad = None
    p.run_control_method(req)
    # success
    p.worker.do_it = MagicMock(return_value={"ok": True})
    req.method, req.args = "do_it", {"x": 1}
    p.run_control_method(req)
    p.worker.do_it.assert_called_once_with(x=1)
    # exception
    p.worker.fail = MagicMock(side_effect=RuntimeError("boom"))
    req.method, req.args = "fail", {}
    p.run_control_method(req)


def test_eplb(pw):
    """_init_eplb_signal + _run_eplb."""
    # Disabled
    p = _make(pw, **{"eplb_config.enable_eplb": False})
    p._init_eplb_signal()
    assert not hasattr(p, "experts_manager")
    p._run_eplb(tp_rank=0)
    # Enabled init
    with patch(f"{WP}.RedundantExpertManager") as rem, patch(f"{WP}.IPCSignal"), patch(f"{WP}.create_mmap"):
        p2 = _make(
            pw,
            **{
                "eplb_config.enable_eplb": True,
                "model_config.num_hidden_layers": 4,
                "model_config.moe_num_experts": 8,
            },
        )
        p2._init_eplb_signal()
        rem.assert_called_once()
    # Enabled run with token stats + weight update path
    p3 = _make(
        pw,
        **{
            "eplb_config.enable_eplb": True,
            "eplb_config.redundant_expert_dump_workload_interval": 10,
        },
    )
    p3.last_dump_expert_workload_ts = 0
    stats = np.ones((4, 8), dtype=np.int32)
    p3.local_experts_token_stats_array = MagicMock()
    p3.local_experts_token_stats_array.value = stats.copy()
    p3.signal_clear_experts_token_stats = MagicMock()
    p3.signal_clear_experts_token_stats.value = np.array([1], dtype=np.int32)
    p3.signal_update_weight_from_tensor_array = MagicMock()
    p3.signal_update_weight_from_tensor_array.value = np.array([1], dtype=np.int32)
    p3.rearrange_experts_signal = MagicMock()
    p3.rearrange_experts_signal.value = np.zeros([1], dtype=np.int32)
    p3.mmap_infos = {"main": "data"}
    p3.worker.get_model().redundant_table_manger.get_expert_tokens_stats.return_value = (
        stats,
        None,
        None,
        None,
    )
    with patch(f"{WP}.time") as t, patch(f"{WP}.paddle") as pdl:
        t.time.return_value = 100.0
        mock_data = MagicMock()
        mock_data.__getitem__ = MagicMock(return_value=147183647)
        pdl.to_tensor.return_value = mock_data
        with patch.object(p3, "update_weights_from_tensor"):
            p3._run_eplb(tp_rank=0)
    assert p3.signal_clear_experts_token_stats.value[0] == 0
    assert p3.signal_update_weight_from_tensor_array.value[0] == 0


def test_barrier_broadcast_update(pw):
    """_tp_barrier_wait + _broadcast_model_weights_signal + update_weights."""
    with patch(f"{WP}.paddle") as pdl:
        p = _make(pw)
        p.enable_overlap_schedule = False
        p._tp_barrier_wait()
        pdl.distributed.barrier.assert_called_once()
    # XPU barrier path
    plat, gw = pw
    plat.is_xpu.return_value = True
    p2 = _make(pw)
    p2.task_queue = MagicMock()
    p2._tp_barrier_wait()
    p2.task_queue.worker_process_tp_barrier.wait.assert_called_once()
    plat.is_xpu.return_value = False
    # broadcast
    with patch(f"{WP}.paddle") as pdl:
        p3 = _make(pw, ranks=2)
        p3.model_weights_signal = np.array([42], dtype=np.int32)
        mock_t = MagicMock()
        mock_t.numpy.return_value = np.array([42])
        pdl.full.return_value = mock_t
        assert p3._broadcast_model_weights_signal(src=0, group=None) == 42
    # update_weights_from_tensor
    with patch(f"{WP}.load_tensor_from_shm_mem") as load, patch(f"{WP}.MODEL_MAIN_NAME", "main"):
        p4 = _make(pw)
        p4.experts_manager = MagicMock()
        p4.experts_manager.tensor_infos = {"x": 1}
        p4.experts_manager.get_ep_rank_to_expert_id_list.return_value = ([1], {0: 1}, 1)
        load.return_value = {"w": MagicMock()}
        p4.update_weights_from_tensor({"main": "data"})
        load.assert_called_once()
        assert p4.experts_manager.tensor_infos is None


# -- run_worker_proc -----------------------------------------------------------


def test_run_worker_proc():
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
        wp = MagicMock()
        cls.return_value = wp
        run_worker_proc()
        cls.assert_called_once()
        wp.init_control.assert_called_once()
        wp.event_loop_normal.assert_called_once()
        # Deterministic mode
        cls.reset_mock()
        env.FD_DETERMINISTIC_MODE = True
        wp2 = MagicMock()
        cls.return_value = wp2
        mock_biao = MagicMock()
        with patch.dict("sys.modules", {"fastdeploy.model_executor.layers.batch_invariant_ops": mock_biao}):
            run_worker_proc()
        mock_biao.init_deterministic_mode.assert_called_once()
        # Iluvatar path
        cls.reset_mock()
        plat.is_iluvatar.return_value = True
        env.FD_DETERMINISTIC_MODE = False
        mock_il = MagicMock()
        with patch.dict("sys.modules", {"fastdeploy.worker.iluvatar_worker": mock_il}):
            run_worker_proc()
        mock_il.IluvatarPaddleDisWorkerProc.assert_called_once()
