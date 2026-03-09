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
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

WP = "fastdeploy.worker.worker_process"


def _cfg(**overrides):
    """Build a minimal FDConfig-like MagicMock."""
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
def proc_factory():
    """Yields a factory that creates PaddleDisWorkerProc with heavy deps mocked."""
    with patch(f"{WP}.current_platform") as plat, patch(f"{WP}.get_worker") as gw:
        plat.is_iluvatar.return_value = False
        plat.is_cuda.return_value = False
        plat.is_xpu.return_value = False

        def make(ranks=1, local_rank=0, **cfg_kw):
            from fastdeploy.worker.worker_process import PaddleDisWorkerProc

            return PaddleDisWorkerProc(_cfg(**cfg_kw), ranks=ranks, local_rank=local_rank)

        make.plat = plat
        make.gw = gw
        yield make


# ── Tests from develop baseline (intercept_paddle_loggers) ──────────────────
class TestInterceptPaddleLoggers:
    def test_paddle_prefix_configured(self):
        from fastdeploy.logger.logger import intercept_paddle_loggers

        lg = logging.getLogger("paddle.test.logger")
        lg.addHandler(logging.StreamHandler())
        lg.addHandler(logging.StreamHandler())
        with intercept_paddle_loggers():
            ilg = logging.getLogger("paddle.test.logger")
            assert len(ilg.handlers) == 1
            assert ilg.level == logging.INFO
        lg.handlers = []

    def test_restores_and_exception_safe(self):
        from fastdeploy.logger.logger import intercept_paddle_loggers

        orig = logging.getLogger
        with intercept_paddle_loggers():
            assert logging.getLogger != orig
        assert logging.getLogger is orig
        # also safe on exception
        try:
            with intercept_paddle_loggers():
                raise ValueError
        except ValueError:
            pass
        assert logging.getLogger is orig


# ── Module-level functions ──────────────────────────────────────────────────
class TestModuleFunctions:
    """Tests for get_worker, update_fd_config_for_mm, parse_args,
    init_distributed_environment, initialize_fd_config."""

    # -- get_worker platform dispatch --
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
    def test_get_worker_dispatch(self, platform, module_path, class_name):
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

    def test_get_worker_logprob_unsupported_raises(self):
        from fastdeploy.worker.worker_process import get_worker

        with patch(f"{WP}.current_platform") as plat:
            for a in ("is_dcu", "is_cuda", "is_xpu", "is_iluvatar", "is_gcu", "is_maca", "is_intel_hpu"):
                getattr(plat, a).return_value = False
            with pytest.raises(NotImplementedError):
                get_worker(_cfg(**{"model_config.enable_logprob": True}), 0, 1)

    # -- update_fd_config_for_mm --
    def test_update_mm_ernie_sets_fields(self):
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
        assert fd.model_config.vision_config.dtype == "float16"

    def test_update_mm_non_ernie_and_disabled_skip(self):
        from fastdeploy.config import ErnieArchitectures
        from fastdeploy.worker.worker_process import update_fd_config_for_mm

        fd = _cfg(**{"model_config.enable_mm": True})
        orig = fd.model_config.tensor_model_parallel_size
        with patch.object(ErnieArchitectures, "contains_ernie_arch", return_value=False):
            update_fd_config_for_mm(fd)
        assert fd.model_config.tensor_model_parallel_size is orig

        fd2 = _cfg(**{"model_config.enable_mm": False})
        orig2 = fd2.model_config.tensor_model_parallel_size
        update_fd_config_for_mm(fd2)
        assert fd2.model_config.tensor_model_parallel_size is orig2

    # -- parse_args --
    def test_parse_args_defaults_and_custom(self):
        from fastdeploy.worker.worker_process import parse_args

        with patch.object(sys, "argv", ["prog"]):
            a = parse_args()
        assert a.model == "./output" and a.dtype == "bfloat16" and a.tensor_parallel_size == 1

        argv = ["prog", "-m", "/tmp/m", "--dtype", "float16", "--do_profile", "--tensor_parallel_size", "4"]
        with patch.object(sys, "argv", argv):
            a = parse_args()
        assert a.model == "/tmp/m" and a.do_profile and a.tensor_parallel_size == 4

    def test_parse_args_json_configs(self):
        from fastdeploy.worker.worker_process import parse_args

        argv = [
            "prog",
            "--speculative_config",
            '{"method":"eagle"}',
            "--quantization",
            '{"quant_type":"wint4"}',
            "--eplb_config",
            '{"enable_eplb":true}',
        ]
        with patch.object(sys, "argv", argv):
            a = parse_args()
        assert a.speculative_config["method"] == "eagle"
        assert a.eplb_config["enable_eplb"]

    # -- init_distributed_environment --
    def test_init_dist_multi_and_single_rank(self):
        from fastdeploy.worker.worker_process import init_distributed_environment

        with patch(f"{WP}.dist") as dist, patch(f"{WP}.fleet") as fleet:
            dist.get_world_size.return_value = 2
            fleet.worker_index.return_value = 1
            r, lr = init_distributed_environment(seed=42)
            assert (r, lr) == (2, 1)
            fleet.init.assert_called_once()

        with patch(f"{WP}.dist") as dist, patch(f"{WP}.fleet"):
            dist.get_world_size.return_value = 0
            r, lr = init_distributed_environment()
            assert (r, lr) == (0, 0)

    # -- initialize_fd_config --
    def test_initialize_fd_config_creates_config(self):
        from fastdeploy.worker.worker_process import parse_args

        with patch.object(sys, "argv", ["prog", "-m", "/tmp/m", "--dtype", "float16"]):
            args = parse_args()
        with (
            patch(f"{WP}.v1_loader_support", return_value=True),
            patch(f"{WP}.parse_quant_config", return_value=None),
            patch(f"{WP}.update_fd_config_for_mm") as upd,
            patch(f"{WP}.current_platform") as plat,
            patch(f"{WP}.paddle"),
            patch(f"{WP}.ModelConfig") as mc,
            patch(f"{WP}.DeviceConfig"),
            patch(f"{WP}.SpeculativeConfig"),
            patch(f"{WP}.ParallelConfig") as pc,
            patch(f"{WP}.CacheConfig"),
            patch(f"{WP}.SchedulerConfig"),
            patch(f"{WP}.EPLBConfig"),
            patch(f"{WP}.LoadConfig") as lc,
            patch(f"{WP}.GraphOptimizationConfig"),
            patch(f"{WP}.PlasAttentionConfig"),
            patch(f"{WP}.EarlyStopConfig"),
            patch(f"{WP}.StructuredOutputsConfig"),
            patch(f"{WP}.RoutingReplayConfig"),
            patch(f"{WP}.FDConfig") as fd,
        ):
            for a in ("is_cuda", "is_xpu", "is_maca", "is_iluvatar", "is_intel_hpu"):
                getattr(plat, a).return_value = a == "is_cuda"
            mc.return_value.num_hidden_layers = 2
            mc.return_value.architectures = ["LlamaForCausalLM"]
            mc.return_value.is_quantized = False
            mc.return_value.quantization_config = None
            mc.return_value.head_dim = 128
            mc.return_value.pretrained_config = MagicMock()
            pc.return_value.tensor_parallel_size = 1
            pc.return_value.data_parallel_size = 1
            pc.return_value.expert_parallel_size = 1
            pc.return_value.use_ep = False
            lc.return_value.dynamic_load_weight = False
            lc.return_value.load_strategy = "ipc_snapshot"
            lc.return_value.rsync_config = None
            lc.return_value.load_choices = "default_v1"
            from fastdeploy.worker.worker_process import initialize_fd_config

            initialize_fd_config(args, ranks=1, local_rank=0)
            fd.assert_called_once()
            upd.assert_called_once()


# ── PaddleDisWorkerProc lifecycle ───────────────────────────────────────────
class TestPaddleDisWorkerProc:
    """End-to-end tests for the worker process class: init → services → model
    lifecycle → kv cache → eplb → control methods → barriers."""

    # -- constructor --
    def test_init_stores_attrs_and_chips(self, proc_factory):
        p = proc_factory(ranks=2, local_rank=1)
        assert p.ranks == 2 and p.local_rank == 1 and p.max_chips_per_node == 8
        proc_factory.gw.assert_called_once()

        proc_factory.plat.is_iluvatar.return_value = True
        proc_factory.gw.reset_mock()
        p2 = proc_factory()
        assert p2.max_chips_per_node == 16

    def test_init_speculative_and_overlap(self, proc_factory):
        proc_factory.plat.is_cuda.return_value = True
        p = proc_factory(
            **{
                "speculative_config.method": "eagle",
                "scheduler_config.enable_overlap_schedule": True,
            }
        )
        assert p.speculative_decoding and not p.enable_overlap_schedule

        proc_factory.gw.reset_mock()
        p2 = proc_factory(**{"scheduler_config.enable_overlap_schedule": True, "speculative_config.method": None})
        assert p2.enable_overlap_schedule

    # -- init_control + health_status + task_queue --
    def test_init_control_creates_fmq_queue(self, proc_factory):
        with patch(f"{WP}.FMQ") as fmq:
            p = proc_factory(**{"parallel_config.local_engine_worker_queue_port": 5555})
            p.local_rank = 3
            p.init_control()
            fmq.return_value.queue.assert_called_once_with("ctrl_w2e_rank3_5555", "producer")

    def test_init_health_status_single_and_multi_dp(self, proc_factory):
        with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.time") as t:
            t.time.return_value = 1000.0
            proc_factory(**{"parallel_config.data_parallel_size": 1}).init_health_status()
            single_count = ipc.call_count

        with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.time") as t, patch(f"{WP}.envs") as env:
            t.time.return_value = 1000.0
            env.FD_ENABLE_MULTI_API_SERVER = False
            sig = MagicMock()
            sig.value.__getitem__ = MagicMock(return_value=1)
            ipc.return_value = sig
            proc_factory(
                **{
                    "parallel_config.data_parallel_size": 2,
                    "parallel_config.local_data_parallel_id": 0,
                }
            ).init_health_status()
            assert ipc.call_count > single_count

    def test_task_queue_shm_and_tcp(self, proc_factory):
        with patch(f"{WP}.envs") as env, patch(f"{WP}.TaskQueue") as tq:
            env.FD_ENGINE_TASK_QUEUE_WITH_SHM = True
            proc_factory(**{"parallel_config.local_engine_worker_queue_port": 7777}).start_task_queue_service()
            assert "fd_task_queue_7777" in tq.call_args[1]["address"]

        with patch(f"{WP}.envs") as env, patch(f"{WP}.TaskQueue") as tq:
            env.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
            proc_factory(
                **{
                    "parallel_config.pod_ip": "10.0.0.1",
                    "parallel_config.local_engine_worker_queue_port": 8888,
                }
            ).start_task_queue_service()
            assert tq.call_args[1]["address"] == ("10.0.0.1", 8888)

    # -- model lifecycle --
    def test_load_model_and_init_device(self, proc_factory):
        with patch(f"{WP}.IPCSignal") as ipc:
            sig = MagicMock()
            sig.value = np.zeros([1], dtype=np.int32)
            ipc.return_value = sig
            p = proc_factory()
            p.load_model()
            p.worker.load_model.assert_called_once()
            assert p.loaded_model_signal.value[0] == 1

        with patch(f"{WP}.IPCSignal") as ipc, patch(f"{WP}.paddle") as pdl:
            sig = MagicMock()
            sig.value = np.zeros([1], dtype=np.int32)
            ipc.return_value = sig
            proc_factory(ranks=2).load_model()
            pdl.distributed.barrier.assert_called_once()

        proc_factory().init_device()

    def test_graph_optimize_and_splitwise(self, proc_factory):
        with patch(f"{WP}.envs") as env:
            env.ENABLE_V1_KVCACHE_SCHEDULER = True
            p = proc_factory()
            p.graph_optimize_and_warm_up_model()
            p.worker.graph_optimize_and_warm_up_model.assert_called_once()

        with patch(f"{WP}.envs") as env, patch(f"{WP}.IPCSignal"):
            env.ENABLE_V1_KVCACHE_SCHEDULER = False
            p = proc_factory(**{"scheduler_config.splitwise_role": "prefill"})
            p.worker.model_runner.device_id = 0
            p.graph_optimize_and_warm_up_model()

    # -- kv cache --
    def test_kv_cache_no_profile(self, proc_factory):
        p = proc_factory(**{"parallel_config.do_profile": False, "cache_config.total_block_num": 42})
        p.initialize_kv_cache()
        p.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=42)

    def test_kv_cache_profile_normal_and_cap(self, proc_factory):
        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
            p = proc_factory(**{"parallel_config.do_profile": True})
            p.worker.determine_available_memory.return_value = 1024**3
            p.worker.cal_theortical_kvcache.return_value = 1024**2
            p.initialize_kv_cache()
            p.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=1024)

        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
            proc_factory.gw.reset_mock()
            p = proc_factory(**{"parallel_config.do_profile": True})
            p.worker.determine_available_memory.return_value = 100 * 1024**3
            p.worker.cal_theortical_kvcache.return_value = 1024
            p.initialize_kv_cache()
            p.worker.initialize_cache.assert_called_once_with(num_gpu_blocks=40000)

    def test_kv_cache_zero_blocks_raises(self, proc_factory):
        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist"):
            p = proc_factory(**{"parallel_config.do_profile": True})
            p.worker.determine_available_memory.return_value = 0
            p.worker.cal_theortical_kvcache.return_value = 1024
            with pytest.raises(ValueError):
                p.initialize_kv_cache()

    def test_kv_cache_multi_rank_all_reduces(self, proc_factory):
        with patch(f"{WP}.IPCSignal"), patch(f"{WP}.dist") as dist, patch(f"{WP}.paddle") as pdl:
            mock_t = MagicMock()
            mock_t.item.return_value = 500
            pdl.full.return_value = mock_t
            p = proc_factory(ranks=2, **{"parallel_config.do_profile": True})
            p.worker.determine_available_memory.return_value = 1024**3
            p.worker.cal_theortical_kvcache.return_value = 1024**2
            p.initialize_kv_cache()
            dist.all_reduce.assert_called_once()

    # -- run_control_method --
    def test_control_method_success_and_errors(self, proc_factory):
        p = proc_factory()
        p._ctrl_output = MagicMock()
        p._ctrl_output.put = AsyncMock()

        # unknown → 400
        p.worker.bad = None
        req = MagicMock()
        req.request_id, req.method, req.args = "r1", "bad", {}
        p.run_control_method(req)

        # success → 200
        p.worker.do_it = MagicMock(return_value={"ok": True})
        req.method, req.args = "do_it", {"x": 1}
        p.run_control_method(req)
        p.worker.do_it.assert_called_once_with(x=1)

        # exception → 500
        p.worker.fail = MagicMock(side_effect=RuntimeError("boom"))
        req.method, req.args = "fail", {}
        p.run_control_method(req)
        p.worker.fail.assert_called_once()

    # -- eplb --
    def test_eplb_disabled_and_enabled(self, proc_factory):
        p = proc_factory(**{"eplb_config.enable_eplb": False})
        p._init_eplb_signal()
        assert not hasattr(p, "experts_manager")
        p._run_eplb(tp_rank=0)

        with patch(f"{WP}.RedundantExpertManager") as rem, patch(f"{WP}.IPCSignal"), patch(f"{WP}.create_mmap"):
            p2 = proc_factory(
                **{
                    "eplb_config.enable_eplb": True,
                    "model_config.num_hidden_layers": 4,
                    "model_config.moe_num_experts": 8,
                }
            )
            p2._init_eplb_signal()
            rem.assert_called_once()

    # -- barrier / broadcast / update_weights --
    def test_tp_barrier_default_and_xpu(self, proc_factory):
        with patch(f"{WP}.paddle") as pdl:
            p = proc_factory()
            p.enable_overlap_schedule = False
            p._tp_barrier_wait()
            pdl.distributed.barrier.assert_called_once()

        proc_factory.plat.is_xpu.return_value = True
        p2 = proc_factory()
        p2.task_queue = MagicMock()
        p2._tp_barrier_wait()
        p2.task_queue.worker_process_tp_barrier.wait.assert_called_once()

    def test_broadcast_model_weights_signal(self, proc_factory):
        with patch(f"{WP}.paddle") as pdl:
            p = proc_factory(ranks=2)
            p.model_weights_signal = np.array([42], dtype=np.int32)
            mock_t = MagicMock()
            mock_t.numpy.return_value = np.array([42])
            pdl.full.return_value = mock_t
            assert p._broadcast_model_weights_signal(src=0, group=None) == 42

    def test_update_weights_from_tensor(self, proc_factory):
        with patch(f"{WP}.load_tensor_from_shm_mem") as load, patch(f"{WP}.MODEL_MAIN_NAME", "main"):
            p = proc_factory()
            p.experts_manager = MagicMock()
            p.experts_manager.tensor_infos = {"x": 1}
            p.experts_manager.get_ep_rank_to_expert_id_list.return_value = ([1], {0: 1}, 1)
            load.return_value = {"w": MagicMock()}
            p.update_weights_from_tensor({"main": "data"})
            load.assert_called_once()
            assert p.experts_manager.tensor_infos is None
