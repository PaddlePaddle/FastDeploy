"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

_DYNAMIC_WEIGHT_MODULE = None


def _install_dynamic_weight_manager_stubs():
    """Install minimal stubs so this unit test can run without Paddle installed."""

    def no_grad():
        def decorator(func):
            return func

        return decorator

    fake_paddle = types.SimpleNamespace(
        Tensor=object,
        no_grad=no_grad,
        distributed=types.SimpleNamespace(
            get_world_size=lambda: 1,
            get_rank=lambda: 0,
            barrier=lambda *args, **kwargs: None,
            restart_process_group=lambda *args, **kwargs: None,
            shutdown_process_group=lambda *args, **kwargs: None,
        ),
        device=types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                synchronize=lambda: None,
                empty_cache=lambda: None,
                max_memory_allocated=lambda: 0,
                max_memory_reserved=lambda: 0,
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
            )
        ),
        base=types.SimpleNamespace(
            core=types.SimpleNamespace(LoDTensor=types.SimpleNamespace(_new_shared_cuda=MagicMock()))
        ),
        load=MagicMock(),
        empty=MagicMock(),
        to_tensor=MagicMock(),
    )
    fake_logger = types.SimpleNamespace(
        info=MagicMock(),
        warning=MagicMock(),
        error=MagicMock(),
        debug=MagicMock(),
    )
    fake_fastdeploy = types.ModuleType("fastdeploy")
    fake_fastdeploy.__path__ = []
    fake_config = types.ModuleType("fastdeploy.config")
    fake_config.FDConfig = object
    fake_model_executor = types.ModuleType("fastdeploy.model_executor")
    fake_model_executor.__path__ = []
    fake_model_executor_utils = types.ModuleType("fastdeploy.model_executor.utils")
    fake_model_executor_utils.process_final_after_loading = MagicMock()
    fake_numpy = types.ModuleType("numpy")
    fake_envs = types.ModuleType("fastdeploy.envs")
    fake_envs.FD_USE_GDR_CHECKPOINT_TRANSFER = False
    fake_inter_communicator = types.ModuleType("fastdeploy.inter_communicator")
    fake_inter_communicator.KVCacheStatus = types.SimpleNamespace()
    fake_inter_communicator.ModelWeightsStatus = types.SimpleNamespace(NORMAL=0, CLEARED=1)
    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = MagicMock(return_value={})
    fake_yaml.YAMLError = Exception

    sys.modules.update(
        {
            "paddle": fake_paddle,
            "numpy": fake_numpy,
            "yaml": fake_yaml,
            "paddleformers": types.ModuleType("paddleformers"),
            "paddleformers.utils": types.ModuleType("paddleformers.utils"),
            "paddleformers.utils.log": types.SimpleNamespace(logger=fake_logger),
            "fastdeploy": fake_fastdeploy,
            "fastdeploy.envs": fake_envs,
            "fastdeploy.config": fake_config,
            "fastdeploy.model_executor": fake_model_executor,
            "fastdeploy.model_executor.utils": fake_model_executor_utils,
            "fastdeploy.inter_communicator": fake_inter_communicator,
        }
    )


def _load_dynamic_weight_manager_from_file():
    module_path = Path(__file__).resolve().parents[2] / "fastdeploy" / "rl" / "dynamic_weight_manager.py"
    spec = importlib.util.spec_from_file_location("dynamic_weight_manager_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dynamic_weight_manager_module():
    global _DYNAMIC_WEIGHT_MODULE
    if _DYNAMIC_WEIGHT_MODULE is not None:
        return _DYNAMIC_WEIGHT_MODULE

    fastdeploy_module = sys.modules.get("fastdeploy")
    if fastdeploy_module is not None and not hasattr(fastdeploy_module, "__path__"):
        _DYNAMIC_WEIGHT_MODULE = _load_dynamic_weight_manager_from_file()
        return _DYNAMIC_WEIGHT_MODULE

    try:
        from fastdeploy.rl import dynamic_weight_manager

        _DYNAMIC_WEIGHT_MODULE = dynamic_weight_manager
        return dynamic_weight_manager
    except ModuleNotFoundError as exc:
        if exc.name not in ("numpy", "paddle", "yaml"):
            raise

    for name in list(sys.modules):
        if name == "fastdeploy" or name.startswith("fastdeploy."):
            sys.modules.pop(name, None)
    _install_dynamic_weight_manager_stubs()

    _DYNAMIC_WEIGHT_MODULE = _load_dynamic_weight_manager_from_file()
    return _DYNAMIC_WEIGHT_MODULE


class _FakeModel:
    def __init__(self):
        self.loaded = []
        self.params = {}

    def load_weights(self, weights_iterator):
        self.loaded.extend(list(weights_iterator))

    def state_dict(self):
        return self.params


class _FakeMTPModel(_FakeModel):
    def __init__(self, mtp_start_layer_idx=2, num_mtp_layers=1):
        super().__init__()
        self.mtp_start_layer_idx = mtp_start_layer_idx
        self.num_mtp_layers = num_mtp_layers


def _make_manager(rsync_config=None, load_strategy="rsync"):
    DynamicWeightManager = _load_dynamic_weight_manager_module().DynamicWeightManager

    manager = object.__new__(DynamicWeightManager)
    fd_config = MagicMock()
    fd_config.load_config.rsync_config = rsync_config or {
        "backend": "mooncake",
        "output_framework": "paddle",
    }
    fd_config.load_config.load_strategy = load_strategy
    fd_config.parallel_config.data_parallel_rank = 2
    fd_config.parallel_config.data_parallel_size = 1
    fd_config.parallel_config.tensor_parallel_rank = 1
    fd_config.parallel_config.tensor_parallel_size = 4
    manager.fd_config = fd_config
    manager.load_config = fd_config.load_config
    manager.parallel_config = fd_config.parallel_config
    manager.local_rank = 5
    manager.nranks = 8
    manager.rdma_handle = None
    manager.model_list = [_FakeModel()]
    manager.state_dict = {}
    manager.use_gdr_checkpoint_transfer = True
    manager._gdr_ct_handle = None
    return manager


class _FakeRole(Enum):
    TRAINER = "trainer"
    INFERENCE = "inference"


class _FakePhase1Backend(Enum):
    GPU_DIRECT = "gpu_direct"
    MOONCAKE = "mooncake"
    IPC = "ipc"


@dataclass
class _FakeTransferConfig:
    role: object
    global_rank: int
    group_size: int = 1
    phase1_backend: object = _FakePhase1Backend.GPU_DIRECT
    phase2_backend: object = None
    phase2_fan_out: int = 4
    bucket_size_mb: int = 512
    num_buffers: int = 2
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    discover_timeout_s: float = 60.0
    redis_ttl_s: int = 60
    recv_bucket_timeout_s: float = 60.0
    session_total_timeout_s: float = 600.0
    device: str = None
    log_level: str = None
    log_file: str = None
    perf_log_file: str = None
    materialize_tensors: bool = True
    qsize: int = 3
    gpu_id: int = -1

    def __post_init__(self):
        self.kwargs = dict(self.__dict__)
        self.kwargs.pop("kwargs", None)


def _patch_gdr_checkpoint_transfer(fake_checkpoint_transfer):
    class FakeCheckpointTransferWithLifecycle(fake_checkpoint_transfer):
        async def initialize(self):
            self.initialized = True

        async def cleanup(self):
            self.cleaned = True

    fake_config_module = types.SimpleNamespace(
        Role=_FakeRole,
        TransferConfig=_FakeTransferConfig,
        Phase1Backend=_FakePhase1Backend,
    )
    fake_transfer_module = types.SimpleNamespace(CheckpointTransfer=FakeCheckpointTransferWithLifecycle)
    return patch.dict(
        sys.modules,
        {
            "checkpoint_transfer.config": fake_config_module,
            "checkpoint_transfer.transfer": fake_transfer_module,
        },
    )


class TestDynamicWeightGDR(unittest.TestCase):
    def test_update_weights_by_gdr_gdr_mode(self):
        created = []

        class FakeCheckpointTransfer:
            def __init__(self, config):
                self.config = config
                self.step_ids = []
                created.append(self)

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                self.step_ids.append(step_id)
                self.output_framework = output_framework
                yield f"model.layers.{len(self.step_ids)}.weight", object()

        manager = _make_manager()

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            result = manager.update_weights_by_gdr(version="step-1")
            second_result = manager.update_weights_by_gdr(version="step-2")

        self.assertEqual(result["version"], "step-1")
        self.assertEqual(second_result["version"], "step-2")
        self.assertEqual(result["update_count"], 1)
        self.assertEqual(second_result["update_count"], 1)
        self.assertIn("total_cost", result)
        self.assertEqual(
            [name for name, _ in manager.model_list[0].loaded], ["model.layers.1.weight", "model.layers.2.weight"]
        )
        self.assertEqual(len(created), 1)
        self.assertIs(manager._gdr_ct_handle, created[0])
        self.assertTrue(created[0].initialized)
        self.assertFalse(hasattr(created[0], "cleaned"))
        self.assertEqual(created[0].step_ids, ["step-1", "step-2"])
        self.assertEqual(created[0].output_framework, "paddle")
        self.assertEqual(created[0].config.kwargs["role"], _FakeRole.INFERENCE)
        self.assertEqual(created[0].config.kwargs["phase1_backend"], _FakePhase1Backend.GPU_DIRECT)
        self.assertEqual(created[0].config.kwargs["global_rank"], 5)
        self.assertEqual(created[0].config.kwargs["group_size"], 8)
        self.assertNotIn("backend", created[0].config.kwargs)
        self.assertNotIn("output_framework", created[0].config.kwargs)

    def test_update_weights_by_gdr_ipc_mode(self):
        created = []

        class FakeCheckpointTransfer:
            def __init__(self, config):
                self.config = config
                created.append(self)

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                self.step_id = step_id
                yield "model.layers.0.weight", object()

        manager = _make_manager(
            rsync_config={"redis_host": "10.0.0.1", "redis_port": 6379},
            load_strategy="ipc",
        )

        with (
            _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer),
            patch.dict("os.environ", {"FLAGS_selected_gpus": "3"}),
        ):
            result = manager.update_weights_by_gdr()

        self.assertEqual(result["version"], "0")
        self.assertEqual(created[0].step_id, "0")
        self.assertEqual(created[0].config.kwargs["phase1_backend"], _FakePhase1Backend.IPC)
        self.assertEqual(created[0].config.kwargs["global_rank"], 3)
        self.assertEqual(created[0].config.kwargs["qsize"], 2)

    def test_gdr_checkpoint_transfer_receive_exception_propagates(self):
        created = []

        class FakeCheckpointTransfer:
            def __init__(self, config):
                created.append(self)

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "model.layers.0.weight", object()
                raise RuntimeError("receive failed")

        class IncrementalModel(_FakeModel):
            def load_weights(self, weights_iterator):
                for item in weights_iterator:
                    self.loaded.append(item)

        manager = _make_manager()
        manager.model_list = [IncrementalModel()]

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            with self.assertRaisesRegex(RuntimeError, "receive failed"):
                manager.update_weights_by_gdr(version="step-error")

        self.assertTrue(created[0].cleaned)
        self.assertIsNone(manager._gdr_ct_handle)

    def test_gdr_checkpoint_transfer_refreshes_state_dict_after_model_loader(self):
        loaded_param = object()

        class FakeCheckpointTransfer:
            def __init__(self, config):
                pass

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "model.weight", loaded_param

        class RefreshingModel(_FakeModel):
            def load_weights(self, weights_iterator):
                super().load_weights(weights_iterator)
                self.params["model.weight"] = loaded_param

        manager = _make_manager()
        manager.model_list = [RefreshingModel()]

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            manager.update_weights_by_gdr(version="step-refresh")

        self.assertIs(manager.state_dict["model.weight"], loaded_param)

    def test_gdr_checkpoint_transfer_caches_mtp_subset_for_auxiliary_model(self):
        objects = [object() for _ in range(4)]

        class FakeCheckpointTransfer:
            def __init__(self, config):
                pass

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "model.layers.0.self_attn.q_proj.weight", objects[0]
                yield "model.layers.2.self_attn.q_proj.weight", objects[1]
                yield "model.layers.20.self_attn.q_proj.weight", objects[2]
                yield "ernie.mtp_linear_proj.0.weight", objects[3]

        manager = _make_manager()
        main_model = _FakeModel()
        mtp_model = _FakeMTPModel(mtp_start_layer_idx=2, num_mtp_layers=1)
        manager.model_list = [main_model, mtp_model]

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            result = manager.update_weights_by_gdr(version="step-5")

        self.assertEqual(result["update_count"], 4)
        self.assertEqual(result["mtp_cache_count"], 2)
        self.assertEqual(
            [name for name, _ in main_model.loaded],
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.2.self_attn.q_proj.weight",
                "model.layers.20.self_attn.q_proj.weight",
                "ernie.mtp_linear_proj.0.weight",
            ],
        )
        self.assertEqual(
            [name for name, _ in mtp_model.loaded],
            [
                "model.layers.2.self_attn.q_proj.weight",
                "ernie.mtp_linear_proj.0.weight",
            ],
        )

    def test_gdr_checkpoint_transfer_flushes_mtp_subset_by_chunk_limit(self):
        class FakeCheckpointTransfer:
            def __init__(self, config):
                pass

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "model.layers.2.self_attn.q_proj.weight", object()
                yield "ernie.mtp_linear_proj.0.weight", object()
                yield "model.layers.2.self_attn.o_proj.weight", object()

        class ChunkRecordingMTPModel(_FakeMTPModel):
            def __init__(self):
                super().__init__(mtp_start_layer_idx=2, num_mtp_layers=1)
                self.load_calls = []

            def load_weights(self, weights_iterator):
                chunk = list(weights_iterator)
                self.load_calls.append([name for name, _ in chunk])
                self.loaded.extend(chunk)

        manager = _make_manager(
            {
                "backend": "mooncake",
                "output_framework": "paddle",
                "gdr_mtp_chunk_size": 2,
            }
        )
        main_model = _FakeModel()
        mtp_model = ChunkRecordingMTPModel()
        manager.model_list = [main_model, mtp_model]

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            result = manager.update_weights_by_gdr(version="step-8")

        self.assertEqual(result["mtp_cache_count"], 3)
        self.assertEqual(
            mtp_model.load_calls,
            [
                [
                    "model.layers.2.self_attn.q_proj.weight",
                    "ernie.mtp_linear_proj.0.weight",
                ],
                ["model.layers.2.self_attn.o_proj.weight"],
            ],
        )

    def test_gdr_checkpoint_transfer_multi_model_requires_mtp_subset(self):
        class FakeCheckpointTransfer:
            def __init__(self, config):
                pass

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "model.layers.0.self_attn.q_proj.weight", object()

        manager = _make_manager()
        manager.model_list = [_FakeModel(), _FakeMTPModel(mtp_start_layer_idx=2, num_mtp_layers=1)]

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            with self.assertRaisesRegex(ValueError, "No MTP weights"):
                manager.update_weights_by_gdr(version="step-5")

    def test_gdr_checkpoint_transfer_config_not_forwarded_to_transfer_config(self):
        created = []

        class FakeCheckpointTransfer:
            def __init__(self, config):
                self.config = config
                created.append(self)

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "w1", object()

        manager = _make_manager(
            {
                "backend": "mooncake",
                "output_framework": "paddle",
            }
        )

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            manager.update_weights_by_gdr(version="step-6")

        self.assertNotIn("gpu_direct", created[0].config.kwargs)
        self.assertNotIn("output_framework", created[0].config.kwargs)
        self.assertEqual(created[0].config.kwargs["phase1_backend"], _FakePhase1Backend.GPU_DIRECT)

    def test_gdr_checkpoint_transfer_computes_global_rank_from_node_index(self):
        created = []

        class FakeCheckpointTransfer:
            def __init__(self, config):
                self.config = config
                created.append(self)

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "w1", object()

        manager = _make_manager(
            {
                "index": 1,
                "backend": "mooncake",
                "output_framework": "paddle",
                "group_size": 16,
            }
        )
        manager.local_rank = 5
        manager.nranks = 8

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            manager.update_weights_by_gdr(version="step-index")

        self.assertEqual(created[0].config.kwargs["global_rank"], 13)
        self.assertEqual(created[0].config.kwargs["group_size"], 16)
        self.assertNotIn("index", created[0].config.kwargs)

    def test_gdr_checkpoint_transfer_config_deep_copied_before_forwarding(self):
        created = []

        class FakeCheckpointTransfer:
            def __init__(self, config):
                self.config = config
                created.append(self)

            def receive_weights_sync(self, step_id, output_framework="paddle"):
                yield "w1", object()

        rsync_config = {
            "backend": "mooncake",
            "output_framework": "paddle",
            "device_name": "mlx5_0",
        }
        manager = _make_manager(rsync_config)

        with _patch_gdr_checkpoint_transfer(FakeCheckpointTransfer):
            manager.update_weights_by_gdr(version="step-7")

        self.assertEqual(created[0].config.kwargs["device"], "mlx5_0")
        self.assertEqual(rsync_config["device_name"], "mlx5_0")

    def test_finalize_update_uses_worker_queue_port_status_suffix(self):
        module = _load_dynamic_weight_manager_module()
        manager = _make_manager()
        manager.first_load = False
        manager.rank = 0
        manager.parallel_config.tensor_parallel_size = 1
        manager.parallel_config.enable_expert_parallel = False
        manager.parallel_config.local_engine_worker_queue_port = 60572
        manager._verify_parameters = MagicMock()

        class FakeArray:
            shape = (1,)
            dtype = "int32"
            nbytes = 4

        class FakeValue:
            def __init__(self):
                self.writes = {}

            def __setitem__(self, key, value):
                self.writes[key] = value

        fake_value = FakeValue()
        with (
            patch.object(module.np, "int32", "int32", create=True),
            patch.object(module.np, "zeros", return_value=FakeArray(), create=True),
            patch.object(module.np, "ndarray", return_value=fake_value, create=True),
            patch.object(module, "SharedMemory") as fake_shared_memory,
        ):
            manager.finalize_update()

        fake_shared_memory.assert_called_once_with(create=False, size=4, name="model_weights_status.60572")
        self.assertEqual(fake_value.writes[0], module.ModelWeightsStatus.NORMAL)


if __name__ == "__main__":
    unittest.main()
