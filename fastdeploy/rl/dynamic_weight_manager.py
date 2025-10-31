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

import os
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import ModelWeightsStatus


class DynamicWeightManager:
    """Manages model weights loading, updating and shared state across processes."""

    def __init__(self, fd_config: FDConfig, models):
        """Initialize with config and model instances."""
        self.fd_config = fd_config
        self.load_config = fd_config.load_config
        self.parallel_config = fd_config.parallel_config
        self.state_dict: Dict[str, paddle.Tensor] = {}
        self.rank = fd_config.parallel_config.tensor_parallel_rank
        self.nranks = paddle.distributed.get_world_size()
        self.meta_src_id = self._get_gpu_id()
        self.first_load = True
        self.ipc_path = f"/shared_ipc_meta/ipc_metas_{self.meta_src_id}"
        if not isinstance(models, List):
            self.model_list = [models]
        else:
            self.model_list = models
        self._capture_model_state()
        self.update_parameters()
        self.finalize_update()

        logger.info(
            f"✅ DynamicLoad model built successfully by {self.load_config.load_strategy}, "
            f" tp rank={self.rank}, dp rank={fd_config.parallel_config.local_data_parallel_id}, ep rank={fd_config.parallel_config.expert_parallel_rank}, ranks={self.nranks}, "
        )

    @paddle.no_grad()
    def _capture_model_state(self):
        """Capture and store initial model parameters state."""
        for model in self.model_list:
            for name, param in model.state_dict().items():
                logger.info(f"Model param: {name}, shape={param.shape}, dtype={param.dtype}")
                self.state_dict[name] = param

    def update_parameters(self, pid: int = 0) -> None:
        """Core method to update model parameters based on strategy."""
        start_time = time.perf_counter()
        paddle.device.cuda.empty_cache()

        # step1 : restart paddle process group
        if not self.first_load:
            paddle.distributed.restart_process_group()
            paddle.distributed.restart_process_group(self.parallel_config.tp_group)
            if self.parallel_config.enable_expert_parallel:
                paddle.distributed.restart_process_group(self.parallel_config.ep_group)

        # step2 : recreat deepep buffer when enable expert parallel
        if self.parallel_config.enable_expert_parallel and not self.first_load:
            from fastdeploy.model_executor.layers.moe.ep import DeepEPBufferManager

            DeepEPBufferManager.recreate_buffer()
            # ep barrier
            paddle.distributed.barrier(self.parallel_config.ep_group)

        # step3 : update model weight
        strategy_handlers = {
            "ipc_snapshot": self._update_ipc_snapshot,
            "ipc": self._update_ipc,
        }

        if handler := strategy_handlers.get(self.load_config.load_strategy):
            handler()
        else:
            raise ValueError(f"Unsupported strategy: {self.load_config.load_strategy}")

        logger.info(f"Update parameters in {time.perf_counter()-start_time:.2f}s")

        # steps in the runner
        # step4: reinitialze kv_cache in the runner
        # step5: recapture cuda_graph
        # step6: update weight status signal

    def _update_ipc_snapshot(self):
        """Update using IPC snapshot strategy for elastic recovery."""
        model_path = os.path.join(
            self.fd_config.model_config.model,
            f"model_state.tp0{self.meta_src_id}.pdparams",
        )

        try:
            ipc_state_dict = paddle.load(model_path)
        except FileNotFoundError:
            fallback_path = f"/shared_ipc_meta/model_state.tp0{self.meta_src_id}.pdparams"
            ipc_state_dict = paddle.load(fallback_path)

        self._update_model_from_state(ipc_state_dict, "snapshot")
        logger.info(f"IPC snapshot update parameters completed from {model_path}")

    def _update_ipc(self):
        """Update using standard IPC strategy (requires Training Worker)."""
        ipc_meta = paddle.load(self.ipc_path)
        state_dict = self._convert_ipc_meta_to_tensor(ipc_meta)
        self._update_model_from_state(state_dict, "raw")
        logger.info(f"IPC update parameters completed from file: {self.ipc_path}")

    def clear_parameters(self, pid: int = 0) -> None:
        """Clear all model parameters and free memory."""

        logger.info("start clear paramaters")

        # step1: release deepep buffer
        if self.parallel_config.enable_expert_parallel:
            from fastdeploy.model_executor.layers.moe.ep import DeepEPBufferManager

            DeepEPBufferManager.clear_buffer()
            # ep barrier
            paddle.distributed.barrier(self.parallel_config.ep_group)
            # shutdown ep group
            paddle.distributed.shutdown_process_group(self.parallel_config.ep_group)

        paddle.device.cuda.empty_cache()
        # step2: release model weight
        for model in self.model_list:
            for param in model.state_dict().values():
                param._clear_data()

        self._verify_parameters("clearance")

        if self.parallel_config.tensor_parallel_size > 1:
            # tp barrier
            paddle.distributed.barrier(self.parallel_config.tp_group)
            paddle.distributed.shutdown_process_group(self.parallel_config.tp_group)
        if self.parallel_config.enable_expert_parallel:
            paddle.distributed.barrier(self.parallel_config.ep_group)
            paddle.distributed.shutdown_process_group(self.parallel_config.ep_group)
        paddle.distributed.shutdown_process_group()
        self._update_shared_status(pid, ModelWeightsStatus.CLEARED)

    def _update_model_from_state(self, state_dict: Dict[str, paddle.Tensor], src_type: str):
        """Update model parameters from given state dictionary."""
        if len(state_dict) == 0:
            raise ValueError(f"No parameter found in state dict {state_dict}")
        update_count = 0
        for name, new_param in state_dict.items():
            if name not in self.state_dict:
                logger.debug(f"Ignoring unmatched {src_type} param: {name}")
                continue

            target_param = self.state_dict[name]
            self._validate_parameter_match(name, new_param, target_param)
            new_param._share_buffer_to(target_param)
            update_count += 1
        logger.info(f"🆗 Updated {update_count}/{len(state_dict)} parameters from {src_type} source")

    def _validate_parameter_match(self, name: str, src: paddle.Tensor, dst: paddle.Tensor):
        """验证参数一致性"""
        if src.dtype != dst.dtype:
            raise TypeError(f"Type mismatch for {name}: {src.dtype} vs {dst.dtype}")
        if src.shape != dst.shape:
            raise ValueError(f"Shape mismatch for {name}: {src.shape} vs {dst.shape}")

    def finalize_update(self, pid: int = 0):
        """Finalize update process with verification."""
        self._verify_parameters("update")

        if self.parallel_config.tensor_parallel_size > 1:
            paddle.distributed.barrier(self.parallel_config.tp_group)

        if self.parallel_config.enable_expert_parallel:
            paddle.distributed.barrier(self.parallel_config.ep_group)

        if not self.first_load:
            self._update_shared_status(pid, ModelWeightsStatus.NORMAL)
        self.first_load = False

    def _get_gpu_id(self) -> int:
        """Get current GPU device ID."""
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")
        return int(visible_devices[int(os.getenv("FLAGS_selected_gpus", "0"))])

    def _verify_parameters(self, operation: str):
        """Verify parameters are in expected state after operation."""
        expected_initialized = operation == "update"
        all_valid = True
        for name, param in self.state_dict.items():
            is_initialized = param._is_initialized()
            if is_initialized != expected_initialized:
                logger.error(
                    f"Verification failed after {operation}: "
                    f"Param {name} initialized={is_initialized} (expected {expected_initialized})"
                )
                all_valid = False

        if all_valid:
            logger.info(f"💡 Model Parameter {operation} verified successfully")
        else:
            raise RuntimeError(f"❌ Model Parameter {operation} verification failed")

    @staticmethod
    def _convert_ipc_meta_to_tensor(
        ipc_meta: Dict[str, Any],
    ) -> Dict[str, paddle.Tensor]:
        """Convert IPC metadata to tensor dictionary."""
        converted = {}
        for name, meta in ipc_meta.items():
            meta[0] = meta[0].encode("latin-1")
            meta[6] = int(os.getenv("FLAGS_selected_gpus", "0"))
            tensor = paddle.base.core.LoDTensor._new_shared_cuda(tuple(meta))
            converted[name] = paddle.to_tensor(tensor)
        return converted

    def _log_memory(self, context: str):
        """Log current GPU memory usage."""
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024**3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024**3)
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024**3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024**3)

        logger.warning(
            f"GPU memory usage {context}:"
            f"max_allocated: {max_alloc:.2f}GB\n"
            f"max_reserved: {max_reserved:.2f}GB\n"
            f"current_allocated: {curr_alloc:.2f}GB\n"
            f"current_reserved: {curr_reserved:.2f}GB"
        )

    def _update_shared_status(self, pid: int, status: int) -> None:
        """Update shared memory status flag for inter-process communication."""
        array = np.zeros([1], dtype=np.int32)
        shm = SharedMemory(create=False, size=array.nbytes, name=f"model_weights_status.{pid}")
        value = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        if self.rank == 0:
            value[self.rank] = status

    @staticmethod
    def check_model_weights_status(model_weights_status, model_runner, pid):
        """
        check model weights status
        """
        logger.info(f"dynamic weight manager is check model weights status! {model_weights_status.value[0]}")
        while model_weights_status.value[0] != ModelWeightsStatus.NORMAL:
            if model_weights_status.value[0] == ModelWeightsStatus.UPDATING:
                logger.info("infer engine stopped! start to load new checkpoint...")
                model_runner.clear_requests()
                model_runner.update_parameters(pid)
                while model_weights_status.value[0] != ModelWeightsStatus.NORMAL:
                    time.sleep(0.01)
                logger.info("finished loading new checkpoint")
            elif model_weights_status.value[0] == ModelWeightsStatus.CLEARING:
                logger.info("infer engine stopped! start to clear checkpoint...")
                model_runner.clear_requests()
                model_runner.clear_parameters(pid)
                while model_weights_status.value[0] != ModelWeightsStatus.CLEARED:
                    time.sleep(0.01)
                logger.info("finished clearing checkpoint")
            time.sleep(0.01)
