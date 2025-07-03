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
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.load_weight_utils import \
    load_composite_checkpoint
from fastdeploy.model_executor.model_loader import MODEL_CLASSES


class DynamicWeightManager:
    """Manages model weights loading, updating and shared state across processes."""

    def __init__(self, fd_config: FDConfig, model: nn.Layer):
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
        self.models: List[nn.Layer] = [model]
        self._capture_model_state()

        if self.load_config.load_strategy != "meta":
            self.update_parameters()

        logger.info(
            f"✅ DynamicLoad model built successfully by {self.load_config.load_strategy}, "
            f" rank={self.rank}, ranks={self.nranks}, "
            f" load ipc weight from {self.ipc_path}.")

    @paddle.no_grad()
    def _capture_model_state(self):
        """Capture and store initial model parameters state."""
        for model in self.models:
            for name, param in model.state_dict().items():
                logger.debug(
                    f"Model param: {name}, shape={param.shape}, dtype={param.dtype}"
                )
                self.state_dict[name] = param

    def add_model(self, model: nn.Layer):
        """"add model"""
        self.models.append(model)
        self._capture_model_state()

    def update_parameters(self, pid: int = 0) -> None:
        """Core method to update model parameters based on strategy."""
        start_time = time.perf_counter()
        paddle.device.cuda.empty_cache()

        if not self.first_load:
            paddle.distributed.restart_process_group()

        strategy_handlers = {
            "ipc_snapshot": self._update_ipc_snapshot,
            "ipc": self._update_ipc,
            "ipc_no_reshard": self._update_ipc_no_reshard,
            "normal": self.load_model,
        }

        if handler := strategy_handlers.get(self.load_config.load_strategy):
            handler()
        else:
            raise ValueError(
                f"Unsupported strategy: {self.load_config.load_strategy}")

        logger.info(
            f"Update parameters in {time.perf_counter()-start_time:.2f}s")

        self._finalize_update(pid)

    def _update_ipc_snapshot(self):
        """Update using IPC snapshot strategy for elastic recovery."""
        model_path = os.path.join(
            self.parallel_config.model_name_or_path,
            f"model_state.tp0{self.meta_src_id}.pdparams")

        try:
            ipc_state_dict = paddle.load(model_path)
        except FileNotFoundError:
            fallback_path = f"/shared_ipc_meta/model_state.tp0{self.meta_src_id}.pdparams"
            ipc_state_dict = paddle.load(fallback_path)

        try:
            self._update_model_from_state(ipc_state_dict, "snapshot")
        except Exception:
            self.models[0].set_state_dict(ipc_state_dict)
            logger.warning(
                "load model from no_reshard weight, maybe need more GPU memory"
            )
        logger.info("IPC snapshot update parameters completed")

    def _update_ipc(self):
        """Update using standard IPC strategy (requires Training Worker)."""
        ipc_meta = paddle.load(self.ipc_path)
        state_dict = self._convert_ipc_meta_to_tensor(ipc_meta)
        self._update_model_from_state(state_dict, "raw")
        logger.info("IPC update parameters completed")

    def _update_ipc_no_reshard(self):
        """Update using no-reshard IPC strategy (faster but uses more memory)."""
        ipc_meta = paddle.load(self.ipc_path)
        state_dict = self._convert_ipc_meta_to_tensor(ipc_meta)
        self.models[0].set_state_dict(state_dict)
        logger.info("IPC no-reshard update parameters completed")

    def load_model(self) -> nn.Layer:
        """Standard model loading without IPC."""
        architectures = self.fd_config.model_config.architectures[0]
        model_class = MODEL_CLASSES[architectures]
        state_dict = load_composite_checkpoint(
            self.fd_config.parallel_config.model_name_or_path,
            model_class,
            self.fd_config.model_config,
            return_numpy=True)
        self.models[0].set_state_dict(state_dict)
        logger.info("normal load update parameters completed")

    def clear_parameters(self, pid: int = 0) -> None:
        """Clear all model parameters and free memory."""
        logger.info("start clear paramaters")
        paddle.device.cuda.empty_cache()
        for model in self.models:
            for param in model.state_dict().values():
                param._clear_data()

        self._verify_parameters("clearance")
        if self.nranks > 1:
            paddle.distributed.barrier()
        paddle.distributed.shutdown_process_group()
        self._update_shared_status(pid, -2)

    def _update_model_from_state(self, state_dict: Dict[str, paddle.Tensor],
                                 src_type: str):
        """Update model parameters from given state dictionary."""
        update_count = 0
        for name, new_param in state_dict.items():
            if name not in self.state_dict:
                logger.debug(f"Ignoring unmatched {src_type} param: {name}")
                continue

            target_param = self.state_dict[name]
            self._validate_parameter_match(name, new_param, target_param)
            new_param._share_buffer_to(target_param)
            update_count += 1
        logger.info(
            f"🆗 Updated {update_count}/{len(state_dict)} parameters from {src_type} source"
        )

    def _validate_parameter_match(self, name: str, src: paddle.Tensor,
                                  dst: paddle.Tensor):
        """验证参数一致性"""
        if src.dtype != dst.dtype:
            raise TypeError(
                f"Type mismatch for {name}: {src.dtype} vs {dst.dtype}")
        if src.shape != dst.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {src.shape} vs {dst.shape}")

    def _finalize_update(self, pid: int):
        """Finalize update process with verification."""
        self._verify_parameters("update")
        if self.nranks > 1:
            paddle.distributed.barrier()
        if not self.first_load:
            self._update_shared_status(pid, 0)
        self.first_load = False

    def _get_gpu_id(self) -> int:
        """Get current GPU device ID."""
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")
        return int(visible_devices[int(os.getenv("FLAGS_selected_gpus", "0"))])

    def _verify_parameters(self, operation: str):
        """Verify parameters are in expected state after operation."""
        expected_initialized = (operation == "update")
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
            raise RuntimeError(
                f"❌ Model Parameter {operation} verification failed")

    @staticmethod
    def _convert_ipc_meta_to_tensor(
            ipc_meta: Dict[str, Any]) -> Dict[str, paddle.Tensor]:
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

        logger.warning(f"GPU memory usage {context}:"
                       f"max_allocated: {max_alloc:.2f}GB\n"
                       f"max_reserved: {max_reserved:.2f}GB\n"
                       f"current_allocated: {curr_alloc:.2f}GB\n"
                       f"current_reserved: {curr_reserved:.2f}GB")

    def _update_shared_status(self, pid: int, status: int) -> None:
        """Update shared memory status flag for inter-process communication."""
        array = np.zeros([1], dtype=np.int32)
        shm = SharedMemory(create=False,
                           size=array.nbytes,
                           name=f"model_weights_status.{pid}")
        value = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        if self.rank == 0:
            value[self.rank] = status

    @staticmethod
    def check_model_weights_status(model_weights_status, model_runner, pid):
        """
        check model weights status
        """
        is_stop = 0
        while model_weights_status.value[0] != 0:
            if model_weights_status.value[0] == 1:
                logger.info(
                    "infer engine stopped! start to load new checkpoint...")
                model_runner.update_parameters(pid)
            elif model_weights_status.value[0] == -1:
                logger.info(
                    "infer engine stopped! start to clear checkpoint...")
                model_runner.clear_parameters(pid)

            while True:
                if model_weights_status.value[0] == 0:
                    logger.info("finished loading new checkpoint")
                    break
                elif is_stop == 1 or (model_weights_status.value[0] == -2
                                      and is_stop == 0):
                    if is_stop == 0:
                        logger.info("finished clearing checkpoint")
                        is_stop = 1
                    time.sleep(0.001)
                    break
                else:
                    time.sleep(0.001)
