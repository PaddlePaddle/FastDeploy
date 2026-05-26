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

import gc
import glob
import os
import re
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List

import numpy as np
import paddle
import yaml
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import KVCacheStatus, ModelWeightsStatus


class DynamicWeightManager:
    """Manages model weights loading, updating and shared state across processes."""

    def __init__(self, fd_config: FDConfig, models, local_rank: int):
        """Initialize with config and model instances."""
        self.fd_config = fd_config
        self.load_config = fd_config.load_config
        self.local_rank = local_rank
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
        self.rdma_handle = None
        if self.load_config.load_strategy == "rsync":
            self.update_weights_by_rdma()
        else:
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
                logger.info(f"Model param: {name}, shape={param.shape}, dtype={param.dtype}, place={param.place}")
                self.state_dict[name] = param

    def update_weights_by_rdma(self, version: str = None, verify_checksum: bool = False):
        def valid_parameters(old_state_dict, new_state_dict):
            is_valid = True
            for key in new_state_dict:
                if key not in old_state_dict:
                    is_valid = False
                    logger.error(f"Invalid parameter: {key} not in old_state_dict")
                elif old_state_dict[key].shape != new_state_dict[key].shape:
                    is_valid = False
                    logger.error(
                        f"Invalid parameter: {key} shape mismatch, "
                        f"new shape:{new_state_dict[key].shape}, "
                        f"old shape:{old_state_dict[key].shape}"
                    )
                elif old_state_dict[key].dtype != new_state_dict[key].dtype:
                    is_valid = False
                    logger.error(
                        f"Invalid parameter: {key} dtype mismatch, old:{old_state_dict[key].dtype}, new:{new_state_dict[key].dtype}"
                    )
            return is_valid

        bootstrap_load = version is None or version == ""
        if bootstrap_load:
            version = self.read_model_version_from_file()
        if version is None or version == "":
            raise Exception(
                "rsync model version not set, please set it in 1) {model_version}/version.yaml "
                "or 2) interface arguments 'version'"
            )

        logger.info(
            f"START rank:{self.local_rank}/{self.nranks} update_weights_by_rdma, "
            f"version:{version}, verify_checksum:{verify_checksum}, bootstrap_load:{bootstrap_load}"
        )

        if self.rdma_handle is None:
            from checkpoint_transfer import CheckpointTransfer

            config = self.fd_config.load_config.rsync_config
            logger.info(f"CheckpointTransfer rsync config:{config}")
            self.rdma_handle = CheckpointTransfer(**config, local_rank=self.local_rank, group_size=self.nranks)
            self.rdma_handle.initialize()

        sync_start = time.perf_counter()
        new_state_dict = dict()
        for key, param in self.rdma_handle.receive_stream(step_id=version, verify_checksum=verify_checksum):
            new_state_dict[key] = param
        sync_cost = time.perf_counter() - sync_start
        logger.info(f"weights sync cost {sync_cost:.2f} seconds")

        old_state_dict = self.state_dict
        if not valid_parameters(old_state_dict, new_state_dict):
            error_msg = "Invalid new_state_dict, update parameters failed"
            logger.error(error_msg)
            raise ValueError(error_msg)

        update_start = time.perf_counter()
        for name, new_param in new_state_dict.items():
            target_param = old_state_dict[name]
            if bootstrap_load and not target_param._is_initialized():
                new_param = new_param.cuda()
                new_param._share_buffer_to(target_param)
            else:
                target_param.set_value(new_param)

        update_cost = time.perf_counter() - update_start
        logger.info(f"params set value cost {update_cost:.2f} seconds")
        total_cost = time.perf_counter() - sync_start
        logger.info(
            f"END update_weights_by_rdma, cost {total_cost:.2f} seconds"
            f" version:{version}, verify_checksum: {verify_checksum}, local_rank: {self.local_rank}",
        )
        return {
            "sync_cost": sync_cost,
            "update_cost": update_cost,
            "total_cost": total_cost,
            "version": version,
            "rank": self.local_rank,
        }

    def update_parameters(self, pid: int = 0, restart_process_group=False) -> None:
        """Core method to update model parameters based on strategy."""
        start_time = time.perf_counter()
        paddle.device.cuda.empty_cache()

        # step1 : restart paddle process group
        if not self.first_load:
            if restart_process_group:
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

    def restart_communication_group(self):
        if not self.first_load:
            start_time = time.perf_counter()
            paddle.distributed.restart_process_group()
            paddle.distributed.restart_process_group(self.parallel_config.tp_group)
            if self.parallel_config.enable_expert_parallel:
                paddle.distributed.restart_process_group(self.parallel_config.ep_group)
            logger.info(f"finish restarting communication groups! time cost: {time.perf_counter()-start_time:.3f}s")

    def recreate_deepep_buffer(self):
        if not self.first_load:
            start_time = time.perf_counter()
            from fastdeploy.model_executor.layers.moe.ep import DeepEPBufferManager

            DeepEPBufferManager.recreate_buffer()
            # ep barrier
            paddle.distributed.barrier(self.parallel_config.ep_group)
            logger.info(f"finish recreating deepep buffer! time cost: {time.perf_counter()-start_time:.3f}s")

    def reload_model_weights(self):
        if not self.first_load:
            start_time = time.perf_counter()
            strategy_handlers = {
                "ipc_snapshot": self._update_ipc_snapshot,
                "ipc": self._update_ipc,
            }

            if handler := strategy_handlers.get(self.load_config.load_strategy):
                handler()
            else:
                raise ValueError(f"Unsupported strategy: {self.load_config.load_strategy}")
            logger.info(f"finish reload model weights! time cost: {time.perf_counter()-start_time:.3f}s")

    def _update_ipc_snapshot(self):
        """Update using IPC snapshot strategy for elastic recovery.

        Loading priority:
          1. Chunked part files  (model_state.tp{rank}.{id}.part{N}.pdparams)
          2. Single full file    (model_state.tp{rank}.{id}.pdparams)
          3. Legacy format       (model_state.tp0{id}.pdparams)
          4. Shared fallback dir (/shared_ipc_meta/...)
        """
        model_dir = self.fd_config.model_config.model
        base_name = f"model_state.tp{paddle.distributed.get_rank()}.{self.meta_src_id}"
        legacy_base_name = f"model_state.tp0{self.meta_src_id}"

        # --- Priority 1: load from chunked part files to avoid memory spike ---
        part_pattern = os.path.join(model_dir, f"{base_name}.part*.pdparams")
        all_part_files = glob.glob(part_pattern)

        valid_part_files = []
        invalid_part_files = []
        part_regex = re.compile(r"\.part(\d+)\.")

        for path in all_part_files:
            match = part_regex.search(path)
            if not match:
                invalid_part_files.append(os.path.basename(path))
                continue
            try:
                part_idx = int(match.group(1))
            except (TypeError, ValueError):
                invalid_part_files.append(os.path.basename(path))
                continue
            valid_part_files.append((part_idx, path))

        if invalid_part_files:
            logger.warning(
                "Found snapshot part files with invalid naming pattern under %s: %s. "
                "These files will be ignored when loading IPC snapshot parts.",
                model_dir,
                ", ".join(invalid_part_files),
            )

        part_files = [p for _, p in sorted(valid_part_files, key=lambda item: item[0])]

        if part_files:
            logger.info(f"Found {len(part_files)} snapshot part files for {base_name}")
            for load_idx, part_path in enumerate(part_files):
                match = re.search(r"\.part(\d+)\.", part_path)
                # Use part index parsed from filename to keep logs and src_type consistent with file naming
                part_index = int(match.group(1)) if match else load_idx
                logger.info(f"Loading snapshot part {part_index+1}/{len(part_files)} from {part_path}")
                ipc_state_dict = paddle.load(part_path, safetensors=True)
                self._update_model_from_state(ipc_state_dict, f"snapshot-part{part_index}")
                del ipc_state_dict
                gc.collect()
            logger.info(f"IPC snapshot update completed from {len(part_files)} part files under {model_dir}")
            return

        # --- Priority 2: single full pdparams file ---
        model_path = os.path.join(model_dir, f"{base_name}.pdparams")
        if os.path.exists(model_path):
            ipc_state_dict = paddle.load(model_path, safetensors=True)
            self._update_model_from_state(ipc_state_dict, "snapshot")
            logger.info(f"IPC snapshot update completed from {model_path}")
            return

        # --- Priority 3: legacy format (model_state.tp0{id}.pdparams) ---
        legacy_path = os.path.join(model_dir, f"{legacy_base_name}.pdparams")
        if os.path.exists(legacy_path):
            ipc_state_dict = paddle.load(legacy_path, safetensors=True)
            self._update_model_from_state(ipc_state_dict, "snapshot")
            logger.info(f"IPC snapshot update completed from legacy format {legacy_path}")
            return

        # --- Priority 4: shared directory fallback ---
        fallback_path = f"/shared_ipc_meta/{base_name}.pdparams"
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                f"No snapshot found for {base_name}: " f"checked {model_dir} (new/legacy) and {fallback_path}"
            )
        logger.info(f"No local snapshot in {model_dir}, fallback to {fallback_path}")
        ipc_state_dict = paddle.load(fallback_path)
        self._update_model_from_state(ipc_state_dict, "snapshot")
        logger.info(f"IPC snapshot update completed from {fallback_path}")

    def _update_ipc(self):
        """Update using standard IPC strategy (requires Training Worker)."""
        ipc_meta = paddle.load(self.ipc_path)
        state_dict = self._convert_ipc_meta_to_tensor(ipc_meta)
        self._update_model_from_state(state_dict, "raw")
        logger.info(f"IPC update parameters completed from file: {self.ipc_path}")

    def clear_parameters(self, pid: int = 0, shutdown_process_group=False) -> None:
        """Clear all model parameters and free memory."""

        logger.info("start clear paramaters")

        # step1: release deepep buffer
        if self.parallel_config.enable_expert_parallel:
            from fastdeploy.model_executor.layers.moe.ep import DeepEPBufferManager

            DeepEPBufferManager.clear_buffer()
            # ep barrier
            paddle.distributed.barrier(self.parallel_config.ep_group)
            if shutdown_process_group:
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
            if shutdown_process_group:
                paddle.distributed.shutdown_process_group(self.parallel_config.tp_group)
        if self.parallel_config.enable_expert_parallel:
            paddle.distributed.barrier(self.parallel_config.ep_group)
            if shutdown_process_group:
                paddle.distributed.shutdown_process_group(self.parallel_config.ep_group)
        if shutdown_process_group:
            # ProcessGroupGloo has no shutdown(); remove it from paddle's registry
            # before the global sweep to avoid AttributeError.
            from paddle.distributed.collective import _get_group_map_by_name

            for name, pg in list(_get_group_map_by_name().items()):
                if pg.process_group is not None and not hasattr(pg.process_group, "shutdown"):
                    _get_group_map_by_name().pop(name, None)
            paddle.distributed.shutdown_process_group()
        self._update_shared_status(pid, ModelWeightsStatus.CLEARED)

    def clear_deepep_buffer(self):
        start_time = time.perf_counter()
        from fastdeploy.model_executor.layers.moe.ep import DeepEPBufferManager

        DeepEPBufferManager.clear_buffer()
        logger.info(f"finish clearing deepep buffer! time cost: {time.perf_counter()-start_time:.3f}s")

    def clear_model_weight(self):
        start_time = time.perf_counter()
        for model in self.model_list:
            for param in model.state_dict().values():
                param._clear_data()
        logger.info(f"finish clearing model weight! time cost: {time.perf_counter()-start_time:.3f}s")

    def clear_communication_group(self):
        start_time = time.perf_counter()
        if self.parallel_config.enable_expert_parallel:
            paddle.distributed.barrier(self.parallel_config.ep_group)
            paddle.distributed.shutdown_process_group(self.parallel_config.ep_group)
        if self.parallel_config.tensor_parallel_size > 1:
            paddle.distributed.barrier(self.parallel_config.tp_group)
            paddle.distributed.shutdown_process_group(self.parallel_config.tp_group)
        logger.info(f"finish clearing communication groups! time cost: {time.perf_counter()-start_time:.3f}s")

    def _update_model_from_state(self, state_dict: Dict[str, paddle.Tensor], src_type: str):
        """Update model parameters from given state dictionary."""
        if len(state_dict) == 0:
            raise ValueError(f"No parameter found in state dict {state_dict}")
        update_count = 0
        with paddle.no_grad():
            for name, new_param in state_dict.items():
                if name not in self.state_dict:
                    logger.debug(f"Ignoring unmatched {src_type} param: {name}")
                    continue

                target_param = self.state_dict[name]
                self._validate_parameter_match(name, new_param, target_param)
                if new_param.stride() != target_param.stride():
                    logger.warning(
                        f"name:[{name}] target_param.stride():[{target_param.stride()}] != new_param.stride():[{new_param.stride()}]"
                    )
                    if not target_param._is_initialized():
                        target_param[...] = paddle.empty(target_param.shape, dtype=target_param.dtype)
                    target_param[...] = new_param
                else:
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

    def read_model_version_from_file(self):
        model_dir = self.fd_config.model_config.model
        version_file = os.path.join(model_dir, "version.yaml")
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                version_info = yaml.safe_load(f) or {}

            if not isinstance(version_info, dict):
                logger.error(f"Failed to read model step from '{version_file}': yaml content is not a mapping")
                return None

            step = version_info.get("step")
            if step is None:
                logger.error(f"Failed to read model step from '{version_file}': missing 'step' field")
                return None

            return str(step)
        except (FileNotFoundError, OSError, IOError, yaml.YAMLError) as e:
            logger.error(f"Failed to read model step from '{version_file}': {e}")
            return None

    @staticmethod
    def check_model_weights_status(model_weights_status, kv_cache_status, model_runner, pid, block):
        """
        A function to handle the state of model weights, check the model weights state,
        and perform corresponding operations as needed.

        - model_weights_status (`IPCSignal`): The signal indicating the status of model weights.
        - kv_cache_status (`IPCSignal`): The signal indicating the status of key-value cache.
        - model_runner (`ModelRunnerBase`): The model runner instance.
        - block (`bool`): Block mode keeps the worker process blocked in the status-check loop,
            avoiding communication operations in the worker event loop.
        """
        logger.info(f"dynamic weight manager is check model weights status! {model_weights_status.value[0]}")
        while model_weights_status.value[0] != ModelWeightsStatus.NORMAL and (
            block or model_weights_status.value[0] != ModelWeightsStatus.CLEARED
        ):
            if model_weights_status.value[0] == ModelWeightsStatus.UPDATING:
                logger.info("infer engine stopped! start to load new checkpoint...")
                if kv_cache_status:
                    kv_cache_status.value[0] = KVCacheStatus.UPDATING
                model_runner.clear_requests()
                model_runner.update_parameters(pid)
                while model_weights_status.value[0] != ModelWeightsStatus.NORMAL:
                    time.sleep(0.01)
                logger.info("finished loading new checkpoint")
            elif model_weights_status.value[0] == ModelWeightsStatus.CLEARING:
                logger.info("infer engine stopped! start to clear checkpoint...")
                if kv_cache_status:
                    kv_cache_status.value[0] = KVCacheStatus.CLEARING
                model_runner.clear_requests()
                model_runner.clear_parameters(pid)
                while model_weights_status.value[0] != ModelWeightsStatus.CLEARED:
                    time.sleep(0.01)
                logger.info("finished clearing checkpoint")
            else:
                time.sleep(0.01)
