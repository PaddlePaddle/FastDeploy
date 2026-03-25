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

import threading
import time
from typing import Dict, List, Optional, Tuple

from fastdeploy import envs
from fastdeploy.cache_manager.cache_data import CacheStatus
from fastdeploy.cache_manager.cache_tasks import DecodeCleanupTask, DecodeOffloadTask, DecodeResumeTask
from fastdeploy.engine.request import Request, RequestStatus
from fastdeploy.utils import offload_logger


class OffloadManager:
    """
    Decode request KV cache offload orchestrator.

    Real KV cache snapshot/restore is executed inside cache_transfer_manager.
    This class only manages request-level state, retry policy and task/result
    synchronization.
    """

    STORAGE_LEVEL_CPU = "L2"
    STORAGE_LEVEL_SSD = "L3"

    def __init__(self, config=None, cache_manager=None, model_runner=None):
        self.config = config
        self.cache_manager = cache_manager
        self.model_runner = model_runner

        self.enable_offload = getattr(config, "enable_decode_offload", False) if config else False
        self.min_steps = 20
        self.cpu_offloading_chunk_size = getattr(envs, "FD_CPU_OFFLOAD_CHUNK_SIZE", 8192)
        self.cpu_memory_limit = getattr(envs, "FD_CPU_MEMORY_LIMIT", 50 * 1024 * 1024 * 1024)

        self._offloaded_requests: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._transfer_events: Dict[Tuple[int, str], threading.Event] = {}
        self._transfer_results: Dict[Tuple[int, str], list] = {}
        self._tensor_parallel_size = getattr(getattr(config, "parallel_config", None), "tensor_parallel_size", 1)

        offload_logger.info(
            f"[DEBUG: offload] OffloadManager initialized: enable_offload={self.enable_offload}, "
            f"min_steps={self.min_steps}"
        )
        if self.cache_manager is not None and hasattr(self.cache_manager, "register_transfer_result_handler"):
            self.cache_manager.register_transfer_result_handler(self._handle_transfer_result)

    def _transfer_key(self, event_type, task_id: str) -> Tuple[int, str]:
        return (event_type.value, task_id)

    def _handle_transfer_result(self, data) -> bool:
        event_type = data[0]
        if event_type.value not in (
            CacheStatus.DECODE_OFFLOAD.value,
            CacheStatus.DECODE_RESUME.value,
            CacheStatus.DECODE_CLEANUP.value,
        ):
            return False

        task_id, rank, ok, meta = data[1:]
        key = self._transfer_key(event_type, task_id)
        with self._lock:
            if key not in self._transfer_results:
                self._transfer_results[key] = []
            self._transfer_results[key].append(
                {
                    "rank": rank,
                    "ok": ok,
                    "meta": meta,
                }
            )
            if len(self._transfer_results[key]) >= self._tensor_parallel_size:
                event = self._transfer_events.get(key)
                if event is not None:
                    event.set()
        return True

    def _issue_transfer_task(self, event_type, task):
        if self.cache_manager is None or not hasattr(self.cache_manager, "cache_task_queue"):
            return None

        key = self._transfer_key(event_type, task.task_id)
        event = threading.Event()
        with self._lock:
            self._transfer_events[key] = event
            self._transfer_results.pop(key, None)
        self.cache_manager.cache_task_queue.put_transfer_task((event_type, task))
        event.wait(timeout=30)
        if not event.is_set():
            offload_logger.error(f"Transfer task {task.task_id} timed out after 30s")
            with self._lock:
                self._transfer_results.pop(key, None)
                self._transfer_events.pop(key, None)
            return None
        with self._lock:
            results = self._transfer_results.pop(key, [])
            self._transfer_events.pop(key, None)
        return {
            "ok": bool(results) and all(item["ok"] for item in results),
            "results": results,
        }

    def can_offload(self, request: Request) -> bool:
        if not self.enable_offload:
            return False
        if request.is_offloaded:
            return False
        if not request.block_tables:
            return False
        if request.need_prefill_tokens is None:
            offload_logger.warning(
                f"[DEBUG: can_offload] {request.request_id}: need_prefill_tokens is None, cannot offload"
            )
            return False
        if request.num_computed_tokens < request.need_prefill_tokens:
            offload_logger.warning(
                f"[DEBUG: can_offload] {request.request_id} is not in decode phase, "
                f"num_computed_tokens={request.num_computed_tokens}, "
                f"need_prefill_tokens={request.need_prefill_tokens}, cannot offload"
            )
            return False
        return True

    def can_resume(self, request: Request) -> bool:
        if not self.enable_offload:
            return False
        if request.request_id not in self._offloaded_requests:
            return False

        offloaded_info = self._offloaded_requests.get(request.request_id)
        if offloaded_info is None or offloaded_info.get("snapshot_handle") is None:
            return False
        if self.cache_manager is None:
            return False

        return self.cache_manager.can_allocate_gpu_blocks(offloaded_info.get("num_blocks_needed", 0))

    def offload_decode(self, running_requests: List[Request], min_steps: int = 20) -> Tuple[List[Request], List[Request]]:
        if not self.enable_offload:
            return [], []

        offloaded_reqs = []
        abort_reqs = []
        remaining_count = len(running_requests)

        for req in running_requests:
            if not self.can_offload(req):
                continue

            if self.offload_req(req):
                offloaded_reqs.append(req)
                remaining_count -= 1
            else:
                abort_reqs.append(req)

            if self.cache_manager is not None and remaining_count > 0:
                block_size = self.cache_manager.cache_config.block_size
                blocks_needed_per_request = (min_steps + block_size - 1) // block_size
                total_blocks_needed = remaining_count * blocks_needed_per_request
                current_free_blocks = len(getattr(self.cache_manager, "gpu_free_block_list", []))
                if current_free_blocks >= total_blocks_needed:
                    break

        return offloaded_reqs, abort_reqs

    def offload_req(self, request: Request) -> bool:
        if not self.enable_offload or self.cache_manager is None:
            return False
        if request.is_offloaded:
            offload_logger.warning(f"[DEBUG: offload_req] Request {request.request_id} already offloaded")
            return False

        start_time = time.perf_counter()
        snapshot_task = DecodeOffloadTask(task_id=request.request_id, gpu_block_ids=list(request.block_tables))
        snapshot_result = self._issue_transfer_task(CacheStatus.DECODE_OFFLOAD, snapshot_task)
        if snapshot_result is None or not snapshot_result.get("ok", False):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            offload_logger.error(
                f"[DEBUG: offload_req] Failed to snapshot request {request.request_id}, "
                f"elapsed_ms={elapsed_ms:.2f}, result={snapshot_result}"
            )
            return False

        with self._lock:
            need_prefill_tokens = request.need_prefill_tokens
            if need_prefill_tokens is None:
                need_prefill_tokens = request.prompt_token_ids_len if request.prompt_token_ids_len else 0
            original_block_tables = list(request.block_tables) if request.block_tables else []
            self._offloaded_requests[request.request_id] = {
                "storage_level": self.STORAGE_LEVEL_CPU,
                "num_tokens": request.num_total_tokens,
                "num_blocks_needed": len(original_block_tables),
                "output_token_ids": list(request.output_token_ids),
                "num_computed_tokens": request.num_computed_tokens,
                "need_prefill_tokens": need_prefill_tokens,
                "prompt_token_ids": list(request.prompt_token_ids) if request.prompt_token_ids else None,
                "prompt_token_ids_len": request.prompt_token_ids_len,
                "sampling_params": request.sampling_params,
                "block_tables": original_block_tables,
                "snapshot_handle": request.request_id,
            }

        self.release_gpu_blocks(request)
        request.is_offloaded = True
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        offload_logger.info(
            f"[DEBUG: offload_req] Request {request.request_id} offloaded to {self.STORAGE_LEVEL_CPU}, "
            f"blocks_needed={len(original_block_tables)}, offload_time_ms={elapsed_ms:.2f}"
        )
        return True

    def offload_kv_cache(self, request: Request, target_level: str = "L2") -> bool:
        """
        Compatibility shim for future multi-level offload.
        """
        if target_level == self.STORAGE_LEVEL_CPU:
            return self.offload_req(request)
        if target_level == self.STORAGE_LEVEL_SSD:
            offload_logger.warning("[DEBUG: offload_kv_cache] SSD offload is not implemented in the first version")
            return False
        offload_logger.error(f"[DEBUG: offload_kv_cache] Invalid target_level: {target_level}")
        return False

    def release_gpu_blocks(self, request: Request) -> None:
        if self.cache_manager is None:
            return
        if request.block_tables:
            blocks_to_release = list(request.block_tables)
            self.cache_manager.recycle_gpu_blocks(blocks_to_release, request.request_id)
            request.block_tables = []

    def save_to_storage(self, kv_cache_cpu) -> Optional[str]:
        """Compatibility placeholder for future SSD offload support."""
        offload_logger.warning("[DEBUG: save_to_storage] SSD offload is not implemented in the first version")
        return None

    def load_from_storage(self, storage_path: str) -> Optional[dict]:
        """Compatibility placeholder for future SSD resume support."""
        offload_logger.warning("[DEBUG: load_from_storage] SSD resume is not implemented in the first version")
        return None

    def resume_decode(self, request: Request) -> Tuple[bool, Optional[int]]:
        if not self.enable_offload:
            return False, None

        start_time = time.perf_counter()
        with self._lock:
            offloaded_info = self._offloaded_requests.get(request.request_id)
            if offloaded_info is None:
                offload_logger.warning(f"[DEBUG: resume_decode] Request {request.request_id} is not offloaded")
                return False, None

            num_blocks_needed = offloaded_info["num_blocks_needed"]
            saved_num_computed_tokens = offloaded_info["num_computed_tokens"]
            saved_need_prefill_tokens = offloaded_info["need_prefill_tokens"]
            snapshot_handle = offloaded_info.get("snapshot_handle")
            output_token_ids = list(offloaded_info.get("output_token_ids", []))
            need_prefill_tokens = offloaded_info.get("need_prefill_tokens")

        if saved_num_computed_tokens <= saved_need_prefill_tokens:
            offload_logger.warning(
                f"[DEBUG: resume_decode] Request {request.request_id} has invalid state: "
                f"num_computed_tokens={saved_num_computed_tokens} <= need_prefill_tokens={saved_need_prefill_tokens}"
            )
            return False, saved_num_computed_tokens
        if self.cache_manager is None:
            return False, saved_num_computed_tokens
        if not self.cache_manager.can_allocate_gpu_blocks(num_blocks_needed):
            offload_logger.debug(
                f"[DEBUG: resume_decode] Not enough GPU blocks for {request.request_id}, "
                f"need={num_blocks_needed}, will retry later"
            )
            return False, saved_num_computed_tokens

        try:
            if snapshot_handle is None:
                offload_logger.warning(
                    f"[DEBUG: resume_decode] Request {request.request_id} has no snapshot handle"
                )
                return False, saved_num_computed_tokens

            new_block_ids = self.cache_manager.allocate_gpu_blocks(num_blocks_needed, request.request_id)
            request.block_tables = new_block_ids
            resume_task = DecodeResumeTask(task_id=snapshot_handle, gpu_block_ids=new_block_ids)
            resume_result = self._issue_transfer_task(CacheStatus.DECODE_RESUME, resume_task)
            if resume_result is None or not resume_result.get("ok", False):
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.cache_manager.recycle_gpu_blocks(new_block_ids, request.request_id)
                request.block_tables = []
                offload_logger.warning(
                    f"[DEBUG: resume_decode] Resume transfer failed for {request.request_id}, "
                    f"elapsed_ms={elapsed_ms:.2f}, result={resume_result}"
                )
                return False, saved_num_computed_tokens

            request.output_token_ids = output_token_ids
            request.num_computed_tokens = saved_num_computed_tokens
            request.need_prefill_tokens = need_prefill_tokens
            request.status = RequestStatus.RUNNING
            request.is_offloaded = False

            with self._lock:
                self._offloaded_requests.pop(request.request_id, None)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            offload_logger.info(
                f"[DEBUG: resume_decode] Request {request.request_id} resumed successfully, "
                f"resume_time_ms={elapsed_ms:.2f}"
            )
            return True, saved_num_computed_tokens
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            offload_logger.error(
                f"[DEBUG: resume_decode] Failed to resume request {request.request_id}, "
                f"elapsed_ms={elapsed_ms:.2f}: {e}"
            )
            return False, saved_num_computed_tokens

    def cleanup_offloaded_request(self, request_id: str) -> None:
        with self._lock:
            offloaded_info = self._offloaded_requests.pop(request_id, None)
        if offloaded_info is None:
            return

        snapshot_handle = offloaded_info.get("snapshot_handle")
        if self.cache_manager is not None and snapshot_handle is not None:
            try:
                self._issue_transfer_task(CacheStatus.DECODE_CLEANUP, DecodeCleanupTask(task_id=snapshot_handle))
            except Exception as e:
                offload_logger.warning(f"[DEBUG: offload] Failed to cleanup snapshot {snapshot_handle}: {e}")
        offload_logger.info(f"[DEBUG: offload] Cleaned up offloaded request: {request_id}")

    def get_offloaded_request_count(self) -> int:
        with self._lock:
            return len(self._offloaded_requests)

    def get_offloaded_request_ids(self) -> List[str]:
        with self._lock:
            return list(self._offloaded_requests.keys())

    def prefetch_ssd_to_cpu(self) -> int:
        """Compatibility placeholder for future SSD prefetch support."""
        return 0
