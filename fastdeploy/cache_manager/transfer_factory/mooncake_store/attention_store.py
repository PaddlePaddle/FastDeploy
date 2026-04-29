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
import traceback
from dataclasses import dataclass
from typing import List

import paddle

from fastdeploy.cache_manager.transfer_factory.kvcache_storage import (
    KVCacheStorage,
    logger,
)
from fastdeploy.platforms import current_platform

try:
    import attentionstore_sdk.api.common.common_pb2 as common_pb2
    from attentionstore_sdk.sdk import AttentionStoreSDK, Tokens
    from attentionstore_sdk.utils.err import AttentionStoreSDKError

    if current_platform.is_cuda():
        from attentionstore_sdk.client.client import AttentionType

    _ATTENTIONSTORE_AVAILABLE = True
except Exception:
    AttentionStoreSDK = None
    Tokens = None
    AttentionStoreSDKError = None
    _ATTENTIONSTORE_AVAILABLE = False


@dataclass
class AttentionStoreConfig:
    namespace: str = "default_ns"
    pod_name: str = "default_pod"
    model_version: str = "v0"
    shard_id: int = 0
    shard_num: int = 1
    layer_num: int = 1
    block_token_size: int = 64
    bytes_per_shard_layer_per_block: int = 1024
    device_id: int = 0
    dp_id: int = 0
    splitwise_role: str = "mixed"


class AttentionStore(KVCacheStorage):
    def __init__(self, **args):

        if not _ATTENTIONSTORE_AVAILABLE:
            raise ImportError("Please install attentionstore_sdk to run Fastdeploy with attentionstore_sdk.")

        self.config = AttentionStoreConfig(**args)

        try:
            self.config.namespace = os.getenv("AS_NAMESPACE", self.config.namespace)
            self.config.pod_name = os.getenv("AS_POD_NAME", self.config.pod_name)
            if int(os.getenv("ENABLE_EP_DP_IN_FD", "1")):
                self.config.pod_name = (
                    self.config.pod_name + "_" + self.config.splitwise_role + "_" + str(self.config.dp_id)
                )
            self.config.model_version = os.getenv("AS_MODEL_VERSION", self.config.model_version)
            logger.info(f"[INIT] Start initializing AttentionStoreSDK with config: {self.config}")
            if current_platform.is_cuda():
                self.sdk = AttentionStoreSDK(
                    self.config.namespace,
                    self.config.pod_name,
                    self.config.model_version,
                    self.config.shard_id,
                    self.config.shard_num,
                    self.config.layer_num,
                    self.config.block_token_size,
                    self.config.bytes_per_shard_layer_per_block,
                    self.config.bytes_per_shard_layer_per_block,
                    self.config.device_id,
                    self.config.dp_id,
                    attention_type=AttentionType.MHA,
                    enable_as_kv_rw=True,
                    gpu_count=0,
                )
            else:
                self.sdk = AttentionStoreSDK(
                    self.config.namespace,
                    self.config.pod_name,
                    self.config.model_version,
                    self.config.shard_id,
                    self.config.shard_num,
                    self.config.layer_num,
                    self.config.block_token_size,
                    self.config.bytes_per_shard_layer_per_block,
                    self.config.device_id,
                    self.config.dp_id,
                )
            self.wait_for_sdk_ready(timeout=300, delta_t=5)
            logger.info("[INIT] ✅ AttentionStore is initialized successfully!")
        except Exception as e:
            logger.error(
                f"[INIT] ❌ AttentionStore initialization failed, error: {e}, traceback:\n{traceback.format_exc()}"
            )
            raise

    def wait_for_sdk_ready(self, timeout: float, delta_t: float):
        t = 0
        while t < timeout:
            try:
                tokens = Tokens(list(range(self.config.block_token_size + 1)), self.config.block_token_size)
                self.sdk.match(tokens, 0, delta_t)
                return
            except AttentionStoreSDKError as e:
                if "cuda memory not ready" in str(e):
                    logger.debug("[INIT] cuda memory not ready, try again..")
                    time.sleep(delta_t)
                    t += delta_t
                    continue
                else:
                    raise RuntimeError(
                        f"Unexpected exception during AttentionStoreSDK initialization: {e}\n{traceback.format_exc()}"
                    )
        raise TimeoutError(f"AttentionStoreSDK initialization timed out after {timeout} seconds")

    def read(
        self,
        task_id: str,
        key_cache: List[paddle.Tensor],
        val_cache: List[paddle.Tensor],
        token_ids: List[int],
        gpu_block_ids: List[int],
        start_read_block_idx: int,
        timeout: float = 30.0,
    ):
        logger.debug(
            f"[READ BEGIN] task_id: {task_id} token_ids: {token_ids} gpu_block_ids: {gpu_block_ids} start_read_block_idx: {start_read_block_idx} timeout: {timeout}"
        )
        tokens = Tokens(token_ids, self.config.block_token_size)
        k_data_ptrs = [k.data_ptr() for k in key_cache]
        v_data_ptrs = [v.data_ptr() for v in val_cache]
        num = 0
        try:
            if current_platform.is_cuda():
                num = self.sdk.read(
                    list(range(self.config.layer_num)),
                    tokens,
                    start_read_block_idx,
                    k_data_ptrs,
                    v_data_ptrs,
                    gpu_block_ids,
                    timeout,
                    remote_addrs=None,
                )
            else:
                num = self.sdk.read(
                    list(range(self.config.layer_num)),
                    tokens,
                    start_read_block_idx,
                    k_data_ptrs,
                    v_data_ptrs,
                    gpu_block_ids,
                    timeout,
                )
            logger.debug(f"[READ END] task_id: {task_id} read_blocks: {num}")
        except AttentionStoreSDKError:
            logger.error(
                f"[READ ERROR] failed to execute sdk read, task_id: {task_id}, traceback:\n{traceback.format_exc()}"
            )
        return num

    def write(
        self,
        task_id: str,
        key_cache: List[paddle.Tensor],
        val_cache: List[paddle.Tensor],
        token_ids: List[int],
        gpu_block_ids: List[int],
        start_write_block_idx: int,
        timeout: float = 30.0,
    ) -> int:
        k_data_ptrs = [k.data_ptr() for k in key_cache]
        v_data_ptrs = [v.data_ptr() for v in val_cache]
        layer_ids = list(range(self.config.layer_num))
        block_token_size = self.config.block_token_size

        total_timeout = float(os.getenv("AS_WRITE_TOTAL_TIMEOUT", str(timeout)))
        slice_block_num = int(os.getenv("AS_WRITE_SLICE_BLOCK_NUM", "100"))
        slice_timeout = float(os.getenv("AS_WRITE_SLICE_TIMEOUT", "10"))
        logger.debug(
            f"[WRITE BEGIN] task_id: {task_id} token_ids: {token_ids} gpu_block_ids: {gpu_block_ids}"
            f" start_write_block_idx: {start_write_block_idx} timeout: {total_timeout}"
        )
        total_blocks = len(gpu_block_ids)
        total_written = 0
        overall_start = time.time()

        for slice_start in range(0, total_blocks, slice_block_num):
            elapsed = time.time() - overall_start
            remaining_timeout = total_timeout - elapsed
            if remaining_timeout <= 0:
                logger.warning(
                    f"[WRITE TIMEOUT] task_id: {task_id} total timeout {total_timeout}s reached, "
                    f"written {total_written}/{total_blocks} blocks"
                )
                break

            slice_end = min(slice_start + slice_block_num, total_blocks)
            slice_gpu_block_ids = gpu_block_ids[slice_start:slice_end]
            slice_write_block_idx = start_write_block_idx + slice_start
            slice_token_ids = token_ids[: (start_write_block_idx + slice_end) * block_token_size]
            slice_tokens = Tokens(slice_token_ids, block_token_size)

            logger.debug(
                f"[WRITE SLICE BEGIN] task_id: {task_id} slice [{slice_start}:{slice_end}] "
                f"block_idx={slice_write_block_idx}, blocks={len(slice_gpu_block_ids)}, "
                f"token_ids_len={len(slice_token_ids)}, timeout={slice_timeout:.2f}s"
            )
            slice_start_time = time.time()
            try:
                if current_platform.is_cuda():
                    written = self.sdk.write(
                        layer_ids,
                        slice_tokens,
                        slice_write_block_idx,
                        k_data_ptrs,
                        v_data_ptrs,
                        slice_gpu_block_ids,
                        slice_timeout,
                        h2h_copy=False,
                        params=None,
                    )
                else:
                    written = self.sdk.write(
                        layer_ids,
                        slice_tokens,
                        slice_write_block_idx,
                        k_data_ptrs,
                        v_data_ptrs,
                        slice_gpu_block_ids,
                        slice_timeout,
                    )
            except AttentionStoreSDKError:
                logger.error(
                    f"[WRITE ERROR] task_id: {task_id} slice [{slice_start}:{slice_end}], "
                    f"traceback:\n{traceback.format_exc()}"
                )
                written = 0
            slice_cost = time.time() - slice_start_time
            total_written += written

            if written < len(slice_gpu_block_ids):
                logger.warning(
                    f"[WRITE SLICE INCOMPLETE] task_id: {task_id} slice [{slice_start}:{slice_end}] "
                    f"({written}/{len(slice_gpu_block_ids)}), cost={slice_cost:.6f}s, "
                    f"total written {total_written}/{total_blocks}, "
                    f"prefix cache continuity broken, skip remaining slices"
                )
                break

            logger.debug(
                f"[WRITE SLICE END] task_id: {task_id} slice [{slice_start}:{slice_end}] "
                f"written={written}, cost={slice_cost:.6f}s"
            )

        total_cost = time.time() - overall_start
        logger.info(
            f"[WRITE END] task_id: {task_id} total cost={total_cost:.6f}s, "
            f"written {total_written}/{total_blocks} blocks"
        )
        return total_written

    def query(self, task_id: str, token_ids: List[int], start_match_block_idx: int, timeout: float = 10.0):
        """
        Given the input ids and starting index to match, get the valid blocks number that
        can be prefetched from storage backend.
        """
        logger.debug(
            f"[QUERY BEGIN] task_id: {task_id} token_ids: {token_ids} start_match_block_idx: {start_match_block_idx} timeout: {timeout}"
        )
        tokens = Tokens(token_ids, self.config.block_token_size)
        num = 0
        try:
            num = self.sdk.match(tokens, start_match_block_idx, timeout)
            logger.debug(f"[QUERY END] task_id: {task_id} matched_blocks: {num}")
        except AttentionStoreSDKError:
            logger.error(
                f"[QUERY ERROR] Failed to execute sdk match, task_id: {task_id}, traceback:\n{traceback.format_exc()}"
            )
        return num

    def flush_token_index(self, task_id: str, token_ids: List[int], start_block_idx: int, reside_in_gpu: bool):
        logger.debug(
            f"[FLUSH BEGIN] task_id: {task_id} token_ids: {token_ids} start_block_idx: {start_block_idx} reside_in_gpu: {reside_in_gpu}"
        )
        tokens = Tokens(token_ids, self.config.block_token_size)
        try:
            if reside_in_gpu:
                self.sdk.flush_token_index(
                    list(range(self.config.layer_num)),
                    tokens,
                    start_block_idx,
                    None,
                    common_pb2.MEDIA_HBM,
                )
            else:
                self.sdk.flush_token_index(
                    list(range(self.config.layer_num)),
                    tokens,
                    start_block_idx,
                    common_pb2.MEDIA_HBM,
                    None,
                )
            logger.debug(f"[FLUSH END] task_id: {task_id}")
        except AttentionStoreSDKError:
            logger.error(
                f"[FLUSH ERROR] Failed to execute sdk flush_token_index, task_id: {task_id}, traceback:\n{traceback.format_exc()}"
            )

    def get(self, **kwargs):
        raise NotImplementedError("AttentionStore does not support this method")

    def batch_get(self, **kwargs):
        raise NotImplementedError("AttentionStore does not support this method")

    def set(self, **kwargs) -> bool:
        raise NotImplementedError("AttentionStore does not support this method")

    def batch_set(self, **kwargs) -> bool:
        raise NotImplementedError("AttentionStore does not support this method")

    def exists(self, keys: List[str]) -> bool:
        raise NotImplementedError("AttentionStore does not support this method")

    def clear(self) -> bool:
        raise NotImplementedError("AttentionStore does not support this method")

    def register_buffer(self, buffer_ptr, buffer_size, buffer_type="none_type") -> None:
        raise NotImplementedError("AttentionStore does not support this method")
