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

import time
import traceback
from dataclasses import dataclass
from typing import List

import paddle

from fastdeploy.cache_manager.transfer_factory.kvcache_storage import (
    KVCacheStorage,
    logger,
)

try:
    import attentionstore_sdk.api.common.common_pb2 as common_pb2
    from attentionstore_sdk.sdk import AttentionStoreSDK, Tokens
    from attentionstore_sdk.utils.err import AttentionStoreSDKError

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


class AttentionStore(KVCacheStorage):
    def __init__(self, **args):

        if not _ATTENTIONSTORE_AVAILABLE:
            raise ImportError("Please install attentionstore_sdk to run Fastdeploy with attentionstore_sdk.")

        self.config = AttentionStoreConfig(**args)

        try:
            logger.info(f"[INIT] Start initializing AttentionStoreSDK with config: {self.config}")
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
        logger.debug(
            f"[WRITE BEGIN] task_id: {task_id} token_ids: {token_ids} gpu_block_ids: {gpu_block_ids} start_write_block_idx: {start_write_block_idx} timeout: {timeout}"
        )
        tokens = Tokens(token_ids, self.config.block_token_size)
        k_data_ptrs = [k.data_ptr() for k in key_cache]
        v_data_ptrs = [v.data_ptr() for v in val_cache]
        num = 0
        try:
            num = self.sdk.write(
                list(range(self.config.layer_num)),
                tokens,
                start_write_block_idx,
                k_data_ptrs,
                v_data_ptrs,
                gpu_block_ids,
                timeout,
            )
            logger.debug(f"[WRITE END] task_id: {task_id} written_blocks: {num}")
        except AttentionStoreSDKError:
            logger.error(
                f"[WRITE ERROR] failed to execute sdk write, task_id: {task_id}, traceback:\n{traceback.format_exc()}"
            )
        return num

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
