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

import ctypes
import os
import pickle
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.cache_manager.transfer_factory.kvcache_storage import (
    KVCacheStorage,
    logger,
)


@dataclass
class FileStoreConfig:
    file_path: str = envs.FILE_BACKEND_STORAGE_DIR
    namespace: Optional[str] = ""
    tp_rank: Optional[int] = 0
    tp_size: Optional[int] = 1


class FileStore(KVCacheStorage):
    def __init__(self, **args):
        try:
            logger.info("Using FileStore storage backend...")

            self.storage_config = FileStoreConfig(**args)
            self.file_path = self.storage_config.file_path
            if self.file_path is None:
                raise ValueError("file_path must be specified for FileStore backend")

            if self.storage_config.namespace:
                self.file_path = os.path.join(self.file_path, self.storage_config.namespace)

            if not os.path.exists(self.file_path):
                if self.storage_config.tp_rank in (None, 0):
                    os.makedirs(self.file_path, exist_ok=True)
                    logger.info(f"Successfully created FileStore storage directory at {self.file_path}")
                else:
                    logger.info(f"Skip mkdir on non-zero tp_rank={self.storage_config.tp_rank}")
            logger.info(
                f"[INIT] FileStore initialized successfully! "
                f"path={self.file_path}, "
                f"config={self.storage_config}"
            )
        except Exception as e:
            logger.error(f"File store initialization failed: {e}, traceback: {traceback.format_exc()}")
            raise

    def register_buffer(self, buffer_ptr, buffer_size) -> None:
        # FileStore does not need to register buffers.
        return None

    def _get_tensor_path(self, key: str) -> str:
        return os.path.join(self.file_path, f"{key}.pd")

    def _tensor_from_ptr(self, ptr: int, size: int) -> paddle.Tensor:
        raw = ctypes.string_at(ptr, size)
        arr = np.frombuffer(raw, dtype="uint8")
        return paddle.to_tensor(arr, place="cpu")

    def _copy_tensor_to_ptr(self, tensor: paddle.Tensor, ptr: int, size: int) -> int:
        if not isinstance(tensor, paddle.Tensor):
            return -1
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        data = tensor.numpy().tobytes()
        actual_size = len(data)
        if actual_size < size:
            logger.error(f"Data size mismatch: tensor {actual_size} < target {size}")
            return -1
        ctypes.memmove(ptr, data, size)
        return size

    def query(
        self, k_cache_keys: Optional[List[str]] = None, v_cache_keys: Optional[List[str]] = None, timeout: float = 10.0
    ) -> int:
        try:
            if not k_cache_keys or not v_cache_keys:
                return 0

            assert len(k_cache_keys) == len(v_cache_keys), "k_cache_keys and v_cache_keys must have the same length."

            all_keys = k_cache_keys + v_cache_keys
            results = self.exists(all_keys)

            matched_count = 0
            for k, v in zip(k_cache_keys, v_cache_keys):
                if results[k] and results[v]:
                    matched_count += 1

            logger.info(
                f"FileStore query: checked {len(k_cache_keys)} block pairs, matched {matched_count} complete blocks"
            )
            return matched_count

        except Exception as e:
            logger.error(f"Failed to query FileStore storage: {e}")
            return 0

    def set(
        self,
        key: str,
        target_location: int,
        target_size: int,
    ) -> int:
        logger.info(f"Set key {key} in FileStore storage...")
        tensor_path = self._get_tensor_path(key)
        if os.path.exists(tensor_path):
            logger.debug(f"Key {key} already exists. Skipped.")
            return 0
        try:
            tensor = self._tensor_from_ptr(target_location, int(target_size))
            paddle.save(tensor, tensor_path)
            file_fd = os.open(tensor_path, os.O_RDONLY)
            try:
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            return 0
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return -1

    def batch_set(
        self,
        keys: List[str],
        target_locations: List[int],
        target_sizes: List[int],
    ) -> List[int]:
        logger.info(f"Batch set {len(keys)} keys in FileStore storage...")
        results = []
        try:
            if len(target_locations) != len(keys) or len(target_sizes) != len(keys):
                logger.error(
                    f"Length of target_locations ({len(target_locations)}) or target_sizes ({len(target_sizes)}) does not match length of keys ({len(keys)})."
                )
                return [-1] * len(keys)

            for key, loc, size in zip(keys, target_locations, target_sizes):
                ok = self.set(key, target_location=loc, target_size=size)
                results.append(ok)
            return results
        except (ValueError, TypeError) as e:
            logger.error(f"Input validation failed in batch_set: {e}")
            return [-1] * len(keys)
        except OSError as e:
            logger.error(f"File system error in batch_set: {e}")
            return [-1] * len(keys)
        except Exception as e:
            logger.error(f"Unexpected error in batch_set: {e}")
            return [-1] * len(keys)

    def get(
        self,
        key: str,
        target_location: int,
        target_size: int,
    ) -> int:
        tensor_path = self._get_tensor_path(key)
        if not os.path.exists(tensor_path):
            logger.warning(f"Failed to fetch {key} from FileStore storage.")
            return -1
        try:
            loaded = paddle.load(tensor_path)

            if target_size <= 0:
                logger.error(f"Invalid target_size: {target_size}")
                return -1
            if not loaded.is_contiguous():
                loaded = loaded.contiguous()
            return self._copy_tensor_to_ptr(loaded, target_location, target_size)

        except (FileNotFoundError, pickle.UnpicklingError, ValueError) as e:
            logger.error(f"Failed to load tensor {key}: {e}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected error loading tensor {key}: {e}")
            return -1

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[int],
        target_sizes: List[int],
    ) -> List[int]:
        num_keys = len(keys)

        if len(target_locations) != num_keys or len(target_sizes) != num_keys:
            logger.error(
                f"Length of target_locations ({len(target_locations)}) or target_sizes ({len(target_sizes)}) "
                f"does not match length of keys ({num_keys})."
            )
            return [-1] * num_keys

        logger.debug(f"{time.localtime()}:[DEBUG] Batch get {num_keys} keys from FileStore storage")
        results = []

        for i in range(num_keys):
            res = self.get(keys[i], target_location=target_locations[i], target_size=target_sizes[i])
            results.append(res)
            if res < 0:
                logger.warning(f"Failed to get key {keys[i]}")

        return results

    def exists(self, keys: List[str]) -> Dict[str, bool]:
        res = {}
        for k in keys:
            p = self._get_tensor_path(k)
            found = os.path.exists(p)
            logger.debug(f"--- [CACHE_CHECK] Key: {k[:10]}... Path: {p} Found: {found} ---")
            res[k] = found
        return res

    def clear(self) -> bool:
        try:
            path = self.file_path
            if path in ("/", ""):
                raise RuntimeError(f"Refuse to clear dangerous path: {path}")
            subprocess.run(["bash", "-c", f"rm -f '{path}'/*.pd"], check=True, stderr=subprocess.DEVNULL)
            logger.info(f"Cleared all .pd entries in FileStore storage at {path}.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clear FileStore storage: {e}")
            return False
