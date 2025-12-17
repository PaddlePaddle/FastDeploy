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

import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

from fastdeploy.cache_manager.transfer_factory.kvcache_storage import (
    KVCacheStorage,
    logger,
)

DEFAULT_GLOBAL_SEGMENT_SIZE = 1024 * 1024 * 1024  # 1 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 128 * 1024 * 1024  # 128MB


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    rdma_devices: str
    master_server_addr: str

    @staticmethod
    def load_from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if file_path is None:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE),
            local_buffer_size=config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE),
            protocol=config.get("protocol", "rdma"),
            rdma_devices=config.get("rdma_devices", ""),
            master_server_addr=config.get("master_server_addr"),
        )


class MooncakeStore(KVCacheStorage):
    def __init__(self):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake store by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/python-api-reference/mooncake-store.html"
                "to run Fastdeploy with mooncake store."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_file()
            logger.info(f"Mooncake Configuration loaded, {self.config}.")

            ret_code = self.store.setup(
                local_hostname=self.config.local_hostname,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size,
                local_buffer_size=self.config.local_buffer_size,
                protocol=self.config.protocol,
                rdma_devices=self.config.rdma_devices,
                master_server_addr=self.config.master_server_addr,
            )
            if ret_code != 0:
                logger.error(f"failed to setup mooncake store, error code: {ret_code}")
                raise RuntimeError(f"failed to setup mooncake store, error code: {ret_code}")
            logger.info("Connect to Mooncake store successfully.")

            self.warmup()
            logger.info("Mooncake store warmup successfully.")
        except Exception as e:
            logger.error(f"Mooncake store initialization failed: {e}, traceback: {traceback.format_exc()}")
            raise

    def warmup(self):
        warmup_key = "fastdeploy_mooncake_store_warmup_key" + str(uuid.uuid4())
        warmup_value = bytes(1 * 1024 * 1024)  # 1 MB
        self.store.put(warmup_key, warmup_value)
        assert self.store.is_exist(warmup_key) == 1
        self.store.get(warmup_key)
        self.store.remove(warmup_key)

    def register_buffer(self, buffer_ptr, buffer_size) -> None:
        try:
            ret_code = self.store.register_buffer(buffer_ptr, buffer_size)
            if ret_code:
                logger.error(f"failed to register buffer, error code: {ret_code}")
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

    def set(
        self,
        key,
        target_location: Optional[List[int]] = None,
        target_size: Optional[List[int]] = None,
    ) -> List[int]:
        pass

    def batch_set(
        self,
        keys: List[str],
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Batch put multiple objects into the store.
        Args:
            keys (list): list of object names to be stored
            target_locations (list): list of memory locations where the data are stored
            target_sizes (list): list of byte sizes corresponding to each object
        Return:
            List[int]: List of status codes for each operation (0 = success, negative = error)
        """
        if not (len(keys) == len(target_locations) == len(target_sizes)):
            err_msg = "The length of keys, target_location and target_sizes must match."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if len(keys) == 0:
            err_msg = "The length of keys, target_location and target_sizes must be greater than zero"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return self._put_batch_zero_copy_impl(keys, target_locations, target_sizes)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> List[int]:
        pass

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[int]:
        """
        Batch get multiple objects from the store.
        Args:
            keys (list): list of object names to be fetched
            target_locations (list): list of memory locations where the data should be stored
            target_sizes (list): list of byte sizes corresponding to each object
        Returns:
            List[int]: List of bytes read for each operation (positive = success, negative = error)
        """
        if not (len(keys) == len(target_locations) == len(target_sizes)):
            err_msg = "The length of keys, target_locations and target_sizes must match."
            logger.error(err_msg)
            raise ValueError(err_msg)

        if len(keys) == 0:
            err_msg = "The length of keys, target_locations and target_sizes must be greater than zero"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return self._get_batch_zero_copy_impl(keys, target_locations, target_sizes)

    def exists(self, keys: List[str]):
        """
        Check existence of multiple objects in a single batch operation.
        Args:
            keys (list): list of object names to be checked
        Returns:
            dict: dictionary mapping each key to its existence status {key: True|False}
        """
        tic = time.time()
        result = {k: v for k, v in zip(keys, self.store.batch_is_exist(keys))}
        cost_time = (time.time() - tic) * 1000
        logger.debug(f"mooncake store exists results: {result}, cost_time: {cost_time:.3f}ms")
        return result

    def delete(self, key, timeout=5) -> bool:
        while timeout:
            result = self.store.remove(key)
            if result == 0:
                logger.info("Successfully removed")
                return True
            else:
                time.sleep(1)
                timeout -= 1
        return False

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def clear(self) -> bool:
        """
        clear all the objects in the store
        """
        count = self.store.remove_all()
        logger.info(f"Removed {count} objects")
        return True

    def _put_batch_zero_copy_impl(self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]) -> int:
        try:
            tic = time.time()
            result = self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)
            # List[int]: List of status codes for each operation (0 = success, negative = error)
            cost_time = time.time() - tic

            success_num = result.count(0)
            if success_num == len(key_strs):
                logger.debug(
                    f"Put all data into Mooncake Store successfully. key_strs: {key_strs}, "
                    f"success_num: {success_num}, cost_time: {cost_time:.6f}s"
                )
            else:
                logger.error(
                    f"Some of the data was not put into Mooncake Store. key_strs: {key_strs}, "
                    f"result: {result}, success_num: {success_num}, cost_time: {cost_time:.6f}s"
                )
            if success_num > 0:
                total_bytes = sum(bi for ri, bi in zip(result, buffer_sizes) if ri == 0)
                total_gb = total_bytes / 1073741824
                speed = total_gb / cost_time
                logger.info(f"Put data into Mooncake Store, total_gb: {total_gb:.6f}GB, speed: {speed:.6f}GB/s")

            return result
        except Exception as err:
            logger.error("Failed to put data into Mooncake Store: %s", err)
            raise

    def _get_batch_zero_copy_impl(self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]) -> int:
        try:
            tic = time.time()
            result = self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
            # List[int]: List of bytes read for each operation (positive = success, negative = error)
            cost_time = time.time() - tic

            success_num = sum(x > 0 for x in result)
            if success_num == len(key_strs):
                logger.debug(
                    f"Get all data from Mooncake Store successfully. key_strs: {key_strs}, "
                    f"success_num: {success_num}, cost_time: {cost_time:.6f}s"
                )
            else:
                logger.error(
                    f"Some of the data was not get from Mooncake Store. key_strs:{key_strs}, "
                    f"result:{result}, success_num: {success_num}, cost_time: {cost_time:.6f}s"
                )
            if success_num > 0:
                total_bytes = sum(bi for ri, bi in zip(result, buffer_sizes) if ri > 0)
                total_gb = total_bytes / 1073741824
                speed = total_gb / cost_time
                logger.info(f"Get data from Mooncake Store, total_gb: {total_gb:.6f}GB, speed: {speed:.6f}GB/s")

            return result
        except Exception as err:
            logger.error("Failed to get data from Mooncake Store: %s", err)
            raise
