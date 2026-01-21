import json
import os
import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import paddle

from fastdeploy.cache_manager.transfer_factory.kvcache_storage import (
    KVCacheStorage,
    logger,
)

import time # for debug

@dataclass
class FileStoreConfig:
    file_path: str
    namespace: Optional[str] = None
    tp_rank: Optional[int] = None
    tp_size: Optional[int] = None

    @staticmethod
    def create() -> "FileStoreConfig":
        config = {}
        file_path = os.getenv("FD_FILESTORE_CONFIG_PATH")

        if file_path is not None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"File path {file_path} for creating FileStoreConfig does not exist."
                )
            with open(file_path) as fin:
                config = json.load(fin)

        storage_dir = config.get(
            "file_path", os.environ.get("FD_FILESTORE_DIR", "/ssd1/cache_manager/file_store")
        )
        namespace = config.get("namespace", os.environ.get("FD_FILESTORE_NAMESPACE"))
        tp_rank = config.get("tp_rank", os.environ.get("FD_FILESTORE_TP_RANK"))
        tp_size = config.get("tp_size", os.environ.get("FD_FILESTORE_TP_SIZE"))

        return FileStoreConfig(
            file_path=storage_dir,
            namespace=namespace,
            tp_rank=None if tp_rank is None else int(tp_rank),
            tp_size=None if tp_size is None else int(tp_size),
        )


class FileStore(KVCacheStorage):
    def __init__(
        self,
        storage_config: FileStoreConfig,
        file_path: str = "/ssd1/cache_manager/file_store",
    ):
        ######## debug ########
        logger.info(f"{time.localtime()}:[DEBUG] Using FileStore storage backend")
        #########################
        self.storage_config = storage_config
        self.file_path = os.getenv("FD_FILESTORE_DIR", storage_config.file_path or file_path)

        suffix_parts = []
        if storage_config.namespace:
            suffix_parts.append(storage_config.namespace)
        if storage_config.tp_rank is not None and storage_config.tp_size is not None:
            suffix_parts.append(f"{storage_config.tp_rank}_{storage_config.tp_size}")
        self.config_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

        if not os.path.exists(self.file_path) and storage_config.tp_rank in (None, 0):
            os.makedirs(self.file_path, exist_ok=True)
            logger.info(f"Created FileStore storage directory at {self.file_path}")

    def register_buffer(self, buffer_ptr, buffer_size) -> None:
        # FileStore does not need to register buffers.
        return None

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_tensor_path(self, key: str) -> str:
        key = self._get_suffixed_key(key)
        return os.path.join(self.file_path, f"{key}.pd")

    def _tensor_from_ptr(self, ptr: int, size: int) -> paddle.Tensor:
        raw = ctypes.string_at(ptr, size)
        arr = np.frombuffer(raw, dtype="uint8")
        return paddle.to_tensor(arr, place="cpu")

    def _copy_tensor_to_ptr(self, tensor: paddle.Tensor, ptr: int, size: int) -> int:
        if not isinstance(tensor, paddle.Tensor):
            return -1
        data = tensor.numpy().tobytes()
        if len(data) < size:
            return -1
        ctypes.memmove(ptr, data, size)
        return size

    def set(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> bool:

        ######## debug ########
        logger.info(f"{time.localtime()}:[DEBUG] Setting key {key} in FileStore storage")
        #########################
        tensor_path = self._get_tensor_path(key)
        if os.path.exists(tensor_path):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        try:
            if isinstance(target_location, paddle.Tensor):
                paddle.save(target_location, tensor_path)
                return True
            if isinstance(target_location, int) and target_size is not None:
                tensor = self._tensor_from_ptr(target_location, int(target_size))
                paddle.save(tensor, tensor_path)
                return True
            raise ValueError("target_location must be a paddle.Tensor or a pointer int with target_size.")
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[int]:
        logger.info(f"{time.localtime()}:[DEBUG] Batch set {len(keys)} keys in FileStore storage")
        results = []
        target_locations = target_locations or [None] * len(keys)
        target_sizes = target_sizes or [None] * len(keys)
        for key, loc, size in zip(keys, target_locations, target_sizes):
            ok = self.set(key, target_location=loc, target_size=size)
            results.append(0 if ok else -1)
        return results

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> Any | None:
        tensor_path = self._get_tensor_path(key)
        if not os.path.exists(tensor_path):
            logger.warning(f"Failed to fetch {key} from FileStore storage.")
            return None
        try:
            loaded = paddle.load(tensor_path)
            if target_location is None:
                return loaded
            if isinstance(target_location, paddle.Tensor):
                paddle.assign(loaded, output=target_location)
                return target_location
            if isinstance(target_location, int) and target_size is not None:
                return self._copy_tensor_to_ptr(loaded, target_location, int(target_size))
            return loaded
        except Exception as e:
            logger.error(f"Failed to load tensor {key}: {e}")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[Any | None]:
        target_locations = target_locations or [None] * len(keys)
        target_sizes = target_sizes or [None] * len(keys)
        logger.info(f"{time.localtime()}:[DEBUG] Batch get {len(keys)} keys from FileStore storage")
        return [
            self.get(key, target_location=loc, target_size=size)
            for key, loc, size in zip(keys, target_locations, target_sizes)
        ]

    def exists(self, keys: List[str]) -> Dict[str, bool]:
        return {key: os.path.exists(self._get_tensor_path(key)) for key in keys}

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in FileStore storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear FileStore storage: {e}")
            return False
