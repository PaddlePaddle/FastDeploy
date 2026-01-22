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
            try:
                with open(file_path) as fin:
                    config = json.load(fin)
            except Exception as e:
                logger.error(f"Load FileStoreConfig failed: {e}")
                raise

        storage_dir = config.get(
            "file_path", os.environ.get("FD_FILESTORE_DIR", "/tmp/fastdeploy_cache")
        )
        namespace = config.get("namespace", os.environ.get("FD_FILESTORE_NAMESPACE"))
        tp_rank = config.get("tp_rank", os.environ.get("FD_FILESTORE_TP_RANK"))
        tp_size = config.get("tp_size", os.environ.get("FD_FILESTORE_TP_SIZE"))

        logger.info(f"File Configuration loaded: {storage_dir}, namespace: {namespace}, tp_rank: {tp_rank}, tp_size: {tp_size}")
        
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
        file_path: str = "/tmp/fastdeploy_cache",
    ):
        try:
            logger.info(f"Using FileStore storage backend...")

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
                logger.info(f"Successfully created FileStore storage directory at {self.file_path}")
        except Exception as e:
            logger.error(f"File store initialization failed: {e}, traceback: {traceback.format_exc()}")
            raise

    def register_buffer(self, buffer_ptr, buffer_size) -> None:
        # FileStore does not need to register buffers.
        return None

    # def _get_suffixed_key(self, key: str) -> str:
    #     return key + self.config_suffix

    def _get_tensor_path(self, key: str) -> str:
        clean_key = key.split('_key_')[0].split('_value_')[0]
        
        # 文件名区分分片。如果是非 TP，会生成 data.pd
        if self.storage_config.tp_rank is not None:
            name = f"rank_{self.storage_config.tp_rank}.pd"
        else:
            name = "data.pd"
            
        return os.path.join(self.file_path, clean_key, name)

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

    def set(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_size: Optional[Any] = None,
    ) -> bool:
        logger.info(f"Set key {key} in FileStore storage...")
        tensor_path = self._get_tensor_path(key)
        if os.path.exists(tensor_path):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        try:
            key_dir = os.path.dirname(tensor_path)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir, exist_ok=True)

            if isinstance(target_location, paddle.Tensor):
                tensor2save = target_location.cpu()
                paddle.save(tensor2save, tensor_path)
                os.fsync(os.open(os.path.dirname(tensor_path), os.O_RDONLY))
            elif isinstance(target_location, int) and target_size is not None:
                tensor = self._tensor_from_ptr(target_location, int(target_size))
                paddle.save(tensor2save, tensor_path)
            else:
                raise ValueError("target_location must be a paddle.Tensor or a pointer int with target_size.")
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        target_locations: Optional[List[Any]] = None,
        target_sizes: Optional[List[Any]] = None,
    ) -> List[int]:
        logger.info(f"Batch set {len(keys)} keys in FileStore storage...")
        results = []
        try:
            target_locations = target_locations or [None] * len(keys)
            target_sizes = target_sizes or [None] * len(keys)

            if len(target_locations) != len(keys) or len(target_sizes) != len(keys):
                logger.error(f"Length of target_locations ({len(target_locations)}) or target_sizes ({len(target_sizes)}) does not match length of keys ({len(keys)}).")
                return [-1] * len(keys)

            for key, loc, size in zip(keys, target_locations, target_sizes):
                ok = self.set(key, target_location=loc, target_size=size)
                results.append(0 if ok else -1)
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
        target_location: Optional[Any] = None,
        target_size: Optional[int] = None,
    ) -> Optional[Any]:
        tensor_path = self._get_tensor_path(key)
        if not os.path.exists(tensor_path):
            logger.warning(f"Failed to fetch {key} from FileStore storage.")
            return None
        try:
            loaded = paddle.load(tensor_path)
            if target_location is None:
                return loaded
            if isinstance(target_location, paddle.Tensor):
                if list(loaded.shape) != list(target_location.shape):
                    loaded = paddle.reshape(loaded, target_location.shape)
                paddle.assign(loaded, output=target_location)
                return target_location
            if isinstance(target_location, int) and target_size is not None:
                if target_size <= 0:
                    logger.error(f"Invalid target_size: {target_size}")
                    return None
                if not loaded.is_contiguous():
                    loaded = loaded.contiguous()
                return self._copy_tensor_to_ptr(loaded, target_location, target_size)
            logger.error(f"Unsupported target_location type: {type(target_location)}")
            return None
        except (FileNotFoundError, pickle.UnpicklingError, ValueError) as e:
            logger.error(f"Failed to load tensor {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading tensor {key}: {e}")
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
        res = {}
        for k in keys:
            p = self._get_tensor_path(k)
            found = os.path.exists(p)
            # 这一行非常重要，能告诉你卡 2 到底有没有来“找”过
            logger.info(f"--- [CACHE_CHECK] Key: {k[:10]}... Path: {p} Found: {found} ---")
            res[k] = found
        return res

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                elif os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in FileStore storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear FileStore storage: {e}")
            return False
