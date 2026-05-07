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

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from fastdeploy.config import CacheConfig

from ..metadata import StorageType
from .base import StorageConnector, StorageScheduler


def create_storage_scheduler(
    config: Any,
) -> Optional[StorageScheduler]:
    """
    Create a StorageScheduler instance based on configuration.

    This is a factory function that creates the appropriate StorageScheduler
    based on the storage backend type specified in the configuration.

    Args:
        config: Configuration object, can be:
            - CacheConfig: FastDeploy configuration object
            - Dict: Dictionary with 'storage_type' and backend-specific settings
            - StorageConfig: StorageConfig dataclass instance

    Returns:
        StorageScheduler instance if successful, None otherwise

    Example:
        # Using CacheConfig
        scheduler = create_storage_scheduler(fd_config)

        # Using dict config
        config = {
            'storage_type': 'mooncake',
            'server_addr': 'localhost:8080',
            'namespace': 'kv_cache',
        }
        scheduler = create_storage_scheduler(config)
    """
    if config.kvcache_storage_backend is None:
        return None

    scheduler: Optional[StorageScheduler] = None

    # Create scheduler based on storage type
    if config.kvcache_storage_backend == "mooncake":
        from .mooncake.connector import MooncakeStorageScheduler

        scheduler = MooncakeStorageScheduler(config)

    elif config.kvcache_storage_backend == "attention_store":
        from .attnstore.connector import AttnStoreScheduler

        scheduler = AttnStoreScheduler(config)

    else:
        raise ValueError(
            f"Unsupported storage type: {config.kvcache_storage_backend}. "
            f"Supported types: mooncake, attention_store, local"
        )

    # Attempt connection
    if scheduler is not None:
        if not scheduler.connect():
            # Log warning but still return the scheduler
            pass

    return scheduler


def create_storage_connector(
    config: Any,
) -> Optional[StorageConnector]:
    """
    Create a StorageConnector instance based on configuration.

    This is a factory function that creates the appropriate StorageConnector
    based on the storage backend type specified in the configuration.

    Args:
        config: Configuration object, can be:
            - CacheConfig: FastDeploy configuration object
            - Dict: Dictionary with 'storage_type' and backend-specific settings
            - StorageConfig: StorageConfig dataclass instance

    Returns:
        StorageConnector instance if successful, None otherwise

    Example:
        # Using CacheConfig
        connector = create_storage_connector(fd_config)

        # Using dict config
        config = {
            'storage_type': 'mooncake',
            'server_addr': 'localhost:8080',
            'buffer_size': 1024 * 1024,
        }
        connector = create_storage_connector(config)
    """
    if config.kvcache_storage_backend is None:
        return None

    connector: Optional[StorageConnector] = None

    # Create connector based on storage type
    if config.kvcache_storage_backend == "mooncake":
        from .mooncake.connector import MooncakeStorageConnector

        connector = MooncakeStorageConnector(config)

    elif config.kvcache_storage_backend == "attention_store":
        from .attnstore.connector import AttnStoreConnector

        connector = AttnStoreConnector(config)

    else:
        raise ValueError(
            f"Unsupported storage type: {config.kvcache_storage_backend}. "
            f"Supported types: mooncake, attention_store, local"
        )

    # Attempt connection
    if connector is not None:
        if not connector.connect():
            # Log warning but still return the connector
            pass

    return connector


def _parse_storage_config(config: "CacheConfig") -> tuple:
    """
    Parse storage configuration from various input types.

    Args:
        config: Configuration object (CacheConfig, Dict, or StorageConfig)

    Returns:
        Tuple of (storage_type, backend_config)
    """
    storage_type = None
    backend_config: Dict[str, Any] = {}

    # Handle CacheConfig
    if hasattr(config, "cache_config") and config.cache_config is not None:
        cache_config = config.cache_config

        # Get storage type from cache_config
        if hasattr(cache_config, "kvcache_storage_backend"):
            storage_backend = cache_config.kvcache_storage_backend
            if storage_backend:
                storage_type = _normalize_storage_type(storage_backend)

        # Extract backend-specific configuration
        if hasattr(cache_config, "kvcache_storage_config"):
            backend_config = cache_config.kvcache_storage_config or {}

    # Handle dict config
    elif isinstance(config, dict):
        if "storage_type" in config:
            storage_type = _normalize_storage_type(config["storage_type"])
            # Copy other keys as backend config
            backend_config = {k: v for k, v in config.items() if k != "storage_type"}
        elif "kvcache_storage_backend" in config:
            storage_type = _normalize_storage_type(config["kvcache_storage_backend"])
            backend_config = config.get("kvcache_storage_config", {})

    # Handle StorageConfig dataclass
    elif hasattr(config, "storage_type"):
        storage_type = config.storage_type
        backend_config = {
            "storage_path": getattr(config, "storage_path", ""),
            "max_size_bytes": getattr(config, "max_size_bytes", 0),
            "enable_compression": getattr(config, "enable_compression", False),
            "compression_algorithm": getattr(config, "compression_algorithm", "lz4"),
            "connection_timeout": getattr(config, "connection_timeout", 30.0),
            "read_timeout": getattr(config, "read_timeout", 60.0),
            "write_timeout": getattr(config, "write_timeout", 60.0),
            "extra_config": getattr(config, "extra_config", {}),
        }

    return storage_type, backend_config


def _normalize_storage_type(storage_type: Any) -> Optional[str]:
    """
    Normalize storage type to lowercase string.

    Args:
        storage_type: Storage type (enum, string, etc.)

    Returns:
        Normalized storage type string
    """
    if storage_type is None:
        return None

    # Handle enum
    if isinstance(storage_type, StorageType):
        return storage_type.value

    # Handle string
    if isinstance(storage_type, str):
        return storage_type.lower()

    # Handle other types
    return str(storage_type).lower()


__all__ = [
    "StorageScheduler",
    "StorageConnector",
    "create_storage_scheduler",
    "create_storage_connector",
]
