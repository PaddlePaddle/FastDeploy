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

from typing import Any, Dict, Optional

from .base import TransferConnector


def create_transfer_connector(
    config: Any,
) -> Optional[TransferConnector]:
    """
    Create a TransferConnector instance based on configuration.

    This is a factory function that creates the appropriate TransferConnector
    based on the transfer backend type specified in the configuration.

    Args:
        config: Configuration object, can be:
            - CacheConfig: FastDeploy configuration object
            - Dict: Dictionary with 'transfer_type' and backend-specific settings

    Returns:
        TransferConnector instance if successful, None otherwise

    Example:
        # Using CacheConfig
        connector = create_transfer_connector(fd_config)

        # Using dict config
        config = {
            'transfer_type': 'rdma',
            'device': 'mlx5_0',
            'port': 1,
        }
        connector = create_transfer_connector(config)
    """
    transfer_type = _get_transfer_type(config)

    if transfer_type is None:
        return None

    connector: Optional[TransferConnector] = None

    # Create connector based on transfer type
    if transfer_type == "rdma":
        from .rdma.connector import RDMAConnector

        connector = RDMAConnector(_get_backend_config(config))

    elif transfer_type == "ipc":
        from .ipc.connector import IPCConnector

        connector = IPCConnector(_get_backend_config(config))

    else:
        raise ValueError(f"Unsupported transfer type: {transfer_type}. " f"Supported types: rdma, ipc")

    # Attempt connection
    if connector is not None:
        if not connector.connect():
            # Log warning but still return the connector
            pass

    return connector


def _get_transfer_type(config: Any) -> Optional[str]:
    """
    Get transfer type from configuration.

    Args:
        config: Configuration object

    Returns:
        Transfer type string or None
    """
    # Handle CacheConfig (from FDConfig)
    if hasattr(config, "kvcache_transfer_backend"):
        transfer_backend = config.kvcache_transfer_backend
        if transfer_backend:
            return _normalize_transfer_type(transfer_backend)

    # Handle dict config
    if isinstance(config, dict):
        if "transfer_type" in config:
            return _normalize_transfer_type(config["transfer_type"])
        elif "kvcache_transfer_backend" in config:
            return _normalize_transfer_type(config["kvcache_transfer_backend"])

    # Handle object with cache_config attribute
    if hasattr(config, "cache_config") and config.cache_config is not None:
        cache_config = config.cache_config
        if hasattr(cache_config, "kvcache_transfer_backend"):
            transfer_backend = cache_config.kvcache_transfer_backend
            if transfer_backend:
                return _normalize_transfer_type(transfer_backend)

    return None


def _get_backend_config(config: Any) -> Dict[str, Any]:
    """
    Extract backend-specific configuration.

    Args:
        config: Configuration object

    Returns:
        Dictionary with backend configuration
    """
    backend_config: Dict[str, Any] = {}

    # Handle CacheConfig
    if hasattr(config, "kvcache_transfer_config"):
        backend_config = config.kvcache_transfer_config or {}

    # Handle dict config
    elif isinstance(config, dict):
        if "transfer_config" in config:
            backend_config = config["transfer_config"]
        elif "kvcache_transfer_config" in config:
            backend_config = config["kvcache_transfer_config"]
        else:
            # Copy all keys except transfer_type
            backend_config = {
                k: v for k, v in config.items() if k not in ("transfer_type", "kvcache_transfer_backend")
            }

    # Handle object with cache_config attribute
    if hasattr(config, "cache_config") and config.cache_config is not None:
        cache_config = config.cache_config
        if hasattr(cache_config, "kvcache_transfer_config"):
            backend_config = cache_config.kvcache_transfer_config or {}

    return backend_config


def _normalize_transfer_type(transfer_type: Any) -> Optional[str]:
    """
    Normalize transfer type to lowercase string.

    Args:
        transfer_type: Transfer type (enum, string, etc.)

    Returns:
        Normalized transfer type string
    """
    if transfer_type is None:
        return None

    # Handle string
    if isinstance(transfer_type, str):
        return transfer_type.lower()

    # Handle other types
    return str(transfer_type).lower()


__all__ = [
    "TransferConnector",
    "create_transfer_connector",
]
