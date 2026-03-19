"""
Cache Manager V1 - Multi-level KV Cache Management System

This module provides a three-level cache hierarchy:
- Device (GPU) → Host (CPU) → Storage

Key components:
- KVCacheBase: Abstract base class defining common interface
- CacheManager: Scheduler-side cache management with block pools
- CacheController: Worker-side cache control for transfer operations
- CacheTransferManager: Manages cache transfer operations
- LayerDoneCounter: Tracks layer-by-layer transfer completion
- create_storage_scheduler: Factory function to create StorageScheduler
- create_storage_connector: Factory function to create StorageConnector
- create_transfer_connector: Factory function to create TransferConnector
"""

from .base import KVCacheBase
from .cache_controller import CacheController
from .cache_manager import CacheManager
from .cache_utils import LayerDoneCounter
from .metadata import (
    AsyncTaskHandler,
    BlockNode,
    CacheBlockMetadata,
    CacheStatus,
    MatchResult,
    PDTransferMetadata,
    StorageConfig,
    StorageMetadata,
    StorageType,
    TransferConfig,
    TransferResult,
    TransferStatus,
    TransferTask,
    TransferType,
)
from .storage import create_storage_connector, create_storage_scheduler
from .transfer import create_transfer_connector
from .transfer_manager import CacheTransferManager

__all__ = [
    # Base classes
    "KVCacheBase",
    # Managers
    "CacheManager",
    "CacheController",
    "CacheTransferManager",
    # Utils
    "LayerDoneCounter",
    # Metadata
    "CacheBlockMetadata",
    "BlockNode",
    "CacheStatus",
    "TransferTask",
    "TransferStatus",
    "TransferConfig",
    "TransferResult",
    "AsyncTaskHandler",
    "MatchResult",
    "StorageMetadata",
    "PDTransferMetadata",
    "StorageConfig",
    "StorageType",
    "TransferType",
    # Factory functions
    "create_storage_scheduler",
    "create_storage_connector",
    "create_transfer_connector",
]
