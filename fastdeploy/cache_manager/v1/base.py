"""
KVCacheBase - Abstract base class for KV cache management

Defines the common interface that both CacheManager (Scheduler) and
CacheController (Worker) must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


class KVCacheBase(ABC):
    """
    Abstract base class for KV cache management.

    This class defines the common interface for cache management operations.
    Subclasses (CacheManager and CacheController) implement specific behaviors
    based on their roles in the system.

    CacheManager (Scheduler process):
        - Manages DeviceBlockPool and HostBlockPool
        - Handles block allocation and release
        - Coordinates storage operations via StorageScheduler

    CacheController (Worker process):
        - Manages cache transfer operations
        - Handles layer-by-layer transfer synchronization
        - Coordinates cross-node transfer via TransferConnector
    """

    def __init__(self, config: "FDConfig"):
        """
        Initialize the KV cache base.

        Args:
            config: FDConfig instance containing all fastdeploy configuration
        """
        self.config = config
        self._initialized = False

    @abstractmethod
    def reset_cache(self) -> bool:
        """
        Reset the cache state.

        This method should be implemented by subclasses to reset their
        specific cache state (e.g., clear block pools, reset transfer state).

        Returns:
            True if reset was successful, False otherwise
        """
        pass

    def is_initialized(self) -> bool:
        """
        Check if the cache has been initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized
