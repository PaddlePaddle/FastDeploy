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

        # Extract configuration from FDConfig
        self.model_config = config.model_config
        self.cache_config = config.cache_config
        self.quant_config = config.quant_config
        self.parallel_config = config.parallel_config

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
