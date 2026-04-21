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
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TransferStatus(Enum):
    """Status of a transfer task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StorageType(Enum):
    """Supported storage backend types."""

    MOONCAKE = "mooncake"
    ATTNSTORE = "attnstore"
    LOCAL = "local"


class TransferType(Enum):
    """Supported transfer mechanism types."""

    RDMA = "rdma"
    IPC = "ipc"


class CacheLevel(Enum):
    """Cache hierarchy levels for transfer operations."""

    DEVICE = "device"
    HOST = "host"
    STORAGE = "storage"


class CacheStatus(Enum):
    """Cache status enum representing the current location and state of a BlockNode.

    Attributes:
        DEVICE: Block is in device (GPU) memory, ready for use. Can be matched.
        HOST: Block is in host (CPU) memory, needs to be loaded to device. Can be matched.
        SWAP_TO_HOST: Block is being evicted from device to host. Cannot be matched.
        SWAP_TO_DEVICE: Block is being loaded from host to device.
        LOADING_FROM_STORAGE: Block is being loaded from storage.
        DELETING: Block is being deleted (removed from host or deleted when no host cache). Cannot be matched.
    """

    DEVICE = auto()
    HOST = auto()
    SWAP_TO_HOST = auto()
    SWAP_TO_DEVICE = auto()
    DELETING = auto()
    LOADING_FROM_STORAGE = auto()


@dataclass
class RadixTreeStats:
    """
    Snapshot of RadixTree statistics.

    Encapsulates all state counters for monitoring and statistics.
    Returns as a snapshot to ensure consistent values across all fields.

    Attributes:
        node_count: Total number of nodes in the tree.
        evictable_device_count: GPU nodes available for eviction (ref_count==0, status==DEVICE).
        evictable_host_count: CPU nodes available for deletion (ref_count==0, status==HOST).
    """

    node_count: int = 0
    evictable_device_count: int = 0
    evictable_host_count: int = 0

    @property
    def evictable_count(self) -> int:
        """Total evictable nodes count."""
        return self.evictable_device_count + self.evictable_host_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_count": self.node_count,
            "evictable_device_count": self.evictable_device_count,
            "evictable_host_count": self.evictable_host_count,
            "evictable_count": self.evictable_count,
        }


@dataclass
class CacheBlockMetadata:
    """
    Metadata for a cache block.

    Attributes:
        block_id: Unique identifier for the block
        device_id: GPU device ID where the block resides
        block_size: Size of the block in bytes
        ref_count: Reference count for the block
        is_pinned: Whether the block is pinned in memory
        layer_indices: List of layer indices stored in this block
        token_count: Number of tokens in this block
        hash_value: Hash value for the block content
        last_access_time: Last access timestamp
    """

    block_id: int
    device_id: int
    block_size: int
    ref_count: int = 0
    is_pinned: bool = False
    layer_indices: List[int] = field(default_factory=list)
    token_count: int = 0
    hash_value: Optional[str] = None
    last_access_time: float = 0.0


@dataclass
class TransferTask:
    """
    Represents a cache transfer task.

    Attributes:
        task_id: Unique identifier for the task
        src_location: Source location (device/host/storage/remote)
        dst_location: Destination location
        block_indices: List of block indices to transfer
        layer_indices: List of layer indices to transfer
        status: Current status of the task
        priority: Task priority (lower is higher priority)
        created_time: Task creation timestamp
        started_time: Task start timestamp
        completed_time: Task completion timestamp
        error_message: Error message if task failed
        metadata: Additional task metadata
    """

    task_id: str
    src_location: str
    dst_location: str
    block_indices: List[int] = field(default_factory=list)
    layer_indices: List[int] = field(default_factory=list)
    status: TransferStatus = TransferStatus.PENDING
    priority: int = 0
    created_time: float = 0.0
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """
    Configuration for storage backend.

    Attributes:
        storage_type: Type of storage backend
        storage_path: Base path for storage
        max_size_bytes: Maximum storage size in bytes
        enable_compression: Whether to enable compression
        compression_algorithm: Compression algorithm to use
        connection_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        write_timeout: Write timeout in seconds
        extra_config: Additional backend-specific configuration
    """

    storage_type: StorageType = StorageType.MOONCAKE
    storage_path: str = ""
    max_size_bytes: int = 0
    enable_compression: bool = False
    compression_algorithm: str = "lz4"
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    write_timeout: float = 60.0
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferConfig:
    """
    Configuration for transfer mechanism.

    Attributes:
        transfer_type: Type of transfer mechanism
        enable_async: Whether to enable async transfer
        max_concurrent_transfers: Maximum concurrent transfer tasks
        buffer_size: Buffer size for transfer in bytes
        enable_checksum: Whether to enable checksum verification
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds
        extra_config: Additional transfer-specific configuration
    """

    transfer_type: TransferType = TransferType.RDMA
    enable_async: bool = True
    max_concurrent_transfers: int = 4
    buffer_size: int = 1024 * 1024  # 1MB
    enable_checksum: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockNode:
    """
    Node in the block management tree.

    Represents a node in the radix tree or block allocation structure,
    tracking block relationships and reference counts.

    Attributes:
        node_id: Globally unique identifier for this node (UUID)
        block_id: Block identifier (may be reused across device/host)
        parent: Parent BlockNode reference (None for root)
        children: Dict mapping hash values to child BlockNodes (for radix tree)
        children_ids: List of child block IDs
        ref_count: Number of references to this block (defaults to 1 on creation)
        token_count: Number of tokens stored in this block
        hash_value: Hash value for prefix matching
        cache_status: Current cache status (DEVICE/HOST/SWAP_TO_HOST/SWAP_TO_DEVICE)
        last_access_time: Last access timestamp (defaults to current time on creation)
        backuped: Whether this block has a backup on host memory
        host_block_id: Host block ID where the backup is stored (if backuped=True)
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    block_id: int = 0
    parent: Optional["BlockNode"] = None
    children: Dict[str, "BlockNode"] = field(default_factory=dict)
    children_ids: List[int] = field(default_factory=list)
    ref_count: int = 0
    token_count: int = 0
    hash_value: Optional[str] = None
    cache_status: CacheStatus = CacheStatus.DEVICE
    last_access_time: float = field(default_factory=time.time)
    # Backup-related fields
    backuped: bool = False  # Whether a backup exists on host memory
    host_block_id: Optional[int] = None  # Host block ID where the backup is stored
    hit_count: int = 1  # triggers backup when reaching the threshold

    def __post_init__(self):
        """Initialize instance with current time if last_access_time not set."""
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()

    def add_child(self, child_id: int) -> None:
        """Add a child block ID."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def remove_child(self, child_id: int) -> bool:
        """Remove a child block ID. Returns True if removed."""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
            return True
        return False

    def increment_ref(self) -> int:
        """Increment reference count and return new count."""
        self.ref_count += 1
        return self.ref_count

    def decrement_ref(self) -> int:
        """Decrement reference count and return new count."""
        if self.ref_count > 0:
            self.ref_count -= 1
        return self.ref_count

    def touch(self) -> None:
        """
        Update last_access_time to current time.

        This method should be called whenever the block is accessed
        to track access recency for eviction policies.
        """
        self.last_access_time = time.time()

    def update_access(self, delta_ref: int = 0) -> None:
        """
        Update reference count and last_access_time.

        Args:
            delta_ref: Change in reference count (positive to increment, negative to decrement)
        """
        if delta_ref > 0:
            self.ref_count += delta_ref
        elif delta_ref < 0:
            self.ref_count = max(0, self.ref_count + delta_ref)
        self.touch()

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children_ids) == 0 and len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this is a root node (no parent)."""
        return self.parent is None

    def is_on_device(self) -> bool:
        """Check if block is on device (GPU) memory."""
        return self.cache_status == CacheStatus.DEVICE

    def is_on_host(self) -> bool:
        """Check if block is on host (CPU) memory."""
        return self.cache_status == CacheStatus.HOST

    def is_swapping(self) -> bool:
        """Check if block is currently being swapped or deleted."""
        return self.cache_status in (
            CacheStatus.SWAP_TO_HOST,
            CacheStatus.SWAP_TO_DEVICE,
            CacheStatus.DELETING,
        )


@dataclass
class MatchResult:
    """
    Three-level cache prefix match result.

    Contains matched nodes from Device, Host, and Storage levels.

    Attributes:
        storage_nodes: List of matched BlockNodes in Storage.
        device_nodes: List of matched BlockNodes in Device.
        host_nodes: List of matched BlockNodes in Host.
    """

    device_nodes: List["BlockNode"] = field(default_factory=list)
    host_nodes: List["BlockNode"] = field(default_factory=list)
    storage_nodes: List["BlockNode"] = field(default_factory=list)
    uncached_block_ids: List[int] = field(default_factory=list)

    @property
    def device_block_ids(self) -> List[int]:
        """Get list of matched device block IDs."""
        return [node.block_id for node in self.device_nodes]

    @property
    def total_matched_blocks(self) -> int:
        """Get total number of matched device blocks."""
        return self.matched_device_nums + self.matched_host_nums + self.matched_storage_nums

    @property
    def matched_device_nums(self) -> int:
        """Get total number of matched device blocks."""
        return len(self.device_nodes)

    @property
    def matched_host_nums(self) -> int:
        """Get total number of matched host blocks."""
        return len(self.host_nodes)

    @property
    def matched_storage_nums(self) -> int:
        """Get total number of matched storage hashes."""
        return len(self.storage_nodes)


@dataclass
class StorageMetadata:
    """
    Base metadata for storage transfer operations.

    Encapsulates all information for storage load/evict operations.
    Different storage implementations can extend this class with additional fields.

    Attributes:
        hash_values: List of hash values to transfer.
        block_ids: Target/source host block IDs (pre-allocated by Scheduler).
        direction: Transfer direction ("load" from storage, "evict" to storage).
        storage_type: Storage type ("mooncake", "attnstore", "rdma", etc.).
        endpoint: Storage service endpoint address.
        timeout: Operation timeout in seconds.
        layer_num: Number of layers to transfer (for layer-by-layer transfer).
        extra_params: Storage-specific extra parameters.
    """

    hash_values: List[str] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
    direction: str = "load"
    storage_type: str = "mooncake"
    endpoint: Optional[str] = None
    timeout: float = 30.0
    layer_num: int = 0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDTransferMetadata:
    """
    Base metadata for PD separation transfer operations.

    Encapsulates all information for cross-node transfer in PD separation architecture.
    Different transfer mechanisms (RDMA, IPC) can extend this class with additional fields.

    Attributes:
        source_node_id: Source node identifier (P node ID).
        target_node_id: Target node identifier (D node ID).
        block_ids: List of block IDs to transfer.
        layer_num: Total number of model layers (for layer-by-layer transfer sync).
        timeout: Operation timeout in seconds.
        extra_params: Transfer-specific extra parameters.
    """

    source_node_id: str = ""
    target_node_id: str = ""
    block_ids: List[int] = field(default_factory=list)
    layer_num: int = 0
    timeout: float = 30.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheSwapMetadata:
    """
    Metadata for cache transfer operations.

    Encapsulates the mapping between source and destination block IDs
    for Host↔Device, Storage→Host, and other transfer operations.

    Attributes:
        src_block_ids: Source block IDs (transfer origin).
        dst_block_ids: Destination block IDs (transfer target).
        src_type: Source cache level (CacheLevel.DEVICE/HOST/STORAGE).
        dst_type: Destination cache level (CacheLevel.DEVICE/HOST/STORAGE).
        hash_values: Corresponding hash values (used for storage-related operations).
        success: Whether the transfer succeeded.
        error_message: Error message if transfer failed.
        async_handler: Async task handler for tracking the swap task execution state.
    """

    src_block_ids: List[int] = field(default_factory=list)
    dst_block_ids: List[int] = field(default_factory=list)
    src_type: Optional[CacheLevel] = None
    dst_type: Optional[CacheLevel] = None
    hash_values: List[str] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    async_handler: Optional["AsyncTaskHandler"] = None

    def is_success(self) -> bool:
        """Return whether the transfer succeeded."""
        return self.success

    @property
    def mapping(self) -> Dict[int, int]:
        """Get the src -> dst block ID mapping dict."""
        if not self.success:
            return {}
        return dict(zip(self.src_block_ids, self.dst_block_ids))


@dataclass
class TransferResult:
    """
    Cache transfer operation result.

    Encapsulates the mapping between source and destination block IDs
    for Host↔Device, Storage→Host, and other transfer operations.

    Attributes:
        src_block_ids: Source block IDs (transfer origin).
        dst_block_ids: Destination block IDs (transfer target).
        src_type: Source cache level (CacheLevel.DEVICE/HOST/STORAGE).
        dst_type: Destination cache level (CacheLevel.DEVICE/HOST/STORAGE).
        success: Whether the transfer succeeded.
        error_message: Error message if transfer failed.
    """

    src_block_ids: List[int] = field(default_factory=list)
    dst_block_ids: List[int] = field(default_factory=list)
    src_type: Optional[CacheLevel] = None
    dst_type: Optional[CacheLevel] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class AsyncTaskHandler:
    """
    Async task handler.

    Used for submitting and tracking the state of async tasks.
    External callers use this handler to check whether a task has completed.

    Attributes:
        task_id: Unique task identifier.
        is_completed: Whether the task has completed.
        result: Task result (available after completion).
        error: Task error message (if failed).
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_completed: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    _event: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize event for synchronization."""
        import threading

        object.__setattr__(self, "_event", threading.Event())

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the task to complete.

        Args:
            timeout: Maximum wait time in seconds. None means wait indefinitely.

        Returns:
            True if completed, False if timed out.
        """
        return self._event.wait(timeout=timeout)

    def cancel(self) -> bool:
        """
        Cancel the task.

        Returns:
            True if successfully cancelled, False otherwise.
        """
        if self.is_completed:
            return False
        self.error = "Task cancelled"
        self.is_completed = True
        self._event.set()
        return True

    def get_result(self) -> Any:
        """
        Get the task result (blocking).

        Returns:
            Task result.

        Raises:
            RuntimeError: If the task failed or was cancelled.
        """
        self._event.wait()
        if self.error:
            raise RuntimeError(self.error)
        return self.result

    def set_result(self, result: Any) -> None:
        """
        Set the task result and mark as completed.

        Args:
            result: Task result.
        """
        self.result = result
        self.is_completed = True
        self._event.set()

    def set_error(self, error: str) -> None:
        """
        Set the error message and mark as completed.

        Args:
            error: Error message.
        """
        self.error = error
        self.is_completed = True
        self._event.set()
