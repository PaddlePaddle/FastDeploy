"""
Metadata definitions for cache management.

This module contains data structures and configurations used across
the cache management system.
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


class CacheStatus(Enum):
    """缓存状态枚举，表示 BlockNode 当前的位置和状态。

    Attributes:
        DEVICE: Block 在 device (GPU) 内存中，可直接使用。可以被命中
        HOST: Block 在 host (CPU) 内存中，需要加载到 device。可以被命中
        SWAP_TO_HOST: Block 正在从 device 驱逐到 host。不可被命中
        SWAP_TO_DEVICE: Block 正在从 host 加载到 device。
        LOADING_FROM_STORAGE: Block 正在从存储加载数据。
        DELETING: Block 正在被删除（从 host 移除或无 host 缓存时删除）。不可被命中
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
    三级缓存前缀匹配结果.

    包含 Device、Host、Storage 三级匹配的节点.

    Attributes:
        storage_nodes: Storage 中匹配的 BlockNode 列表.
        device_nodes: Device 中匹配的 BlockNode 列表.
        host_nodes: Host 中匹配的 BlockNode 列表.
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
    Storage 传输元数据基类.

    封装 storage 加载/驱逐操作的所有信息.
    不同 storage 实现可以通过继承此类添加特定字段.

    Attributes:
        hash_values: 要传输的 hash 值列表.
        block_ids: 目标/源 host block IDs（由 Scheduler 预先分配）.
        direction: 传输方向（"load" 从 storage 加载，"evict" 驱逐到 storage）.
        storage_type: Storage 类型（"mooncake", "attnstore", "rdma" 等）.
        endpoint: Storage 服务端点地址.
        timeout: 操作超时时间（秒）.
        layer_num: 传输的层数（用于逐层传输）.
        extra_params: Storage 特定的额外参数.
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
    PD 分离传输元数据基类.

    封装 PD 分离架构下跨节点传输的所有信息.
    不同传输方式（RDMA、IPC）可以通过继承此类添加特定字段.

    Attributes:
        source_node_id: 源节点标识（P 节点 ID）.
        target_node_id: 目标节点标识（D 节点 ID）.
        block_ids: 要传输的 block IDs 列表.
        layer_num: 模型总层数（用于逐层传输同步）.
        timeout: 操作超时时间（秒）.
        extra_params: 传输特定的额外参数.
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
    Cache 传输操作元数据.

    包装源 block IDs 和目标 block IDs 的映射关系，
    用于 Host↔Device、Storage→Host 等传输操作.

    Attributes:
        src_block_ids: 源 block IDs（传输来源）.
        dst_block_ids: 目标 block IDs（传输目的地）.
        src_type: 源缓存类型（"device", "host", "storage"）.
        dst_type: 目标缓存类型（"device", "host", "storage"）.
        hash_values: 对应的 hash 值列表（storage 相关操作时使用）.
        success: 传输是否成功.
        error_message: 错误信息（如果失败）.
        async_handler: 异步任务处理器，用于追踪该 swap 任务的执行状态.
    """

    src_block_ids: List[int] = field(default_factory=list)
    dst_block_ids: List[int] = field(default_factory=list)
    src_type: str = ""
    dst_type: str = ""
    hash_values: List[str] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    async_handler: Optional["AsyncTaskHandler"] = None

    def is_success(self) -> bool:
        """成功传输的 block 数量."""
        return self.success

    @property
    def mapping(self) -> Dict[int, int]:
        """获取 src -> dst 的映射字典."""
        if not self.success:
            return {}
        return dict(zip(self.src_block_ids, self.dst_block_ids))


@dataclass
class TransferResult:
    """
    Cache 传输操作结果.

    包装源 block IDs 和目标 block IDs 的映射关系，
    用于 Host↔Device、Storage→Host 等传输操作.

    Attributes:
        src_block_ids: 源 block IDs（传输来源）.
        dst_block_ids: 目标 block IDs（传输目的地）.
        src_type: 源缓存类型（"device", "host", "storage"）.
        dst_type: 目标缓存类型（"device", "host", "storage"）.
        success: 传输是否成功.
        error_message: 错误信息（如果失败）.
    """

    src_block_ids: List[int] = field(default_factory=list)
    dst_block_ids: List[int] = field(default_factory=list)
    src_type: str = ""
    dst_type: str = ""
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class AsyncTaskHandler:
    """
    异步任务处理器.

    用于异步任务的提交和状态追踪.
    外部通过此 handler 判断任务是否完成.

    Attributes:
        task_id: 任务唯一标识.
        is_completed: 任务是否已完成.
        result: 任务结果（完成后可用）.
        error: 任务错误信息（如果失败）.
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
        等待任务完成.

        Args:
            timeout: 最大等待时间（秒），None 表示无限等待.

        Returns:
            True 表示完成，False 表示超时.
        """
        return self._event.wait(timeout=timeout)

    def cancel(self) -> bool:
        """
        取消任务.

        Returns:
            成功取消返回 True，否则返回 False.
        """
        if self.is_completed:
            return False
        self.error = "Task cancelled"
        self.is_completed = True
        self._event.set()
        return True

    def get_result(self) -> Any:
        """
        获取任务结果（阻塞）.

        Returns:
            任务结果.

        Raises:
            RuntimeError: 任务失败或被取消.
        """
        self._event.wait()
        if self.error:
            raise RuntimeError(self.error)
        return self.result

    def set_result(self, result: Any) -> None:
        """
        设置任务结果并标记完成.

        Args:
            result: 任务结果.
        """
        self.result = result
        self.is_completed = True
        self._event.set()

    def set_error(self, error: str) -> None:
        """
        设置错误信息并标记完成.

        Args:
            error: 错误信息.
        """
        self.error = error
        self.is_completed = True
        self._event.set()
