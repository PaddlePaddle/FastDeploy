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
from typing import Any, Dict, List, Optional

from fastdeploy.utils import get_host_ip

from ..base import StorageConnector, StorageScheduler

DEFAULT_GLOBAL_SEGMENT_SIZE = 1024 * 1024 * 1024  # 1 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1024 * 1024  # 1 MB
DEFAULT_MC_MAX_MR_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB
MIN_MC_MAX_MR_SIZE = 1024 * 1024 * 1024  # 1 GB
MAX_MC_MAX_MR_SIZE = 6 * 1024 * 1024 * 1024  # 6 GB


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MooncakeStorageConfig:
    """
    Configuration for Mooncake distributed store.

    Loaded with the following priority (highest first):
      1. Explicit keyword arguments passed to ``from_config``
      2. JSON config file at ``MOONCAKE_CONFIG_PATH``
      3. Individual environment variables
    """

    local_hostname: str
    metadata_server: str
    master_server_addr: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    rdma_devices: str

    # ---------------------------------------------------------------------------

    @staticmethod
    def create(extra: Optional[Dict[str, Any]] = None) -> "MooncakeStorageConfig":
        """
        Load config from (in priority order):
          1. ``extra`` dict (e.g. from CacheConfig.kvcache_storage_config)
          2. JSON file at ``MOONCAKE_CONFIG_PATH``
          3. Environment variables

        Args:
            extra: Optional dict of override values (takes highest priority).

        Returns:
            Populated ``MooncakeStorageConfig`` instance.
        """
        extra = extra or {}

        # --- base from env / file ---
        file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        host_ip = get_host_ip()
        file_cfg: Dict[str, Any] = {}

        if file_path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"MOONCAKE_CONFIG_PATH points to non-existent file: {file_path}")
            with open(file_path) as f:
                file_cfg = json.load(f)

        def _get(key: str, default: Any = None) -> Any:
            """extra > file > env > default"""
            if key in extra:
                return extra[key]
            if key in file_cfg:
                return file_cfg[key]
            env_map = {
                "local_hostname": "MOONCAKE_LOCAL_HOSTNAME",
                "metadata_server": "MOONCAKE_METADATA_SERVER",
                "master_server_addr": "MOONCAKE_MASTER_SERVER_ADDR",
                "global_segment_size": "MOONCAKE_GLOBAL_SEGMENT_SIZE",
                "local_buffer_size": "MOONCAKE_LOCAL_BUFFER_SIZE",
                "protocol": "MOONCAKE_PROTOCOL",
                "rdma_devices": "MOONCAKE_RDMA_DEVICES",
            }
            if key in env_map:
                return os.environ.get(env_map[key], default)
            return default

        local_hostname = _get("local_hostname", host_ip)
        metadata_server = _get("metadata_server")
        master_server_addr = _get("master_server_addr")
        global_segment_size = int(_get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE))
        local_buffer_size = int(_get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE))
        protocol = _get("protocol", "rdma")
        rdma_devices = _get("rdma_devices", "")

        if metadata_server is None or master_server_addr is None:
            raise ValueError(
                "Both MOONCAKE_METADATA_SERVER and MOONCAKE_MASTER_SERVER_ADDR must be provided "
                "(via extra config, config file, or environment variables)."
            )
        if local_hostname == "localhost":
            raise ValueError("local_hostname must not be 'localhost'; Mooncake requires a real IP or hostname.")

        # Auto-detect RDMA NICs if not provided
        if rdma_devices == "" and protocol == "rdma":
            try:
                from fastdeploy.cache_manager.v1.cache_utils import get_rdma_nics

                rdma_devices = get_rdma_nics()
            except Exception:
                pass

        return MooncakeStorageConfig(
            local_hostname=local_hostname,
            metadata_server=metadata_server,
            master_server_addr=master_server_addr,
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=protocol,
            rdma_devices=rdma_devices,
        )

    def select_rdma_device(self, tp_rank: int) -> None:
        """Select a single RDMA device from a comma-separated list by TP rank."""
        devices = [d.strip() for d in self.rdma_devices.split(",") if d.strip()]
        if devices:
            self.rdma_devices = devices[tp_rank % len(devices)]


# ---------------------------------------------------------------------------
# Shared helper — wraps the raw MooncakeDistributedStore
# ---------------------------------------------------------------------------


class _MooncakeStoreBase:
    """
    Thin wrapper around ``mooncake.store.MooncakeDistributedStore`` shared by
    both the Scheduler and Connector implementations.
    """

    def __init__(self, logger) -> None:
        self._store = None  # MooncakeDistributedStore instance
        self.logger = logger
        self.mc_max_mr_size = DEFAULT_MC_MAX_MR_SIZE

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # Minimal segment size for the Scheduler process.
    # The Scheduler only calls batch_is_exist (metadata query, no RDMA transfer),
    # so there is no need to allocate a large global segment.
    _SCHEDULER_SEGMENT_SIZE = 64 * 1024 * 1024  # 64 MB

    def _setup_store(
        self,
        cfg: MooncakeStorageConfig,
        tp_rank: Optional[int] = None,
        scheduler_mode: bool = False,
    ) -> None:
        """
        Import the SDK and call ``store.setup()``.

        Args:
            cfg: Populated ``MooncakeStorageConfig``.
            tp_rank: When provided, selects a single RDMA device from the
                comma-separated ``cfg.rdma_devices`` list by rank modulo.
                Must be provided for Connector (worker) instances.
            scheduler_mode: When True, selects the first RDMA device from the
                list (scheduler has no tp_rank) and uses a small
                ``global_segment_size`` since no actual data transfer happens.
        """
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "mooncake package not found. Install it by following "
                "https://kvcache-ai.github.io/Mooncake/python-api-reference/mooncake-store.html"
            ) from e

        if tp_rank is not None:
            # Worker path: pick one device per TP rank
            cfg.select_rdma_device(tp_rank)
        elif scheduler_mode:
            # Scheduler path: Mooncake setup() expects a single device name,
            # not a comma-separated list.  Pick the first available device.
            cfg.select_rdma_device(0)

        # Scheduler does not transfer data — avoid allocating a large segment.
        if scheduler_mode:
            cfg.global_segment_size = self._SCHEDULER_SEGMENT_SIZE

        host_ip = get_host_ip()
        os.environ.setdefault("MC_TCP_BIND_ADDRESS", host_ip)

        # Configure MC_MAX_MR_SIZE for buffer registration
        raw_mr_size = int(os.environ.get("MC_MAX_MR_SIZE", 0))
        if raw_mr_size == 0:
            self.mc_max_mr_size = DEFAULT_MC_MAX_MR_SIZE
        elif raw_mr_size < MIN_MC_MAX_MR_SIZE:
            self.mc_max_mr_size = MIN_MC_MAX_MR_SIZE
        elif raw_mr_size > MAX_MC_MAX_MR_SIZE:
            self.mc_max_mr_size = MAX_MC_MAX_MR_SIZE
        else:
            self.mc_max_mr_size = raw_mr_size
        os.environ["MC_MAX_MR_SIZE"] = str(self.mc_max_mr_size)

        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            local_hostname=cfg.local_hostname,
            metadata_server=cfg.metadata_server,
            global_segment_size=cfg.global_segment_size,
            local_buffer_size=cfg.local_buffer_size,
            protocol=cfg.protocol,
            rdma_devices=cfg.rdma_devices,
            master_server_addr=cfg.master_server_addr,
        )
        if ret != 0:
            raise RuntimeError(f"MooncakeDistributedStore.setup() returned error code {ret}")
        self.logger.info("MooncakeDistributedStore connected successfully.")

    def _teardown_store(self) -> None:
        """Release the store (destructor handles cleanup)."""
        self._store = None

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, prefix: str = "fd") -> None:
        """Send a small test key to verify connectivity."""
        key = f"{prefix}_mooncake_warmup_{uuid.uuid4().hex}"
        value = bytes(1 * 1024 * 1024)  # 1 MB
        rc = self._store.put(key, value)
        if rc != 0:
            raise RuntimeError(f"Warmup put failed for key={key}, rc={rc}")
        rc = self._store.is_exist(key)
        if rc != 1:
            raise RuntimeError(f"Warmup exists check failed for key={key}, rc={rc}")
        self._store.get(key)
        self._store.remove(key)

    # ------------------------------------------------------------------
    # Low-level zero-copy primitives
    # ------------------------------------------------------------------

    def _batch_put(
        self,
        keys: List[str],
        src_ptrs: List[int],
        sizes: List[int],
    ) -> List[int]:
        """
        Call ``store.batch_put_from``.

        Returns:
            List of ints: 0 = success, negative = error.
        """
        tic = time.perf_counter()
        results: List[int] = self._store.batch_put_from(keys, src_ptrs, sizes)
        elapsed = time.perf_counter() - tic
        success = results.count(0)
        total = len(keys)
        if success == total:
            self.logger.debug(f"batch_put {total} keys in {elapsed:.4f}s")
        else:
            self.logger.error(f"batch_put: {total - success}/{total} keys failed, elapsed={elapsed:.4f}s")
        if success > 0:
            total_bytes = sum(s for r, s in zip(results, sizes) if r == 0)
            speed_gbs = total_bytes / (elapsed * 1024**3) if elapsed > 0 else float("inf")
            self.logger.debug(f"batch_put throughput: {total_bytes / 1024**3:.4f} GB @ {speed_gbs:.4f} GB/s")
        return results

    def _batch_get(
        self,
        keys: List[str],
        dst_ptrs: List[int],
        sizes: List[int],
    ) -> List[int]:
        """
        Call ``store.batch_get_into``.

        Returns:
            List of ints: bytes_read (> 0) = success, negative = error.
        """
        tic = time.perf_counter()
        results: List[int] = self._store.batch_get_into(keys, dst_ptrs, sizes)
        elapsed = time.perf_counter() - tic
        success = sum(1 for r in results if r > 0)
        total = len(keys)
        if success == total:
            self.logger.debug(f"batch_get {total} keys in {elapsed:.4f}s")
        else:
            self.logger.error(f"batch_get: {total - success}/{total} keys failed, elapsed={elapsed:.4f}s")
        if success > 0:
            total_bytes = sum(s for r, s in zip(results, sizes) if r > 0)
            speed_gbs = total_bytes / (elapsed * 1024**3) if elapsed > 0 else float("inf")
            self.logger.debug(f"batch_get throughput: {total_bytes / 1024**3:.4f} GB @ {speed_gbs:.4f} GB/s")
        return results

    def _batch_exists(self, keys: List[str]) -> tuple:
        """
        Call ``store.batch_is_exist``.

        Returns:
            Tuple of (results, elapsed_ms):
                results: List of ints, 1 = exists, 0 = not found.
                elapsed_ms: Time taken in milliseconds.
        """
        tic = time.perf_counter()
        results: List[int] = self._store.batch_is_exist(keys)
        elapsed_exists_ms = (time.perf_counter() - tic) * 1000
        return results, elapsed_exists_ms


# ---------------------------------------------------------------------------
# StorageScheduler implementation — Scheduler process
# ---------------------------------------------------------------------------


class MooncakeStorageScheduler(StorageScheduler):
    """
    Mooncake storage scheduler for the Scheduler (controller) process.

    Only performs existence queries and metadata lookups — never transfers data.
    Uses the same underlying ``MooncakeDistributedStore`` so it can call
    ``batch_is_exist`` efficiently via RDMA.
    """

    def __init__(self, config: Any = None):
        """
        Args:
            config: Either a ``CacheConfig``-style object (with
                ``kvcache_storage_config`` attribute) or a plain dict.
        """
        super().__init__(config)
        self._base = _MooncakeStoreBase(self.logger)
        self._mc_config: Optional[MooncakeStorageConfig] = None

    # ------------------------------------------------------------------
    # StorageScheduler interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to Mooncake store."""
        if self._connected:
            return True
        try:
            extra = self._extract_extra_config(self.config)
            self._mc_config = MooncakeStorageConfig.create(extra)
            self._base._setup_store(self._mc_config, scheduler_mode=True)
            self._base._warmup("fd_scheduler")
            self._connected = True
            self.logger.info("MooncakeStorageScheduler connected.")
            return True
        except Exception as e:
            self.logger.error(f"MooncakeStorageScheduler connect failed: {e}\n{traceback.format_exc()}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Mooncake store."""
        self._base._teardown_store()
        self._connected = False

    def exists(self, key: str) -> bool:
        """Check if a single key exists."""
        if not self._connected or self._base._store is None:
            return False
        results, _ = self._base._batch_exists([key])
        return results[0] == 1

    def batch_exists(self, keys: List[str]) -> List[bool]:
        """Batch check key existence."""
        if not self._connected or self._base._store is None:
            return [False] * len(keys)
        results, _ = self._base._batch_exists(keys)
        return [r == 1 for r in results]

    def query_prefix_count(
        self,
        k_keys: List[str],
        v_keys: List[str],
        k_scale_keys: Optional[List[str]] = None,
        v_scale_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Return the number of consecutive valid KV cache blocks from the start.

        Mirrors the logic of ``MooncakeStore.query()`` in the v1 transfer_factory.
        """
        if not self._connected or self._base._store is None:
            return 0

        assert len(k_keys) == len(v_keys), "k_keys and v_keys must have the same length"

        has_scale = k_scale_keys is not None and v_scale_keys is not None
        all_keys = k_keys + v_keys
        if has_scale:
            assert (
                len(k_scale_keys) == len(v_scale_keys) == len(k_keys)
            ), "scale key lists must have the same length as k/v key lists"
            all_keys = all_keys + k_scale_keys + v_scale_keys

        exist_map = dict(zip(all_keys, self._base._batch_exists(all_keys)[0]))

        count = 0
        if has_scale:
            for k, v, ks, vs in zip(k_keys, v_keys, k_scale_keys, v_scale_keys):
                if not (exist_map[k] and exist_map[v] and exist_map[ks] and exist_map[vs]):
                    break
                count += 1
        else:
            for k, v in zip(k_keys, v_keys):
                if not (exist_map[k] and exist_map[v]):
                    break
                count += 1

        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_extra_config(config: Any) -> Dict[str, Any]:
        """Extract the mooncake-specific sub-config from a CacheConfig or dict."""
        if config is None:
            return {}
        if isinstance(config, dict):
            return config.get("kvcache_storage_config", config)
        # CacheConfig-style object
        return getattr(config, "kvcache_storage_config", None) or {}


# ---------------------------------------------------------------------------
# StorageConnector implementation — Worker process
# ---------------------------------------------------------------------------


class MooncakeStorageConnector(StorageConnector):
    """
    Mooncake storage connector for Worker processes.

    Performs zero-copy data transfer using ``batch_put_from`` / ``batch_get_into``
    from ``MooncakeDistributedStore``.

    Memory model
    ------------
    Data flows between Mooncake distributed store and the **CPU cache** (pinned
    host memory), never directly to/from GPU blocks.  The typical lifecycle is:

    1. The CacheController allocates a contiguous pinned-CPU memory pool.
    2. It calls ``register_buffer(pool_ptr, pool_size)`` once to register the
       entire pool with Mooncake for zero-copy RDMA access.
    3. For each eviction / prefetch it calls ``batch_set`` / ``batch_get``
       with raw pointers into that pool.

    ``global_segment_size`` must be at least as large as the registered buffer.
    Pass the actual per-rank CPU cache size via ``cpu_cache_size`` so the value
    is set correctly at setup time.
    """

    def __init__(
        self,
        config: Any = None,
        tp_rank: Optional[int] = None,
        cpu_cache_size: Optional[int] = None,
    ):
        """
        Args:
            config: Either a ``CacheConfig``-style object or a plain dict.
            tp_rank: Tensor-parallel rank used for RDMA NIC selection.
            cpu_cache_size: Size in bytes of the pinned CPU memory pool that
                will be registered via ``register_buffer``.  When provided,
                overrides ``global_segment_size`` from config so that the
                Mooncake segment exactly covers the registered buffer.
                If omitted, the value from config / env is used as-is.
        """
        super().__init__(config)
        self._base = _MooncakeStoreBase(self.logger)
        self._mc_config: Optional[MooncakeStorageConfig] = None
        self._tp_rank = tp_rank
        self._cpu_cache_size = cpu_cache_size

    # ------------------------------------------------------------------
    # StorageConnector interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to Mooncake store."""
        if self._connected:
            return True
        try:
            extra = self._extract_extra_config(self.config)
            self._mc_config = MooncakeStorageConfig.create(extra)

            # Override global_segment_size with the actual CPU cache size when
            # provided.  This ensures the Mooncake segment covers the buffer
            # that will be registered via register_buffer().
            if self._cpu_cache_size is not None:
                self.logger.info(
                    f"Overriding global_segment_size with cpu_cache_size="
                    f"{self._cpu_cache_size / 1024**3:.3f} GB (tp_rank={self._tp_rank})"
                )
                self._mc_config.global_segment_size = self._cpu_cache_size

            self._base._setup_store(self._mc_config, tp_rank=self._tp_rank)
            self._base._warmup("fd_worker")
            self._connected = True
            self.logger.info(f"MooncakeStorageConnector connected (tp_rank={self._tp_rank}).")
            return True
        except Exception as e:
            self.logger.error(f"MooncakeStorageConnector connect failed: {e}\n{traceback.format_exc()}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Mooncake store."""
        self._base._teardown_store()
        self._connected = False

    def register_buffer(self, buffer_ptr: int, buffer_size: int) -> None:
        """
        Register a memory buffer with the Mooncake store for zero-copy RDMA.

        Must be called before using ``buffer_ptr`` in any get/set operation.
        If buffer_size exceeds ``mc_max_mr_size`` the buffer is split into
        multiple chunks, each registered separately.

        Args:
            buffer_ptr: Raw pointer (int) to the memory region start.
            buffer_size: Size in bytes.

        Raises:
            RuntimeError: If the store is not connected or registration fails.
        """
        if self._base._store is None:
            raise RuntimeError("MooncakeStorageConnector is not connected; call connect() first.")

        max_mr_size = self._base.mc_max_mr_size
        if buffer_size <= max_mr_size:
            ret = self._base._store.register_buffer(buffer_ptr, buffer_size)
            if ret != 0:
                raise RuntimeError(f"MooncakeDistributedStore.register_buffer() failed with error code {ret}")
            self.logger.debug(f"Registered buffer ptr=0x{buffer_ptr:x} size={buffer_size} bytes.")
        else:
            num_chunks = (buffer_size + max_mr_size - 1) // max_mr_size
            self.logger.info(
                f"Registering buffer of {buffer_size / 1024**3:.2f} GB in {num_chunks} chunks "
                f"(max_mr_size={max_mr_size / 1024**3:.2f} GB per chunk)"
            )
            for i in range(num_chunks):
                chunk_ptr = buffer_ptr + i * max_mr_size
                chunk_size = min(max_mr_size, buffer_size - i * max_mr_size)
                ret = self._base._store.register_buffer(chunk_ptr, chunk_size)
                if ret != 0:
                    raise RuntimeError(
                        f"MooncakeDistributedStore.register_buffer() chunk {i}/{num_chunks} failed "
                        f"with error code {ret}"
                    )

    # ------------------------------------------------------------------
    # Single-key operations (delegates to batch for consistency)
    # ------------------------------------------------------------------

    def get(self, key: str, dst_ptr: int, size: int) -> bool:
        """Get a single object via zero-copy into ``dst_ptr``."""
        if not self._connected or self._base._store is None:
            return False
        results = self._base._batch_get([key], [dst_ptr], [size])
        return results[0] > 0

    def set(self, key: str, src_ptr: int, size: int) -> bool:
        """Set a single object via zero-copy from ``src_ptr``."""
        if not self._connected or self._base._store is None:
            return False
        results = self._base._batch_put([key], [src_ptr], [size])
        return results[0] == 0

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_get(
        self,
        keys: List[str],
        dst_ptrs: List[int],
        sizes: List[int],
    ) -> List[bool]:
        """
        Batch get multiple objects via zero-copy.

        Args:
            keys: Storage keys to retrieve.
            dst_ptrs: Destination memory pointers (must be registered for RDMA).
            sizes: Expected sizes in bytes for each key.

        Returns:
            List of booleans: True if the corresponding key was retrieved successfully.
        """
        if not self._connected or self._base._store is None:
            return [False] * len(keys)
        if not keys:
            return []
        if not (len(keys) == len(dst_ptrs) == len(sizes)):
            raise ValueError("keys, dst_ptrs, and sizes must have the same length")

        results = self._base._batch_get(keys, dst_ptrs, sizes)
        return [r > 0 for r in results]

    def batch_set(
        self,
        keys: List[str],
        src_ptrs: List[int],
        sizes: List[int],
    ) -> List[bool]:
        """
        Batch set multiple objects via zero-copy.

        Skips keys that already exist in the store to avoid redundant writes.

        Args:
            keys: Storage keys.
            src_ptrs: Source memory pointers (must be registered for RDMA).
            sizes: Data sizes in bytes.

        Returns:
            List of booleans: True if the corresponding key was stored successfully.
        """
        if not self._connected or self._base._store is None:
            return [False] * len(keys)
        if not keys:
            return []
        if not (len(keys) == len(src_ptrs) == len(sizes)):
            raise ValueError("keys, src_ptrs, and sizes must have the same length")

        put_results = self._base._batch_put(keys, src_ptrs, sizes)
        final_results = [r == 0 for r in put_results]
        success = put_results.count(0)
        total_bytes = sum(s for r, s in zip(put_results, sizes) if r == 0)
        self.logger.debug(
            f"batch_set {len(keys)} keys: " f"written={success}/{len(keys)}, " f"data={total_bytes / 1024**3:.4f} GB"
        )

        return final_results

    def batch_exists(self, keys: List[str]) -> List[bool]:
        """Batch check key existence."""
        if not self._connected or self._base._store is None:
            return [False] * len(keys)
        if not keys:
            return []
        results, _ = self._base._batch_exists(keys)
        return [r == 1 for r in results]

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def delete(self, key: str, timeout: int = 5) -> bool:
        """
        Delete a key from the store, retrying up to ``timeout`` seconds.

        Args:
            key: Key to delete.
            timeout: Retry window in seconds.

        Returns:
            True if deletion succeeded within the timeout.
        """
        if not self._connected or self._base._store is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rc = self._base._store.remove(key)
            if rc == 0:
                return True
            time.sleep(1)
        self.logger.error(f"delete({key!r}) timed out after {timeout}s")
        return False

    def batch_delete(self, keys: List[str]) -> List[bool]:
        """
        Delete multiple keys from the store (single attempt, no retry).

        Used for cleaning up partial writes where some kinds succeeded
        and others failed. Returns per-key success flags.
        """
        if not self._connected or self._base._store is None:
            return [False] * len(keys)
        results = []
        for key in keys:
            rc = self._base._store.remove(key)
            results.append(rc == 0)
        return results

    def clear(self) -> int:
        """
        Remove all objects from the store.

        Returns:
            Number of objects removed (as reported by the store).
        """
        if not self._connected or self._base._store is None:
            return 0
        count: int = self._base._store.remove_all()
        self.logger.info(f"Cleared {count} objects from Mooncake store.")
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_extra_config(config: Any) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, dict):
            return config.get("kvcache_storage_config", config)
        return getattr(config, "kvcache_storage_config", None) or {}
