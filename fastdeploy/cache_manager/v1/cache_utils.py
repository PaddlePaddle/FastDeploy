"""
Utility classes and functions for cache management.
"""

import hashlib
import logging
import pickle
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

logger = logging.getLogger("cache_utils_debug")


class LayerDoneCounter:
    """
    Counter for tracking layer-by-layer transfer completion using CUDA events.

    Used in CacheController to synchronize layer transfers during
    multi-level cache operations. Each layer must complete before
    the next layer can be processed.

    Thread-safe implementation for use in async environments.
    Uses CUDA events for efficient waiting (no polling).
    """

    def __init__(self, num_layers: int = 0):
        """
        Initialize the layer done counter.

        Args:
            num_layers: Total number of layers to track
        """
        self._num_layers = num_layers
        self._lock = threading.RLock()
        self._completed_layers: Dict[str, Set[int]] = defaultdict(set)
        self._callbacks: Dict[str, List[Callable[[int], None]]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}

        # ============ CUDA Events for efficient waiting (no polling) ============
        self._cuda_events: Dict[str, List[Any]] = {}  # transfer_id -> list of events per layer
        self._layer_complete_times: Dict[str, Dict[int, float]] = {}  # transfer_id -> {layer_idx: complete_time}

        # ============ Reference count for active waiters (prevents premature clear) ============
        # Tracks how many wait_for_layer calls are actively waiting for each transfer
        self._wait_counts: Dict[str, int] = defaultdict(int)

    def get_num_layers(self) -> int:
        """Get the total number of layers."""
        return self._num_layers

    def start_transfer(self, transfer_id: str) -> None:
        """
        Mark the start of a transfer.

        Args:
            transfer_id: Unique identifier for the transfer
        """
        with self._lock:
            self._completed_layers[transfer_id] = set()
            self._start_times[transfer_id] = time.time()
            self._layer_complete_times[transfer_id] = {}

            # Create CUDA events for each layer
            try:
                import paddle
                self._cuda_events[transfer_id] = [
                    paddle.device.cuda.Event() if paddle.is_compiled_with_cuda() else None
                    for _ in range(self._num_layers)
                ]
            except Exception as e:
                logger.warning(f"Failed to create CUDA events for transfer {transfer_id}: {e}")
                self._cuda_events[transfer_id] = [None] * self._num_layers

    def mark_layer_done(self, transfer_id: str, layer_idx: int, cuda_event: Any = None) -> bool:
        """
        Mark a layer as completed.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the completed layer
            cuda_event: Optional CUDA event to record completion

        Returns:
            True if this was the last layer, False otherwise
        """
        with self._lock:
            if transfer_id not in self._completed_layers:
                logger.error(f"[mark_layer_done] FAILED: transfer_id={transfer_id} not in _completed_layers. Available keys: {list(self._completed_layers.keys())}")
                return False

            self._completed_layers[transfer_id].add(layer_idx)
            self._layer_complete_times[transfer_id][layer_idx] = time.time()

            # Record CUDA event if provided
            if cuda_event is not None and transfer_id in self._cuda_events:
                try:
                    cuda_event.record()
                except Exception as e:
                    logger.warning(f"Failed to record CUDA event for layer {layer_idx}: {e}")

            # Execute callbacks for this layer
            for callback in self._callbacks.get(transfer_id, []):
                try:
                    callback(layer_idx)
                except Exception:
                    pass  # Ignore callback errors

            return len(self._completed_layers[transfer_id]) >= self._num_layers

    def mark_all_layers_done(self, transfer_id: str, cuda_event: Any = None) -> bool:
        """
        Mark all layers as completed at once (optimization for swap_all_layers mode).

        Args:
            transfer_id: Unique identifier for the transfer
            cuda_event: Optional CUDA event to record completion

        Returns:
            True (always returns True since all layers are marked done)
        """
        with self._lock:
            if transfer_id not in self._completed_layers:
                logger.error(f"[mark_all_layers_done] FAILED: transfer_id={transfer_id} not in _completed_layers. Available keys: {list(self._completed_layers.keys())}")
                return False

            now = time.time()
            self._completed_layers[transfer_id] = set(range(self._num_layers))
            self._layer_complete_times[transfer_id] = {i: now for i in range(self._num_layers)}

            # Record CUDA event if provided
            if cuda_event is not None and transfer_id in self._cuda_events:
                try:
                    cuda_event.record()
                except Exception as e:
                    logger.warning(f"Failed to record CUDA event for transfer {transfer_id}: {e}")

            # Execute all callbacks (call with -1 to indicate all layers done)
            for callback in self._callbacks.get(transfer_id, []):
                try:
                    callback(-1)
                except Exception:
                    pass  # Ignore callback errors

            return True

    def is_layer_done(self, transfer_id: str, layer_idx: int) -> bool:
        """
        Check if a specific layer is completed.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the layer to check

        Returns:
            True if the layer is completed, False otherwise
        """
        with self._lock:
            return layer_idx in self._completed_layers.get(transfer_id, set())

    def is_transfer_complete(self, transfer_id: str) -> bool:
        """
        Check if all layers for a transfer are completed.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            True if all layers are completed, False otherwise
        """
        with self._lock:
            if transfer_id not in self._completed_layers:
                return False
            return len(self._completed_layers[transfer_id]) >= self._num_layers

    def get_completed_count(self, transfer_id: str) -> int:
        """
        Get the number of completed layers for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            Number of completed layers
        """
        with self._lock:
            return len(self._completed_layers.get(transfer_id, set()))

    def get_pending_layers(self, transfer_id: str) -> List[int]:
        """
        Get list of pending layer indices for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            List of pending layer indices
        """
        with self._lock:
            if transfer_id not in self._completed_layers:
                return list(range(self._num_layers))
            completed = self._completed_layers[transfer_id]
            return [i for i in range(self._num_layers) if i not in completed]

    def register_callback(self, transfer_id: str, callback: Callable[[int], None]) -> None:
        """
        Register a callback to be called when each layer completes.

        Args:
            transfer_id: Unique identifier for the transfer
            callback: Function to call with layer index when completed
        """
        with self._lock:
            self._callbacks[transfer_id].append(callback)

    def increment_wait_count(self, transfer_id: str) -> None:
        """
        Increment the wait count for a transfer.
        Called when wait_for_layer starts waiting.

        Args:
            transfer_id: Unique identifier for the transfer
        """
        with self._lock:
            self._wait_counts[transfer_id] += 1
            logger.debug(f"[increment_wait_count] transfer_id={transfer_id}, count={self._wait_counts[transfer_id]}")

    def decrement_wait_count(self, transfer_id: str) -> None:
        """
        Decrement the wait count for a transfer.
        Called when wait_for_layer finishes waiting.

        Args:
            transfer_id: Unique identifier for the transfer
        """
        with self._lock:
            if self._wait_counts.get(transfer_id, 0) > 0:
                self._wait_counts[transfer_id] -= 1
                logger.debug(f"[decrement_wait_count] transfer_id={transfer_id}, count={self._wait_counts[transfer_id]}")

                # If count reaches 0, try to clear (in case clear_transfer was deferred)
                if self._wait_counts[transfer_id] == 0:
                    self._completed_layers.pop(transfer_id, None)
                    self._callbacks.pop(transfer_id, None)
                    self._start_times.pop(transfer_id, None)
                    self._cuda_events.pop(transfer_id, None)
                    self._layer_complete_times.pop(transfer_id, None)
                    self._wait_counts.pop(transfer_id, None)
                    logger.debug(f"[decrement_wait_count] auto-cleared transfer_id={transfer_id}")

    def clear_transfer(self, transfer_id: str) -> None:
        """
        Clear tracking for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer
        """
        with self._lock:
            # Check if there are active waiters - if so, defer clearing
            if self._wait_counts.get(transfer_id, 0) > 0:
                logger.debug(f"[clear_transfer] deferred for {transfer_id}, wait_count={self._wait_counts[transfer_id]}")
                return

            self._completed_layers.pop(transfer_id, None)
            self._callbacks.pop(transfer_id, None)
            self._start_times.pop(transfer_id, None)
            self._cuda_events.pop(transfer_id, None)
            self._layer_complete_times.pop(transfer_id, None)
            self._wait_counts.pop(transfer_id, None)
            logger.debug(f"[clear_transfer] completed for {transfer_id}")

    # ============ CUDA Event Methods ============

    def get_layer_cuda_event(self, transfer_id: str, layer_idx: int) -> Any:
        """
        Get the CUDA event for a specific layer.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the layer

        Returns:
            CUDA event for the layer, or None if not available
        """
        with self._lock:
            if transfer_id not in self._cuda_events:
                return None
            events = self._cuda_events[transfer_id]
            if layer_idx < len(events):
                return events[layer_idx]
            return None

    def get_layer_complete_time(self, transfer_id: str, layer_idx: int) -> Optional[float]:
        """
        Get the completion time for a specific layer.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the layer

        Returns:
            Completion time as Unix timestamp, or None if not completed
        """
        with self._lock:
            if transfer_id not in self._layer_complete_times:
                return None
            return self._layer_complete_times[transfer_id].get(layer_idx)

    def get_layer_wait_time(self, transfer_id: str, layer_idx: int) -> Optional[float]:
        """
        Get the time from transfer start to layer completion.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the layer

        Returns:
            Time in seconds, or None if transfer not found or layer not completed
        """
        with self._lock:
            if transfer_id not in self._start_times:
                return None
            complete_time = self._layer_complete_times.get(transfer_id, {}).get(layer_idx)
            if complete_time is None:
                return None
            return complete_time - self._start_times[transfer_id]

    def get_all_layer_times(self, transfer_id: str) -> Dict[int, float]:
        """
        Get completion times for all layers.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            Dictionary mapping layer_idx to completion time
        """
        with self._lock:
            return self._layer_complete_times.get(transfer_id, {}).copy()

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._completed_layers.clear()
            self._callbacks.clear()
            self._start_times.clear()
            self._cuda_events.clear()
            self._layer_complete_times.clear()

    def get_elapsed_time(self, transfer_id: str) -> Optional[float]:
        """
        Get elapsed time for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            Elapsed time in seconds, or None if transfer not found
        """
        with self._lock:
            if transfer_id not in self._start_times:
                return None
            return time.time() - self._start_times[transfer_id]

    def get_stats(self) -> Dict:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "num_layers": self._num_layers,
                "active_transfers": len(self._completed_layers),
                "transfer_ids": list(self._completed_layers.keys()),
            }


# ============ Block Hash Computation ============


def hash_block_tokens(
    token_ids: Sequence[int],
    parent_block_hash: str | None = None,
    extra_keys: Any = None,
) -> str:
    """
    Compute hash value for a single block.

    Reference: vLLM's hash_block_tokens implementation using chained hash:
    hash = SHA256((parent_block_hash, token_ids_tuple, extra_keys))

    Args:
        token_ids: Token IDs of the current block.
        parent_block_hash: Hash of the parent block (chained hash).
        extra_keys: Additional keys (e.g., multimodal info, LoRA).

    Returns:
        Computed block hash as hex string.
    """
    if parent_block_hash is None:
        parent_block_hash = ""

    value = (parent_block_hash, tuple(token_ids), extra_keys)
    return hashlib.sha256(pickle.dumps(value)).hexdigest()


def get_request_block_hasher(
    block_size: int,
) -> Callable[[Any], List[str]]:
    """
    Factory function: returns a block hash calculator bound to block_size.

    The returned function computes hashes for new complete blocks in a request.
    Computation logic:
    1. Get all token IDs (prompt + output)
    2. Determine starting position based on existing block_hashes count
    3. Compute hashes for new complete blocks (chained hash)

    Usage:
        # Create hasher at service startup
        block_hasher = get_request_block_hasher(block_size=64)

        # Use in Request.prompt_hashes property
        new_hashes = block_hasher(self)
        self._prompt_hashes.extend(new_hashes)

    Args:
        block_size: Number of tokens per block.

    Returns:
        A function that takes a request and returns a list of newly computed
        block hashes.
    """

    def request_block_hasher(request: Any) -> List[str]:
        """
        Compute hashes for uncomputed complete blocks in a request.

        Args:
            request: Request object with the following attributes:
                - prompt_token_ids: Input token IDs.
                - _prompt_hashes: List of existing block hashes (private attr).
                - output_token_ids: Output token IDs (optional).

        Returns:
            List of newly computed block hashes (only new complete blocks).
        """
        # Get prompt token IDs
        prompt_ids = request.prompt_token_ids
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids is None:
            prompt_ids = []

        # Get output token IDs
        output_ids = getattr(request, "output_token_ids", [])
        if hasattr(output_ids, "tolist"):
            output_ids = output_ids.tolist()
        if output_ids is None:
            output_ids = []

        # Combine all token IDs
        all_token_ids = list(prompt_ids) + list(output_ids)
        num_tokens = len(all_token_ids)

        # Get existing block hashes
        existing_hashes = getattr(request, "_prompt_hashes", [])
        if existing_hashes is None:
            existing_hashes = []

        # Calculate starting position (skip already computed blocks)
        start_token_idx = len(existing_hashes) * block_size

        # Return empty if no new complete blocks
        if start_token_idx + block_size > num_tokens:
            return []

        new_block_hashes: List[str] = []
        prev_block_hash = existing_hashes[-1] if existing_hashes else None

        # Compute hashes for new complete blocks
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                break

            # Get tokens for current block
            block_tokens = all_token_ids[start_token_idx:end_token_idx]

            # TODO: Add extra_keys support (multimodal, LoRA, etc.)

            # Compute hash (chained hash)
            block_hash = hash_block_tokens(block_tokens, prev_block_hash, None)
            new_block_hashes.append(block_hash)

            # Update state
            start_token_idx += block_size
            prev_block_hash = block_hash

        return new_block_hashes

    return request_block_hasher
