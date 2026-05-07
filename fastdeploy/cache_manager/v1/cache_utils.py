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

import hashlib
import pickle
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from paddleformers.utils.log import logger


class LayerDoneCounter:
    """
    Independent synchronization primitive for tracking layer completion of a single transfer.

    Used in compute-transfer overlap scenarios:
    - Each LayerDoneCounter instance tracks layer completion for one transfer task.
    - Uses CUDA Events for efficient waiting (no polling).
    - Thread-safe.

    Attributes:
        _num_layers: Total number of layers.
        _lock: Thread lock.
        _completed_layers: Set of completed layer indices.
        _callbacks: List of layer-completion callbacks.
        _cuda_events: CUDA event per layer.
        _layer_complete_times: Mapping of layer index to completion time.
        _wait_count: Count of active waiters.
    """

    def __init__(self, num_layers: int):
        """
        Initialize the layer done counter.

        Args:
            num_layers: Total number of layers to track
        """
        self._num_layers = num_layers
        self._lock = threading.RLock()
        self._completed_layers: Set[int] = set()
        self._callbacks: List[Callable[[int], None]] = []
        self._start_time: float = time.time()

        # ============ CUDA Events for efficient waiting (no polling) ============
        # Initialized to None; set by set_layer_event() after kernel submission to transfer stream.
        # None means no event recorded yet for that layer (must fall back to polling).
        self._cuda_events: List[Any] = [None] * num_layers
        self._layer_complete_times: Dict[int, float] = {}

        # ============ Reference count for active waiters (prevents premature cleanup) ============
        self._wait_count: int = 0

    def get_num_layers(self) -> int:
        """Get the total number of layers."""
        return self._num_layers

    # ============ Mark Methods (called by transfer thread) ============

    def set_layer_event(self, layer_idx: int, cuda_event: Any) -> None:
        """
        Set the CUDA event for a specific layer (used for cross-stream synchronization).

        Called by transfer thread after submitting a layer's kernel to a non-default
        stream (e.g., input_stream), so that wait_for_layer() can correctly synchronize
        on the actual stream where the transfer runs.

        Args:
            layer_idx: Index of the layer
            cuda_event: CUDA event recorded on the transfer stream after kernel submission
        """
        with self._lock:
            if 0 <= layer_idx < len(self._cuda_events):
                self._cuda_events[layer_idx] = cuda_event

    def mark_layer_done(self, layer_idx: int, cuda_event: Any = None) -> bool:
        """
        Mark a layer as completed.

        Args:
            layer_idx: Index of the completed layer
            cuda_event: Optional CUDA event to record completion

        Returns:
            True if this was the last layer, False otherwise
        """
        with self._lock:
            if layer_idx in self._completed_layers:
                logger.warning(f"[mark_layer_done] layer {layer_idx} already marked done")
                return len(self._completed_layers) >= self._num_layers

            self._completed_layers.add(layer_idx)
            self._layer_complete_times[layer_idx] = time.time()

            # Record CUDA event if provided
            if cuda_event is not None:
                try:
                    cuda_event.record()
                except Exception as e:
                    logger.warning(f"Failed to record CUDA event for layer {layer_idx}: {e}")

            # Execute callbacks for this layer
            for callback in self._callbacks:
                try:
                    callback(layer_idx)
                except Exception:
                    pass

            return len(self._completed_layers) >= self._num_layers

    def mark_all_done(self, cuda_event: Any = None) -> bool:
        """
        Mark all layers as completed at once (used for D2H all-layers evict mode).

        Args:
            cuda_event: Optional CUDA event to record completion

        Returns:
            True (always returns True since all layers are marked done)
        """
        with self._lock:
            now = time.time()
            self._completed_layers = set(range(self._num_layers))
            self._layer_complete_times = {i: now for i in range(self._num_layers)}

            # Record CUDA event if provided
            if cuda_event is not None:
                try:
                    cuda_event.record()
                except Exception as e:
                    logger.warning(f"Failed to record CUDA event: {e}")

            # Execute all callbacks (call with -1 to indicate all layers done)
            for callback in self._callbacks:
                try:
                    callback(-1)
                except Exception:
                    pass

            return True

    # ============ Query Methods ============

    def is_layer_done(self, layer_idx: int) -> bool:
        """
        Check if a specific layer is completed.

        Args:
            layer_idx: Index of the layer to check

        Returns:
            True if the layer is completed, False otherwise
        """
        with self._lock:
            return layer_idx in self._completed_layers

    def is_all_done(self) -> bool:
        """
        Check if all layers are completed.

        Returns:
            True if all layers are completed, False otherwise
        """
        with self._lock:
            return len(self._completed_layers) >= self._num_layers

    def get_completed_count(self) -> int:
        """
        Get the number of completed layers.

        Returns:
            Number of completed layers
        """
        with self._lock:
            return len(self._completed_layers)

    def get_pending_layers(self) -> List[int]:
        """
        Get list of pending layer indices.

        Returns:
            List of pending layer indices
        """
        with self._lock:
            return [i for i in range(self._num_layers) if i not in self._completed_layers]

    # ============ Wait Methods (called by forward thread) ============

    def wait_for_layer(self, layer_idx: int, timeout: Optional[float] = None) -> bool:
        """
        Wait for a specific layer to complete (CUDA Event synchronization).

        Always synchronizes the CUDA event before returning to guarantee the GPU
        transfer has actually completed, not just that the kernel was submitted.
        The fast path that only checked is_layer_done() was unsafe because
        mark_layer_done() is called immediately after kernel submission (async),
        before the GPU has finished the transfer.

        Args:
            layer_idx: Index of the layer to wait for
            timeout: Maximum wait time in seconds (default: 1s)

        Returns:
            True if layer completed

        Raises:
            LayerSwapTimeoutError: If timeout occurs before layer completes
        """
        self._increment_wait_count()
        try:
            start_time = time.time()
            timeout = timeout if timeout is not None else 1.0
            while True:
                # Always try CUDA event sync first: set_layer_event() is called before
                # mark_layer_done(), so once is_layer_done() is True the event is present.
                cuda_event = self._cuda_events[layer_idx] if layer_idx < len(self._cuda_events) else None
                if cuda_event is not None:
                    try:
                        cuda_event.synchronize()
                        return True
                    except Exception as e:
                        logger.warning(f"CUDA event sync failed for layer {layer_idx}: {e}")
                        # Event sync failed; fall through to is_layer_done check

                # No event yet (or sync failed): check software state as fallback
                # (covers non-cupy scenarios where events are never set)
                if self.is_layer_done(layer_idx):
                    return True

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.error(f"[WaitForLayer] layer={layer_idx} TIMEOUT after {elapsed:.2f}s")
                    raise LayerSwapTimeoutError(f"Layer swap timeout: layer={layer_idx}, elapsed={elapsed:.2f}s")

                time.sleep(0.001)
        finally:
            self._decrement_wait_count()

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all layers to complete (used for D2H all-layers evict mode).

        Always synchronizes _cuda_events[-1] (set by set_layer_event for the last layer)
        before returning, for the same reason as wait_for_layer.

        Args:
            timeout: Maximum wait time in seconds (default: 300s)

        Returns:
            True if all layers completed

        Raises:
            LayerSwapTimeoutError: If timeout occurs
        """
        self._increment_wait_count()
        try:
            start_time = time.time()
            timeout = timeout if timeout is not None else 300.0
            while True:
                # _cuda_events[-1] is set by set_layer_event(num_layers-1, ...) before mark_all_done()
                last_event = self._cuda_events[-1] if self._cuda_events else None
                if last_event is not None:
                    try:
                        last_event.synchronize()
                        return True
                    except Exception as e:
                        logger.warning(f"CUDA event sync failed for wait_all: {e}")

                # No event yet (or sync failed): check software state as fallback
                if self.is_all_done():
                    return True

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.error(f"[wait_all] TIMEOUT after {elapsed:.2f}s")
                    raise LayerSwapTimeoutError(f"wait_all timeout: elapsed={elapsed:.2f}s")

                time.sleep(0.001)
        finally:
            self._decrement_wait_count()

    # ============ Callback Methods ============

    def register_callback(self, callback: Callable[[int], None]) -> None:
        """
        Register a callback to be called when each layer completes.

        Args:
            callback: Function to call with layer index when completed
        """
        with self._lock:
            self._callbacks.append(callback)

    # ============ Internal Helper Methods ============

    def _increment_wait_count(self) -> None:
        """Increment the wait count."""
        with self._lock:
            self._wait_count += 1

    def _decrement_wait_count(self) -> None:
        """Decrement the wait count."""
        with self._lock:
            if self._wait_count > 0:
                self._wait_count -= 1

    def _should_cleanup(self) -> bool:
        """Check if cleanup is safe (no active waiters and all done)."""
        with self._lock:
            return self._wait_count == 0 and self.is_all_done()

    # ============ Time Tracking Methods ============

    def get_layer_complete_time(self, layer_idx: int) -> Optional[float]:
        """
        Get the completion time for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Completion time as Unix timestamp, or None if not completed
        """
        with self._lock:
            return self._layer_complete_times.get(layer_idx)

    def get_layer_wait_time(self, layer_idx: int) -> Optional[float]:
        """
        Get the time from transfer start to layer completion.

        Args:
            layer_idx: Index of the layer

        Returns:
            Time in seconds, or None if not completed
        """
        with self._lock:
            complete_time = self._layer_complete_times.get(layer_idx)
            if complete_time is None:
                return None
            return complete_time - self._start_time

    def get_all_layer_times(self) -> Dict[int, float]:
        """
        Get completion times for all layers.

        Returns:
            Dictionary mapping layer_idx to completion time
        """
        with self._lock:
            return self._layer_complete_times.copy()

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since transfer start.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self._start_time

    def get_stats(self) -> Dict:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "num_layers": self._num_layers,
                "completed_layers": len(self._completed_layers),
                "pending_layers": self._num_layers - len(self._completed_layers),
                "wait_count": self._wait_count,
            }

    # ============ Cleanup Methods ============

    def cleanup(self) -> None:
        """
        Explicit cleanup method to release CUDA events.

        Called when the transfer is complete and no more waiting is needed.
        """
        with self._lock:
            # Check if safe to cleanup
            if self._wait_count > 0:
                return

            # Clear CUDA events
            self._cuda_events.clear()

    def __del__(self) -> None:
        """
        Destructor to ensure CUDA events are released.

        Note: This is a fallback. For explicit cleanup, call cleanup() method.
        """
        try:
            if self._cuda_events:
                self._cuda_events.clear()
        except Exception:
            pass  # Ignore errors during destruction


class LayerSwapTimeoutError(Exception):
    """Exception raised when layer swap operation times out."""

    pass


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


def get_block_hash_extra_keys(
    request: Any,
    start_idx: int,
    end_idx: int,
    mm_idx: int,
) -> tuple:
    """
    Retrieve additional hash keys for a block based on multimodal information.

    Mirrors the logic from prefix_cache_manager.PrefixCacheManager.get_block_hash_extra_keys.

    For each block [start_idx, end_idx), scans the multimodal positions starting
    from mm_idx and collects hashes of any multimodal items that overlap with the block.

    Args:
        request: Request object.  Must expose a ``multimodal_inputs`` attribute which
            is either None or a dict with keys:
                - ``mm_positions``: list of objects with ``.offset`` and ``.length``
                - ``mm_hashes``:    list of hash strings, one per multimodal item
        start_idx: Token index of the block start (inclusive).
        end_idx:   Token index of the block end (exclusive).
        mm_idx:    Index into mm_positions / mm_hashes to start scanning from
                   (avoids re-scanning already-processed items).

    Returns:
        (next_mm_idx, hash_keys):
            next_mm_idx: updated mm_idx for the next block.
            hash_keys  : list of multimodal hash strings that fall within this block.
    """
    hash_keys: List[str] = []
    mm_inputs = getattr(request, "multimodal_inputs", None)
    if (
        mm_inputs is None
        or "mm_positions" not in mm_inputs
        or "mm_hashes" not in mm_inputs
        or len(mm_inputs["mm_positions"]) == 0
    ):
        return mm_idx, hash_keys

    mm_positions = mm_inputs["mm_positions"]
    mm_hashes = mm_inputs["mm_hashes"]

    # Fast exit: last multimodal item ends before this block starts
    if mm_positions[-1].offset + mm_positions[-1].length <= start_idx:
        return mm_idx, hash_keys

    for img_idx in range(mm_idx, len(mm_positions)):
        image_offset = mm_positions[img_idx].offset
        image_length = mm_positions[img_idx].length

        if image_offset + image_length <= start_idx:
            # Multimodal item ends before block starts – skip
            continue
        elif image_offset >= end_idx:
            # Multimodal item starts after block ends – stop
            return img_idx, hash_keys
        elif image_offset + image_length > end_idx:
            # Multimodal item spans beyond block end – include hash, stop at this item
            hash_keys.append(mm_hashes[img_idx])
            return img_idx, hash_keys
        else:
            # Multimodal item is fully contained within the block
            hash_keys.append(mm_hashes[img_idx])

    return len(mm_positions) - 1, hash_keys


def get_request_block_hasher(
    block_size: int,
) -> Callable[[Any], List[str]]:
    """
    Factory function: returns a block hash calculator bound to block_size.

    The returned function computes hashes for new complete blocks in a request.
    Computation logic:
    1. Get all token IDs (prompt + output)
    2. Determine starting position based on existing block_hashes count
    3. Compute hashes for new complete blocks (chained hash, with multimodal extra_keys)

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
                - multimodal_inputs (optional): Multimodal info dict with
                  ``mm_positions`` and ``mm_hashes``.

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

        # mm_idx tracks which multimodal item to scan from, avoiding redundant iteration
        mm_idx = 0

        # Compute hashes for new complete blocks
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                break

            # Get tokens for current block
            block_tokens = all_token_ids[start_token_idx:end_token_idx]

            # Collect multimodal extra_keys for this block
            mm_idx, extra_keys = get_block_hash_extra_keys(
                request=request,
                start_idx=start_token_idx,
                end_idx=end_token_idx,
                mm_idx=mm_idx,
            )
            extra_keys_value = tuple(extra_keys) if extra_keys else None

            # Compute hash (chained hash)
            block_hash = hash_block_tokens(block_tokens, prev_block_hash, extra_keys_value)
            new_block_hashes.append(block_hash)

            # Update state
            start_token_idx += block_size
            prev_block_hash = block_hash

        return new_block_hashes

    return request_block_hasher
