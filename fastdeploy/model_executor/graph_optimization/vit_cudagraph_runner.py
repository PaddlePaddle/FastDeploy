"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import paddle
from paddle.device.cuda import graphs

from fastdeploy.utils import get_logger

logger = get_logger("vit_cudagraph_runner", "vit_cudagraph_runner.log")


@dataclass
class _Qwen25GraphEntry:
    graph: Optional[graphs.CUDAGraph] = None
    input_buffer: Optional[paddle.Tensor] = None
    rotary_buffer: Optional[paddle.Tensor] = None
    cu_full_buffer: Optional[paddle.Tensor] = None
    cu_window_buffer: Optional[paddle.Tensor] = None
    output: Optional[paddle.Tensor] = None
    captured: bool = False
    disabled: bool = False
    capture_error: str = ""
    replay_count: int = 0


@dataclass
class _Qwen3GraphEntry:
    graph: Optional[graphs.CUDAGraph] = None
    input_buffer: Optional[paddle.Tensor] = None
    rotary_buffer: Optional[paddle.Tensor] = None
    cu_buffer: Optional[paddle.Tensor] = None
    output: Optional[paddle.Tensor] = None
    captured: bool = False
    disabled: bool = False
    capture_error: str = ""
    replay_count: int = 0


class Qwen25ViTCudaGraphRunner:
    """ViT CUDA graph runner for Qwen2.5-VL vision encoder."""

    def __init__(self, vit, warmup_iters: int = 2, max_graph_entries: int = 8) -> None:
        self.vit = vit
        self.warmup_iters = warmup_iters
        self.max_graph_entries = max(1, int(max_graph_entries))
        self._entries: OrderedDict[Tuple[int, int, int, int, int], _Qwen25GraphEntry] = OrderedDict()
        self._fullatt_block_indexes = set(int(x) for x in getattr(vit, "fullatt_block_indexes", []))
        logger.info(
            "[ViT CG][Qwen2.5] Initialized. warmup_iters=%s, max_graph_entries=%s, fullatt_block_indexes=%s",
            self.warmup_iters,
            self.max_graph_entries,
            sorted(self._fullatt_block_indexes),
        )

    def _release_entry(self, key: Tuple[int, int, int, int, int], entry: _Qwen25GraphEntry) -> None:
        entry.graph = None
        entry.input_buffer = None
        entry.rotary_buffer = None
        entry.cu_full_buffer = None
        entry.cu_window_buffer = None
        entry.output = None
        logger.info("[ViT CG][Qwen2.5] Evicted LRU graph entry. key=%s", key)

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_graph_entries:
            old_key, old_entry = self._entries.popitem(last=False)
            self._release_entry(old_key, old_entry)

    def _graph_key(
        self,
        seq_len: int,
        cu_full: paddle.Tensor,
        cu_window: paddle.Tensor,
        max_seqlen_full: int,
        max_seqlen_window: int,
    ) -> Tuple[int, int, int, int, int]:
        return (
            int(seq_len),
            int(cu_full.shape[0]),
            int(cu_window.shape[0]),
            int(max_seqlen_full),
            int(max_seqlen_window),
        )

    def _forward_blocks_and_merger(
        self,
        hidden_states: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
        cu_full: paddle.Tensor,
        cu_window: paddle.Tensor,
        max_seqlen_full: int,
        max_seqlen_window: int,
    ) -> paddle.Tensor:
        for layer_num, block in enumerate(self.vit.blocks):
            if layer_num in self._fullatt_block_indexes:
                cu_seqlens_now = cu_full
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window
                max_seqlen_now = max_seqlen_window

            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                max_seqlen=max_seqlen_now,
                rotary_pos_emb=rotary_pos_emb,
            )

        return self.vit.merger(hidden_states)

    def _create_entry(
        self,
        key: Tuple[int, int, int, int, int],
        hidden_states: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
        cu_full: paddle.Tensor,
        cu_window: paddle.Tensor,
        max_seqlen_full: int,
        max_seqlen_window: int,
    ) -> _Qwen25GraphEntry:
        entry = _Qwen25GraphEntry()
        entry.input_buffer = paddle.empty_like(hidden_states)
        entry.rotary_buffer = paddle.empty_like(rotary_pos_emb)
        entry.cu_full_buffer = paddle.empty_like(cu_full)
        entry.cu_window_buffer = paddle.empty_like(cu_window)
        entry.input_buffer.copy_(hidden_states, False)
        entry.rotary_buffer.copy_(rotary_pos_emb, False)
        entry.cu_full_buffer.copy_(cu_full, False)
        entry.cu_window_buffer.copy_(cu_window, False)

        try:
            for _ in range(self.warmup_iters):
                self._forward_blocks_and_merger(
                    entry.input_buffer,
                    entry.rotary_buffer,
                    entry.cu_full_buffer,
                    entry.cu_window_buffer,
                    max_seqlen_full,
                    max_seqlen_window,
                )

            graph = graphs.CUDAGraph()
            paddle.device.synchronize()
            graph.capture_begin()
            entry.output = self._forward_blocks_and_merger(
                entry.input_buffer,
                entry.rotary_buffer,
                entry.cu_full_buffer,
                entry.cu_window_buffer,
                max_seqlen_full,
                max_seqlen_window,
            )
            graph.capture_end()
            paddle.device.synchronize()
            entry.graph = graph
            entry.captured = True
            logger.info("[ViT CG][Qwen2.5] Capture success. key=%s", key)
        except Exception as err:  # pragma: no cover - best effort fallback on runtime failures
            entry.disabled = True
            entry.capture_error = repr(err)
            logger.warning(
                "[ViT CG][Qwen2.5] Capture failed, fallback to eager. key=%s, err=%s",
                key,
                entry.capture_error,
            )

        self._entries[key] = entry
        self._entries.move_to_end(key, last=True)
        self._evict_if_needed()
        return entry

    def run(
        self,
        hidden_states: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor,
        cu_full: paddle.Tensor,
        cu_window: paddle.Tensor,
        max_seqlen_full: int,
        max_seqlen_window: int,
    ) -> paddle.Tensor:
        if not hidden_states.place.is_gpu_place():
            logger.debug("[ViT CG][Qwen2.5] Non-GPU place detected, run eager path.")
            return self._forward_blocks_and_merger(
                hidden_states,
                rotary_pos_emb,
                cu_full,
                cu_window,
                max_seqlen_full,
                max_seqlen_window,
            )

        seq_len = int(hidden_states.shape[0])
        key = self._graph_key(
            seq_len=seq_len,
            cu_full=cu_full,
            cu_window=cu_window,
            max_seqlen_full=max_seqlen_full,
            max_seqlen_window=max_seqlen_window,
        )
        entry = self._entries.get(key)
        if entry is None:
            logger.info(
                "[ViT CG][Qwen2.5] Cache miss. key=%s, seq_len=%s, cu_full_len=%s, cu_window_len=%s",
                key,
                seq_len,
                int(cu_full.shape[0]),
                int(cu_window.shape[0]),
            )
            entry = self._create_entry(
                key=key,
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                cu_full=cu_full,
                cu_window=cu_window,
                max_seqlen_full=max_seqlen_full,
                max_seqlen_window=max_seqlen_window,
            )
        else:
            self._entries.move_to_end(key, last=True)

        if entry.disabled or (not entry.captured) or entry.graph is None or entry.output is None:
            logger.info("[ViT CG][Qwen2.5] Eager fallback. key=%s", key)
            return self._forward_blocks_and_merger(
                hidden_states,
                rotary_pos_emb,
                cu_full,
                cu_window,
                max_seqlen_full,
                max_seqlen_window,
            )

        entry.input_buffer.copy_(hidden_states, False)
        entry.rotary_buffer.copy_(rotary_pos_emb, False)
        entry.cu_full_buffer.copy_(cu_full, False)
        entry.cu_window_buffer.copy_(cu_window, False)
        entry.graph.replay()
        entry.replay_count += 1
        if entry.replay_count <= 5:
            logger.info("[ViT CG][Qwen2.5] Replay hit. key=%s, replay_count=%s", key, entry.replay_count)
        else:
            logger.debug("[ViT CG][Qwen2.5] Replay hit. key=%s, replay_count=%s", key, entry.replay_count)
        return entry.output


class Qwen3ViTCudaGraphRunner:
    """ViT CUDA graph runner for Qwen3-VL vision encoder."""

    def __init__(self, vit, warmup_iters: int = 2, max_graph_entries: int = 8) -> None:
        self.vit = vit
        self.warmup_iters = warmup_iters
        self.max_graph_entries = max(1, int(max_graph_entries))
        self._entries: OrderedDict[Tuple[int, int, int], _Qwen3GraphEntry] = OrderedDict()
        self._deepstack_visual_indexes = [int(x) for x in getattr(vit, "deepstack_visual_indexes", [])]
        self._deepstack_merge_map = {layer_id: idx for idx, layer_id in enumerate(self._deepstack_visual_indexes)}
        logger.info(
            "[ViT CG][Qwen3] Initialized. warmup_iters=%s, max_graph_entries=%s, deepstack_visual_indexes=%s",
            self.warmup_iters,
            self.max_graph_entries,
            self._deepstack_visual_indexes,
        )

    def _release_entry(self, key: Tuple[int, int, int], entry: _Qwen3GraphEntry) -> None:
        entry.graph = None
        entry.input_buffer = None
        entry.rotary_buffer = None
        entry.cu_buffer = None
        entry.output = None
        logger.info("[ViT CG][Qwen3] Evicted LRU graph entry. key=%s", key)

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_graph_entries:
            old_key, old_entry = self._entries.popitem(last=False)
            self._release_entry(old_key, old_entry)

    def _graph_key(
        self,
        seq_len: int,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
    ) -> Tuple[int, int, int]:
        return (
            int(seq_len),
            # int(cu_seqlens.shape[0]),
            # int(max_seqlen),
        )

    def _forward_blocks_and_merger(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
        rotary_pos_emb: paddle.Tensor,
    ) -> paddle.Tensor:
        deepstack_features = []
        for layer_id, block in enumerate(self.vit.blocks):
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rotary_pos_emb)
            deepstack_merge_idx = self._deepstack_merge_map.get(layer_id)
            if deepstack_merge_idx is not None:
                deepstack_features.append(self.vit.deepstack_merger_list[deepstack_merge_idx](hidden_states))

        hidden_states = self.vit.merger(hidden_states)
        if deepstack_features:
            hidden_states = paddle.concat([hidden_states] + deepstack_features, axis=1)
        return hidden_states

    def _create_entry(
        self,
        key: Tuple[int, int, int],
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
        rotary_pos_emb: paddle.Tensor,
    ) -> _Qwen3GraphEntry:
        entry = _Qwen3GraphEntry()
        entry.input_buffer = paddle.empty_like(hidden_states)
        entry.rotary_buffer = paddle.empty_like(rotary_pos_emb)
        entry.cu_buffer = paddle.empty_like(cu_seqlens)
        entry.input_buffer.copy_(hidden_states, False)
        entry.rotary_buffer.copy_(rotary_pos_emb, False)
        entry.cu_buffer.copy_(cu_seqlens, False)

        try:
            for _ in range(self.warmup_iters):
                self._forward_blocks_and_merger(
                    entry.input_buffer,
                    entry.cu_buffer,
                    max_seqlen,
                    entry.rotary_buffer,
                )

            graph = graphs.CUDAGraph()
            paddle.device.synchronize()
            graph.capture_begin()
            entry.output = self._forward_blocks_and_merger(
                entry.input_buffer,
                entry.cu_buffer,
                max_seqlen,
                entry.rotary_buffer,
            )
            graph.capture_end()
            paddle.device.synchronize()
            entry.graph = graph
            entry.captured = True
            logger.info("[ViT CG][Qwen3] Capture success. key=%s", key)
        except Exception as err:  # pragma: no cover - best effort fallback on runtime failures
            entry.disabled = True
            entry.capture_error = repr(err)
            logger.warning(
                "[ViT CG][Qwen3] Capture failed, fallback to eager. key=%s, err=%s",
                key,
                entry.capture_error,
            )

        self._entries[key] = entry
        self._entries.move_to_end(key, last=True)
        self._evict_if_needed()
        return entry

    def run(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        max_seqlen: int,
        rotary_pos_emb: paddle.Tensor,
    ) -> paddle.Tensor:
        t_total_begin = time.perf_counter()
        if not hidden_states.place.is_gpu_place():
            logger.info("[ViT CG][Qwen3] Non-GPU place detected, run eager path.")
            out = self._forward_blocks_and_merger(hidden_states, cu_seqlens, max_seqlen, rotary_pos_emb)
            t_total_end = time.perf_counter()
            logger.info(
                "[ViT CG][Qwen3][TIME] path=eager_non_gpu total_ms=%.3f seq_len=%s",
                (t_total_end - t_total_begin) * 1000.0,
                int(hidden_states.shape[0]),
            )
            return out

        seq_len = int(hidden_states.shape[0])
        key = self._graph_key(
            seq_len=seq_len,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        entry = self._entries.get(key)
        if entry is None:
            t_create_begin = time.perf_counter()
            logger.info(
                "[ViT CG][Qwen3] Cache miss. key=%s, seq_len=%s, cu_len=%s, max_seqlen=%s",
                key,
                seq_len,
                int(cu_seqlens.shape[0]),
                int(max_seqlen),
            )
            entry = self._create_entry(
                key=key,
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_pos_emb=rotary_pos_emb,
            )
            t_create_end = time.perf_counter()
            logger.info(
                "[ViT CG][Qwen3][TIME] stage=create_entry elapsed_ms=%.3f key=%s captured=%s disabled=%s",
                (t_create_end - t_create_begin) * 1000.0,
                key,
                entry.captured,
                entry.disabled,
            )
        else:
            self._entries.move_to_end(key, last=True)

        if entry.disabled or (not entry.captured) or entry.graph is None or entry.output is None:
            logger.info("[ViT CG][Qwen3] Eager fallback. key=%s", key)
            out = self._forward_blocks_and_merger(hidden_states, cu_seqlens, max_seqlen, rotary_pos_emb)
            t_total_end = time.perf_counter()
            logger.info(
                "[ViT CG][Qwen3][TIME] path=eager_fallback total_ms=%.3f key=%s seq_len=%s",
                (t_total_end - t_total_begin) * 1000.0,
                key,
                seq_len,
            )
            return out

        entry.input_buffer.copy_(hidden_states, False)
        entry.rotary_buffer.copy_(rotary_pos_emb, False)
        entry.cu_buffer.copy_(cu_seqlens, False)
        t_replay_begin = time.perf_counter()
        entry.graph.replay()
        t_replay_end = time.perf_counter()
        entry.replay_count += 1
        if entry.replay_count <= 5:
            logger.info("[ViT CG][Qwen3] Replay hit. key=%s, replay_count=%s", key, entry.replay_count)
        else:
            logger.debug("[ViT CG][Qwen3] Replay hit. key=%s, replay_count=%s", key, entry.replay_count)
        logger.info(
            "[ViT CG][Qwen3][TIME] path=replay replay_ms=%.3f total_ms=%.3f key=%s replay_count=%s",
            (t_replay_end - t_replay_begin) * 1000.0,
            (t_replay_end - t_total_begin) * 1000.0,
            key,
            entry.replay_count,
        )
        return entry.output
