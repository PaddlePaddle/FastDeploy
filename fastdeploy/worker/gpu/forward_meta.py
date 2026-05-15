"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import paddle

from fastdeploy.model_executor.forward_meta import ForwardMode
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)


@dataclass
class ForwardMetaV1:
    # ------------------------------------------------------------------ #
    # Required fields                                                      #
    # ------------------------------------------------------------------ #

    # Structured input batch (runtime type: InputBatch).
    # Contains token IDs, positions, seq_lens, block_table_tensor, etc.
    input_batch: Any

    # KV cache tensors for all layers.
    # Standard layout : [key_0, val_0, key_1, val_1, ...]  → 2 * num_layers entries.
    # block_wise_fp8   : [key_0, val_0, key_scale_0, val_scale_0, ...]  → 4 * num_layers entries.
    caches: List[paddle.Tensor]

    # Block table tensor, shape [num_seqs, max_blocks].
    # Contains the block mapping for each sequence.
    block_table_tensor: paddle.Tensor

    # Slot mapping for paged KV attention, shape [num_tokens].
    # Maps each query token position to the physical cache slot where its
    # key/value projection should be written.
    slot_mapping: paddle.Tensor

    # ------------------------------------------------------------------ #
    # Optional / control fields                                            #
    # ------------------------------------------------------------------ #

    # Whether to replay a previously captured CUDA Graph for this step.
    # True only when the batch contains *only* decode sequences and its size
    # matches one of the pre-captured bucket sizes.
    step_use_cudagraph: bool = False

    # Attention backend instance (FlashInferAttentionBackend).
    attn_backend: Optional[AttentionBackend] = None

    # Execution mode: EXTEND (prefill only), DECODE (decode only), or MIXED.
    forward_mode: ForwardMode = ForwardMode.MIXED

    # Suppress expensive initialisations during warm-up / memory-profiling.
    is_dummy_or_profile_run: bool = False

    # True when the effective token count is zero (e.g. an EP worker that
    # holds no expert shards for this batch).
    is_zero_size: bool = False

    # ------------------------------------------------------------------ #
    # Compatibility shim                                                   #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        """
        Return ``None`` for any attribute that is not explicitly declared.

        This makes ``ForwardMetaV1`` duck-type compatible with ``ForwardMeta``
        so that attention backends can safely access optional fields such as
        ``rotary_embs``, ``attn_mask``, ``position_ids``, etc. without raising
        ``AttributeError``.
        """
        # Avoid infinite recursion for dunder attributes that Python itself
        # looks up (e.g. ``__repr__``, ``__class__``, etc.).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self.__dict__[name] = None
        return None
