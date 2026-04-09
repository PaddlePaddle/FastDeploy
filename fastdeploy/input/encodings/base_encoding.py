# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Abstract base class for multimodal encoding strategies.

Each encoding strategy handles model-family-specific logic such as
position ID computation, image/video preprocessing, and token counting.
New model families should subclass ``BaseEncoding`` and implement all
abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseEncoding(ABC):
    """Contract that every encoding strategy must fulfil.

    Required (abstract) methods cover the core encoding pipeline.
    Optional methods (``init_extra``, ``get_mm_max_tokens_per_item``) have
    default no-op implementations so subclasses only override when needed.
    """

    def __init__(self, processor):
        self.p = processor

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    @abstractmethod
    def add_image(self, img, outputs: dict, uuid, token_len=None):
        """Process a raw image and append results to *outputs*."""

    @abstractmethod
    def add_processed_image(self, img_cache, outputs: dict, uuid, token_len=None):
        """Append a pre-processed (cached) image to *outputs*."""

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    @abstractmethod
    def add_video(self, frames, outputs: dict, uuid, token_len=None, meta: Optional[dict] = None):
        """Process video frames and append results to *outputs*.

        Parameters
        ----------
        frames : array-like
            Decoded video frames.
        outputs : dict
            Mutable accumulator for input_ids, position_ids, etc.
        uuid : str | None
            Unique identifier for cache lookup.
        token_len : int | None
            Expected token count (for validation against pre-tokenised prompts).
        meta : dict | None
            Video metadata (fps, duration, ...).  Encoding strategies that
            need metadata (e.g. Qwen) read from this dict; those that don't
            (e.g. Ernie) simply ignore it.
        """

    @abstractmethod
    def add_processed_video(self, frames_cache, outputs: dict, uuid, token_len=None):
        """Append a pre-processed (cached) video to *outputs*."""

    @abstractmethod
    def load_video(self, url, item: dict) -> Tuple[Any, dict]:
        """Decode a video from *url* and return ``(frames, meta)``.

        All implementations must return a 2-tuple so that the caller
        (``MultiModalProcessor.text2ids``) can unpack uniformly.
        """

    # ------------------------------------------------------------------
    # Text / position helpers
    # ------------------------------------------------------------------
    @abstractmethod
    def add_text_positions(self, outputs: dict, num_tokens: int):
        """Append text position IDs to *outputs*."""

    @abstractmethod
    def append_completion_tokens(self, multimodal_inputs: dict, completion_token_ids):
        """Append completion token IDs (and their positions) to *multimodal_inputs*."""

    # ------------------------------------------------------------------
    # Prompt-token-ids path
    # ------------------------------------------------------------------
    @abstractmethod
    def prompt_token_ids2outputs(self, request: dict) -> dict:
        """Build outputs dict from pre-tokenised ``prompt_token_ids``."""

    # ------------------------------------------------------------------
    # Token counting & packing
    # ------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def mm_num_tokens(grid_thw):
        """Return the number of multimodal tokens for a given grid_thw."""

    @abstractmethod
    def pack_position_ids(self, outputs: dict):
        """Convert intermediate position ID lists into final packed format."""

    # ------------------------------------------------------------------
    # Optional hooks — subclasses override only when needed
    # ------------------------------------------------------------------
    def init_extra(self, processor_kwargs: dict):
        """Model-specific extra initialisation (called once after ``__init__``)."""

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Optional[Dict[str, int]]:
        """Per-modality max token counts for the scheduler.  ``None`` = not applicable."""
        return None
