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

"""MMProcessor abstract base class for multimodal processing.

Only one public method: process(request).
Responsible for converting prompt + multimodal_data into token IDs and
pixel features, writing them back into the request dict.
"""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq

from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.multimodal.hasher import MultimodalHasher
from fastdeploy.utils import data_processor_logger

_DEFAULT_MM_LIMITS = {"image": 1, "video": 1, "audio": 1}


# ------------------------------------------------------------------
# Data classes for structured multimodal context
# ------------------------------------------------------------------


class TokenizationPath(Enum):
    """Processing path for multimodal requests."""

    PRETOKENIZED = "pretokenized"  # request already has prompt_token_ids
    FROM_TEXT = "from_text"  # prompt text + multimodal_data -> tokenize


@dataclass
class MMItem:
    """A normalized multimodal element (image or video)."""

    type: str  # "image" | "video"
    data: Any = None  # raw data (PIL Image, frames, etc.) or None if pending cache fetch
    uuid: Optional[str] = None


@dataclass
class MMContext:
    """Normalized multimodal context passed between process() steps."""

    images: List[MMItem] = field(default_factory=list)
    videos: List[MMItem] = field(default_factory=list)
    mm_order: List[str] = field(default_factory=list)  # interleaved type order: ["image", "video", ...]
    path: TokenizationPath = TokenizationPath.FROM_TEXT
    prompt_token_ids: Optional[List[int]] = None  # used by PRETOKENIZED path


# ------------------------------------------------------------------
# Cache client (centralized ZMQ connection management)
# ------------------------------------------------------------------


class _CacheClient:
    """Lazy-initialized ZMQ DEALER client for processor cache."""

    _IPC_ADDR = "ipc:///dev/shm/processor_cache.ipc"

    def __init__(self):
        self._socket = None

    @property
    def socket(self):
        if self._socket is None:
            ctx = zmq.Context()
            self._socket = ctx.socket(zmq.DEALER)
            self._socket.connect(self._IPC_ADDR)
        return self._socket

    def get(self, hashes: list) -> list:
        """Retrieve cached multimodal data by hash list."""
        req = pickle.dumps(hashes)
        self.socket.send_multipart([b"", req])
        _, resp = self.socket.recv_multipart()
        items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {hashes}")
        return items

    def put(self, hashes: list, items: list) -> None:
        """Write processed multimodal items to cache."""
        req = pickle.dumps((hashes, items))
        self.socket.send_multipart([b"", req])
        data_processor_logger.info(f"Update cache of mm_hashes: {hashes}")


# ------------------------------------------------------------------
# MMProcessor abstract base class
# ------------------------------------------------------------------


class MMProcessor(ABC):
    """Abstract base class for multimodal processors.

    Only public method: process(request) -> None
    Uses a template method pattern: base class provides the orchestration
    flow, subclasses implement hooks for model-specific logic.
    """

    # ---- Subclass must declare ----
    image_placeholder: str = ""
    video_placeholder: str = ""
    image_token_str: str = ""
    video_token_str: str = ""
    tokenizer_type: str = "auto"

    # ---- Video defaults (subclass can override) ----
    default_min_frames: int = 4
    default_max_frames: int = 768
    default_target_frames: int = -1
    default_fps: float = 2.0
    default_frames_sample: str = "leading"

    # ---- processor_kwargs type validation whitelist ----
    expected_kwargs: Dict[str, type] = {}

    def __init__(
        self,
        tokenizer,
        model_name_or_path: str,
        config=None,
        processor_kwargs: Optional[dict] = None,
        limit_mm_per_prompt: Optional[dict] = None,
        enable_processor_cache: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.enable_processor_cache = enable_processor_cache
        self._cache = _CacheClient() if enable_processor_cache else None

        kw = processor_kwargs or {}
        self.fps = kw.get("video_fps", self.default_fps)
        self.min_frames = kw.get("video_min_frames", self.default_min_frames)
        self.max_frames = kw.get("video_max_frames", self.default_max_frames)
        self.target_frames = kw.get("video_target_frames", self.default_target_frames)

        self.role_prefixes = self._init_role_prefixes()
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)

        # Subclass extra init hook
        self._init_extra(processor_kwargs)

    # ------------------------------------------------------------------
    # Public interface (only method exposed to Processor)
    # ------------------------------------------------------------------

    def process(self, request: dict) -> None:
        """Multimodal data processing (template method).

        Reads from request:
            request["prompt"] or request["prompt_token_ids"]
            request["multimodal_data"]
            request["messages"] (for prompt_token_ids path with media items)

        Writes into request:
            request["prompt_token_ids"]
            request["multimodal_inputs"]
        """
        # Step 1: Resolve and normalize multimodal data
        mm_context = self._resolve_mm_data(request)
        # Step 2: Fetch missing data from cache (if enabled)
        self._fetch_from_cache(mm_context)
        # Step 3: Core tokenization + preprocessing
        outputs = self._tokenize_and_preprocess(request, mm_context)
        # Step 4: Append completion tokens (speculative decoding)
        self._process_post_tokens(request, outputs)
        # Step 5: Compute mm_hashes and update processor cache (before packing)
        self._update_cache(mm_context, outputs)
        # Step 6: Pack to numpy
        outputs = self._pack_outputs(outputs)
        # Step 7: Write back (subclass can override)
        self._write_back(request, outputs)

    # ------------------------------------------------------------------
    # Step 1: Resolve multimodal data
    # ------------------------------------------------------------------

    def _resolve_mm_data(self, request: dict) -> MMContext:
        """Parse request and build a normalized MMContext.

        Multimodal data is read from request["multimodal_data"] (populated by
        Processor.process_messages when messages are present).
        Path is determined by whether prompt_token_ids exists.
        """
        if not request.get("prompt_token_ids") and not request.get("prompt"):
            raise ValueError("Request must contain 'prompt_token_ids', 'prompt', or 'messages'")

        mm_data = request.get("multimodal_data") or {}
        raw_images = mm_data.get("image", [])
        raw_videos = mm_data.get("video", [])
        self._check_mm_limits({"image": raw_images, "video": raw_videos})

        images = []
        for img in raw_images:
            if isinstance(img, dict):
                images.append(MMItem(type="image", data=img.get("data"), uuid=img.get("uuid")))
            else:
                images.append(MMItem(type="image", data=img, uuid=None))

        videos = []
        for vid in raw_videos:
            if isinstance(vid, dict):
                videos.append(MMItem(type="video", data=vid.get("data"), uuid=vid.get("uuid")))
            else:
                videos.append(MMItem(type="video", data=vid, uuid=None))

        # Interleaved type order: must be provided when multimodal items exist.
        mm_order = mm_data.get("mm_order")
        if not mm_order:
            if images or videos:
                raise ValueError(
                    "multimodal_data must contain 'mm_order' specifying the interleaved order of images and videos"
                )
            mm_order = []

        if request.get("prompt_token_ids"):
            return MMContext(
                images=images,
                videos=videos,
                mm_order=mm_order,
                path=TokenizationPath.PRETOKENIZED,
                prompt_token_ids=request["prompt_token_ids"],
            )

        if not request.get("prompt"):
            raise ValueError("Request must contain 'prompt_token_ids', 'prompt', or 'messages'")

        return MMContext(images=images, videos=videos, mm_order=mm_order, path=TokenizationPath.FROM_TEXT)

    # ------------------------------------------------------------------
    # Step 2: Fetch from cache
    # ------------------------------------------------------------------

    def _fetch_from_cache(self, mm_context: MMContext) -> None:
        """Retrieve missing multimodal data from processor cache."""
        missing_hashes = []
        missing_items_ref = []

        for item in mm_context.images + mm_context.videos:
            if item.data is None and item.uuid is not None:
                missing_hashes.append(item.uuid)
                missing_items_ref.append(item)

        if not missing_hashes:
            return

        if not self._cache:
            raise ValueError("Missing items cannot be retrieved without processor cache.")

        cached_data = self._cache.get(missing_hashes)
        for i, data in enumerate(cached_data):
            if not data:
                raise ValueError(f"Missing item {i} not found in processor cache")
            missing_items_ref[i].data = data

    # ------------------------------------------------------------------
    # Step 3: Tokenize and preprocess
    # ------------------------------------------------------------------

    def _tokenize_and_preprocess(self, request: dict, mm_context: MMContext) -> dict:
        """Core tokenization and preprocessing, dispatching by path."""
        if mm_context.path == TokenizationPath.PRETOKENIZED:
            return self.prompt_token_ids2outputs(mm_context)
        else:
            return self._build_outputs_from_text(request["prompt"], mm_context)

    def _build_outputs_from_text(self, text: str, mm_context: MMContext) -> dict:
        """Build outputs by scanning text for placeholders and tokenizing segments.

        All multimodal data in mm_context is already resolved (no cache logic here).
        """
        outputs = self._make_outputs()

        IMAGE_PLACEHOLDER = self.image_placeholder
        VIDEO_PLACEHOLDER = self.video_placeholder
        IMAGE_PLACEHOLDER_LEN = len(IMAGE_PLACEHOLDER)
        VIDEO_PLACEHOLDER_LEN = len(VIDEO_PLACEHOLDER)

        st, image_idx, video_idx = 0, 0, 0
        while st < len(text):
            image_pos = text.find(IMAGE_PLACEHOLDER, st)
            image_pos = len(text) if image_pos == -1 else image_pos
            video_pos = text.find(VIDEO_PLACEHOLDER, st)
            video_pos = len(text) if video_pos == -1 else video_pos
            ed = min(image_pos, video_pos)

            self._add_text(text[st:ed], outputs)
            if ed == len(text):
                break

            if ed == image_pos:
                if image_idx >= len(mm_context.images):
                    raise ValueError("prompt has more image placeholders than provided images")
                mm_item = mm_context.images[image_idx]
                if not isinstance(mm_item.data, tuple):
                    self.preprocess_image(mm_item.data, outputs, mm_item.uuid)
                else:
                    self.preprocess_cached_image(mm_item.data, outputs, mm_item.uuid)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                if video_idx >= len(mm_context.videos):
                    raise ValueError("prompt has more video placeholders than provided videos")
                mm_item = mm_context.videos[video_idx]
                if not isinstance(mm_item.data, tuple):
                    if isinstance(mm_item.data, dict):
                        frames, meta = self.load_video(mm_item.data.get("video", mm_item.data), mm_item.data)
                    else:
                        frames, meta = self.load_video(mm_item.data, {})
                    self.preprocess_video(frames, outputs, mm_item.uuid, meta=meta)
                else:
                    self.preprocess_cached_video(mm_item.data, outputs, mm_item.uuid)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return outputs

    # ------------------------------------------------------------------
    # Step 4: Post-tokens
    # ------------------------------------------------------------------

    def _process_post_tokens(self, request, outputs):
        """Handle completion_token_ids for speculative decoding."""
        completion_token_ids = request.get("completion_token_ids") or request.get("generated_token_ids")
        if completion_token_ids:
            self.append_completion_tokens(outputs, completion_token_ids)

    # ------------------------------------------------------------------
    # Step 6: Pack outputs
    # ------------------------------------------------------------------

    def _pack_outputs(self, outputs) -> dict:
        """Convert lists to numpy arrays."""
        if not outputs["images"]:
            outputs["images"] = None
            outputs["grid_thw"] = None
            outputs["image_type_ids"] = None
        else:
            outputs["images"] = np.vstack(outputs["images"])
            outputs["grid_thw"] = np.vstack(outputs["grid_thw"])
            outputs["image_type_ids"] = np.array(outputs["image_type_ids"])

        outputs["input_ids"] = np.array(outputs["input_ids"], dtype=np.int64)
        outputs["token_type_ids"] = np.array(outputs["token_type_ids"], dtype=np.int64)
        outputs["mm_num_token_func"] = self.mm_num_tokens

        # Position IDs: delegate to subclass
        self.pack_position_ids(outputs)

        return outputs

    # ------------------------------------------------------------------
    # Step 5: Compute hashes and update cache
    # ------------------------------------------------------------------

    def _update_cache(self, mm_context: MMContext, outputs: dict) -> None:
        """Compute mm_hashes for all items and optionally update processor cache.

        Hash computation is centralized here: use item.uuid if available,
        otherwise compute hash from the processed pixel_values.
        outputs["mm_hashes"] is always populated (needed by downstream engine).
        Processor cache is only updated when self._cache is enabled.

        NOTE: Must run BEFORE _pack_outputs(), because outputs["images"] is
        still a per-item list at this point (not yet vstack'd).
        """
        # Reconstruct interleaved item list using mm_order
        all_items = []
        img_idx, vid_idx = 0, 0
        for t in mm_context.mm_order:
            if t == "image":
                all_items.append(mm_context.images[img_idx])
                img_idx += 1
            else:
                all_items.append(mm_context.videos[vid_idx])
                vid_idx += 1

        hashes_to_cache, items_to_cache = [], []
        for idx, item in enumerate(all_items):
            if outputs["images"] is None or idx >= len(outputs["images"]):
                continue
            pixel_values = outputs["images"][idx]
            # Compute hash: prefer uuid, fallback to content hash
            cache_key = item.uuid if item.uuid else MultimodalHasher.hash_features(pixel_values)
            outputs["mm_hashes"].append(cache_key)

            # Only cache newly-processed items (not those fetched from cache)
            if self._cache and not isinstance(item.data, tuple):
                meta = {}
                grid_thw_list = outputs.get("grid_thw")
                if grid_thw_list is not None and idx < len(grid_thw_list):
                    grid_thw = np.asarray(outputs["grid_thw"][idx]) if outputs["grid_thw"] is not None else None
                    if grid_thw is not None:
                        if grid_thw.ndim > 1:
                            t_val, h, w = grid_thw[0]
                        else:
                            t_val, h, w = grid_thw
                        meta["thw"] = (int(t_val), int(h), int(w))
                if "fps" in outputs and idx < len(outputs.get("fps", [])):
                    meta["fps"] = outputs["fps"][idx]
                hashes_to_cache.append(cache_key)
                items_to_cache.append((pixel_values, meta))

        if hashes_to_cache:
            self._cache.put(hashes_to_cache, items_to_cache)

    # ------------------------------------------------------------------
    # Step 7: Write-back hook (subclass can override)
    # ------------------------------------------------------------------

    def _write_back(self, request: dict, outputs: dict) -> None:
        """Write processing results back to request.

        Default: unconditionally overwrite prompt_token_ids.
        Subclasses can override to customize write-back behavior.
        """
        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["multimodal_inputs"] = outputs

    # ------------------------------------------------------------------
    # Text tokenization helper
    # ------------------------------------------------------------------

    def _add_text(self, tokens, outputs):
        """Tokenize text and add to outputs."""
        if not tokens:
            return
        if isinstance(tokens, str):
            tokens_str = self.tokenizer.tokenize(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)
        num_tokens = len(tokens)
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)
        self.add_text_positions(outputs, num_tokens)

    # ------------------------------------------------------------------
    # Outputs accumulator
    # ------------------------------------------------------------------

    def _make_outputs(self) -> dict:
        """Create the mutable accumulator dict. Subclass can override to add fields."""
        return {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
        }

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_role_prefixes(self) -> dict:
        """Set up role prefixes for message parsing. Subclass can override."""
        return {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
        }

    def _parse_limits(self, limits: Optional[dict]) -> dict:
        if not limits:
            return dict(_DEFAULT_MM_LIMITS)
        try:
            if not isinstance(limits, dict):
                raise ValueError("limit-mm-per-prompt must be a dictionary")
            data_processor_logger.info(f"_parse_limits:{limits}")
            return {**_DEFAULT_MM_LIMITS, **limits}
        except Exception as e:
            data_processor_logger.warning(f"Invalid limit-mm-per-prompt format: {e}, using default limits")
            return dict(_DEFAULT_MM_LIMITS)

    def _check_mm_limits(self, mm_data):
        """Validate that request does not exceed per-modality limits."""
        if isinstance(mm_data, dict):
            for modality, data in mm_data.items():
                if modality in self.limit_mm_per_prompt and data:
                    limit = self.limit_mm_per_prompt[modality]
                    if len(data) > limit:
                        raise ValueError(f"Too many {modality} items in prompt, got {len(data)} but limit is {limit}")

    def _init_extra(self, processor_kwargs):
        """Model-specific extra initialization. Override in subclass."""
        pass

    # ------------------------------------------------------------------
    # Public helpers (called by Processor)
    # ------------------------------------------------------------------

    def get_mm_max_tokens_per_item(self, seq_len: int) -> Optional[Mapping[str, int]]:
        """Per-modality max token counts for the scheduler. None = not applicable."""
        return None

    def append_completion_tokens(self, multimodal_inputs: dict, completion_token_ids):
        """Append completion tokens. Must be implemented by subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Abstract methods (subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess_image(self, img, outputs: dict, uuid, token_len=None):
        """Process a raw image and append results to outputs."""

    @abstractmethod
    def preprocess_cached_image(self, img_cache, outputs: dict, uuid, token_len=None):
        """Append a pre-processed (cached) image to outputs."""

    @abstractmethod
    def preprocess_video(self, frames, outputs: dict, uuid, token_len=None, meta=None):
        """Process video frames and append results to outputs."""

    @abstractmethod
    def preprocess_cached_video(self, frames_cache, outputs: dict, uuid, token_len=None):
        """Append a pre-processed (cached) video to outputs."""

    @abstractmethod
    def load_video(self, url, item: dict) -> Tuple[Any, dict]:
        """Decode a video and return (frames, meta)."""

    @abstractmethod
    def add_text_positions(self, outputs: dict, num_tokens: int):
        """Append text position IDs to outputs."""

    @abstractmethod
    def pack_position_ids(self, outputs: dict):
        """Convert intermediate position ID lists into final packed format."""

    @staticmethod
    @abstractmethod
    def mm_num_tokens(grid_thw) -> int:
        """Calculate number of multimodal tokens for given grid_thw."""

    def prompt_token_ids2outputs(self, mm_context: "MMContext") -> dict:
        """Build outputs from pre-tokenized prompt_token_ids. Override if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support prompt_token_ids path")
