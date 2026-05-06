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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq

from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.utils import data_processor_logger

_DEFAULT_MM_LIMITS = {"image": 1, "video": 1, "audio": 1}


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
        # Step 1: Route to tokenization path
        outputs = self._route_tokenization(request)
        # Step 2: Append completion tokens (speculative decoding)
        self._process_post_tokens(request, outputs)
        # Step 3: Pack to numpy
        outputs = self._pack_outputs(outputs)
        # Step 4: Write back (subclass can override)
        self._write_back(request, outputs)

    # ------------------------------------------------------------------
    # Write-back hook (subclass can override)
    # ------------------------------------------------------------------

    def _write_back(self, request: dict, outputs: dict) -> None:
        """Write processing results back to request.

        Default: unconditionally overwrite prompt_token_ids.
        ErnieVLProcessor overrides to preserve original token_ids on Path A.
        """
        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["multimodal_inputs"] = outputs

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_tokenization(self, request: dict) -> dict:
        """Route to one of two tokenization paths.

        - Path A: already has prompt_token_ids -> _process_prompt_token_ids(request)
        - Path B: prompt text + multimodal_data -> _text2ids(prompt, images, videos)
        """
        if request.get("prompt_token_ids"):
            return self._process_prompt_token_ids(request)
        else:
            mm_data = request.get("multimodal_data") or {}
            images = mm_data.get("image", [])
            videos = mm_data.get("video", [])
            self._check_mm_limits({"image": images, "video": videos})
            return self._text2ids(request["prompt"], images, videos)

    # ------------------------------------------------------------------
    # Core tokenization loop (Path B)
    # ------------------------------------------------------------------

    def _text2ids(self, text, images, videos) -> dict:
        """Scan text for image/video placeholders and build outputs.

        Integrates with processor cache when enabled.
        """
        outputs = self._make_outputs()

        IMAGE_PLACEHOLDER = self.image_placeholder
        VIDEO_PLACEHOLDER = self.video_placeholder
        IMAGE_PLACEHOLDER_LEN = len(IMAGE_PLACEHOLDER)
        VIDEO_PLACEHOLDER_LEN = len(VIDEO_PLACEHOLDER)

        # Handle cache: retrieve missing items
        all_mm_items = []
        image_uuid = []
        video_uuid = []
        for img in images:
            if isinstance(img, dict):
                all_mm_items.append(img)
                image_uuid.append(img.get("uuid"))
            else:
                all_mm_items.append({"type": "image", "data": img, "uuid": None})
                image_uuid.append(None)
        for vid in videos:
            if isinstance(vid, dict):
                all_mm_items.append(vid)
                video_uuid.append(vid.get("uuid"))
            else:
                all_mm_items.append({"type": "video", "data": vid, "uuid": None})
                video_uuid.append(None)

        # Retrieve from cache if needed
        missing_hashes, missing_idx = [], []
        for idx, item in enumerate(all_mm_items):
            if not item.get("data"):
                missing_hashes.append(item.get("uuid"))
                missing_idx.append(idx)

        dealer = None
        if missing_hashes and not self.enable_processor_cache:
            raise ValueError("Missing items cannot be retrieved without processor cache.")

        if self.enable_processor_cache:
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")
            if missing_hashes:
                missing_items = self._get_cached_mm_data(dealer, missing_hashes)
                for idx_i in range(len(missing_items)):
                    if not missing_items[idx_i]:
                        raise ValueError(f"Missing item {idx_i} not found in processor cache")
                    all_mm_items[missing_idx[idx_i]]["data"] = missing_items[idx_i]

        # Rebuild images/videos lists with resolved data
        resolved_images = []
        resolved_videos = []
        for item in all_mm_items:
            if item.get("type") == "image" or (not item.get("type") and item in images):
                resolved_images.append(item.get("data", item))
            elif item.get("type") == "video":
                resolved_videos.append(item.get("data", item))

        # Use original lists if no dict items were present
        if not any(isinstance(img, dict) for img in images):
            resolved_images = images
        if not any(isinstance(vid, dict) for vid in videos):
            resolved_videos = videos

        # Scan and process
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
                image = resolved_images[image_idx]
                uuid = image_uuid[image_idx] if image_idx < len(image_uuid) else None
                if not isinstance(image, tuple):
                    self.preprocess_image(image, outputs, uuid)
                else:
                    self.preprocess_cached_image(image, outputs, uuid)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = resolved_videos[video_idx]
                uuid = video_uuid[video_idx] if video_idx < len(video_uuid) else None
                if not isinstance(item, tuple):
                    if isinstance(item, dict):
                        frames, meta = self.load_video(item.get("video", item), item)
                    else:
                        frames, meta = self.load_video(item, {})
                    self.preprocess_video(frames, outputs, uuid, meta=meta)
                else:
                    self.preprocess_cached_video(item, outputs, uuid)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        # Update cache with newly processed items
        if self.enable_processor_cache and dealer:
            self._update_mm_cache(dealer, missing_idx, all_mm_items, outputs)

        return outputs

    # ------------------------------------------------------------------
    # Path A: prompt_token_ids
    # ------------------------------------------------------------------

    def _process_prompt_token_ids(self, request) -> dict:
        """Handle pre-tokenized prompt_token_ids path."""
        prompt_token_ids = request.get("prompt_token_ids", [])

        if not request.get("messages"):
            return self.prompt_token_ids2outputs(prompt_token_ids)

        # Extract mm items from messages for cache
        mm_items = self._extract_mm_items_from_messages(request)
        outputs = self.prompt_token_ids2outputs(prompt_token_ids, mm_items)

        if self.enable_processor_cache:
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")
            missing_idx_set = set()
            for idx, item in enumerate(mm_items):
                if not item.get("data"):
                    missing_idx_set.add(idx)
            self._update_mm_cache(dealer, list(missing_idx_set), mm_items, outputs)

        return outputs

    def _extract_mm_items_from_messages(self, request) -> list:
        """Extract multimodal items from request messages."""
        from fastdeploy.entrypoints.chat_utils import parse_chat_messages

        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            role = msg.get("role")
            if role not in self.role_prefixes:
                raise ValueError(f"Unsupported role: {role}")
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content]
            for item in content:
                if isinstance(item, dict) and item.get("type") in ["image", "video"]:
                    mm_items.append(item)

        # Resolve missing via cache
        missing_hashes, missing_idx = [], []
        for idx, item in enumerate(mm_items):
            if not item.get("data"):
                missing_hashes.append(item.get("uuid"))
                missing_idx.append(idx)

        if missing_hashes:
            if not self.enable_processor_cache:
                raise ValueError("Missing items cannot be retrieved without processor cache.")
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")
            missing_items = self._get_cached_mm_data(dealer, missing_hashes)
            for idx_i in range(len(missing_items)):
                if not missing_items[idx_i]:
                    raise ValueError(f"Missing item {idx_i} not found in processor cache")
                mm_items[missing_idx[idx_i]]["data"] = missing_items[idx_i]

        return mm_items

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
    # Post-tokens and packing
    # ------------------------------------------------------------------

    def _process_post_tokens(self, request, outputs):
        """Handle completion_token_ids for speculative decoding."""
        completion_token_ids = request.get("completion_token_ids") or request.get("generated_token_ids")
        if completion_token_ids:
            self.append_completion_tokens(outputs, completion_token_ids)

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
    # Cache methods
    # ------------------------------------------------------------------

    def _get_cached_mm_data(self, socket, mm_hashes) -> list:
        """Retrieve cached multimodal data via ZMQ."""
        req = pickle.dumps(mm_hashes)
        socket.send_multipart([b"", req])
        _, resp = socket.recv_multipart()
        mm_items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {mm_hashes}")
        return mm_items

    def _update_mm_cache(self, dealer, missing_idx, mm_items, outputs):
        """Write newly-processed multimodal items to the processor cache."""
        missing_idx_set = set(missing_idx)
        hashes_to_cache, items_to_cache = [], []
        for idx in range(len(mm_items)):
            if idx in missing_idx_set:
                continue
            meta = {}
            if idx < len(outputs.get("grid_thw", [])):
                grid_thw = np.asarray(outputs["grid_thw"][idx])
                if grid_thw.ndim > 1:
                    t, h, w = grid_thw[0]
                else:
                    t, h, w = grid_thw
                meta["thw"] = (int(t), int(h), int(w))
            if "fps" in outputs and idx < len(outputs.get("fps", [])):
                meta["fps"] = outputs["fps"][idx]
            if idx < len(outputs.get("mm_hashes", [])):
                hashes_to_cache.append(outputs["mm_hashes"][idx])
                if idx < len(outputs.get("images", []) or []):
                    items_to_cache.append((outputs["images"][idx] if outputs["images"] else None, meta))
                else:
                    items_to_cache.append((None, meta))
        if hashes_to_cache:
            req = pickle.dumps((hashes_to_cache, items_to_cache))
            dealer.send_multipart([b"", req])
            data_processor_logger.info(f"Update cache of mm_hashes: {hashes_to_cache}")

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
                        raise ValueError(
                            f"Too many {modality} items in prompt, got {len(data)} but limit is {limit}"
                        )

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

    def prompt_token_ids2outputs(self, prompt_token_ids, mm_items=None) -> dict:
        """Build outputs from pre-tokenized prompt_token_ids. Override if supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support prompt_token_ids path")
