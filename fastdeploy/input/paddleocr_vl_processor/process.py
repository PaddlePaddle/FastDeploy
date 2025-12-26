"""
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
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq
from paddleformers.transformers import AutoTokenizer
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.ernie4_5_vl_processor import read_video_decord
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.multimodal.hasher import MultimodalHasher
from fastdeploy.utils import data_processor_logger

from .image_processor import ImageProcessor
from .process_video import sample_frames


class DataProcessor:
    """
    Processes multimodal inputs (text, images, videos) into model-ready formats.

    Handles:
    - Tokenization of text with special tokens for visual content
    - Image and video preprocessing
    - Generation of 3D positional embeddings
    - Conversion of chat messages to model inputs

    Attributes:
        tokenizer: Text tokenizer instance
        image_processor: Image/video preprocessor
        image_token: Special token for image placeholders
        video_token: Special token for video placeholders
        vision_start: Token marking start of visual content
    """

    def __init__(
        self,
        model_path: str,
        enable_processor_cache: bool = False,
        video_min_frames: int = 4,
        video_max_frames: int = 768,
        video_target_frames: int = -1,
        video_fps: int = -1,
        tokens_per_second: int = 2,
        tokenizer=None,
        **kwargs,
    ) -> None:
        """
        Initialize the data processor.

        Args:
            model_path: Path to pretrained model
            video_min_frames: Minimum frames to sample from videos
            video_max_frames: Maximum frames to sample from videos
            tokens_per_second: Temporal resolution for positional embeddings
            **kwargs: Additional configuration
        """
        self.min_frames = video_min_frames
        self.max_frames = video_max_frames
        self.target_frames = video_target_frames
        self.fps = video_fps

        # Initialize tokenizer with left padding and fast tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
            self.tokenizer.ignored_index = -100  # Set ignored index for loss calculation
        else:
            self.tokenizer = tokenizer
        self.image_processor = ImageProcessor.from_pretrained(model_path)  # Initialize image processor
        self.enable_processor_cache = enable_processor_cache

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = self.image_processor.merge_size
        self.temporal_conv_size = self.image_processor.temporal_patch_size

        # Special tokens and IDs
        self.image_token = "<|IMAGE_PLACEHOLDER|>"
        self.video_token = "<|video_pad|>"

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
        self.image_patch_id = self.image_token_id

        self.vision_start = "<|IMAGE_START|>"
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids(self.vision_start)

        self.tokens_per_second = tokens_per_second

        self.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
        }

    def text2ids(self, text, images=None, videos=None, image_uuid=None, video_uuid=None):
        """
        Convert text with image/video placeholders into model inputs.

        Args:
            text: Input text with <|image@placeholder|> and <|video@placeholder|> markers
            images: List of PIL Images corresponding to image placeholders
            videos: List of video data corresponding to video placeholders
            image_uuid: List of unique identifiers for each image, used for caching or hashing.
            video_uuid: List of unique identifiers for each video, used for caching or hashing.

        Returns:
            Dict containing:
                - input_ids: Token IDs
                - token_type_ids: Type identifiers (text/image/video)
                - position_ids: 3D positional embeddings
                - images: Preprocessed visual features
                - grid_thw: Spatial/temporal dimensions
                - image_type_ids: Visual content type (0=image, 1=video)
        """

        outputs = {
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
            "fps": [],
            "mm_positions": [],
            "mm_hashes": [],
            "vit_seqlen": [],
            "vit_position_ids": [],
        }

        # Define placeholders and their lengths
        IMAGE_PLACEHOLDER = self.image_token
        VIDEO_PLACEHOLDER = self.video_token
        IMAGE_PLACEHOLDER_LEN = len(IMAGE_PLACEHOLDER)
        VIDEO_PLACEHOLDER_LEN = len(VIDEO_PLACEHOLDER)

        # Initialize tracking variables for text parsing
        st, image_idx, video_idx = 0, 0, 0  # Start position, image counter, video counter
        while st < len(text):
            # Find next image or video placeholder in text
            image_pos = text.find(IMAGE_PLACEHOLDER, st)
            image_pos = len(text) if image_pos == -1 else image_pos  # Set to end if not found
            video_pos = text.find(VIDEO_PLACEHOLDER, st)
            video_pos = len(text) if video_pos == -1 else video_pos  # Set to end if not found
            ed = min(image_pos, video_pos)  # End position is first placeholder found

            self._add_text(text[st:ed], outputs)
            if ed == len(text):
                break

            if ed == image_pos:
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                if not isinstance(image, tuple):
                    self._add_image(image, outputs, uuid)
                else:
                    self._add_processed_image(image, outputs, uuid)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                if not isinstance(item, tuple):
                    if isinstance(item, dict):
                        frames, meta = self._load_and_process_video(item["video"], item)
                    else:
                        frames, meta = self._load_and_process_video(item, {})
                    self._add_video(frames, meta, outputs, uuid)
                else:
                    # cached frames are already processed
                    self._add_processed_video(item, outputs, uuid)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return outputs

    def request2ids(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """
        Convert chat request with multimodal messages into model inputs.

        Args:
            request: Dictionary containing:
                - messages: List of chat messages with text/image/video content
                - request_id: Unique identifier for logging
            tgts: Optional target sequences

        Returns:
            Dict with same structure as text2ids() output
        """

        # Parse and validate chat messages
        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"

            # Normalize content to list format
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content]
            # Collect all visual content items
            for item in content:
                if item.get("type") in ["image", "video"]:
                    mm_items.append(item)

        missing_hashes, missing_idx = [], []
        for idx, item in enumerate(mm_items):
            if not item.get("data"):
                # raw data not provided, should be retrieved from processor cache
                missing_hashes.append(item.get("uuid"))
                missing_idx.append(idx)

        if len(missing_hashes) > 0 and not self.enable_processor_cache:
            raise ValueError("Missing items cannot be retrieved without processor cache.")

        if self.enable_processor_cache:
            context = zmq.Context()
            dealer = context.socket(zmq.DEALER)
            dealer.connect("ipc:///dev/shm/processor_cache.ipc")

            missing_items = self.get_processor_cache(dealer, missing_hashes)
            for idx in range(len(missing_items)):
                if not missing_items[idx]:
                    raise ValueError(f"Missing item {idx} not found in processor cache")
                mm_items[missing_idx[idx]]["data"] = missing_items[idx]

        images, videos = [], []
        image_uuid, video_uuid = [], []
        for item in mm_items:
            if item.get("type") == "image":
                images.append(item["data"])
                image_uuid.append(item["uuid"])
            elif item.get("type") == "video":
                videos.append(item["data"])
                video_uuid.append(item["uuid"])
            else:
                raise ValueError(f"Unsupported multimodal type: {item.get('type')}")

        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat template.")

        chat_template_kwargs = request.get("chat_template_kwargs", {})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=request.get("add_generation_prompt", True),
            **chat_template_kwargs,
        )
        request["prompt_tokens"] = prompt

        outputs = self.text2ids(prompt, images, videos, image_uuid, video_uuid)

        if self.enable_processor_cache:
            missing_idx = set(missing_idx)
            hashes_to_cache, items_to_cache = [], []
            for idx in range(len(mm_items)):
                if idx in missing_idx:
                    continue
                meta = {}
                t, h, w = outputs["grid_thw"][idx]
                meta["thw"] = (t, h, w)
                meta["fps"] = outputs["fps"][idx]
                hashes_to_cache.append(outputs["mm_hashes"][idx])
                items_to_cache.append((outputs["images"][idx], meta))
            self.update_processor_cache(dealer, hashes_to_cache, items_to_cache)

        return outputs

    def _add_text(self, tokens, outputs: Dict) -> None:
        """
        Add text tokens to model inputs dictionary.

        Args:
            tokens: Text string or already tokenized IDs
            outputs: Dictionary accumulating model inputs

        Note:
            - Handles both raw text and pre-tokenized inputs
            - Updates position IDs for 3D embeddings
        """
        if not tokens:
            return None

        if isinstance(tokens, str):
            tokens_str = self.tokenizer.tokenize(tokens)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens_str)

        num_tokens = len(tokens)
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        pos_ids = self._compute_text_positions(outputs["cur_position"], num_tokens)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

    def _compute_text_positions(self, start_pos: int, num_tokens: int) -> np.ndarray:
        """
        Generate 3D positional embeddings for text tokens.

        Args:
            start_pos: Starting position index
            num_tokens: Number of tokens to generate positions for

        Returns:
            numpy.ndarray: 3D position IDs shaped (3, num_tokens)
        """
        text_array = np.arange(num_tokens).reshape(1, -1)
        text_index = np.broadcast_to(text_array, (3, num_tokens))
        position = text_index + start_pos
        return position

    def _add_image(self, img, outputs: Dict, uuid: Optional[str]) -> None:
        """
        Add image data to model inputs dictionary.

        Args:
            img: PIL Image to process
            outputs: Dictionary accumulating model inputs

        Note:
            - Preprocesses image and calculates spatial dimensions
            - Adds image token IDs and type markers
            - Generates appropriate position embeddings
        """
        ret = self.image_processor.preprocess(images=[img.convert("RGB")])
        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += int(num_tokens)

        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].append(0)

        # position_ids
        t, h, w = grid_thw
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1
        outputs["fps"].append(0)
        numel = h * w
        outputs["vit_seqlen"].append(numel)
        outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def _add_processed_image(self, img_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str) -> None:
        img, meta = img_cache
        num_tokens = img.shape[0] // self.image_processor.merge_size**2

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        _, h, w = meta["thw"]
        pos_ids = self._compute_vision_positions(outputs["cur_position"], 1, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

        outputs["fps"].append(0)

    def _add_video(self, frames, meta: Dict, outputs: Dict, uuid: Optional[str]) -> None:
        """
        Add video data to model inputs dictionary.

        Args:
            frames: Video frames as numpy array
            meta: Video metadata containing fps/duration
            outputs: Dictionary accumulating model inputs

        Note:
            - Handles temporal dimension in position embeddings
            - Uses video-specific token IDs and type markers
        """
        ret = self.image_processor.preprocess(images=frames)

        num_tokens = ret["image_grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["image_grid_thw"].tolist()

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.video_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += int(num_tokens)

        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].extend([1] * grid_thw[0])

        fps = meta["fps"]
        second_per_grid_t = self.temporal_conv_size / fps
        t, h, w = grid_thw
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)

        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1
        outputs["fps"].append(fps)
        numel = h * w
        outputs["vit_seqlen"].append(numel)
        outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def _add_processed_video(self, frames_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str) -> None:
        frames, meta = frames_cache
        num_tokens = frames.shape[0] // self.image_processor.merge_size**2

        t, h, w = meta["thw"]
        outputs["images"].append(frames)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[t, h, w]]))

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["image_type_ids"].extend([1] * t)

        fps = meta["fps"]
        second_per_grid_t = self.temporal_conv_size / fps
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

    def _compute_vision_positions(
        self, start_pos: int, t: int, h: int, w: int, second_per_grid_t: float
    ) -> np.ndarray:
        """
        Generate 3D position IDs for visual inputs.

        Args:
            start_pos: Base position in sequence
            t: Temporal patches (1 for images)
            h: Height in patches
            w: Width in patches
            second_per_grid_t: Time per temporal patch

        Returns:
            np.ndarray: Position IDs for [t,h,w] dimensions
        """
        h //= self.spatial_conv_size
        w //= self.spatial_conv_size

        tn = np.arange(t).reshape(-1, 1)
        tn = np.broadcast_to(tn, (t, h * w))
        tn = tn * int(second_per_grid_t) * self.tokens_per_second
        t_index = tn.flatten()

        hn = np.arange(h).reshape(1, -1, 1)
        h_index = np.broadcast_to(hn, (t, h, w)).flatten()

        wn = np.arange(w).reshape(1, 1, -1)
        w_index = np.broadcast_to(wn, (t, h, w)).flatten()

        position = np.stack([t_index, h_index, w_index]) + start_pos
        return position

    def _load_and_process_video(self, url: str, item: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Load and preprocess video into frames.

        Args:
            url: Video file path or bytes
            item: Dictionary containing processing parameters

        Returns:
            tuple: (frames, metadata) where:
                - frames: Processed video frames as numpy array
                - metadata: Updated video metadata dictionary
        """
        reader, meta, _ = read_video_decord(url, save_to_disk=False)

        # Apply frame sampling if fps or target_frames specified
        fps = item.get("fps", self.fps)
        num_frames = item.get("target_frames", self.target_frames)

        frame_indices = list(range(meta["num_of_frame"]))
        if fps > 0 or num_frames > 0:
            # Get frame sampling constraints
            min_frames = item.get("min_frames", self.min_frames)
            max_frames = item.get("max_frames", self.max_frames)

            # Sample frames according to specifications
            frame_indices = sample_frames(
                frame_factor=self.temporal_conv_size,  # Ensure divisible by temporal patch size
                min_frames=min_frames,
                max_frames=max_frames,
                metadata=meta,
                fps=fps,
                num_frames=num_frames,
            )

            # Update metadata with new frame count and fps
            meta["num_of_frame"] = len(frame_indices)
            if fps is not None:
                meta["fps"] = fps  # Use specified fps
                meta["duration"] = len(frame_indices) / fps
            else:
                meta["fps"] = len(frame_indices) / meta["duration"]  # Calculate fps from sampled frames

        frames = []
        for idx in frame_indices:
            frame = reader[idx].asnumpy()
            image = Image.fromarray(frame, "RGB")
            frames.append(image)
        frames = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)

        return frames, meta

    def get_processor_cache(self, socket, mm_hashes: list[str]) -> list:
        """
        get cache correspond to given hash values
        """
        req = pickle.dumps(mm_hashes)
        socket.send_multipart([b"", req])
        _, resp = socket.recv_multipart()
        mm_items = pickle.loads(resp)
        data_processor_logger.info(f"Get cache of mm_hashes: {mm_hashes}")

        return mm_items

    def update_processor_cache(self, socket, mm_hashes: list[str], mm_items):
        """
        update cache data
        """
        req = pickle.dumps((mm_hashes, mm_items))
        socket.send_multipart([b"", req])
        data_processor_logger.info(f"Update cache of mm_hashes: {mm_hashes}")

    def apply_chat_template(self, request):
        """
        Apply chat template to convert messages into token sequence.

        Args:
            request: Dictionary containing chat messages

        Returns:
            List of token IDs

        Raises:
            ValueError: If model doesn't support chat templates
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")

        raw_prompt = self.tokenizer.apply_chat_template(
            request["messages"],
            tokenize=False,
            add_generation_prompt=request.get("add_generation_prompt", True),
            chat_template=request.get("chat_template", None),
        )
        prompt_token_str = raw_prompt.replace(self.image_token, "").replace(self.video_token, "")
        request["text_after_process"] = raw_prompt

        tokens = self.tokenizer.tokenize(prompt_token_str)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(
            f"req_id:{request.get('request_id', ''), } prompt: {raw_prompt} tokens: {tokens}, token_ids: {token_ids}"
        )
        return token_ids
