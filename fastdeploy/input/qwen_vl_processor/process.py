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

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from paddleformers.transformers import AutoTokenizer

from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.utils import data_processor_logger

from .image_processor import ImageProcessor
from .process_video import read_frames, sample_frames


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
        video_min_frames: int = 4,
        video_max_frames: int = 768,
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

        # Initialize tokenizer with left padding and fast tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
            self.tokenizer.ignored_index = -100  # Set ignored index for loss calculation
        else:
            self.tokenizer = tokenizer
        self.image_processor = ImageProcessor.from_pretrained(model_path)  # Initialize image processor

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = self.image_processor.merge_size
        self.temporal_conv_size = self.image_processor.temporal_patch_size

        # Special tokens and IDs
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)

        self.vision_start = "<|vision_start|>"
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids(self.vision_start)

        self.tokens_per_second = tokens_per_second

        self.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
        }

    def _pack_outputs(self, outputs):
        """
        Pack and convert all output data into numpy arrays with appropriate types.

        Args:
            outputs (dict): Dictionary containing model outputs with keys:
                - images: List of visual features
                - grid_thw: List of spatial dimensions
                - image_type_ids: List of content type indicators
                - input_ids: List of token IDs
                - token_type_ids: List of type identifiers
                - position_ids: List of position embeddings

        Returns:
            dict: Processed outputs with all values converted to numpy arrays
        """
        # Process visual outputs - stack if exists or set to None if empty
        if not outputs["images"]:
            outputs["images"] = None  # No images case
            outputs["grid_thw"] = None  # No spatial dimensions
            outputs["image_type_ids"] = None  # No type IDs
        else:
            outputs["images"] = np.vstack(outputs["images"])  # Stack image features vertically
            outputs["grid_thw"] = np.vstack(outputs["grid_thw"])  # Stack spatial dimensions
            outputs["image_type_ids"] = np.array(outputs["image_type_ids"])  # Convert to numpy array

        # Convert all outputs to numpy arrays with appropriate types
        outputs["input_ids"] = np.array(outputs["input_ids"], dtype=np.int64)  # Token IDs as int64
        outputs["token_type_ids"] = np.array(outputs["token_type_ids"], dtype=np.int64)  # Type IDs as int64
        outputs["position_ids"] = np.concatenate(
            outputs["position_ids"], axis=1, dtype=np.int64
        )  # Concatenate position IDs
        return outputs

    def text2ids(self, text, images=None, videos=None):
        """
        Convert text with image/video placeholders into model inputs.

        Args:
            text: Input text with <|image@placeholder|> and <|video@placeholder|> markers
            images: List of PIL Images corresponding to image placeholders
            videos: List of video data corresponding to video placeholders

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
            "pic_cnt": 0,
            "video_cnt": 0,
        }

        # Define placeholders and their lengths
        IMAGE_PLACEHOLDER = "<|image_pad|>"
        VIDEO_PLACEHOLDER = "<|video_pad|>"
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
                outputs["pic_cnt"] += 1
                self._add_image(images[image_idx], outputs)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = videos[video_idx]
                if isinstance(item, dict):
                    frames, meta = self._load_and_process_video(item["video"], item)
                else:
                    frames, meta = self._load_and_process_video(item, {})

                outputs["video_cnt"] += 1
                self._add_video(frames, meta, outputs)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return self._pack_outputs(outputs)

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

        outputs = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "pic_cnt": 0,
            "video_cnt": 0,
        }

        # Parse and validate chat messages
        messages = parse_chat_messages(request.get("messages"))
        image_message_list = []  # Store visual content messages

        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"

            # Normalize content to list format
            content_items = msg.get("content")
            if not isinstance(content_items, list):
                content_items = [content_items]

            # Collect all visual content items
            for item in content_items:
                if isinstance(item, dict) and item.get("type") in ["image", "video"]:
                    image_message_list.append(item)

        raw_messages = request["messages"]
        request["messages"] = messages

        prompt_token_ids = self.apply_chat_template(request)
        if len(prompt_token_ids) == 0:
            raise ValueError("Invalid input: prompt_token_ids must be a non-empty sequence of token IDs")
        request["messages"] = raw_messages

        vision_start_index = 0
        vision_message_index = 0
        for i in range(len(prompt_token_ids)):
            if prompt_token_ids[i] == self.vision_start_id:
                self._add_text(prompt_token_ids[vision_start_index : i + 1], outputs)

                vision_start_index = i + 1
                image_message = image_message_list[vision_message_index]

                if image_message["type"] == "image":
                    img = image_message.get("image")
                    if img is None:
                        continue
                    outputs["pic_cnt"] += 1
                    self._add_image(img, outputs)

                elif image_message["type"] == "video":
                    video_bytes = image_message.get("video")
                    if video_bytes is None:
                        continue
                    frames, meta = self._load_and_process_video(video_bytes, image_message)

                    outputs["video_cnt"] += 1
                    self._add_video(frames, meta, outputs)

                vision_message_index += 1

        self._add_text(prompt_token_ids[vision_start_index:], outputs)
        return self._pack_outputs(outputs)

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

        position_ids = self._compute_text_positions(outputs["cur_position"], num_tokens)
        outputs["position_ids"].append(position_ids)
        outputs["cur_position"] = position_ids.max() + 1

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

    def _add_image(self, img, outputs: Dict) -> None:
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

        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].append(0)

        t, h, w = grid_thw
        position_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, 0)

        outputs["position_ids"].append(position_ids)
        outputs["cur_position"] = position_ids.max() + 1

    def _add_video(self, frames, meta: Dict, outputs: Dict) -> None:
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

        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()

        outputs["input_ids"].extend([self.video_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)

        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].extend([1] * grid_thw[0])

        fps = meta["fps"]
        second_per_grid_t = self.temporal_conv_size / fps
        t, h, w = grid_thw
        position_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)

        outputs["position_ids"].append(position_ids)
        outputs["cur_position"] = position_ids.max() + 1

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
        frames, meta = read_frames(url)

        # Apply frame sampling if fps or target_frames specified
        fps = item.get("fps", None)
        num_frames = item.get("target_frames", None)

        if fps is not None or num_frames is not None:
            # Get frame sampling constraints
            min_frames = item.get("min_frames", self.min_frames)
            max_frames = item.get("max_frames", self.max_frames)

            # Sample frames according to specifications
            frames = sample_frames(
                video=frames,
                frame_factor=self.temporal_conv_size,  # Ensure divisible by temporal patch size
                min_frames=min_frames,
                max_frames=max_frames,
                metadata=meta,
                fps=fps,
                num_frames=num_frames,
            )

            # Update metadata with new frame count and fps
            meta["num_of_frame"] = frames.shape[0]
            if fps is not None:
                meta["fps"] = fps  # Use specified fps
                meta["duration"] = frames.shape[0] / fps
            else:
                meta["fps"] = frames.shape[0] / meta["duration"]  # Calculate fps from sampled frames

        return frames, meta

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
        )
        prompt_token_str = raw_prompt.replace(self.image_token, "").replace(self.video_token, "")
        request["prompt_tokens"] = raw_prompt

        tokens = self.tokenizer.tokenize(prompt_token_str)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(
            f"req_id:{request.get('request_id', ''), } prompt: {raw_prompt} tokens: {tokens}, token_ids: {token_ids}"
        )
        return token_ids
