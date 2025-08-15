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

""" process.py """
from typing import Any, Dict, List, Union
import numpy as np
from PIL import Image
from paddleformers.transformers import AutoTokenizer
from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.mm_processor import IDS_TYPE_FLAG
from fastdeploy.utils import data_processor_logger

from .image_processor import ImageProcessor
from .process_video import read_video_decord, sample_frames


class DataProcessor:
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    def __init__(
        self,
        model_path: str,
        video_min_frames: int = 4,
        video_max_frames: int = 768,
        tokens_per_second: int = 2,
        **kwargs,
    ) -> None:
        self.min_frames = video_min_frames
        self.max_frames = video_max_frames

        # Tokenizer and image preprocessor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
        self.tokenizer.ignored_index = -100
        self.image_processor = ImageProcessor.from_pretrained(model_path)

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

    def text2ids(self, text, images=None, videos=None):
        """
        Convert chat text into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
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

        IMAGE_PLACEHOLDER = "<|image@placeholder|>"
        VIDEO_PLACEHOLDER = "<|video@placeholder|>"
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
                self._add_image(images[image_idx], outputs)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = videos[video_idx]
                if isinstance(item, dict):
                    frames = self._load_and_process_video(item["video"], item)
                else:
                    frames = self._load_and_process_video(item, {})

                self._add_video(frames, outputs)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return outputs

    def request2ids(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
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

        messages = parse_chat_messages(request.get("messages"))
        image_message_list = []
        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"
            content_items = msg.get("content")
            if not isinstance(content_items, list):
                content_items = [content_items]
            for item in content_items:
                if isinstance(item, dict) and item.get("type") in [
                    "image",
                    "video",
                ]:
                    image_message_list.append(item)
        request["messages"] = messages

        prompt_token_ids = self.apply_chat_template(request)
        if len(prompt_token_ids) == 0:
            raise ValueError("Invalid input: prompt_token_ids must be a non-empty sequence of token IDs")

        vision_start_index = 0
        vision_message_index = 0
        for i in range(len(prompt_token_ids)):
            if prompt_token_ids[i] == self.vision_start_id :
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
                    # -----------
                    # from fastdeploy.entrypoints.chat_utils import MultiModalPartParser
                    # mock_frames = []
                    # mm_parser = MultiModalPartParser()
                    # fimg = mm_parser.parse_image("file:///home/liudongdong/github/llm/data/images/demo.jpeg")
                    # for i in range(frames.shape[0]):
                    #     mock_frames.append(fimg.copy())
                    # mock_frames = np.stack([np.array(f.convert("RGB")) for f in mock_frames], axis=0)
                    # meta["fps"] = 3.0
                    # frames = mock_frames

                    outputs["video_cnt"] += 1
                    self._add_video(frames, meta, outputs)

                vision_message_index += 1

        self._add_text(prompt_token_ids[vision_start_index:], outputs)
        return outputs

    def _add_text(self, tokens, outputs: Dict) -> None:
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, add_special_tokens=False)["input_ids"]

        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        position_ids = self._compute_text_positions(outputs["cur_position"], len(tokens))
        outputs["position_ids"].append(position_ids)
        outputs["cur_position"] = position_ids.max() + 1

    def _compute_text_positions(self, start_pos: int, num_tokens: int) -> np.ndarray:
        text_array = np.arange(num_tokens).reshape(1, -1)
        text_index = np.broadcast_to(text_array, (3, num_tokens))
        position = text_index + start_pos
        return position

    def _add_image(self, img, outputs: Dict) -> None:
        ret = self.image_processor.preprocess(images=[img.convert("RGB")])
        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()

        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        outputs["images"].append(ret["pixel_values"])
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].append(0)

        t, h, w = grid_thw
        position_ids = self._compute_vision_positions(outputs["cur_position"], t,h,w, 0)

        outputs["position_ids"].append(position_ids)
        outputs["cur_position"] = position_ids.max() + 1

    def _add_video(self, frames, meta: Dict, outputs: Dict) -> None:
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
        position_ids = self._compute_vision_positions(outputs["cur_position"], t,h,w, second_per_grid_t)

        outputs["position_ids"].append(position_ids)
        outputs["cur_position"] = position_ids.max() + 1

    def _compute_vision_positions(self, start_pos: int, t: int, h: int, w: int, second_per_grid_t:float) -> np.ndarray:
        h //= self.spatial_conv_size
        w //= self.spatial_conv_size

        tn = np.arange(t).reshape(-1, 1)
        tn = np.broadcast_to(tn, (t, h * w))
        tn = tn * second_per_grid_t * self.tokens_per_second
        t_index = tn.flatten()

        hn = np.arange(h).reshape(1, -1, 1)
        h_index = np.broadcast_to(hn, (t, h, w)).flatten()

        wn = np.arange(w).reshape(1, 1, -1)
        w_index = np.broadcast_to(wn, (t, h, w)).flatten()

        position = np.stack([t_index, h_index, w_index]) + start_pos
        return position

    def _load_and_process_video(self, url: str, item: Dict) -> np.ndarray:
        reader, meta = read_video_decord(url)

        frames = []
        for i in range(meta["num_of_frame"]):
            frame = reader[i].asnumpy()
            image = Image.fromarray(frame, "RGB")
            frames.append(image)
        frames = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)

        fps = item.get("fps", None)
        num_frames = item.get("target_frames", None)
        if fps is not None or num_frames is not None:
            min_frames = item.get("min_frames", self.min_frames)
            max_frames = item.get("max_frames", self.max_frames)
            frames = sample_frames(video=frames,
                                   frame_factor=self.temporal_conv_size,
                                   min_frames=min_frames,
                                   max_frames=max_frames,
                                   metadata=meta,
                                   fps=fps,
                                   num_frames=num_frames)
            
            meta["num_of_frame"] = frames.shape[0]
            if fps is not None:
                meta["fps"] = fps 
            else:
                meta["fps"] = frames.shape[0] / meta["duration"]

        return frames, meta

    def apply_chat_template(self, request):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages: Either a request dict containing 'messages' field,
                                or a list of message dicts directly

        Returns:
            List of token IDs as strings (converted from token objects)
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")

        prompt_token_str = self.tokenizer.apply_chat_template(
            request["messages"],
            tokenize=False,
            add_generation_prompt=request.get("add_generation_prompt", True),
        )
        prompt_token_str = prompt_token_str.replace(self.image_token, "").replace(self.video_token, "")

        tokens = self.tokenizer.tokenize(prompt_token_str)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(
            f"req_id:{request.get('request_id', ''), } tokens: {tokens}, token_ids: {token_ids}"
        )
        return token_ids
