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
import copy
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zmq
from paddleformers.transformers.image_utils import ChannelDimension
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.multimodal.hasher import MultimodalHasher
from fastdeploy.utils import data_processor_logger

from .image_preprocessor.image_preprocessor_adaptive import AdaptiveImageProcessor
from .process_video import read_frames_decord, read_video_decord
from .utils.render_timestamp import render_frame_timestamp


def fancy_print(input_ids, tokenizer, image_patch_id=None):
    """
    input_ids: input_ids
    tokenizer: the tokenizer of models
    """
    i = 0
    res = ""
    text_ids = []
    real_image_token_len = 0
    while i < len(input_ids):
        if input_ids[i] == image_patch_id:
            if len(text_ids) > 0:
                res += tokenizer.decode(text_ids)
                text_ids = []

            real_image_token_len += 1
        else:
            if real_image_token_len != 0:
                res += f"<|IMAGE@{real_image_token_len}|>"
                real_image_token_len = 0

            text_ids.append(input_ids[i])

        i += 1
    if len(text_ids) > 0:

        res += tokenizer.decode(text_ids)
        text_ids = []
    return res


class DataProcessor:
    """
    Processes multimodal chat messages into model-ready inputs,
    handling text, images, and videos with 3D positional embeddings.
    """

    CLS_TOKEN = "<|begin_of_sentence|>"
    SEP_TOKEN = "<|end_of_sentence|>"
    EOS_TOKEN = "</s>"
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def __init__(
        self,
        tokenizer_name: str,
        image_preprocessor_name: str,
        enable_processor_cache: bool = False,
        spatial_conv_size: int = 2,
        temporal_conv_size: int = 2,
        image_min_pixels: int = 4 * 28 * 28,
        image_max_pixels: int = 6177 * 28 * 28,
        video_min_pixels: int = 299 * 28 * 28,
        video_max_pixels: int = 1196 * 28 * 28,
        video_target_frames: int = -1,
        video_frames_sample: str = "leading",
        video_max_frames: int = 180,
        video_min_frames: int = 16,
        video_fps: int = 2,
        **kwargs,
    ) -> None:
        # Tokenizer and image preprocessor
        self.model_name_or_path = tokenizer_name
        self._load_tokenizer()
        self.tokenizer.ignored_index = -100
        self.image_preprocessor = AdaptiveImageProcessor.from_pretrained(image_preprocessor_name)
        self.enable_processor_cache = enable_processor_cache

        # Convolution sizes for patch aggregation
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size

        # Pixel constraints
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

        # Video sampling parameters
        self.target_frames = video_target_frames
        self.frames_sample = video_frames_sample
        self.max_frames = video_max_frames
        self.min_frames = video_min_frames
        self.fps = video_fps

        # Special tokens and IDs
        self.cls_token = self.CLS_TOKEN
        self.sep_token = self.SEP_TOKEN
        self.eos_token = self.EOS_TOKEN
        self.image_start = self.IMG_START
        self.image_end = self.IMG_END
        self.video_start = self.VID_START
        self.video_end = self.VID_END
        self.image_patch_id = self.tokenizer.convert_tokens_to_ids("<|IMAGE_PLACEHOLDER|>")
        self.image_start_id = self.tokenizer.convert_tokens_to_ids(self.image_start)
        self.image_end_id = self.tokenizer.convert_tokens_to_ids(self.image_end)
        self.video_start_id = self.tokenizer.convert_tokens_to_ids(self.video_start)
        self.video_end_id = self.tokenizer.convert_tokens_to_ids(self.video_end)
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)

        self.token_type_mapping = self._build_token_type_mapping()
        self.is_training = True
        self.role_prefixes = {
            "system": "",
            "user": "User: ",
            "bot": "Assistant: ",
            "assistant": "Assistant: ",
            "tool": "Tool: ",
        }

    def _build_token_type_mapping(self) -> Dict[Any, int]:
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (
            self.IMG_START,
            self.IMG_END,
            self.VID_START,
            self.VID_END,
        ):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_patch_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def train(self) -> None:
        """Enable training mode (produces labels)."""
        self.is_training = True

    def eval(self) -> None:
        """Enable evaluation mode (doesn't produce labels)."""
        self.is_training = False

    def text2ids(self, text, images=None, videos=None, image_uuid=None, video_uuid=None):
        """
        Convert chat text into model inputs.

        Args:
            text (str): The chat text containing placeholders for images and videos.
            images (list, optional): List of images to be processed and inserted at image placeholders.
            videos (list, optional): List of videos to be processed and inserted at video placeholders.
            image_uuid (list, optional): List of unique identifiers for each image, used for caching or hashing.
            video_uuid (list, optional): List of unique identifiers for each video, used for caching or hashing.
        Returns:
            dict: A dictionary with keys input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels, etc.
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
            "mm_positions": [],
            "mm_hashes": [],
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
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                if not isinstance(image, tuple):
                    self._add_image(image, outputs, uuid)
                else:
                    # cached images are already processed
                    self._add_processed_image(image, outputs, uuid)
                image_idx += 1
                st = ed + IMAGE_PLACEHOLDER_LEN
            else:
                item = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                if not isinstance(item, tuple):
                    if isinstance(item, dict):
                        frames = self._load_and_process_video(item["video"], item)
                    else:
                        frames = self._load_and_process_video(item, {})
                    self._add_video(frames, outputs, uuid)
                else:
                    # cached frames are already processed
                    self._add_processed_video(item, outputs, uuid)
                video_idx += 1
                st = ed + VIDEO_PLACEHOLDER_LEN

        return outputs

    def extract_mm_items(self, request: Dict[str, Any]):
        messages = parse_chat_messages(request.get("messages"))
        mm_items = []
        for msg in messages:
            role = msg.get("role")
            assert role in self.role_prefixes, f"Unsupported role: {role}"
            content = msg.get("content")
            if not isinstance(content, list):
                content = [content]
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

        dealer = None
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
        return images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items

    def request2ids(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
        """
        Convert chat messages into model inputs.
        Returns a dict with input_ids, token_type_ids, position_ids, images, grid_thw, image_type_ids, labels.
        """
        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self.extract_mm_items(request)

        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat template.")

        chat_template_kwargs = request.get("chat_template_kwargs", {})
        prompt = self.tokenizer.apply_chat_template(
            request,
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
                t, h, w = outputs["grid_thw"][idx][0]
                meta["thw"] = (t, h, w)
                hashes_to_cache.append(outputs["mm_hashes"][idx])
                items_to_cache.append((outputs["images"][idx], meta))
            self.update_processor_cache(dealer, hashes_to_cache, items_to_cache)

        if self.is_training:
            assert tgts, "Training must give tgt"
            self._extract_labels(outputs, tgts)

        return outputs

    def prompt_token_ids2outputs(
        self, request: Dict[str, Any], tgts: List[str] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], None]]:
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
            "mm_positions": [],
            "mm_hashes": [],
        }
        prompt_token_ids = request.get("prompt_token_ids", [])
        prompt_token_ids_len = len(prompt_token_ids)
        if not request.get("messages"):
            outputs["input_ids"].extend(prompt_token_ids)
            outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * prompt_token_ids_len)
            for i in range(prompt_token_ids_len):
                outputs["position_ids"].append([i] * 3)
            outputs["cur_position"] += prompt_token_ids_len
            return outputs
        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self.extract_mm_items(request)
        st, image_idx, video_idx = 0, 0, 0
        while st < prompt_token_ids_len:
            cur_token_id = prompt_token_ids[st]
            if cur_token_id == self.image_start_id:
                if image_idx >= len(images):
                    raise ValueError("prompt token ids has more image placeholder than in messages")
                # append image_start_id
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
                # process placeholder token ids
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != self.image_end_id:
                    cur_idx += 1
                if cur_idx >= prompt_token_ids_len:
                    raise ValueError("image token ids not complete")
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                token_len = cur_idx - st
                if not isinstance(image, tuple):
                    self._add_image(image, outputs, uuid, token_len)
                else:
                    self._add_processed_image(image, outputs, uuid, token_len)
                image_idx += 1
                st = cur_idx
            elif cur_token_id == self.video_start_id:
                if video_idx >= len(videos):
                    raise ValueError("prompt token ids has more video placeholder than in messages")
                # append video_start_id
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
                # process placeholder token ids
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != self.video_end_id:
                    cur_idx += 1
                if cur_idx >= prompt_token_ids_len:
                    raise ValueError("video token ids not complete")
                video = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                token_len = cur_idx - st
                if not isinstance(video, tuple):
                    if isinstance(video, dict):
                        frames = self._load_and_process_video(video["video"], video)
                    else:
                        frames = self._load_and_process_video(video, {})
                    self._add_video(frames, outputs, uuid, token_len)
                else:
                    self._add_processed_video(video, outputs, uuid, token_len)
                video_idx += 1
                st = cur_idx
            else:
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
        if image_idx != len(images):
            raise ValueError("number of images does not match")
        if video_idx != len(videos):
            raise ValueError("number of videos does not match")

        if self.enable_processor_cache:
            missing_idx = set(missing_idx)
            hashes_to_cache, items_to_cache = [], []
            for idx in range(len(mm_items)):
                if idx in missing_idx:
                    continue
                meta = {}
                t, h, w = outputs["grid_thw"][idx][0]
                meta["thw"] = (t, h, w)
                hashes_to_cache.append(outputs["mm_hashes"][idx])
                items_to_cache.append((outputs["images"][idx], meta))
            self.update_processor_cache(dealer, hashes_to_cache, items_to_cache)

        return outputs

    def _add_special_token(self, token: Union[str, int], outputs: Dict) -> None:
        token_id = token if isinstance(token, int) else self.tokenizer.convert_tokens_to_ids(token)
        outputs["input_ids"].append(token_id)
        outputs["token_type_ids"].append(self.token_type_mapping[token])
        pos = outputs["cur_position"]
        outputs["position_ids"].append([pos] * 3)
        outputs["cur_position"] += 1

    def _add_text(self, tokens, outputs: Dict) -> None:
        if isinstance(tokens, str):
            tokens = self.tokenizer.encode(tokens, add_special_tokens=False)["input_ids"]
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(tokens))

        start = outputs["cur_position"]
        for i in range(len(tokens)):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += len(tokens)

    def _add_image(self, img, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        patches_h, patches_w = self.image_preprocessor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)
        if token_len and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += num_tokens

        pos_ids = self._compute_3d_positions(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        # Preprocess pixels
        ret = self.image_preprocessor.preprocess(
            images=[img.convert("RGB")],
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]]),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(ret["image_grid_thw"])
        outputs["image_type_ids"].append(0)

    def _add_processed_image(
        self, img_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len=None
    ) -> None:
        img, meta = img_cache
        num_tokens = img.shape[0] // (self.spatial_conv_size**2)
        if token_len and num_tokens != token_len:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)

        _, h, w = meta["thw"]
        pos_ids = self._compute_3d_positions(1, h, w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

    def _add_video(self, frames, outputs: Dict, uuid: Optional[str], token_len=None) -> None:
        patches_h, patches_w = self.image_preprocessor.get_smarted_resize(
            frames[0].height,
            frames[0].width,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        num_frames = len(frames)
        num_tokens = (num_frames * patches_h * patches_w) // (self.spatial_conv_size**2 * self.temporal_conv_size)
        if token_len and num_tokens != token_len:
            raise ValueError("video tokens num not match the size")

        pixel_stack = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)
        ret = self.image_preprocessor.preprocess(
            images=None,
            videos=pixel_stack,
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[patches_h, patches_w]] * num_frames),
            do_convert_rgb=True,
            input_data_format=ChannelDimension.LAST,
        )
        outputs["images"].append(ret["pixel_values_videos"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values_videos"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(ret["video_grid_thw"])
        outputs["image_type_ids"].extend([1] * num_frames)

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += num_tokens

        pos_ids = self._compute_3d_positions(num_frames, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def _add_processed_video(
        self, frames_cache: Tuple[np.ndarray, dict], outputs: Dict, uuid: str, token_len=None
    ) -> None:
        frames, meta = frames_cache
        num_tokens = frames.shape[0] // (self.spatial_conv_size**2 * self.temporal_conv_size)
        if token_len and num_tokens != token_len:
            raise ValueError("video tokens num not match the size")

        t, h, w = meta["thw"]
        outputs["images"].append(frames)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[t, h, w]]))

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_patch_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["image_type_ids"].extend([1] * t)

        pos_ids = self._compute_3d_positions(t, h, w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def _extract_labels(self, outputs: Dict, tgts: List[str]) -> None:
        input_ids = copy.deepcopy(outputs["input_ids"])
        labels = [self.tokenizer.ignored_index] * len(input_ids)

        tgt_count = input_ids.count(self.sep_token_id)
        assert tgt_count == len(tgts), f"len(tgts) != len(src) {len(tgts)} vs {tgt_count}"

        tgt_index = 0
        for i, token_id in enumerate(input_ids):
            if token_id == self.sep_token_id:
                labels_token = self.tokenizer.tokenize(tgts[tgt_index])
                labels_token_id = self.tokenizer.convert_tokens_to_ids(labels_token)
                labels[i - len(labels_token_id) : i] = labels_token_id
                labels[i] = self.eos_token_id  # </s>
                tgt_index += 1

        outputs["labels"] = labels

    def _load_and_process_video(self, url: str, item: Dict) -> List[Image.Image]:
        reader, meta, path = read_video_decord(url, save_to_disk=False)

        video_frame_args = dict()
        video_frame_args["fps"] = item.get("fps", self.fps)
        video_frame_args["min_frames"] = item.get("min_frames", self.min_frames)
        video_frame_args["max_frames"] = item.get("max_frames", self.max_frames)
        video_frame_args["target_frames"] = item.get("target_frames", self.target_frames)
        video_frame_args["frames_sample"] = item.get("frames_sample", self.frames_sample)

        video_frame_args = self._set_video_frame_args(video_frame_args, meta)

        frames_data, _, timestamps = read_frames_decord(
            path,
            reader,
            meta,
            target_frames=video_frame_args["target_frames"],
            target_fps=video_frame_args["fps"],
            frames_sample=video_frame_args["frames_sample"],
            save_to_disk=False,
        )

        frames: List[Image.Image] = []
        for img_array, ts in zip(frames_data, timestamps):
            frames.append(render_frame_timestamp(img_array, ts))
        # Ensure even number of frames for temporal conv
        if len(frames) % 2 != 0:
            frames.append(copy.deepcopy(frames[-1]))
        return frames

    def _set_video_frame_args(self, video_frame_args, video_meta):
        """
        根据已知参数和优先级，设定最终的抽帧参数
        """
        # 优先级：video_target_frames > (video_min_frames, video_max_frames) > video_fps
        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] >= 0:
                raise ValueError("fps must be negative if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] < 0:
                raise ValueError("Must provide either positive target_fps or positive target_frames.")
            # 先计算在video_fps下抽到的帧数
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])
            # 判断是否在目标区间内，如果不是，则取target_frames为上界或下界
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if video_frame_args["min_frames"] > 0 and frames_to_extract < video_frame_args["min_frames"]:
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
                video_frame_args["fps"] = -1
            if video_frame_args["max_frames"] > 0 and frames_to_extract > video_frame_args["max_frames"]:
                video_frame_args["target_frames"] = video_frame_args["max_frames"]
                video_frame_args["fps"] = -1

        return video_frame_args

    def _compute_3d_positions(self, t: int, h: int, w: int, start_idx: int) -> List[List[int]]:
        # Downsample time if needed
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords]

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        vocab_file_names = [
            "tokenizer.model",
            "spm.model",
            "ernie_token_100k.model",
        ]
        for i in range(len(vocab_file_names)):
            if os.path.exists(os.path.join(self.model_name_or_path, vocab_file_names[i])):
                Ernie4_5Tokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                break
        self.tokenizer = Ernie4_5Tokenizer.from_pretrained(self.model_name_or_path)

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
