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

"""Ernie4.5-VL encoding strategy for MultiModalProcessor."""

import copy
from collections import defaultdict

import numpy as np
import paddle
from paddleformers.transformers.image_utils import ChannelDimension

from fastdeploy.engine.request import ImagePosition
from fastdeploy.input.encodings.base_encoding import BaseEncoding
from fastdeploy.input.encodings.registry import EncodingRegistry
from fastdeploy.input.mm_model_config import ERNIE4_5_VL
from fastdeploy.input.utils import IDS_TYPE_FLAG, MAX_IMAGE_DIMENSION
from fastdeploy.multimodal.hasher import MultimodalHasher


@EncodingRegistry.register(ERNIE4_5_VL)
class ErnieEncoding(BaseEncoding):
    """Encoding strategy for Ernie4.5-VL models."""

    # Boundary token constants
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    VID_START = "<|VIDEO_START|>"
    VID_END = "<|VIDEO_END|>"

    def init_extra(self, processor_kwargs):
        """Ernie-specific extra initialisation (pixel params, token type mapping, etc.)."""
        self.image_min_pixels = processor_kwargs.get("image_min_pixels", 4 * 28 * 28)
        self.image_max_pixels = processor_kwargs.get("image_max_pixels", 6177 * 28 * 28)
        self.video_min_pixels = processor_kwargs.get("video_min_pixels", 299 * 28 * 28)
        self.video_max_pixels = processor_kwargs.get("video_max_pixels", 1196 * 28 * 28)
        self.frames_sample = processor_kwargs.get("video_frames_sample", self.cfg.default_frames_sample)

        # Build token-type mapping for ernie boundary tokens
        self.token_type_mapping = self._build_token_type_mapping()

    def _build_token_type_mapping(self):
        mapping = defaultdict(lambda: IDS_TYPE_FLAG["text"])
        for token in (self.IMG_START, self.IMG_END, self.VID_START, self.VID_END):
            mapping[token] = IDS_TYPE_FLAG["image"]
        mapping[self.image_token_id] = IDS_TYPE_FLAG["image"]
        return mapping

    def add_image(self, img, outputs, uuid, token_len=None):
        patches_h, patches_w = self.image_processor.get_smarted_resize(
            img.height,
            img.width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        num_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)
        if token_len and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += num_tokens

        pos_ids = self._compute_3d_positions(1, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        ret = self.image_processor.preprocess(
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

    def add_processed_image(self, img_cache, outputs, uuid, token_len=None):
        img, meta = img_cache
        num_tokens = img.shape[0] // (self.spatial_conv_size**2)
        if token_len and num_tokens != token_len:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += num_tokens

        _, h, w = meta["thw"]
        pos_ids = self._compute_3d_positions(1, h, w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

    def add_video(self, frames, outputs, uuid, token_len=None, meta=None):
        patches_h, patches_w = self.image_processor.get_smarted_resize(
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
        ret = self.image_processor.preprocess(
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
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += num_tokens

        pos_ids = self._compute_3d_positions(num_frames, patches_h, patches_w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def add_processed_video(self, frames_cache, outputs, uuid, token_len=None):
        frames, meta = frames_cache
        num_tokens = frames.shape[0] // (self.spatial_conv_size**2 * self.temporal_conv_size)
        if token_len and num_tokens != token_len:
            raise ValueError("video tokens num not match the size")

        t, h, w = meta["thw"]
        outputs["images"].append(frames)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[t, h, w]]))

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += num_tokens
        outputs["image_type_ids"].extend([1] * t)

        pos_ids = self._compute_3d_positions(t, h, w, outputs["cur_position"])
        outputs["position_ids"].extend(pos_ids)
        outputs["cur_position"] = np.max(pos_ids) + 1

    def load_video(self, url, item):
        from fastdeploy.input.utils.render_timestamp import render_frame_timestamp
        from fastdeploy.input.utils.video import (
            read_frames_paddlecodec,
            read_video_paddlecodec,
        )

        reader, meta, path = read_video_paddlecodec(url, save_to_disk=False)

        video_frame_args = {
            "fps": item.get("fps", self.fps),
            "min_frames": item.get("min_frames", self.min_frames),
            "max_frames": item.get("max_frames", self.max_frames),
            "target_frames": item.get("target_frames", self.target_frames),
            "frames_sample": item.get("frames_sample", self.frames_sample),
        }
        video_frame_args = self.set_video_frame_args(video_frame_args, meta)

        frames_data, _, timestamps = read_frames_paddlecodec(
            path,
            reader,
            meta,
            target_frames=video_frame_args["target_frames"],
            target_fps=video_frame_args["fps"],
            frames_sample=video_frame_args["frames_sample"],
            save_to_disk=False,
        )

        frames = []
        for img_array, ts in zip(frames_data, timestamps):
            frames.append(render_frame_timestamp(img_array, ts))
        # Ensure even number of frames for temporal conv
        if len(frames) % 2 != 0:
            frames.append(copy.deepcopy(frames[-1]))
        return frames, {}

    def set_video_frame_args(self, video_frame_args, video_meta):
        """Set final frame sampling args based on priorities."""
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
            frames_to_extract = int(video_meta["duration"] * video_frame_args["fps"])
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

    def add_text_positions(self, outputs, num_tokens):
        """Write text position IDs in ernie [pos, pos, pos] format."""
        start = outputs["cur_position"]
        for i in range(num_tokens):
            outputs["position_ids"].append([start + i] * 3)
        outputs["cur_position"] += num_tokens

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def _compute_3d_positions(self, t, h, w, start_idx):
        """Compute 3D position IDs as list-of-lists for ernie format."""
        t_eff = t // self.temporal_conv_size if t != 1 else 1
        gh, gw = h // self.spatial_conv_size, w // self.spatial_conv_size
        time_idx = np.repeat(np.arange(t_eff), gh * gw)
        h_idx = np.tile(np.repeat(np.arange(gh), gw), t_eff)
        w_idx = np.tile(np.arange(gw), t_eff * gh)

        coords = list(zip(time_idx, h_idx, w_idx))
        return [[start_idx + ti, start_idx + hi, start_idx + wi] for ti, hi, wi in coords]

    def prompt_token_ids2outputs(self, prompt_token_ids, mm_items=None):
        outputs = self._make_outputs()
        prompt_token_ids_len = len(prompt_token_ids)

        if mm_items is None:
            outputs["input_ids"].extend(prompt_token_ids)
            outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * prompt_token_ids_len)
            for i in range(prompt_token_ids_len):
                outputs["position_ids"].append([i] * 3)
            outputs["cur_position"] += prompt_token_ids_len
            return outputs

        images, videos = [], []
        image_uuid, video_uuid = [], []
        for item in mm_items:
            if item.get("type") == "image":
                images.append(item["data"])
                image_uuid.append(item.get("uuid"))
            elif item.get("type") == "video":
                videos.append(item["data"])
                video_uuid.append(item.get("uuid"))

        image_start_id = self.tokenizer.convert_tokens_to_ids(self.IMG_START)
        image_end_id = self.tokenizer.convert_tokens_to_ids(self.IMG_END)
        video_start_id = self.tokenizer.convert_tokens_to_ids(self.VID_START)
        video_end_id = self.tokenizer.convert_tokens_to_ids(self.VID_END)

        st, image_idx, video_idx = 0, 0, 0
        while st < prompt_token_ids_len:
            cur_token_id = prompt_token_ids[st]
            if cur_token_id == image_start_id:
                if image_idx >= len(images):
                    raise ValueError("prompt token ids has more image placeholder than in messages")
                # append image_start_id
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
                # process placeholder token ids
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != image_end_id:
                    cur_idx += 1
                if cur_idx >= prompt_token_ids_len:
                    raise ValueError("image token ids not complete")
                image = images[image_idx]
                uuid = image_uuid[image_idx] if image_uuid else None
                token_len = cur_idx - st
                if not isinstance(image, tuple):
                    self.add_image(image, outputs, uuid, token_len)
                else:
                    self.add_processed_image(image, outputs, uuid, token_len)
                image_idx += 1
                st = cur_idx
            elif cur_token_id == video_start_id:
                if video_idx >= len(videos):
                    raise ValueError("prompt token ids has more video placeholder than in messages")
                # append video_start_id
                outputs["input_ids"].extend([cur_token_id])
                outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1
                # process placeholder token ids
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != video_end_id:
                    cur_idx += 1
                if cur_idx >= prompt_token_ids_len:
                    raise ValueError("video token ids not complete")
                video = videos[video_idx]
                uuid = video_uuid[video_idx] if video_uuid else None
                token_len = cur_idx - st
                if not isinstance(video, tuple):
                    if isinstance(video, dict):
                        frames, _ = self.load_video(video["video"], video)
                    else:
                        frames, _ = self.load_video(video, {})
                    self.add_video(frames, outputs, uuid, token_len=token_len)
                else:
                    self.add_processed_video(video, outputs, uuid, token_len)
                video_idx += 1
                st = cur_idx
            else:
                outputs["input_ids"].extend([cur_token_id])
                type_flag = (
                    IDS_TYPE_FLAG["image"] if cur_token_id in (image_end_id, video_end_id) else IDS_TYPE_FLAG["text"]
                )
                outputs["token_type_ids"].extend([type_flag])
                outputs["position_ids"].append([outputs["cur_position"]] * 3)
                outputs["cur_position"] += 1
                st += 1

        if image_idx != len(images):
            raise ValueError("number of images does not match")
        if video_idx != len(videos):
            raise ValueError("number of videos does not match")

        return outputs

    @staticmethod
    def mm_num_tokens(grid_thw):
        """Ernie mm_num_tokens: video (t>1) divides by an extra 2."""
        if isinstance(grid_thw, paddle.Tensor):
            grid_thw = grid_thw.numpy()
        if len(grid_thw) == 0:
            return 0

        def calc_one(thw):
            t, h, w = map(int, thw)
            if t == 1:
                return t * h * w // 4
            else:
                return t * h * w // 4 // 2

        if isinstance(grid_thw[0], (list, tuple, np.ndarray)):
            return [calc_one(x) for x in grid_thw]
        return calc_one(grid_thw)

    def pack_position_ids(self, outputs):
        """Ernie: position_ids is np.array (list-of-lists -> ndarray)."""
        outputs["position_ids"] = np.array(outputs["position_ids"], dtype=np.int64)
        outputs["image_patch_id"] = self.image_token_id

    def get_mm_max_tokens_per_item(self, seq_len):
        """Per-modality max token counts for ernie."""
        target_height, target_width = self._get_image_size_with_most_features()
        # image
        patches_h, patches_w = self.image_processor.get_smarted_resize(
            height=target_height,
            width=target_width,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[1]
        max_image_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2)
        max_image_tokens = min(max_image_tokens, seq_len)
        # video
        patches_h, patches_w = self.image_processor.get_smarted_resize(
            height=target_height,
            width=target_width,
            min_pixels=self.video_min_pixels,
            max_pixels=self.video_max_pixels,
        )[1]
        max_video_tokens = (patches_h * patches_w) // (self.spatial_conv_size**2 * self.temporal_conv_size)
        max_video_tokens = min(max_video_tokens, seq_len)
        return {"image": max_image_tokens, "video": max_video_tokens}

    def _get_image_size_with_most_features(self):
        resized_height, resized_width = self.image_processor.get_smarted_resize(
            height=MAX_IMAGE_DIMENSION,
            width=MAX_IMAGE_DIMENSION,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )[0]
        return (resized_height, resized_width)
