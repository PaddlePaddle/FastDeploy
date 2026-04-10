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

"""Qwen-family (qwen_vl / qwen3_vl) encoding strategy."""


import numpy as np
import paddle
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.input.encodings.base_encoding import BaseEncoding
from fastdeploy.input.encodings.registry import EncodingRegistry
from fastdeploy.input.mm_model_config import QWEN3_VL, QWEN_VL
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.input.utils.video import read_video_decord
from fastdeploy.input.utils.video import sample_frames_qwen as _sample_qwen
from fastdeploy.multimodal.hasher import MultimodalHasher


@EncodingRegistry.register(QWEN_VL, QWEN3_VL)
class QwenEncoding(BaseEncoding):
    """Encoding strategy for qwen_vl and qwen3_vl."""

    FRAME_FACTOR = 2

    def _make_outputs(self) -> dict:
        outputs = super()._make_outputs()
        outputs["fps"] = []
        return outputs

    def add_image(self, img, outputs, uuid, token_len=None):
        ret = self.image_processor.preprocess(images=[img.convert("RGB")])
        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

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

        t, h, w = grid_thw
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(0)

    def add_processed_image(self, img_cache, outputs, uuid, token_len=None):
        img, meta = img_cache
        num_tokens = img.shape[0] // self.image_processor.merge_size**2
        if token_len is not None and token_len != num_tokens:
            raise ValueError("image tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["image"]] * num_tokens)
        outputs["num_input_image_tokens"] += num_tokens

        _, h, w = meta["thw"]
        pos_ids = self._compute_vision_positions(outputs["cur_position"], 1, h, w, 0)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["images"].append(img)
        outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(np.array([[1, h, w]]))
        outputs["image_type_ids"].append(0)

        outputs["fps"].append(0)

    def add_video(self, frames, outputs, uuid, token_len=None, meta=None):
        preprocess_kwargs = {}
        # qwen3_vl passes min/max pixels for video
        if self.cfg.video_min_pixels is not None:
            preprocess_kwargs["min_pixels"] = self.cfg.video_min_pixels
            preprocess_kwargs["max_pixels"] = self.cfg.video_max_pixels

        ret = self.image_processor.preprocess(images=frames, **preprocess_kwargs)

        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("video tokens num not match the size")

        outputs["mm_positions"].append(ImagePosition(len(outputs["input_ids"]), num_tokens))
        outputs["input_ids"].extend([self.image_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += int(num_tokens)

        outputs["images"].append(ret["pixel_values"])
        if not uuid:
            outputs["mm_hashes"].append(MultimodalHasher.hash_features(ret["pixel_values"]))
        else:
            outputs["mm_hashes"].append(uuid)
        outputs["grid_thw"].append(grid_thw)
        outputs["image_type_ids"].extend([1] * grid_thw[0])

        fps = meta["fps"] if meta else 0
        second_per_grid_t = self.temporal_conv_size / fps if fps else 0
        t, h, w = grid_thw
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

    def add_processed_video(self, frames_cache, outputs, uuid, token_len=None):
        frames, meta = frames_cache
        num_tokens = frames.shape[0] // self.image_processor.merge_size**2
        if token_len is not None and token_len != num_tokens:
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

        fps = meta["fps"]
        second_per_grid_t = self.temporal_conv_size / fps
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

    def load_video(self, url, item):
        reader, meta, _ = read_video_decord(url, save_to_disk=False)

        fps = item.get("fps", self.fps)
        num_frames = item.get("target_frames", self.target_frames)

        frame_indices = list(range(meta["num_of_frame"]))
        if fps > 0 or num_frames > 0:
            min_frames = item.get("min_frames", self.min_frames)
            max_frames = item.get("max_frames", self.max_frames)

            frame_indices = _sample_qwen(
                frame_factor=self.FRAME_FACTOR,
                min_frames=min_frames,
                max_frames=max_frames,
                metadata=meta,
                fps=-1 if num_frames > 0 else fps,
                num_frames=num_frames,
            )

            meta["num_of_frame"] = len(frame_indices)
            if fps is not None:
                meta["fps"] = fps
                meta["duration"] = len(frame_indices) / fps
            else:
                meta["fps"] = len(frame_indices) / meta["duration"]

        frames = []
        for idx in frame_indices:
            frame = reader[idx].asnumpy()
            image = Image.fromarray(frame, "RGB")
            frames.append(image)
        frames = np.stack([np.array(f.convert("RGB")) for f in frames], axis=0)

        return frames, meta

    def add_text_positions(self, outputs, num_tokens):
        """Write text position IDs in qwen 3xN ndarray format."""
        pos_ids = self._compute_text_positions(outputs["cur_position"], num_tokens)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        num_tokens = len(completion_token_ids)
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        pos_ids = self._compute_text_positions(multimodal_inputs["cur_position"], num_tokens)
        multimodal_inputs["position_ids"].append(pos_ids)
        multimodal_inputs["cur_position"] += num_tokens

    def prompt_token_ids2outputs(self, prompt_token_ids, mm_items=None):
        """Build outputs from prompt_token_ids. Only qwen3_vl supports this."""
        outputs = self._make_outputs()
        prompt_token_ids_len = len(prompt_token_ids)

        if mm_items is None:
            self._add_text_tokens(prompt_token_ids, outputs)
            return outputs

        st, mm_idx = 0, 0
        while st < prompt_token_ids_len:
            if prompt_token_ids[st] != self.image_token_id:
                cur_idx = st
                while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] != self.image_token_id:
                    cur_idx += 1
                self._add_text_tokens(prompt_token_ids[st:cur_idx], outputs)
                st = cur_idx
                continue

            if mm_idx >= len(mm_items):
                raise ValueError("prompt token ids has more multimodal placeholder than in messages")

            cur_idx = st
            while cur_idx < prompt_token_ids_len and prompt_token_ids[cur_idx] == self.image_token_id:
                cur_idx += 1

            item = mm_items[mm_idx]
            uuid = item.get("uuid")
            token_len = cur_idx - st
            if item.get("type") == "image":
                image = item.get("data")
                if not isinstance(image, tuple):
                    self.add_image(image, outputs, uuid, token_len)
                else:
                    self.add_processed_image(image, outputs, uuid, token_len)
            elif item.get("type") == "video":
                video = item.get("data")
                if not isinstance(video, tuple):
                    if isinstance(video, dict):
                        frames, meta = self.load_video(video["video"], video)
                    else:
                        frames, meta = self.load_video(video, {})
                    self.add_video(frames, outputs, uuid, token_len=token_len, meta=meta)
                else:
                    self.add_processed_video(video, outputs, uuid, token_len)
            else:
                raise ValueError(f"Unsupported multimodal type: {item.get('type')}")
            mm_idx += 1
            st = cur_idx

        if mm_idx != len(mm_items):
            raise ValueError("number of multimodal items does not match prompt token ids")

        return outputs

    def _add_text_tokens(self, tokens, outputs):
        """Helper: add text tokens with position IDs."""
        if not tokens:
            return
        num_tokens = len(tokens)
        outputs["input_ids"].extend(tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)
        self.add_text_positions(outputs, num_tokens)

    def _compute_text_positions(self, start_pos, num_tokens):
        """3xN ndarray for qwen-family text positions."""
        text_array = np.arange(num_tokens).reshape(1, -1)
        text_index = np.broadcast_to(text_array, (3, num_tokens))
        return text_index + start_pos

    def _compute_vision_positions(self, start_pos, t, h, w, second_per_grid_t):
        """3D position IDs as 3xN ndarray for qwen-family."""
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

        return np.stack([t_index, h_index, w_index]) + start_pos

    @staticmethod
    def mm_num_tokens(grid_thw):
        """Qwen mm_num_tokens: t * h * w // 4."""
        if isinstance(grid_thw, paddle.Tensor):
            grid_thw = grid_thw.numpy()
        if len(grid_thw) == 0:
            return 0

        def calc_one(thw):
            t, h, w = map(int, thw)
            return t * h * w // 4

        if isinstance(grid_thw[0], (list, tuple, np.ndarray)):
            return [calc_one(x) for x in grid_thw]
        return calc_one(grid_thw)

    def pack_position_ids(self, outputs):
        """Qwen: concatenate 3xN arrays, then transpose to Nx3."""
        outputs["position_ids"] = np.concatenate(outputs["position_ids"], axis=1, dtype=np.int64)
        outputs["image_patch_id"] = self.image_token_id
        outputs["video_patch_id"] = self.video_token_id
        outputs["position_ids"] = outputs["position_ids"].transpose(1, 0)
