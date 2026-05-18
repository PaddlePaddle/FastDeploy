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

"""PaddleOCR-VL encoding strategy."""

import numpy as np
from PIL import Image

from fastdeploy.engine.request import ImagePosition
from fastdeploy.input.encodings.qwen_encoding import QwenEncoding
from fastdeploy.input.encodings.registry import EncodingRegistry
from fastdeploy.input.mm_model_config import PADDLEOCR_VL
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.input.utils.video import read_video_paddlecodec
from fastdeploy.input.utils.video import sample_frames_paddleocr as _sample_paddleocr
from fastdeploy.multimodal.hasher import MultimodalHasher


@EncodingRegistry.register(PADDLEOCR_VL)
class PaddleOCREncoding(QwenEncoding):
    """Encoding strategy for paddleocr_vl.

    Inherits from QwenEncoding and overrides methods that differ:
    - _make_outputs: add vit_seqlen / vit_position_ids
    - add_image / add_video: append vit_fields (vit_seqlen, vit_position_ids)
    - add_video / add_processed_video: use video_token_id instead of image_token_id
    - load_video: use sample_frames_paddleocr instead of sample_frames_qwen
    """

    def _make_outputs(self) -> dict:
        outputs = super()._make_outputs()
        outputs["vit_seqlen"] = []
        outputs["vit_position_ids"] = []
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

        # paddleocr vit fields
        numel = h * w
        outputs["vit_seqlen"].append(numel)
        outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def add_processed_image(self, img_cache, outputs, uuid, token_len=None):
        super().add_processed_image(img_cache, outputs, uuid, token_len)
        _, h, w = img_cache[1]["thw"]
        numel = h * w
        outputs["vit_seqlen"].append(numel)
        outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def add_video(self, frames, outputs, uuid, token_len=None, meta=None):
        preprocess_kwargs = {}
        if self.cfg.video_min_pixels is not None:
            preprocess_kwargs["min_pixels"] = self.cfg.video_min_pixels
            preprocess_kwargs["max_pixels"] = self.cfg.video_max_pixels

        ret = self.image_processor.preprocess(images=frames, **preprocess_kwargs)

        num_tokens = ret["grid_thw"].prod() // self.image_processor.merge_size**2
        grid_thw = ret["grid_thw"].tolist()
        if token_len is not None and token_len != num_tokens:
            raise ValueError("video tokens num not match the size")

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

        fps = meta["fps"] if meta else 0
        second_per_grid_t = self.temporal_conv_size / fps if fps else 0
        t, h, w = grid_thw
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

        # paddleocr vit fields
        numel = h * w
        outputs["vit_seqlen"].append(numel)
        outputs["vit_position_ids"].append(np.arange(numel) % numel)

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
        outputs["input_ids"].extend([self.video_token_id] * num_tokens)
        outputs["token_type_ids"].extend([IDS_TYPE_FLAG["video"]] * num_tokens)
        outputs["num_input_video_tokens"] += num_tokens
        outputs["image_type_ids"].extend([1] * t)

        fps = meta["fps"]
        second_per_grid_t = self.temporal_conv_size / fps
        pos_ids = self._compute_vision_positions(outputs["cur_position"], t, h, w, second_per_grid_t)
        outputs["position_ids"].append(pos_ids)
        outputs["cur_position"] = pos_ids.max() + 1

        outputs["fps"].append(fps)

        # paddleocr vit fields
        numel = h * w
        outputs["vit_seqlen"].append(numel)
        outputs["vit_position_ids"].append(np.arange(numel) % numel)

    def load_video(self, url, item):
        reader, meta, _ = read_video_paddlecodec(url, save_to_disk=False)

        fps = item.get("fps", self.fps)
        num_frames = item.get("target_frames", self.target_frames)

        frame_indices = list(range(meta["num_of_frame"]))
        if fps > 0 or num_frames > 0:
            min_frames = item.get("min_frames", self.min_frames)
            max_frames = item.get("max_frames", self.max_frames)

            frame_indices = _sample_paddleocr(
                frame_factor=self.temporal_conv_size,
                min_frames=min_frames,
                max_frames=max_frames,
                metadata=meta,
                fps=fps,
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
