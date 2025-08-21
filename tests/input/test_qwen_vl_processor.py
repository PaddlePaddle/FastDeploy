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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from fastdeploy.engine.request import Request
from fastdeploy.input.qwen_mm_processor import DataProcessor
from fastdeploy.input.qwen_vl_processor import QwenVLProcessor


def mock_pil_image(height, width):
    """Generate mock random RGB image

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        PIL.Image object with random RGB data
    """
    rgb_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(rgb_image)


def mock_parse_chat_messages():
    """Generate mock chat messages with image, video and text content

    Returns:
        List of chat message dictionaries containing:
        - Mock image data (480x640 pixels)
        - Mock video data (dummy bytes)
        - Sample text prompt
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image_url": {},
                    "image": mock_pil_image(480, 640),
                },
                {
                    "type": "video",
                    "video_url": {},
                    "video": b"123",
                },
                {"type": "text", "text": "Describe image and video."},
            ],
        }
    ]
    return messages


def mock_video_frames(num_frames, height, width):
    """Generate mock video frames with random pixel data

    Args:
        num_frames: Number of frames to generate
        height: Frame height in pixels
        width: Frame width in pixels

    Returns:
        Numpy array of shape (num_frames, height, width, 3)
        containing random RGB frames
    """
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frames.append(frame)
    return np.stack(frames, axis=0)


def mock_load_and_process_video():
    """Mock video loading and processing

    Returns:
        Tuple containing:
        - frames: 3 mock video frames (480x640 resolution)
        - meta: Dictionary with mock video metadata:
            * fps: 1
            * duration: 3 seconds
            * num_of_frame: 3
    """
    frames = mock_video_frames(num_frames=3, height=480, width=640)
    meta = {
        "fps": 1,
        "duration": 3,
        "num_of_frame": 3,
    }
    return frames, meta


class TestQwenVLProcessor(unittest.TestCase):
    """Unit tests for Qwen Vision-Language Processor functionality"""

    def setUp(self):
        """Initialize test case with:
        - Mock configuration
        - Patched message parsing and video processing methods
        - QwenVLProcessor instance with test parameters
        """
        config = MagicMock()
        config.vision_config.tokens_per_second = 2

        self.patcher_parse_chat_messages = patch.object(
            DataProcessor, "_parse_chat_messages", return_value=mock_parse_chat_messages()
        )
        self.patcher_parse_chat_messages.start()

        self.patcher_load_and_process_video = patch.object(
            DataProcessor, "_load_and_process_video", return_value=mock_load_and_process_video()
        )
        self.patcher_load_and_process_video.start()

        mm_processor_kwargs = {
            "video_max_frames": 10,
            "video_min_frames": 1,
        }
        limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}

        # model_name_or_path = "/ModelData/Qwen2.5-VL-7B-Instruct"
        model_name_or_path = "./data/models/paddle/Qwen2.5-VL-3B-Instruct"
        self.processor = QwenVLProcessor(
            config=config,
            model_name_or_path=model_name_or_path,
            limit_mm_per_prompt=limit_mm_per_prompt,
            mm_processor_kwargs=mm_processor_kwargs,
            reasoning_parser_obj=None,
            tool_parser_obj=None,
        )

    def tearDown(self) -> None:
        """Clean up test case by stopping all mock patches"""
        self.patcher_parse_chat_messages.stop()
        self.patcher_load_and_process_video.stop()

    def test_process_request(self):
        """Test processing of Request object with multimodal input

        Validates:
        1. Token ID lengths match position_ids and token_type_ids shapes
        2. Image processing produces expected output dimensions
        3. Video processing produces expected output dimensions
        4. Correct counts for images (1) and videos (1)
        """
        prompt = {
            "request_id": "12345",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "file://demo.jpeg"}},
                        {"type": "video_url", "video_url": {"url": "file://3_frame_video.mp4"}},
                        {"type": "text", "text": "Describe image and video."},
                    ],
                }
            ],
        }

        request = Request.from_dict(prompt)
        result = self.processor.process_request(request, 1024 * 100)

        self.assertEqual(result.prompt_token_ids_len, result.multimodal_inputs["position_ids"].shape[0])
        self.assertEqual(result.prompt_token_ids_len, result.multimodal_inputs["token_type_ids"].shape[0])
        self.assertEqual(
            result.multimodal_inputs["images"].shape[0],
            sum(map(lambda x: x.prod(), result.multimodal_inputs["grid_thw"])),
        )
        self.assertEqual(
            result.multimodal_inputs["image_type_ids"].shape[0], result.multimodal_inputs["grid_thw"][:, 0].sum()
        )
        self.assertEqual(result.multimodal_inputs["pic_cnt"], 1)
        self.assertEqual(result.multimodal_inputs["video_cnt"], 1)

    def test_process_request_dict(self):
        """Test processing of dictionary-format request with multimodal input

        Validates:
        1. Token ID lengths match position_ids and token_type_ids shapes
        2. Image processing produces expected output dimensions
        3. Video processing produces expected output dimensions
        4. Correct counts for images (1) and videos (1)
        """
        num_generated_token_ids = 10
        request = {
            "request_id": "12345",
            "metadata": {
                "generated_token_ids": [1] * num_generated_token_ids,
            },
            "stop": ["stop", "eof"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "file://demo.jpeg"}},
                        {"type": "video_url", "video_url": {"url": "file://3_frame_video.mp4"}},
                        {"type": "text", "text": "Describe image and video."},
                    ],
                }
            ],
        }

        result = self.processor.process_request_dict(request, 1024 * 100)

        self.assertEqual(result["prompt_token_ids_len"], result["multimodal_inputs"]["position_ids"].shape[0])
        self.assertEqual(result["prompt_token_ids_len"], result["multimodal_inputs"]["token_type_ids"].shape[0])
        self.assertEqual(
            result["multimodal_inputs"]["images"].shape[0],
            sum(map(lambda x: x.prod(), result["multimodal_inputs"]["grid_thw"])),
        )
        self.assertEqual(
            result["multimodal_inputs"]["image_type_ids"].shape[0], result["multimodal_inputs"]["grid_thw"][:, 0].sum()
        )
        self.assertEqual(result["multimodal_inputs"]["pic_cnt"], 1)
        self.assertEqual(result["multimodal_inputs"]["video_cnt"], 1)

    def test_prompt(self):
        """Test processing of prompt with image and video placeholders

        Validates:
        1. Token ID lengths match position_ids and token_type_ids shapes
        2. Image processing produces expected output dimensions
        3. Video processing produces expected output dimensions
        4. Correct counts for images (1) and videos (1)
        """
        prompt = {
            "request_id": "12345",
            "prompt": "<|image@placeholder|><|video@placeholder|>Describe image and video.",
            "multimodal_data": {
                "image": [mock_pil_image(10, 2100)],
                "video": [b"123"],
            },
        }

        request = Request.from_dict(prompt)
        result = self.processor.process_request(request, 1024 * 100)

        self.assertEqual(result.prompt_token_ids_len, result.multimodal_inputs["position_ids"].shape[0])
        self.assertEqual(result.prompt_token_ids_len, result.multimodal_inputs["token_type_ids"].shape[0])
        self.assertEqual(
            result.multimodal_inputs["images"].shape[0],
            sum(map(lambda x: x.prod(), result.multimodal_inputs["grid_thw"])),
        )
        self.assertEqual(
            result.multimodal_inputs["image_type_ids"].shape[0], result.multimodal_inputs["grid_thw"][:, 0].sum()
        )
        self.assertEqual(result.multimodal_inputs["pic_cnt"], 1)
        self.assertEqual(result.multimodal_inputs["video_cnt"], 1)


if __name__ == "__main__":
    unittest.main()
