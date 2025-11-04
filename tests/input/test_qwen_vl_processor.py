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
from fastdeploy.input.qwen_vl_processor import QwenVLProcessor


def mock_pil_image(height, width):
    """
    Generate mock random RGB image

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        PIL.Image object with random RGB data
    """
    rgb_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(rgb_image)


def mock_read_frames(height: int, width: int, nums_frame: int, fps: int):
    """
    Generate mock video frames with metadata for testing purposes

    Creates synthetic video data by generating random RGB frames and constructing
    corresponding metadata to simulate real video processing.

    Args:
        height (int): Height of video frames in pixels
        width (int): Width of video frames in pixels
        nums_frame (int): Number of frames to generate
        fps (int): Frames per second for the mock video

    Returns:
        tuple: A tuple containing:
            frames (numpy.ndarray): Array of shape (nums_frame, height, width, 3)
                containing randomly generated RGB frames
            meta (dict): Dictionary with video metadata:
                - fps (int): Frames per second (same as input)
                - duration (float): Calculated duration in seconds (nums_frame/fps)
                - num_of_frame (int): Number of frames (same as nums_frame input)
    """
    frames = []
    for _ in range(nums_frame):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frames.append(frame)
    frames = np.stack(frames, axis=0)

    meta = {
        "fps": fps,
        "duration": nums_frame / fps,
        "num_of_frame": nums_frame,
    }
    return frames, meta


class TestQwenVLProcessor(unittest.TestCase):
    """
    Unit tests for Qwen Vision-Language Processor functionality
    """

    def setUp(self):
        """
        Initialize test case with:
        - Mock configuration
        - Patched message parsing and video processing methods
        - QwenVLProcessor instance with test parameters
        """
        config = MagicMock()
        config.vision_config.tokens_per_second = 2

        self.patcher_parse_image = patch(
            "fastdeploy.entrypoints.chat_utils.MultimodalPartParser.parse_image", return_value=mock_pil_image(480, 640)
        )
        self.patcher_parse_image.start()

        self.patcher_parse_video = patch(
            "fastdeploy.entrypoints.chat_utils.MultimodalPartParser.parse_video", return_value=b"123"
        )
        self.patcher_parse_video.start()

        self.patcher_read_frames = patch(
            "fastdeploy.input.qwen_vl_processor.process.DataProcessor._load_and_process_video",
            return_value=mock_read_frames(480, 640, 5, 2),
        )
        self.patcher_read_frames.start()

        mm_processor_kwargs = {
            "video_max_frames": 10,
            "video_min_frames": 1,
        }
        limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}

        self.model_name_or_path = "/ModelData/Qwen2.5-VL-7B-Instruct"
        self.processor = QwenVLProcessor(
            config=config,
            model_name_or_path=self.model_name_or_path,
            limit_mm_per_prompt=limit_mm_per_prompt,
            mm_processor_kwargs=mm_processor_kwargs,
            reasoning_parser_obj=None,
            tool_parser_obj=None,
        )

    def tearDown(self) -> None:
        """Clean up test case by stopping all mock patches"""
        self.patcher_read_frames.stop()
        self.patcher_parse_image.stop()
        self.patcher_parse_video.stop()

    def test_process_request(self):
        """
        Test processing of Request object with multimodal input

        Validates:
        1. Token ID lengths match position_ids and token_type_ids shapes
        2. Image processing produces expected output dimensions
        3. Video processing produces expected output dimensions
        4. Correct counts for images (1) and videos (1)
        """
        message = {
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

        request = Request.from_dict(message)
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

    def test_process_request_dict(self):
        """
        Test processing of dictionary-format request with multimodal input

        Validates:
        1. Token ID lengths match position_ids and token_type_ids shapes
        2. Image processing produces expected output dimensions
        3. Video processing produces expected output dimensions
        4. Correct counts for images (1) and videos (1)
        """
        num_completion_token_ids = 10
        request = {
            "request_id": "12345",
            "completion_token_ids": [1] * num_completion_token_ids,
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

    def test_process_request_dict_enable_thinking(self):
        num_completion_token_ids = 10
        request = {
            "request_id": "12345",
            "completion_token_ids": [1] * num_completion_token_ids,
            "stop": ["stop", "eof"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                    ],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }

        result = self.processor.process_request_dict(request, 100)
        self.assertEqual(result.get("enable_thinking"), False)

    def test_prompt(self):
        """
        Test processing of prompt with image and video placeholders

        Validates:
        1. Token ID lengths match position_ids and token_type_ids shapes
        2. Image processing produces expected output dimensions
        3. Video processing produces expected output dimensions
        4. Correct counts for images (1) and videos (1)
        """
        IMAGE_PLACEHOLDER = "<|image_pad|>"
        VIDEO_PLACEHOLDER = "<|video_pad|>"
        prompt = {
            "request_id": "12345",
            "prompt": f"{IMAGE_PLACEHOLDER}{VIDEO_PLACEHOLDER}Describe image and video.",
            "multimodal_data": {
                "image": [mock_pil_image(10, 2100)],
                "video": [{"video": b"123", "fps": 5}],
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

    def test_message_and_prompt(self):
        """
        Test consistency between message-based and prompt-based processing

        Validates that processing a request through:
        1. The message format (with image/video URLs)
        2. The prompt format (with direct image/video data)
        produces identical tokenization and multimodal input results.

        Checks:
        1. Prompt token IDs match between both processing methods
        2. Grid dimensions (THW) match between both methods
        3. Position IDs match between both methods
        """
        # Create test request in message format
        request = {
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
        result = self.processor.process_request_dict(request, 1024 * 100)

        # Create equivalent request in prompt format
        prompt = {
            "request_id": "12345",
            "prompt": request["prompt_tokens"],
            "multimodal_data": {
                "image": [mock_pil_image(480, 640)],
                "video": [{"video": b"123"}],
            },
        }
        request2 = Request.from_dict(prompt)
        result2 = self.processor.process_request(request2, 1024 * 100)

        # Verify both processing methods produce identical results
        self.assertEqual(result["prompt_token_ids"], result2.prompt_token_ids)
        self.assertTrue(np.equal(result["multimodal_inputs"]["grid_thw"], result2.multimodal_inputs["grid_thw"]).all())
        self.assertTrue(
            np.equal(result["multimodal_inputs"]["position_ids"], result2.multimodal_inputs["position_ids"]).all()
        )

    def test_apply_chat_template(self):
        """
        Test the consistency between:
        1. Directly applying chat template using HuggingFace tokenizer
        2. Applying chat template through the processor's request processing

        This test verifies that:
        - The processor correctly handles multimodal messages (image, video, text)
        - The prompt_tokens field matches the output from direct tokenizer application
        - The chat template application preserves the message structure and content

        Test Steps:
        1. Create sample multimodal messages with image, video and text content
        2. Apply chat template directly using the tokenizer
        3. Process the same messages through the processor
        4. Compare the outputs to ensure consistency
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # Sample multimodal messages containing image, video and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://demo.jpeg"}},
                    {"type": "video", "video": {"url": "file://3_frame_video.mp4"}},
                    {"type": "text", "text": "Describe image and video."},
                ],
            }
        ]

        # Apply chat template directly using the tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Create equivalent request dictionary
        request = {
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

        # Process request through the processor
        self.processor.process_request_dict(request, 1024 * 100)
        prompt2 = request["prompt_tokens"]

        # Verify both methods produce identical prompt strings
        self.assertEqual(prompt, prompt2)


if __name__ == "__main__":
    unittest.main()
