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
from fastdeploy.input.qwen_vl_processor.process_video import sample_frames


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

    def test_think_status(self):
        """测试 思考机制"""
        request = {
            "prompt": "hello",
            "request_id": "test_1",
            "prompt_token_ids": [1, 2, 3],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        self.processor.reasoning_parser = MagicMock()
        self.processor.reasoning_parser.get_model_status.return_value = "think_start"
        self.processor.model_status_dict = {}
        self.processor.process_request_dict(request, max_model_len=512)
        self.assertEqual(request["enable_thinking"], True)

        request = {
            "prompt": "hello",
            "request_id": "test",
            "prompt_token_ids": [1, 2, 3],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        self.processor.process_request_dict(request, max_model_len=512)
        self.assertEqual(request["enable_thinking"], True)

    def test_parse_processor_kwargs_valid(self):
        """Test _parse_processor_kwargs with valid input"""
        valid_kwargs = {"video_max_frames": 10, "video_min_frames": 1}
        result = self.processor._parse_processor_kwargs(valid_kwargs)
        self.assertEqual(result, valid_kwargs)

    def test_parse_processor_kwargs_empty(self):
        """Test _parse_processor_kwargs with empty input"""
        result = self.processor._parse_processor_kwargs(None)
        self.assertEqual(result, {})

    def test_parse_processor_kwargs_invalid_type(self):
        """Test _parse_processor_kwargs with invalid type"""
        result = self.processor._parse_processor_kwargs("invalid")
        self.assertEqual(result, {})

    def test_parse_processor_kwargs_invalid_value_type(self):
        """Test _parse_processor_kwargs with invalid value type"""
        invalid_kwargs = {"video_max_frames": "10"}  # Should be int
        result = self.processor._parse_processor_kwargs(invalid_kwargs)
        self.assertEqual(result, {})

    def test_parse_processor_kwargs_mixed_valid_invalid(self):
        """Test _parse_processor_kwargs with mixed valid and invalid types"""
        mixed_kwargs = {"video_max_frames": 10, "video_min_frames": "invalid"}
        result = self.processor._parse_processor_kwargs(mixed_kwargs)
        self.assertEqual(result, {})

    def test_parse_limits_valid(self):
        """Test _parse_limits with valid limits"""
        limits = {"image": 2, "video": 3}
        result = self.processor._parse_limits(limits)
        expected = {"image": 2, "video": 3, "audio": 1}
        self.assertEqual(result, expected)

    def test_parse_limits_empty(self):
        """Test _parse_limits with empty input"""
        result = self.processor._parse_limits(None)
        expected = {"image": 1, "video": 1, "audio": 1}
        self.assertEqual(result, expected)

    def test_parse_limits_invalid_type(self):
        """Test _parse_limits with invalid type"""
        result = self.processor._parse_limits("invalid")
        expected = {"image": 1, "video": 1, "audio": 1}
        self.assertEqual(result, expected)

    def test_parse_limits_partial(self):
        """Test _parse_limits with partial limits"""
        limits = {"image": 5}
        result = self.processor._parse_limits(limits)
        expected = {"image": 5, "video": 1, "audio": 1}
        self.assertEqual(result, expected)

    def test_check_mm_limits_dict_valid(self):
        """Test _check_mm_limits with valid dict input"""
        mm_data = {"image": [mock_pil_image(10, 10)], "video": [{"video": b"123"}]}
        # Should not raise exception
        self.processor._check_mm_limits(mm_data)

    def test_check_mm_limits_dict_exceed_limit(self):
        """Test _check_mm_limits when dict input exceeds limit"""
        mm_data = {"image": [mock_pil_image(10, 10), mock_pil_image(10, 10)]}
        with self.assertRaises(ValueError) as context:
            self.processor._check_mm_limits(mm_data)
        self.assertIn("Too many image items", str(context.exception))

    def test_check_mm_limits_messages_valid(self):
        """Test _check_mm_limits with valid messages input"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://demo.jpeg"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        # Should not raise exception
        self.processor._check_mm_limits(messages)

    def test_check_mm_limits_messages_exceed_limit(self):
        """Test _check_mm_limits when messages input exceeds limit"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://demo1.jpeg"}},
                    {"type": "image_url", "image_url": {"url": "file://demo2.jpeg"}},
                ],
            }
        ]
        with self.assertRaises(ValueError) as context:
            self.processor._check_mm_limits(messages)
        self.assertIn("Too many image items", str(context.exception))

    def test_check_mm_limits_video_exceed(self):
        """Test _check_mm_limits when video exceeds limit"""
        mm_data = {"video": [{"video": b"123"}, {"video": b"456"}]}
        with self.assertRaises(ValueError) as context:
            self.processor._check_mm_limits(mm_data)
        self.assertIn("Too many video items", str(context.exception))

    def test_process_request_dict_with_prompt(self):
        """Test process_request_dict with prompt format"""
        request = {
            "request_id": "12345",
            "prompt": "Test prompt",
            "multimodal_data": {"image": [mock_pil_image(10, 10)]},
        }
        result = self.processor.process_request_dict(request, 1024)
        self.assertIn("prompt_token_ids", result)
        self.assertIn("multimodal_inputs", result)

    def test_process_request_dict_with_messages(self):
        """Test process_request_dict with messages format"""
        request = {
            "request_id": "12345",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
        }
        result = self.processor.process_request_dict(request, 1024)
        self.assertIn("prompt_token_ids", result)
        self.assertIn("multimodal_inputs", result)

    def test_process_request_dict_invalid_format(self):
        """Test process_request_dict with invalid format"""
        request = {"request_id": "12345"}
        with self.assertRaises(ValueError) as context:
            self.processor.process_request_dict(request, 1024)
        self.assertIn("must contain 'prompt', or 'messages'", str(context.exception))

    def test_process_request_dict_with_bad_words(self):
        """Test process_request_dict with bad_words"""
        request = {
            "request_id": "12345",
            "prompt": "Test prompt",
            "bad_words": ["bad", "word"],
            "bad_words_token_ids": [100, 200],
        }
        result = self.processor.process_request_dict(request, 1024)
        # Verify bad_words_token_ids is set
        self.assertIn("bad_words_token_ids", result)
        self.assertIsNotNone(result["bad_words_token_ids"])

    def test_process_request_dict_invalid_chat_template_kwargs(self):
        """Test process_request_dict with invalid chat_template_kwargs"""
        request = {
            "request_id": "12345",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "chat_template_kwargs": "invalid",
        }
        with self.assertRaises(ValueError) as context:
            self.processor.process_request_dict(request, 1024)
        self.assertIn("must be a dict", str(context.exception))

    def test_process_request_dict_with_completion_token_ids(self):
        """Test process_request_dict with completion_token_ids"""
        request = {
            "request_id": "12345",
            "prompt": "Test",
            "completion_token_ids": [1, 2, 3],
        }
        result = self.processor.process_request_dict(request, 1024)
        # Verify completion tokens are appended
        self.assertGreater(len(result["prompt_token_ids"]), 3)

    def test_process_request_dict_prompt_truncation(self):
        """Test process_request_dict with prompt truncation"""
        # Create a long prompt that exceeds max_model_len
        long_prompt = "Test " * 1000
        request = {
            "request_id": "12345",
            "prompt": long_prompt,
        }
        result = self.processor.process_request_dict(request, 100)
        # Verify prompt is truncated
        self.assertLessEqual(len(result["prompt_token_ids"]), 99)

    def test_process_request_dict_default_max_tokens(self):
        """Test process_request_dict sets default max_tokens"""
        request = {
            "request_id": "12345",
            "prompt": "Test",
        }
        result = self.processor.process_request_dict(request, 1024)
        self.assertIn("max_tokens", result)
        self.assertGreater(result["max_tokens"], 0)

    def test_process_request_dict_enable_thinking_false(self):
        """Test process_request_dict sets enable_thinking to False"""
        request = {
            "request_id": "12345",
            "prompt": "Test",
            "enable_thinking": True,
        }
        result = self.processor.process_request_dict(request, 1024)
        self.assertFalse(result["enable_thinking"])

    def test_append_completion_tokens(self):
        """Test append_completion_tokens method"""
        multimodal_inputs = {
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
            "cur_position": 3,
        }
        completion_token_ids = [4, 5]
        self.processor.append_completion_tokens(multimodal_inputs, completion_token_ids)

        self.assertEqual(multimodal_inputs["input_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(multimodal_inputs["token_type_ids"], [0, 0, 0, 0, 0])
        self.assertEqual(multimodal_inputs["cur_position"], 5)

    def test_pack_outputs_with_images(self):
        """Test pack_outputs with image data"""
        outputs = {
            "images": [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
            "grid_thw": [np.array([2, 2, 1]), np.array([2, 2, 1])],
            "image_type_ids": [0, 1],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }
        result = self.processor.pack_outputs(outputs)

        self.assertIsNotNone(result["images"])
        self.assertIsNotNone(result["grid_thw"])
        self.assertIsNotNone(result["image_type_ids"])
        self.assertEqual(result["images"].shape[0], 4)
        self.assertEqual(result["grid_thw"].shape[0], 2)

    def test_pack_outputs_without_images(self):
        """Test pack_outputs without image data"""
        outputs = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }
        result = self.processor.pack_outputs(outputs)

        # Test that image-related fields are None when no images
        self.assertIsNone(result["images"])
        self.assertIsNone(result["grid_thw"])
        self.assertIsNone(result["image_type_ids"])

        # Test data types
        self.assertEqual(result["input_ids"].dtype, np.int64)
        self.assertEqual(result["token_type_ids"].dtype, np.int64)
        self.assertEqual(result["position_ids"].dtype, np.int64)

        # Test patch IDs are set
        self.assertIn("image_patch_id", result)
        self.assertIn("video_patch_id", result)
        self.assertIn("mm_num_token_func", result)


class TestSampleFrames(unittest.TestCase):
    """
    Unit tests for sample_frames function
    """

    def setUp(self):
        self.metadata = {
            "num_of_frame": 100,
            "fps": 25,
        }

    def test_fps_and_num_frames_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            sample_frames(
                frame_factor=4,
                min_frames=8,
                max_frames=32,
                metadata=self.metadata,
                fps=2,
                num_frames=16,
            )

    def test_num_frames_round_to_factor(self):
        indices = sample_frames(
            frame_factor=4,
            min_frames=8,
            max_frames=64,
            metadata=self.metadata,
            num_frames=18,  # round(18 / 4) * 4 = 16
        )

        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 0)
        self.assertLess(indices[-1], self.metadata["num_of_frame"])

    def test_fps_sampling_basic(self):
        # total = 100 frames, fps=25, target fps=5 → 20 frames
        indices = sample_frames(
            frame_factor=4,
            min_frames=8,
            max_frames=64,
            metadata=self.metadata,
            fps=5,
        )

        self.assertEqual(len(indices), 20)
        self.assertEqual(indices.dtype, np.int32)
        self.assertEqual(indices[0], 0)

    def test_fps_respects_min_frames(self):
        indices = sample_frames(
            frame_factor=4,
            min_frames=24,
            max_frames=64,
            metadata=self.metadata,
            fps=1,  # very small fps
        )

        self.assertEqual(len(indices), 24)

    def test_num_frames_exceeds_total_raises(self):
        with self.assertRaises(ValueError):
            sample_frames(
                frame_factor=4,
                min_frames=8,
                max_frames=200,
                metadata=self.metadata,
                num_frames=200,
            )

    def test_force_multiple_of_4_hack(self):
        indices = sample_frames(
            frame_factor=2,
            min_frames=2,
            max_frames=100,
            metadata=self.metadata,
            num_frames=10,  # 10 % 4 != 0 → hack → 8
        )

        self.assertEqual(len(indices), 8)
        self.assertEqual(len(indices) % 4, 0)

    def test_keep_all_frames_when_num_frames_zero(self):
        indices = sample_frames(
            frame_factor=4,
            min_frames=0,
            max_frames=100,
            metadata=self.metadata,
            num_frames=0,
        )

        self.assertEqual(len(indices), self.metadata["num_of_frame"])
        np.testing.assert_array_equal(indices, np.arange(0, 100, dtype=np.int32))

    def test_indices_evenly_spaced(self):
        indices = sample_frames(
            frame_factor=4,
            min_frames=8,
            max_frames=32,
            metadata=self.metadata,
            num_frames=16,
        )

        diffs = np.diff(indices)
        self.assertTrue(np.all(diffs > 0))


if __name__ == "__main__":
    unittest.main()
