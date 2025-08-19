import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from fastdeploy.engine.request import Request
from fastdeploy.input.qwen_mm_processor import DataProcessor
from fastdeploy.input.qwen_vl_processor import QwenVLProcessor


def mock_pil_image(height, width):
    rgb_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(rgb_image)


def mock_parse_chat_messages():
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
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frames.append(frame)
    return np.stack(frames, axis=0)


def mock_load_and_process_video():
    frames = mock_video_frames(num_frames=3, height=480, width=640)
    meta = {
        "fps": 1,
        "duration": 3,
        "num_of_frame": 3,
    }
    return frames, meta


class TestQwenVLProcessor(unittest.TestCase):

    def setUp(self):
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
            "video_max_frames": 20,
            "video_min_frames": 1,
        }
        limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}

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
        self.patcher_parse_chat_messages.stop()
        self.patcher_load_and_process_video.stop()

    def test_process_request(self):
        prompt = {
            "request_id": "123",
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
        num_generated_token_ids = 10
        request = {
            "metadata": {
                "generated_token_ids": [1] * num_generated_token_ids,
            },
            "stop": ["stop", "eof"],
            "request_id": "123",
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


if __name__ == "__main__":
    unittest.main()
