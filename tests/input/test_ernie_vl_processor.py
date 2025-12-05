import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
from fastdeploy.input.ernie4_5_vl_processor import Ernie4_5_VLProcessor
from fastdeploy.input.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive import (
    AdaptiveImageProcessor,
)
from fastdeploy.input.ernie4_5_vl_processor.process import DataProcessor
from fastdeploy.input.utils import IDS_TYPE_FLAG


class TestErnie4_5_vl_ProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # Create mock object for Ernie4_5Processor instance
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5_VLProcessor("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        # Set necessary attributes
        self.processor.tokenizer = MagicMock()
        self.processor.tokenizer.eos_token_id = 1
        self.processor.decode_status = {}
        self.processor.reasoning_end_dict = {}
        self.processor.tool_parser_dict = {}
        self.processor.generation_config = MagicMock()
        self.processor.eos_token_ids = [1]
        self.processor.reasoning_parser = MagicMock()
        self.processor._check_mm_limits = MagicMock()
        self.processor.ernie4_5_processor = MagicMock()
        self.processor.pack_outputs = MagicMock()

        # Mock ids2tokens method
        def mock_ids2tokens(token_ids, task_id):
            self.processor.decode_status[task_id] = "mock_decode_status"
            return "delta_text", [2, 3], "previous_texts"

        self.processor.ids2tokens = mock_ids2tokens

        def mock_messages2ids(request, **kwargs):
            if "chat_template" in kwargs:
                return [1]
            else:
                return [0]

        def mock_apply_default_parameters(request):
            return request

        self.processor._apply_default_parameters = mock_apply_default_parameters

        # Mock reasoning parser
        self.mock_reasoning_parser = MagicMock()
        self.mock_reasoning_parser.__class__.__name__ = "ErnieX1ReasoningParser"
        # self.mock_reasoning_parser.extract_reasoning_content_streaming.return_value = ("reasoning", "text")
        self.processor.reasoning_parser = self.mock_reasoning_parser

        # Mock tool parser
        self.mock_tool_parser = MagicMock()
        self.mock_tool_parser.extract_tool_calls_streaming.return_value = None
        self.mock_tool_parser_obj = MagicMock()
        self.mock_tool_parser_obj.return_value = self.mock_tool_parser
        self.processor.tool_parser_obj = self.mock_tool_parser_obj

    def test_process_request_dict_with_options(self):
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "open"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "close"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "false"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "123"}},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "close"}},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "false"}},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)


class TestDataProcessorTargetMethods(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock(spec=Ernie4_5Tokenizer)
        self.mock_tokenizer.ignored_index = -100
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = self._mock_convert_tokens_to_ids
        self.mock_tokenizer.chat_template = "mock_template"
        self.mock_tokenizer.apply_chat_template.return_value = "User: Hello<|image@placeholder|>"
        # Mock encode method for _add_text
        self.mock_tokenizer.encode = MagicMock(return_value={"input_ids": [1, 2, 3]})

        def mock_load_tokenizer(dp_instance):
            dp_instance.tokenizer = self.mock_tokenizer

        with patch.object(DataProcessor, "_load_tokenizer", side_effect=mock_load_tokenizer, autospec=True):
            with patch.object(AdaptiveImageProcessor, "from_pretrained") as mock_image_preprocessor:
                mock_image_preprocessor.return_value = MagicMock()
                self.data_processor = DataProcessor(
                    tokenizer_name="mock_tokenizer",
                    image_preprocessor_name="mock_image_preprocessor",
                    enable_processor_cache=False,
                )
        self.data_processor.image_patch_id = 1001
        self.data_processor.image_start_id = 1002
        self.data_processor.image_end_id = 1003
        self.data_processor.video_start_id = 1004
        self.data_processor.video_end_id = 1005
        self.data_processor.role_prefixes = {"user": "User: ", "assistant": "Assistant: "}
        self.data_processor.enable_processor_cache = False
        # Note: extract_mm_items is not mocked by default, only when needed
        self.data_processor.extract_mm_items = MagicMock(return_value=([], [], [], [], None, [], []))

    def _restore_real_extract_mm_items(self):
        """Helper method to restore real extract_mm_items method for testing"""
        from fastdeploy.input.ernie4_5_vl_processor.process import DataProcessor

        original_extract_mm_items = DataProcessor.extract_mm_items
        self.data_processor.extract_mm_items = original_extract_mm_items.__get__(self.data_processor, DataProcessor)

    def _mock_convert_tokens_to_ids(self, token):
        token_id_map = {
            "<|begin_of_sentence|>": 101,
            "<|end_of_sentence|>": 102,
            "</s>": 103,
            "<|IMAGE_PLACEHOLDER|>": 1001,
            "<|IMAGE_START|>": 1002,
            "<|IMAGE_END|>": 1003,
            "<|VIDEO_START|>": 1004,
            "<|VIDEO_END|>": 1005,
        }
        return token_id_map.get(token, 999)

    def test_prompt_token_ids2outputs_only_prompt_token_ids(self):
        test_prompt_token_ids = [101, 999, 998, 997, 102]
        request = {
            "prompt_token_ids": test_prompt_token_ids,
        }

        outputs = self.data_processor.prompt_token_ids2outputs(request)

        prompt_len = len(test_prompt_token_ids)

        self.assertEqual(
            outputs["input_ids"],
            test_prompt_token_ids,
            f"input_ids mismatch: actual {outputs['input_ids']}, expected {test_prompt_token_ids}",
        )

        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * prompt_len)

        expected_position_ids = [[i] * 3 for i in range(prompt_len)]
        self.assertEqual(outputs["position_ids"], expected_position_ids)

        self.assertEqual(outputs["cur_position"], prompt_len)

        self.assertEqual(len(outputs["images"]), 0)
        self.assertEqual(len(outputs["grid_thw"]), 0)
        self.assertEqual(len(outputs["mm_positions"]), 0)
        self.assertEqual(len(outputs["mm_hashes"]), 0)
        self.assertEqual(outputs["video_cnt"], 0)
        self.assertEqual(outputs["num_input_image_tokens"], 0)
        self.assertEqual(outputs["num_input_video_tokens"], 0)

    def test_prompt_token_ids2outputs_with_messages_no_mm(self):
        test_prompt_token_ids = [101, 999, 998, 997, 102]
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [{"role": "user", "content": "Hello World"}],
        }

        self.data_processor.extract_mm_items.return_value = ([], [], [], [], None, [], [])

        outputs = self.data_processor.prompt_token_ids2outputs(request)

        prompt_len = len(test_prompt_token_ids)

        self.assertEqual(outputs["input_ids"], test_prompt_token_ids)

        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * prompt_len)

        expected_position_ids = [[i] * 3 for i in range(prompt_len)]
        self.assertEqual(outputs["position_ids"], expected_position_ids)

        self.assertEqual(outputs["cur_position"], prompt_len)

        self.assertEqual(len(outputs["images"]), 0)
        self.assertEqual(outputs["video_cnt"], 0)
        self.assertEqual(outputs["num_input_image_tokens"], 0)

    def test_prompt_token_ids2outputs_add_image(self):
        test_prompt_token_ids = [101, 1002, 1001, 1001, 1003, 102]
        mock_img = MagicMock()
        mock_img.height = 224
        mock_img.width = 224
        mock_img.convert.return_value = mock_img
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": mock_img, "uuid": "img_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [mock_img],
            [],
            ["img_uuid"],
            [],
            None,
            [],
            [{"type": "image", "data": mock_img}],
        )
        mock_resize = (None, (2, 4))
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = mock_resize
        mock_preprocess = {"pixel_values": np.random.randn(1, 16, 16, 3), "image_grid_thw": np.array([[2, 4]])}
        self.data_processor.image_preprocessor.preprocess.return_value = mock_preprocess
        # self.data_processor._compute_3d_positions = MagicMock(return_value=[[i]*3 for i in range(4)])
        outputs = self.data_processor.prompt_token_ids2outputs(request)
        self.assertEqual(outputs["input_ids"], [101, 1002, 1001, 1001, 1003, 102])
        self.assertEqual(
            outputs["token_type_ids"],
            [
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["image"],
                IDS_TYPE_FLAG["image"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
            ],
        )
        self.assertEqual(len(outputs["position_ids"]), 6)
        self.assertEqual(outputs["cur_position"], 6)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertIsNotNone(outputs["images"][0])
        self.assertEqual(outputs["num_input_image_tokens"], 2)
        self.assertEqual(len(outputs["mm_positions"]), 1)
        self.assertEqual(len(outputs["mm_hashes"]), 1)
        self.assertEqual(len(outputs["grid_thw"]), 1)
        self.assertEqual(len(outputs["image_type_ids"]), 1)

    def test_prompt_token_ids2outputs_add_processed_image(self):
        test_prompt_token_ids = [101, 1002, 1001, 1001, 1003, 102]
        mock_img_data = np.random.randn(8, 28, 28)
        mock_img_cache = (mock_img_data, {"thw": (1, 8, 8)})
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": mock_img_cache, "uuid": "img_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [mock_img_cache],
            [],
            ["img_uuid"],
            [],
            None,
            [],
            [{"type": "image", "data": mock_img_cache}],
        )
        outputs = self.data_processor.prompt_token_ids2outputs(request)
        self.assertEqual(outputs["input_ids"], [101, 1002, 1001, 1001, 1003, 102])
        self.assertEqual(
            outputs["token_type_ids"],
            [
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["image"],
                IDS_TYPE_FLAG["image"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
            ],
        )
        self.assertEqual(len(outputs["position_ids"]), 20)
        self.assertEqual(outputs["cur_position"], 8)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertIsNotNone(outputs["images"][0])
        self.assertEqual(len(outputs["mm_positions"]), 1)
        self.assertEqual(outputs["mm_hashes"][0], "img_uuid")
        self.assertEqual(len(outputs["grid_thw"]), 1)
        self.assertEqual(len(outputs["image_type_ids"]), 1)

    def test_prompt_token_ids2outputs_add_video(self):
        test_prompt_token_ids = [101, 1004, 1001, 1001, 1001, 1001, 1005, 102]
        mock_frame1 = MagicMock()
        mock_frame1.height = 224
        mock_frame1.width = 224
        mock_frame1.convert.return_value = mock_frame1
        mock_frame2 = MagicMock()
        mock_frame2.height = 224
        mock_frame2.width = 224
        mock_frame2.convert.return_value = mock_frame2
        frames = [mock_frame1, mock_frame2]
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "video_url", "video_url": frames, "uuid": "vid_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [],
            [frames],
            [],
            ["vid_uuid"],
            None,
            [],
            [{"type": "video", "data": frames}],
        )
        self.data_processor._load_and_process_video = MagicMock(return_value=frames)
        patches_h, patches_w = 4, 4
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (None, (patches_h, patches_w))
        mock_preprocess = {
            "pixel_values_videos": np.random.randn(2, patches_h, patches_w, 3),
            "video_grid_thw": np.array([[patches_h, patches_w]] * 2),
        }
        self.data_processor.image_preprocessor.preprocess.return_value = mock_preprocess
        outputs = self.data_processor.prompt_token_ids2outputs(request)
        self.assertEqual(outputs["input_ids"], [101, 1004, 1001, 1001, 1001, 1001, 1005, 102])
        self.assertEqual(
            outputs["token_type_ids"],
            [
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
            ],
        )
        self.assertEqual(len(outputs["position_ids"]), 8)
        self.assertEqual(outputs["cur_position"], 6)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertIsNotNone(outputs["images"][0])
        self.assertEqual(len(outputs["mm_positions"]), 1)
        self.assertEqual(outputs["mm_hashes"][0], "vid_uuid")
        self.assertEqual(len(outputs["grid_thw"]), 1)
        self.assertEqual(len(outputs["image_type_ids"]), 2)
        self.assertEqual(outputs["num_input_video_tokens"], 4)

    def test_prompt_token_ids2outputs_add_processed_video(self):
        test_prompt_token_ids = [101, 1004, 1001, 1001, 1001, 1001, 1005, 102]
        t, h, w = 2, 4, 4
        spatial_conv_size = self.data_processor.spatial_conv_size
        temporal_conv_size = self.data_processor.temporal_conv_size
        token_per_frame = (h // spatial_conv_size) * (w // spatial_conv_size)
        num_tokens = (t // temporal_conv_size) * token_per_frame
        mock_frames_data = np.random.randn(num_tokens * spatial_conv_size**2 * temporal_conv_size, 28, 28)
        mock_frames_cache = (mock_frames_data, {"thw": (t, h, w)})
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "video", "data": mock_frames_cache, "uuid": "vid_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [],
            [mock_frames_cache],
            [],
            ["vid_uuid"],
            None,
            [],
            [{"type": "video", "data": mock_frames_cache}],
        )
        outputs = self.data_processor.prompt_token_ids2outputs(request)
        self.assertEqual(outputs["input_ids"], [101, 1004, 1001, 1001, 1001, 1001, 1005, 102])
        self.assertEqual(
            outputs["token_type_ids"],
            [
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["video"],
                IDS_TYPE_FLAG["text"],
                IDS_TYPE_FLAG["text"],
            ],
        )
        self.assertEqual(len(outputs["position_ids"]), 8)
        self.assertEqual(outputs["cur_position"], 6)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertIsNotNone(outputs["images"][0])
        self.assertEqual(len(outputs["mm_positions"]), 1)
        self.assertEqual(outputs["mm_hashes"][0], "vid_uuid")
        self.assertEqual(len(outputs["grid_thw"]), 1)
        self.assertEqual(len(outputs["image_type_ids"]), 2)

    def test_prompt_token_ids2outputs_add_image_token_len_mismatch(self):
        test_prompt_token_ids = [101, 1002, 1001, 1001, 1001, 1003, 102]
        mock_img = MagicMock()
        mock_img.height = 224
        mock_img.width = 224
        mock_img.convert.return_value = mock_img
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": mock_img, "uuid": "img_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [mock_img],
            [],
            ["img_uuid"],
            [],
            None,
            [],
            [{"type": "image", "data": mock_img}],
        )
        patches_h, patches_w = 8, 8
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (None, (patches_h, patches_w))
        mock_preprocess = {
            "pixel_values": np.random.randn(1, patches_h, patches_w, 3),
            "image_grid_thw": np.array([[patches_h, patches_w]]),
        }
        self.data_processor.image_preprocessor.preprocess.return_value = mock_preprocess
        with self.assertRaises(ValueError) as ctx:
            self.data_processor.prompt_token_ids2outputs(request)
        self.assertIn("image tokens num not match the size", str(ctx.exception))

    def test_prompt_token_ids2outputs_add_processed_image_token_len_mismatch(self):
        test_prompt_token_ids = [101, 1002, 1001, 1001, 1003, 102]
        spatial_conv_size = self.data_processor.spatial_conv_size
        num_tokens = 4
        mock_img_data = np.random.randn(num_tokens * (spatial_conv_size**2), 28, 28)
        mock_img_cache = (mock_img_data, {"thw": (1, 8, 8)})
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": mock_img_cache, "uuid": "img_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [mock_img_cache],
            [],
            ["img_uuid"],
            [],
            None,
            [],
            [{"type": "image", "data": mock_img_cache}],
        )
        with self.assertRaises(ValueError) as ctx:
            self.data_processor.prompt_token_ids2outputs(request)
        self.assertIn("image tokens num not match the size", str(ctx.exception))

    def test_prompt_token_ids2outputs_add_video_token_len_mismatch(self):
        test_prompt_token_ids = [101, 1004, 1001, 1001, 1005, 102]
        mock_frame1 = MagicMock()
        mock_frame1.height = 224
        mock_frame1.width = 224
        mock_frame1.convert.return_value = mock_frame1
        mock_frame2 = MagicMock()
        mock_frame2.height = 224
        mock_frame2.width = 224
        mock_frame2.convert.return_value = mock_frame2
        frames = [mock_frame1, mock_frame2]
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "video_url", "video_url": frames, "uuid": "vid_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [],
            [frames],
            [],
            ["vid_uuid"],
            None,
            [],
            [{"type": "video", "data": frames}],
        )
        self.data_processor._load_and_process_video = MagicMock(return_value=frames)
        patches_h, patches_w = 8, 8
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (None, (patches_h, patches_w))
        mock_preprocess = {
            "pixel_values_videos": np.random.randn(2, patches_h, patches_w, 3),
            "video_grid_thw": np.array([[patches_h, patches_w]] * 2),
        }
        self.data_processor.image_preprocessor.preprocess.return_value = mock_preprocess
        with self.assertRaises(ValueError) as ctx:
            self.data_processor.prompt_token_ids2outputs(request)
        self.assertIn("video tokens num not match the size", str(ctx.exception))

    def test_prompt_token_ids2outputs_add_processed_video_token_len_mismatch(self):
        test_prompt_token_ids = [101, 1004, 1001, 1005, 102]
        t, h, w = 2, 8, 8
        spatial_conv_size = self.data_processor.spatial_conv_size
        temporal_conv_size = self.data_processor.temporal_conv_size

        num_tokens = 4
        mock_frames_data = np.random.randn(num_tokens * spatial_conv_size**2 * temporal_conv_size, 28, 28)
        mock_frames_cache = (mock_frames_data, {"thw": (t, h, w)})
        request = {
            "prompt_token_ids": test_prompt_token_ids,
            "messages": [
                {"role": "user", "content": [{"type": "video", "data": mock_frames_cache, "uuid": "vid_uuid"}]}
            ],
        }
        self.data_processor.extract_mm_items.return_value = (
            [],
            [mock_frames_cache],
            [],
            ["vid_uuid"],
            None,
            [],
            [{"type": "video", "data": mock_frames_cache}],
        )
        with self.assertRaises(ValueError) as ctx:
            self.data_processor.prompt_token_ids2outputs(request)
        self.assertIn("video tokens num not match the size", str(ctx.exception))

    def test_extract_mm_items(self):
        """Test extract_mm_items with various scenarios: basic items, video, and missing data error"""
        self._restore_real_extract_mm_items()

        # Test basic multimodal items (image + video)
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image", "data": Image.new("RGB", (224, 224)), "uuid": "img1"},
                        {"type": "video", "data": [Image.new("RGB", (224, 224))], "uuid": "vid1"},
                    ],
                }
            ]
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = request["messages"]
            images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = (
                self.data_processor.extract_mm_items(request)
            )
            self.assertEqual(len(images), 1)
            self.assertEqual(len(videos), 1)
            self.assertEqual(image_uuid[0], "img1")
            self.assertEqual(video_uuid[0], "vid1")
            self.assertEqual(len(mm_items), 2)

        # Test missing data error when cache is disabled
        self.data_processor.enable_processor_cache = False
        request = {"messages": [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = request["messages"]
            with self.assertRaises(ValueError) as ctx:
                self.data_processor.extract_mm_items(request)
            self.assertIn("Missing items cannot be retrieved", str(ctx.exception))


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_tokenizer = MagicMock()

        def mock_convert_tokens_to_ids(x):
            if isinstance(x, list):
                return [hash(str(token)) % 10000 for token in x]
            return hash(str(x)) % 10000

        self.mock_tokenizer.convert_tokens_to_ids = MagicMock(side_effect=mock_convert_tokens_to_ids)
        self.mock_tokenizer.encode = MagicMock(return_value={"input_ids": [1, 2, 3]})
        self.mock_tokenizer.decode = MagicMock(return_value="decoded_text")
        self.mock_tokenizer.tokenize = MagicMock(return_value=["token1", "token2"])
        self.mock_tokenizer.ignored_index = -100
        self.mock_tokenizer.chat_template = MagicMock()
        self.mock_tokenizer.apply_chat_template = MagicMock(return_value="formatted_prompt")

        self.mock_image_preprocessor = MagicMock()
        self.mock_image_preprocessor.get_smarted_resize = MagicMock(return_value=((224, 224), (16, 16)))
        self.mock_image_preprocessor.preprocess = MagicMock(
            return_value={
                "pixel_values": np.random.rand(256, 3 * 14 * 14).astype(np.float32),
                "image_grid_thw": np.array([[1, 16, 16]]),
            }
        )
        self.mock_image_preprocessor.from_pretrained = MagicMock(return_value=self.mock_image_preprocessor)

        with patch(
            "fastdeploy.input.ernie4_5_vl_processor.process.AdaptiveImageProcessor",
            self.mock_image_preprocessor,
        ):
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.Ernie4_5Tokenizer") as mock_tokenizer_class:
                mock_tokenizer_class.from_pretrained = MagicMock(return_value=self.mock_tokenizer)
                mock_tokenizer_class.resource_files_names = {"vocab_file": "tokenizer.model"}
                with patch("os.path.exists", return_value=True):
                    self.processor = DataProcessor(
                        tokenizer_name="test_model",
                        image_preprocessor_name="test_model",
                    )

    def _create_outputs(self):
        """Helper to create outputs dict"""
        return {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "mm_positions": [],
            "mm_hashes": [],
            "cur_position": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
        }

    def _mock_video_processing(self, mock_frames=None):
        """Helper to mock video processing"""
        if mock_frames is None:
            mock_frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        mock_read = patch("fastdeploy.input.ernie4_5_vl_processor.process.read_video_decord")
        mock_frames_read = patch("fastdeploy.input.ernie4_5_vl_processor.process.read_frames_decord")
        mock_render = patch("fastdeploy.input.ernie4_5_vl_processor.process.render_frame_timestamp")
        return mock_read, mock_frames_read, mock_render, mock_frames

    def _setup_video_mocks(self, mock_read, mock_frames_read, mock_render, mock_frames):
        """Setup video processing mocks"""
        mock_read.return_value = (None, {"duration": 2.0}, "test_path")
        mock_frames_read.return_value = (
            [np.array(f) for f in mock_frames],
            None,
            [0.0, 0.5, 1.0, 1.5] if len(mock_frames) == 4 else [float(i) * 0.5 for i in range(len(mock_frames))],
        )
        mock_render.side_effect = lambda img, ts: (Image.fromarray(img) if isinstance(img, np.ndarray) else img)
        self.mock_image_preprocessor.preprocess.return_value = {
            "pixel_values_videos": np.random.rand(len(mock_frames), 256, 3 * 14 * 14).astype(np.float32),
            "video_grid_thw": np.array([[len(mock_frames), 16, 16]]),
        }

    def test_train_and_eval(self):
        """Test training and evaluation mode switching"""
        self.assertTrue(self.processor.is_training)
        self.processor.eval()
        self.assertFalse(self.processor.is_training)
        self.processor.train()
        self.assertTrue(self.processor.is_training)

    def test_build_token_type_mapping(self):
        """Test token type mapping construction"""
        mapping = self.processor._build_token_type_mapping()
        for token in [
            self.processor.IMG_START,
            self.processor.IMG_END,
            self.processor.VID_START,
            self.processor.VID_END,
        ]:
            self.assertEqual(mapping[token], IDS_TYPE_FLAG["image"])
        self.assertEqual(mapping[self.processor.image_patch_id], IDS_TYPE_FLAG["image"])

    def test_add_text_and_special_token(self):
        """Test adding text and special tokens"""
        outputs = self._create_outputs()
        self.processor._add_text("hello", outputs)
        self.assertEqual(len(outputs["input_ids"]), 3)
        self.assertEqual(outputs["cur_position"], 3)

        outputs2 = self._create_outputs()
        self.processor._add_text([1, 2, 3, 4, 5], outputs2)
        self.assertEqual(len(outputs2["input_ids"]), 5)

        outputs3 = self._create_outputs()
        self.processor._add_special_token("<|begin_of_sentence|>", outputs3)
        self.processor._add_special_token(12345, outputs3)
        self.assertEqual(len(outputs3["input_ids"]), 2)

    def test_compute_3d_positions(self):
        """Test 3D position computation"""
        pos_ids = self.processor._compute_3d_positions(t=2, h=16, w=16, start_idx=10)
        self.assertIsInstance(pos_ids, list)
        self.assertGreater(len(pos_ids), 0)
        self.assertEqual(len(pos_ids[0]), 3)

        pos_ids2 = self.processor._compute_3d_positions(t=1, h=16, w=16, start_idx=0)
        expected_len = 1 * (16 // self.processor.spatial_conv_size) ** 2
        self.assertEqual(len(pos_ids2), expected_len)

    def test_set_video_frame_args_comprehensive(self):
        """Test _set_video_frame_args with various scenarios"""
        # Valid cases
        result = self.processor._set_video_frame_args(
            {
                "target_frames": 32,
                "fps": -1,
                "min_frames": 16,
                "max_frames": 64,
                "frames_sample": "leading",
            },
            {"duration": 10.0},
        )
        self.assertEqual(result["target_frames"], 32)

        result = self.processor._set_video_frame_args(
            {
                "target_frames": -1,
                "fps": 2,
                "min_frames": 16,
                "max_frames": 64,
                "frames_sample": "leading",
            },
            {"duration": 10.0},
        )
        self.assertIsNotNone(result)

        # Error cases
        with self.assertRaises(ValueError):
            self.processor._set_video_frame_args(
                {
                    "target_frames": -1,
                    "fps": -1,
                    "min_frames": 16,
                    "max_frames": 64,
                    "frames_sample": "leading",
                },
                {"duration": 10.0},
            )
        with self.assertRaises(ValueError):
            self.processor._set_video_frame_args(
                {
                    "target_frames": 10,
                    "fps": 2,
                    "min_frames": 1,
                    "max_frames": 100,
                    "frames_sample": "leading",
                },
                {"duration": 10.0},
            )
        with self.assertRaises(ValueError):
            self.processor._set_video_frame_args(
                {
                    "target_frames": 5,
                    "fps": -1,
                    "min_frames": 10,
                    "max_frames": 100,
                    "frames_sample": "leading",
                },
                {"duration": 10.0},
            )
        with self.assertRaises(ValueError):
            self.processor._set_video_frame_args(
                {
                    "target_frames": 200,
                    "fps": -1,
                    "min_frames": 1,
                    "max_frames": 100,
                    "frames_sample": "leading",
                },
                {"duration": 10.0},
            )
        with self.assertRaises(ValueError):
            self.processor._set_video_frame_args(
                {
                    "target_frames": -1,
                    "fps": 2,
                    "min_frames": 100,
                    "max_frames": 10,
                    "frames_sample": "leading",
                },
                {"duration": 10.0},
            )

        # Adjustment cases
        result = self.processor._set_video_frame_args(
            {
                "target_frames": -1,
                "fps": 1,
                "min_frames": 10,
                "max_frames": 100,
                "frames_sample": "leading",
            },
            {"duration": 1.0},
        )
        self.assertEqual(result["target_frames"], 10)
        self.assertEqual(result["fps"], -1)

        result = self.processor._set_video_frame_args(
            {
                "target_frames": -1,
                "fps": 10,
                "min_frames": 1,
                "max_frames": 100,
                "frames_sample": "leading",
            },
            {"duration": 100.0},
        )
        self.assertEqual(result["target_frames"], 100)
        self.assertEqual(result["fps"], -1)

    def test_text2ids_comprehensive(self):
        """Test text2ids with various scenarios"""
        # Text only
        outputs = self.processor.text2ids("Hello world")
        self.assertIn("input_ids", outputs)
        self.assertEqual(len(outputs["images"]), 0)

        # Empty text
        outputs = self.processor.text2ids("")
        self.assertEqual(len(outputs["input_ids"]), 0)

        # With image placeholder
        mock_image = Image.new("RGB", (224, 224))
        outputs = self.processor.text2ids("Hello <|image@placeholder|> world", images=[mock_image])
        self.assertGreater(len(outputs["input_ids"]), 0)
        self.assertGreater(len(outputs["images"]), 0)

        # With cached image
        cached_image = (
            np.random.rand(256, 3 * 14 * 14).astype(np.float32),
            {"thw": (1, 16, 16)},
        )
        outputs = self.processor.text2ids(
            "Hello <|image@placeholder|> world",
            images=[cached_image],
            image_uuid=["uuid"],
        )
        self.assertGreater(len(outputs["input_ids"]), 0)

        # Multiple images
        outputs = self.processor.text2ids(
            "Hello <|image@placeholder|> world <|image@placeholder|> end",
            images=[mock_image, mock_image],
        )
        self.assertEqual(len(outputs["images"]), 2)

        # With video placeholder
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = (
                [np.array(f) for f in mock_frames],
                None,
                [0.0, 0.5, 1.0, 1.5],
            )
            mren.side_effect = lambda img, ts: (Image.fromarray(img) if isinstance(img, np.ndarray) else img)
            self.mock_image_preprocessor.preprocess.return_value = {
                "pixel_values_videos": np.random.rand(4, 256, 3 * 14 * 14).astype(np.float32),
                "video_grid_thw": np.array([[4, 16, 16]]),
            }
            outputs = self.processor.text2ids("Hello <|video@placeholder|> world", videos=["test_video.mp4"])
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Cached video
        cached_video = (
            np.random.rand(256, 3 * 14 * 14).astype(np.float32),
            {"thw": (4, 16, 16)},
        )
        outputs = self.processor.text2ids(
            "Hello <|video@placeholder|> world",
            videos=[cached_video],
            video_uuid=["uuid"],
        )
        self.assertGreater(len(outputs["input_ids"]), 0)

        # Video dict format
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = (
                [np.array(f) for f in mock_frames],
                None,
                [0.0, 0.5, 1.0, 1.5],
            )
            mren.side_effect = lambda img, ts: (Image.fromarray(img) if isinstance(img, np.ndarray) else img)
            self.mock_image_preprocessor.preprocess.return_value = {
                "pixel_values_videos": np.random.rand(4, 256, 3 * 14 * 14).astype(np.float32),
                "video_grid_thw": np.array([[4, 16, 16]]),
            }
            outputs = self.processor.text2ids(
                "Hello <|video@placeholder|> world",
                videos=[{"video": "test.mp4", "fps": 2}],
            )
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Image and video together
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = (
                [np.array(f) for f in mock_frames],
                None,
                [0.0, 0.5, 1.0, 1.5],
            )
            mren.side_effect = lambda img, ts: (Image.fromarray(img) if isinstance(img, np.ndarray) else img)
            self.mock_image_preprocessor.preprocess.side_effect = [
                {
                    "pixel_values": np.random.rand(256, 3 * 14 * 14).astype(np.float32),
                    "image_grid_thw": np.array([[1, 16, 16]]),
                },
                {
                    "pixel_values_videos": np.random.rand(4, 256, 3 * 14 * 14).astype(np.float32),
                    "video_grid_thw": np.array([[4, 16, 16]]),
                },
            ]
            outputs = self.processor.text2ids(
                "Hello <|image@placeholder|> world <|video@placeholder|> end",
                images=[mock_image],
                videos=["test_video.mp4"],
            )
            self.assertGreater(len(outputs["input_ids"]), 0)
            self.mock_image_preprocessor.preprocess.side_effect = None

    def test_request2ids_comprehensive(self):
        """Test request2ids with various scenarios"""
        self.processor.is_training = False

        # Basic request with multimodal content - covers both text and image branches in one call
        mock_image = Image.new("RGB", (224, 224))
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image", "data": mock_image, "uuid": "img1"},
                    ],
                }
            ],
            "add_generation_prompt": True,
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = request["messages"]
            outputs = self.processor.request2ids(request)
            self.assertIn("input_ids", outputs)

        # Error case: missing chat_template
        self.processor.tokenizer.chat_template = None
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            with self.assertRaises(ValueError):
                self.processor.request2ids(request)
        self.processor.tokenizer.chat_template = MagicMock()

        # Error case: unsupported role
        request = {
            "messages": [{"role": "invalid_role", "content": "Hello"}],
            "add_generation_prompt": True,
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "invalid_role", "content": [{"type": "text", "text": "Hello"}]}]
            with self.assertRaises(AssertionError):
                self.processor.request2ids(request)

        # Error case: missing cache when cache is disabled
        self.processor.enable_processor_cache = False
        request = {"messages": [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = request["messages"]
            with self.assertRaises(ValueError):
                self.processor.request2ids(request)

    def test_extract_labels(self):
        """Test label extraction"""
        outputs = {"input_ids": [1, 2, 3, self.processor.sep_token_id, 4, 5], "labels": []}
        self.processor.is_training = True
        self.processor._extract_labels(outputs, ["target text"])
        self.assertEqual(len(outputs["labels"]), len(outputs["input_ids"]))

        # Multiple targets
        outputs2 = {
            "input_ids": [1, 2, 3, self.processor.sep_token_id, 4, 5, self.processor.sep_token_id, 6, 7],
            "labels": [],
        }
        self.processor._extract_labels(outputs2, ["target1", "target2"])
        self.assertEqual(len(outputs2["labels"]), len(outputs2["input_ids"]))

        # Error case
        outputs3 = {"input_ids": [1, 2, 3, self.processor.sep_token_id], "labels": []}
        with self.assertRaises(AssertionError):
            self.processor._extract_labels(outputs3, ["target1", "target2"])

    def test_fancy_print(self):
        """Test fancy_print function"""
        from fastdeploy.input.ernie4_5_vl_processor.process import fancy_print

        test_cases = [
            ([1, 2, 3, self.processor.image_patch_id, 4, 5], self.processor.image_patch_id, None),
            (
                [
                    1,
                    2,
                    self.processor.image_patch_id,
                    self.processor.image_patch_id,
                    self.processor.image_patch_id,
                    4,
                    5,
                ],
                self.processor.image_patch_id,
                "<|IMAGE@",
            ),
            ([1, 2, 3, 4, 5], self.processor.image_patch_id, None),
        ]
        for input_ids, image_patch_id, expected_contains in test_cases:
            result = fancy_print(input_ids, self.mock_tokenizer, image_patch_id)
            self.assertIsInstance(result, str)
            if expected_contains:
                self.assertIn(expected_contains, result)

    def test_processor_cache_operations(self):
        """Test processor cache get/update and request2ids with cache"""
        # Test get_processor_cache
        mock_socket = MagicMock()
        mock_socket.recv_multipart = MagicMock(return_value=(b"", b"pickled_data"))
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.pickle") as mock_pickle:
            mock_pickle.loads = MagicMock(return_value=[{"data": "cached_item"}])
            result = self.processor.get_processor_cache(mock_socket, ["hash1", "hash2"])
            self.assertEqual(len(result), 1)

        # Test update_processor_cache
        mock_socket2 = MagicMock()
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.pickle"):
            self.processor.update_processor_cache(
                mock_socket2,
                ["hash1"],
                [(np.array([1, 2, 3]), {"meta": "data"})],
            )
            mock_socket2.send_multipart.assert_called_once()

        # Test request2ids with processor cache update
        self.processor.is_training = False
        self.processor.enable_processor_cache = True
        mock_image = Image.new("RGB", (224, 224))
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image", "data": mock_image, "uuid": "img1"},
                    ],
                }
            ],
            "add_generation_prompt": True,
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_socket.recv_multipart = MagicMock(return_value=(b"", b"pickled_data"))
            mock_context.socket.return_value = mock_socket
            mock_zmq.Context.return_value = mock_context
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
                mock_parse.return_value = request["messages"]
                with patch("fastdeploy.input.ernie4_5_vl_processor.process.pickle") as mock_pickle:
                    mock_pickle.loads = MagicMock(return_value=[])
                    with patch.object(self.processor, "text2ids") as mock_text2ids:
                        mock_text2ids.return_value = {
                            "input_ids": [1, 2, 3],
                            "token_type_ids": [0] * 3,
                            "position_ids": [[i] * 3 for i in range(3)],
                            "images": [np.random.rand(256, 3 * 14 * 14).astype(np.float32)],
                            "grid_thw": [np.array([[1, 16, 16]])],
                            "image_type_ids": [0],
                            "cur_position": 3,
                            "video_cnt": 0,
                            "num_input_image_tokens": 0,
                            "num_input_video_tokens": 0,
                            "mm_positions": [],
                            "mm_hashes": ["hash1"],
                        }
                        with patch.object(self.processor, "update_processor_cache") as mock_update:
                            self.processor.request2ids(request)
                            mock_update.assert_called_once()
        self.processor.enable_processor_cache = False


if __name__ == "__main__":
    unittest.main()
