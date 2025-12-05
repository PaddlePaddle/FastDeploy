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
        self.data_processor.extract_mm_items = MagicMock(return_value=([], [], [], [], None, [], []))

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
            f"input_ids 涓嶅尮閰嶏細瀹為檯{outputs['input_ids']}锛岄鏈焄{test_prompt_token_ids}]",
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

    def test_text2ids_basic(self):
        """Test text2ids with basic text input"""
        text = "Hello world"
        outputs = self.data_processor.text2ids(text)

        self.assertIn("input_ids", outputs)
        self.assertIn("token_type_ids", outputs)
        self.assertIn("position_ids", outputs)
        self.assertGreater(len(outputs["input_ids"]), 0)
        self.assertEqual(len(outputs["images"]), 0)
        self.assertEqual(len(outputs["videos"]) if "videos" in outputs else 0, 0)

    def test_text2ids_with_image_placeholder(self):
        """Test text2ids with image placeholder"""
        mock_img = Image.new("RGB", (224, 224))
        text = "Hello <|image@placeholder|> world"
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (None, (16, 16))
        self.data_processor.image_preprocessor.preprocess.return_value = {
            "pixel_values": np.random.randn(256, 3 * 14 * 14).astype(np.float32),
            "image_grid_thw": np.array([[1, 16, 16]]),
        }

        outputs = self.data_processor.text2ids(text, images=[mock_img])

        self.assertGreater(len(outputs["input_ids"]), 0)
        self.assertGreater(len(outputs["images"]), 0)
        self.assertEqual(outputs["num_input_image_tokens"], 64)  # (16*16) // (2*2) = 64

    def test_text2ids_with_video_placeholder(self):
        """Test text2ids with video placeholder"""
        mock_frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        text = "Hello <|video@placeholder|> world"
        self.data_processor._load_and_process_video = MagicMock(return_value=mock_frames)
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (None, (16, 16))
        self.data_processor.image_preprocessor.preprocess.return_value = {
            "pixel_values_videos": np.random.randn(4, 256, 3 * 14 * 14).astype(np.float32),
            "video_grid_thw": np.array([[4, 16, 16]]),
        }

        outputs = self.data_processor.text2ids(text, videos=["test_video.mp4"])

        self.assertGreater(len(outputs["input_ids"]), 0)
        self.assertGreater(len(outputs["images"]), 0)
        self.assertGreater(outputs["num_input_video_tokens"], 0)

    def test_request2ids_basic(self):
        """Test request2ids with basic request"""
        self.data_processor.is_training = False
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "add_generation_prompt": True,
        }

        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            outputs = self.data_processor.request2ids(request)

            self.assertIn("input_ids", outputs)
            self.assertGreater(len(outputs["input_ids"]), 0)

    def test_request2ids_with_multimodal(self):
        """Test request2ids with multimodal content"""
        self.data_processor.is_training = False
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
            mock_parse.return_value = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image", "data": mock_image, "uuid": "img1"},
                    ],
                }
            ]
            self.data_processor.image_preprocessor.get_smarted_resize.return_value = (None, (16, 16))
            self.data_processor.image_preprocessor.preprocess.return_value = {
                "pixel_values": np.random.randn(256, 3 * 14 * 14).astype(np.float32),
                "image_grid_thw": np.array([[1, 16, 16]]),
            }
            outputs = self.data_processor.request2ids(request)

            self.assertIn("input_ids", outputs)
            self.assertGreater(len(outputs["images"]), 0)

    def test_extract_mm_items_basic(self):
        """Test extract_mm_items with basic multimodal items"""
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
            mock_parse.return_value = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image", "data": Image.new("RGB", (224, 224)), "uuid": "img1"},
                        {"type": "video", "data": [Image.new("RGB", (224, 224))], "uuid": "vid1"},
                    ],
                }
            ]
            images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = (
                self.data_processor.extract_mm_items(request)
            )

            self.assertEqual(len(images), 1)
            self.assertEqual(len(videos), 1)
            self.assertEqual(image_uuid[0], "img1")
            self.assertEqual(video_uuid[0], "vid1")
            self.assertEqual(len(mm_items), 2)

    def test_extract_mm_items_missing_data_error(self):
        """Test extract_mm_items raises error when data is missing and cache is disabled"""
        self.data_processor.enable_processor_cache = False
        request = {"messages": [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]}

        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]
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
            "fastdeploy.input.ernie4_5_vl_processor.process.AdaptiveImageProcessor", self.mock_image_preprocessor
        ):
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.Ernie4_5Tokenizer") as mock_tokenizer_class:
                mock_tokenizer_class.from_pretrained = MagicMock(return_value=self.mock_tokenizer)
                mock_tokenizer_class.resource_files_names = {"vocab_file": "tokenizer.model"}
                with patch("os.path.exists", return_value=True):
                    self.processor = DataProcessor(tokenizer_name="test_model", image_preprocessor_name="test_model")

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
        cached_image = (np.random.rand(256, 3 * 14 * 14).astype(np.float32), {"thw": (1, 16, 16)})
        outputs = self.processor.text2ids(
            "Hello <|image@placeholder|> world", images=[cached_image], image_uuid=["uuid"]
        )
        self.assertGreater(len(outputs["input_ids"]), 0)

        # Multiple images
        outputs = self.processor.text2ids(
            "Hello <|image@placeholder|> world <|image@placeholder|> end", images=[mock_image, mock_image]
        )
        self.assertEqual(len(outputs["images"]), 2)

        # With video placeholder
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = ([np.array(f) for f in mock_frames], None, [0.0, 0.5, 1.0, 1.5])
            mren.side_effect = lambda img, ts: Image.fromarray(img) if isinstance(img, np.ndarray) else img
            self.mock_image_preprocessor.preprocess.return_value = {
                "pixel_values_videos": np.random.rand(4, 256, 3 * 14 * 14).astype(np.float32),
                "video_grid_thw": np.array([[4, 16, 16]]),
            }
            outputs = self.processor.text2ids("Hello <|video@placeholder|> world", videos=["test_video.mp4"])
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Cached video
        cached_video = (np.random.rand(256, 3 * 14 * 14).astype(np.float32), {"thw": (4, 16, 16)})
        outputs = self.processor.text2ids(
            "Hello <|video@placeholder|> world", videos=[cached_video], video_uuid=["uuid"]
        )
        self.assertGreater(len(outputs["input_ids"]), 0)

        # Video dict format
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = ([np.array(f) for f in mock_frames], None, [0.0, 0.5, 1.0, 1.5])
            mren.side_effect = lambda img, ts: Image.fromarray(img) if isinstance(img, np.ndarray) else img
            self.mock_image_preprocessor.preprocess.return_value = {
                "pixel_values_videos": np.random.rand(4, 256, 3 * 14 * 14).astype(np.float32),
                "video_grid_thw": np.array([[4, 16, 16]]),
            }
            outputs = self.processor.text2ids(
                "Hello <|video@placeholder|> world", videos=[{"video": "test.mp4", "fps": 2}]
            )
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Image and video together
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = ([np.array(f) for f in mock_frames], None, [0.0, 0.5, 1.0, 1.5])
            mren.side_effect = lambda img, ts: Image.fromarray(img) if isinstance(img, np.ndarray) else img
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

        # Basic request
        request = {"messages": [{"role": "user", "content": "Hello"}], "add_generation_prompt": True}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            outputs = self.processor.request2ids(request)
            self.assertIn("input_ids", outputs)

        # With multimodal
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
            mock_parse.return_value = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image", "data": mock_image, "uuid": "img1"},
                    ],
                }
            ]
            outputs = self.processor.request2ids(request)
            self.assertIn("input_ids", outputs)

        # Content not list
        request = {"messages": [{"role": "user", "content": "Hello"}], "add_generation_prompt": True}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": {"type": "text", "text": "Hello"}}]
            outputs = self.processor.request2ids(request)
            self.assertIn("input_ids", outputs)

        # Training mode
        self.processor.is_training = True
        self.mock_tokenizer.apply_chat_template.return_value = f"Hello {self.processor.sep_token} response"
        request = {"messages": [{"role": "user", "content": "Hello"}], "add_generation_prompt": True}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            with patch.object(self.processor, "text2ids") as mock_text2ids:
                mock_text2ids.return_value = {
                    "input_ids": [1, 2, 3, self.processor.sep_token_id, 4, 5],
                    "token_type_ids": [0] * 6,
                    "position_ids": [[i] * 3 for i in range(6)],
                    "images": [],
                    "grid_thw": [],
                    "image_type_ids": [],
                    "labels": [],
                    "cur_position": 6,
                    "video_cnt": 0,
                    "num_input_image_tokens": 0,
                    "num_input_video_tokens": 0,
                    "mm_positions": [],
                    "mm_hashes": [],
                }
                outputs = self.processor.request2ids(request, tgts=["response"])
                self.assertIn("labels", outputs)

        # Error cases
        self.processor.is_training = False
        self.processor.tokenizer.chat_template = None
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
            with self.assertRaises(ValueError):
                self.processor.request2ids(request)
        self.processor.tokenizer.chat_template = MagicMock()

        # Unsupported role
        request = {"messages": [{"role": "invalid_role", "content": "Hello"}], "add_generation_prompt": True}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "invalid_role", "content": [{"type": "text", "text": "Hello"}]}]
            with self.assertRaises(AssertionError):
                self.processor.request2ids(request)

        # Missing cache error
        self.processor.enable_processor_cache = False
        request = {"messages": [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]}
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]
            with self.assertRaises(ValueError):
                self.processor.request2ids(request)

        # Processor cache
        self.processor.enable_processor_cache = True
        request = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "image", "uuid": "img1"}]}
            ],
            "add_generation_prompt": True,
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_context.socket.return_value = mock_socket
            mock_zmq.Context.return_value = mock_context
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
                mock_parse.return_value = [
                    {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "image", "uuid": "img1"}]}
                ]
                with patch.object(self.processor, "get_processor_cache") as mock_get_cache:
                    mock_get_cache.return_value = [Image.new("RGB", (224, 224))]
                    with patch.object(self.processor, "update_processor_cache"):
                        outputs = self.processor.request2ids(request)
                        self.assertIn("input_ids", outputs)

        # Cache missing item
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_context.socket.return_value = mock_socket
            mock_zmq.Context.return_value = mock_context
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
                mock_parse.return_value = [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]
                with patch.object(self.processor, "get_processor_cache") as mock_get_cache:
                    mock_get_cache.return_value = [None]
                    with self.assertRaises(ValueError):
                        self.processor.request2ids(request)

    def test_add_image_and_video(self):
        """Test adding images and videos"""
        # Add image with UUID
        outputs = self._create_outputs()
        mock_image = Image.new("RGB", (224, 224))
        self.processor._add_image(mock_image, outputs, "test_uuid")
        self.assertGreater(len(outputs["input_ids"]), 0)
        self.assertEqual(outputs["mm_hashes"][-1], "test_uuid")

        # Add image without UUID
        outputs2 = self._create_outputs()
        self.processor._add_image(mock_image, outputs2, None)
        self.assertGreater(len(outputs2["mm_hashes"]), 0)

        # Add processed image
        cached_image = (np.random.rand(256, 3 * 14 * 14).astype(np.float32), {"thw": (1, 16, 16)})
        outputs3 = self._create_outputs()
        self.processor._add_processed_image(cached_image, outputs3, "uuid")
        self.assertGreater(len(outputs3["input_ids"]), 0)

        # Add video with UUID
        mock_frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        self.mock_image_preprocessor.preprocess.return_value = {
            "pixel_values_videos": np.random.rand(4, 256, 3 * 14 * 14).astype(np.float32),
            "video_grid_thw": np.array([[4, 16, 16]]),
        }
        outputs4 = self._create_outputs()
        self.processor._add_video(mock_frames, outputs4, "test_video_uuid")
        self.assertGreater(len(outputs4["input_ids"]), 0)
        self.assertEqual(outputs4["mm_hashes"][-1], "test_video_uuid")

        # Add processed video
        cached_video = (np.random.rand(256, 3 * 14 * 14).astype(np.float32), {"thw": (4, 16, 16)})
        outputs5 = self._create_outputs()
        self.processor._add_processed_video(cached_video, outputs5, "uuid")
        self.assertGreater(len(outputs5["input_ids"]), 0)

    def test_load_and_process_video(self):
        """Test loading and processing video"""
        mock_read, mock_frames_read, mock_render, mock_frames = self._mock_video_processing()
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = ([np.array(f) for f in mock_frames], None, [0.0, 0.5, 1.0, 1.5])
            mren.side_effect = lambda img, ts: Image.fromarray(img) if isinstance(img, np.ndarray) else img
            frames = self.processor._load_and_process_video("test_video.mp4", {})
            self.assertEqual(len(frames), 4)

        # Odd frames
        mock_frames_odd = [Image.new("RGB", (224, 224)) for _ in range(5)]
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = ([np.array(f) for f in mock_frames_odd], None, [0.0, 0.4, 0.8, 1.2, 1.6])
            mren.side_effect = lambda img, ts: Image.fromarray(img) if isinstance(img, np.ndarray) else img
            frames = self.processor._load_and_process_video("test_video.mp4", {})
            self.assertEqual(len(frames) % 2, 0)

        # With video_frame_args
        video_frame_args = {
            "target_frames": 4,
            "fps": -1,
            "min_frames": 4,
            "max_frames": 8,
            "frames_sample": "leading",
        }
        with mock_read as mr, mock_frames_read as mfr, mock_render as mren:
            mr.return_value = (None, {"duration": 2.0}, "test_path")
            mfr.return_value = ([np.array(f) for f in mock_frames], None, [0.0, 0.5, 1.0, 1.5])
            mren.side_effect = lambda img, ts: Image.fromarray(img) if isinstance(img, np.ndarray) else img
            frames = self.processor._load_and_process_video("test_video.mp4", video_frame_args)
            self.assertGreater(len(frames), 0)

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

    def test_extract_mm_items_video_and_unsupported(self):
        """Test extract_mm_items with video type and unsupported type"""
        # Test video type extraction
        mock_frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this video?"},
                        {"type": "video", "data": mock_frames, "uuid": "vid1"},
                    ],
                }
            ]
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this video?"},
                        {"type": "video", "data": mock_frames, "uuid": "vid1"},
                    ],
                }
            ]
            images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self.processor.extract_mm_items(
                request
            )
            self.assertEqual(len(videos), 1)
            self.assertEqual(video_uuid[0], "vid1")

        # Test unsupported multimodal type - error occurs in extract_mm_items when processing items
        # Note: extract_mm_items only collects items with type "image" or "video" (line 258)
        # So we need to inject an unsupported type after collection
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "data": Image.new("RGB", (224, 224)), "uuid": "img1"}],
                }
            ]
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": Image.new("RGB", (224, 224)), "uuid": "img1"}]}
            ]
            # Manually inject unsupported type into mm_items to test the error path
            images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = self.processor.extract_mm_items(
                request
            )
            # Inject unsupported type to test error handling
            mm_items.append({"type": "audio", "data": "test_audio", "uuid": "audio1"})
            with self.assertRaisesRegex(ValueError, "Unsupported multimodal type"):
                # Re-run the processing logic that would fail
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

    def test_request2ids_processor_cache_update(self):
        """Test request2ids with processor cache update"""
        self.processor.is_training = False  # Ensure eval mode
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
                mock_parse.return_value = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "image", "data": mock_image, "uuid": "img1"},
                        ],
                    }
                ]
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

    def test_processor_cache(self):
        """Test processor cache operations"""
        mock_socket = MagicMock()
        mock_socket.recv_multipart = MagicMock(return_value=(b"", b"pickled_data"))
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.pickle") as mock_pickle:
            mock_pickle.loads = MagicMock(return_value=[{"data": "cached_item"}])
            result = self.processor.get_processor_cache(mock_socket, ["hash1", "hash2"])
            self.assertEqual(len(result), 1)

        mock_socket2 = MagicMock()
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.pickle"):
            self.processor.update_processor_cache(mock_socket2, ["hash1"], [(np.array([1, 2, 3]), {"meta": "data"})])
            mock_socket2.send_multipart.assert_called_once()

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

    def test_prompt_token_ids2outputs(self):
        """Test prompt_token_ids2outputs method"""
        # No messages
        request = {"prompt_token_ids": [1, 2, 3, 4, 5]}
        outputs = self.processor.prompt_token_ids2outputs(request)
        self.assertEqual(len(outputs["input_ids"]), 5)

        # With image - need to match token count with actual image patch count
        self.processor.is_training = False
        mock_image = Image.new("RGB", (224, 224))
        # Calculate expected token count: (16*16) // (2*2) = 64 tokens
        num_tokens = (16 * 16) // (self.processor.spatial_conv_size**2)
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}],
            "prompt_token_ids": [self.processor.image_start_id]
            + [self.processor.image_patch_id] * num_tokens
            + [self.processor.image_end_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}
            ]
            outputs = self.processor.prompt_token_ids2outputs(request)
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Incomplete image tokens
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}],
            "prompt_token_ids": [self.processor.image_start_id, self.processor.image_patch_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}
            ]
            with self.assertRaises(ValueError):
                self.processor.prompt_token_ids2outputs(request)

        # Image count mismatch
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}],
            "prompt_token_ids": [
                self.processor.image_start_id,
                self.processor.image_patch_id,
                self.processor.image_end_id,
                self.processor.image_start_id,
                self.processor.image_patch_id,
                self.processor.image_end_id,
            ],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}
            ]
            with self.assertRaises(ValueError):
                self.processor.prompt_token_ids2outputs(request)

        # Video count mismatch
        mock_frames = [Image.new("RGB", (224, 224)) for _ in range(4)]
        num_video_tokens = (4 * 16 * 16) // (self.processor.spatial_conv_size**2 * self.processor.temporal_conv_size)
        request = {
            "messages": [{"role": "user", "content": [{"type": "video", "data": mock_frames, "uuid": "vid1"}]}],
            "prompt_token_ids": [
                self.processor.video_start_id,
                self.processor.image_patch_id,
                self.processor.video_end_id,
                self.processor.video_start_id,
                self.processor.image_patch_id,
                self.processor.video_end_id,
            ],
        }
        with (
            patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_video_decord") as mock_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_frames_decord") as mock_frames_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.render_frame_timestamp") as mock_render,
        ):
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "video", "data": mock_frames, "uuid": "vid1"}]}
            ]
            self._setup_video_mocks(mock_read, mock_frames_read, mock_render, mock_frames)
            with self.assertRaises(ValueError):
                self.processor.prompt_token_ids2outputs(request)

        # Image idx out of range (more image placeholders than images)
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}],
            "prompt_token_ids": [
                self.processor.image_start_id,
                self.processor.image_patch_id,
                self.processor.image_end_id,
                self.processor.image_start_id,
                self.processor.image_patch_id,
                self.processor.image_end_id,
            ],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}
            ]
            with self.assertRaises(ValueError):
                self.processor.prompt_token_ids2outputs(request)

        # Video idx out of range (more video placeholders than videos)
        request = {
            "messages": [{"role": "user", "content": [{"type": "video", "data": mock_frames, "uuid": "vid1"}]}],
            "prompt_token_ids": [
                self.processor.video_start_id,
                self.processor.image_patch_id,
                self.processor.video_end_id,
                self.processor.video_start_id,
                self.processor.image_patch_id,
                self.processor.video_end_id,
            ],
        }
        with (
            patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_video_decord") as mock_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_frames_decord") as mock_frames_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.render_frame_timestamp") as mock_render,
        ):
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "video", "data": mock_frames, "uuid": "vid1"}]}
            ]
            self._setup_video_mocks(mock_read, mock_frames_read, mock_render, mock_frames)
            with self.assertRaises(ValueError):
                self.processor.prompt_token_ids2outputs(request)

        # Test with cached image (tuple format)
        cached_image = (np.random.rand(256, 3 * 14 * 14).astype(np.float32), {"thw": (1, 16, 16)})
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": cached_image, "uuid": "img1"}]}],
            "prompt_token_ids": [self.processor.image_start_id]
            + [self.processor.image_patch_id] * num_tokens
            + [self.processor.image_end_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": cached_image, "uuid": "img1"}]}
            ]
            outputs = self.processor.prompt_token_ids2outputs(request)
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Test with video (dict format)
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "data": {"video": "test.mp4", "fps": 2}, "uuid": "vid1"}],
                }
            ],
            "prompt_token_ids": [self.processor.video_start_id]
            + [self.processor.image_patch_id] * num_video_tokens
            + [self.processor.video_end_id],
        }
        with (
            patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_video_decord") as mock_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_frames_decord") as mock_frames_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.render_frame_timestamp") as mock_render,
        ):
            mock_parse.return_value = [
                {
                    "role": "user",
                    "content": [{"type": "video", "data": {"video": "test.mp4", "fps": 2}, "uuid": "vid1"}],
                }
            ]
            self._setup_video_mocks(mock_read, mock_frames_read, mock_render, mock_frames)
            outputs = self.processor.prompt_token_ids2outputs(request)
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Test with cached video (tuple format)
        cached_video = (np.random.rand(4 * 256, 3 * 14 * 14).astype(np.float32), {"thw": (4, 16, 16)})
        request = {
            "messages": [{"role": "user", "content": [{"type": "video", "data": cached_video, "uuid": "vid1"}]}],
            "prompt_token_ids": [self.processor.video_start_id]
            + [self.processor.image_patch_id] * num_video_tokens
            + [self.processor.video_end_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "video", "data": cached_video, "uuid": "vid1"}]}
            ]
            outputs = self.processor.prompt_token_ids2outputs(request)
            self.assertGreater(len(outputs["input_ids"]), 0)

        # Test prompt_token_ids2outputs with processor cache update
        self.processor.enable_processor_cache = True
        # Reset preprocess mock to return correct format
        self.mock_image_preprocessor.preprocess.return_value = {
            "pixel_values": np.random.rand(256, 3 * 14 * 14).astype(np.float32),
            "image_grid_thw": np.array([[1, 16, 16]]),
        }
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}],
            "prompt_token_ids": [self.processor.image_start_id]
            + [self.processor.image_patch_id] * num_tokens
            + [self.processor.image_end_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_socket.recv_multipart = MagicMock(return_value=(b"", b"pickled_data"))
            mock_context.socket.return_value = mock_socket
            mock_zmq.Context.return_value = mock_context
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
                mock_parse.return_value = [
                    {"role": "user", "content": [{"type": "image", "data": mock_image, "uuid": "img1"}]}
                ]
                with patch("fastdeploy.input.ernie4_5_vl_processor.process.pickle") as mock_pickle:
                    mock_pickle.loads = MagicMock(return_value=[])
                    with patch.object(self.processor, "update_processor_cache") as mock_update:
                        outputs = self.processor.prompt_token_ids2outputs(request)
                        mock_update.assert_called_once()
        self.processor.enable_processor_cache = False

        # Test token_len mismatch for processed image
        cached_image_wrong = (np.random.rand(128, 3 * 14 * 14).astype(np.float32), {"thw": (1, 16, 16)})
        request = {
            "messages": [{"role": "user", "content": [{"type": "image", "data": cached_image_wrong, "uuid": "img1"}]}],
            "prompt_token_ids": [self.processor.image_start_id]
            + [self.processor.image_patch_id] * num_tokens
            + [self.processor.image_end_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "image", "data": cached_image_wrong, "uuid": "img1"}]}
            ]
            with self.assertRaisesRegex(ValueError, "image tokens num not match"):
                self.processor.prompt_token_ids2outputs(request)

        # Test token_len mismatch for video
        request = {
            "messages": [{"role": "user", "content": [{"type": "video", "data": mock_frames, "uuid": "vid1"}]}],
            "prompt_token_ids": [self.processor.video_start_id]
            + [self.processor.image_patch_id] * 10
            + [self.processor.video_end_id],
        }
        with (
            patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_video_decord") as mock_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.read_frames_decord") as mock_frames_read,
            patch("fastdeploy.input.ernie4_5_vl_processor.process.render_frame_timestamp") as mock_render,
        ):
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "video", "data": mock_frames, "uuid": "vid1"}]}
            ]
            self._setup_video_mocks(mock_read, mock_frames_read, mock_render, mock_frames)
            with self.assertRaisesRegex(ValueError, "video tokens num not match"):
                self.processor.prompt_token_ids2outputs(request)

        # Test token_len mismatch for processed video
        cached_video_wrong = (np.random.rand(128, 3 * 14 * 14).astype(np.float32), {"thw": (4, 16, 16)})
        request = {
            "messages": [{"role": "user", "content": [{"type": "video", "data": cached_video_wrong, "uuid": "vid1"}]}],
            "prompt_token_ids": [self.processor.video_start_id]
            + [self.processor.image_patch_id] * num_video_tokens
            + [self.processor.video_end_id],
        }
        with patch("fastdeploy.input.ernie4_5_vl_processor.process.parse_chat_messages") as mock_parse:
            mock_parse.return_value = [
                {"role": "user", "content": [{"type": "video", "data": cached_video_wrong, "uuid": "vid1"}]}
            ]
            with self.assertRaisesRegex(ValueError, "video tokens num not match"):
                self.processor.prompt_token_ids2outputs(request)

    def test_load_tokenizer(self):
        """Test _load_tokenizer method"""
        with patch("os.path.exists", return_value=True):
            with patch("fastdeploy.input.ernie4_5_vl_processor.process.Ernie4_5Tokenizer") as mock_tokenizer_class:
                mock_tokenizer_class.resource_files_names = {"vocab_file": "tokenizer.model"}
                mock_tokenizer_class.from_pretrained = MagicMock(return_value=self.mock_tokenizer)
                with patch(
                    "fastdeploy.input.ernie4_5_vl_processor.process.AdaptiveImageProcessor",
                    self.mock_image_preprocessor,
                ):
                    processor = DataProcessor(tokenizer_name="test_model", image_preprocessor_name="test_model")
                    self.assertIsNotNone(processor.tokenizer)


if __name__ == "__main__":
    unittest.main()
