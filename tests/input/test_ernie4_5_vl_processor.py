import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
from fastdeploy.input.ernie4_5_vl_processor import Ernie4_5_VLProcessor
from fastdeploy.input.ernie4_5_vl_processor.image_preprocessor.image_preprocessor_adaptive import (
    AdaptiveImageProcessor,
)
from fastdeploy.input.ernie4_5_vl_processor.process import DataProcessor
from fastdeploy.input.utils import IDS_TYPE_FLAG


class TestErnie4_5_vl_ProcessorProcessResponseDictStreaming(unittest.TestCase):
    def setUp(self):
        # 创建 Ernie4_5Processor 实例的模拟对象
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5_VLProcessor("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        # 设置必要的属性
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

        # 模拟 ids2tokens 方法
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

        # 模拟推理解析器
        self.mock_reasoning_parser = MagicMock()
        self.mock_reasoning_parser.__class__.__name__ = "ErnieX1ReasoningParser"
        # self.mock_reasoning_parser.extract_reasoning_content_streaming.return_value = ("reasoning", "text")
        self.processor.reasoning_parser = self.mock_reasoning_parser

        # 模拟工具解析器
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

    def test_parse_processor_kwargs_valid(self):
        """Test _parse_processor_kwargs with valid kwargs (lines 128-163)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._parse_processor_kwargs = Ernie4_5_VLProcessor._parse_processor_kwargs.__get__(
                processor, Ernie4_5_VLProcessor
            )

            valid_kwargs = {
                "spatial_conv_size": 14,
                "temporal_conv_size": 2,
                "image_min_pixels": 1000,
                "image_max_pixels": 10000,
            }
            result = processor._parse_processor_kwargs(valid_kwargs)
            self.assertEqual(result, valid_kwargs)

    def test_parse_processor_kwargs_invalid_type(self):
        """Test _parse_processor_kwargs with invalid type (line 155)

        Note: The implementation catches ValueError and returns empty dict with warning log,
        rather than raising the exception.
        """
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")

            invalid_kwargs = {"spatial_conv_size": "invalid"}  # Should be int
            # Implementation catches exception and returns empty dict
            result = Ernie4_5_VLProcessor._parse_processor_kwargs(processor, invalid_kwargs)
            self.assertEqual(result, {})

    def test_parse_processor_kwargs_not_dict(self):
        """Test _parse_processor_kwargs with non-dict input (line 135)

        Note: The implementation catches ValueError and returns empty dict with warning log,
        rather than raising the exception.
        """
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")

            # Implementation catches exception and returns empty dict
            result = Ernie4_5_VLProcessor._parse_processor_kwargs(processor, "not a dict")
            self.assertEqual(result, {})

    def test_parse_processor_kwargs_exception_handling(self):
        """Test _parse_processor_kwargs exception handling (line 162)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._parse_processor_kwargs = Ernie4_5_VLProcessor._parse_processor_kwargs.__get__(
                processor, Ernie4_5_VLProcessor
            )

            # This should return empty dict on exception
            with patch("fastdeploy.input.ernie4_5_vl_processor.ernie4_5_vl_processor.data_processor_logger"):
                result = processor._parse_processor_kwargs(None)
                self.assertEqual(result, {})

    def test_parse_limits_valid(self):
        """Test _parse_limits with valid input (lines 165-179)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._parse_limits = Ernie4_5_VLProcessor._parse_limits.__get__(processor, Ernie4_5_VLProcessor)

            valid_limits = {"image": 5, "video": 3}
            result = processor._parse_limits(valid_limits)
            self.assertEqual(result["image"], 5)
            self.assertEqual(result["video"], 3)
            self.assertEqual(result["audio"], 1)  # Default value

    def test_parse_limits_invalid_type(self):
        """Test _parse_limits with invalid type (line 174)

        Note: The implementation catches ValueError and returns DEFAULT_LIMITS with warning log,
        rather than raising the exception.
        """
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")

            # Implementation catches exception and returns default limits
            result = Ernie4_5_VLProcessor._parse_limits(processor, "not a dict")
            # Should return DEFAULT_LIMITS = {"image": 1, "video": 1, "audio": 1}
            self.assertEqual(result["image"], 1)
            self.assertEqual(result["video"], 1)
            self.assertEqual(result["audio"], 1)

    def test_parse_limits_empty(self):
        """Test _parse_limits with empty input (line 170)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._parse_limits = Ernie4_5_VLProcessor._parse_limits.__get__(processor, Ernie4_5_VLProcessor)

            result = processor._parse_limits(None)
            self.assertEqual(result["image"], 1)
            self.assertEqual(result["video"], 1)
            self.assertEqual(result["audio"], 1)

    def test_check_mm_limits_with_dict(self):
        """Test _check_mm_limits with dict input (lines 182-184)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.limit_mm_per_prompt = {"image": 2, "video": 1}
            processor._check_mm_limits = Ernie4_5_VLProcessor._check_mm_limits.__get__(processor, Ernie4_5_VLProcessor)

            mm_data = {"image": [1, 2], "video": [1]}
            # Should not raise
            processor._check_mm_limits(mm_data)

    def test_check_mm_limits_with_messages(self):
        """Test _check_mm_limits with messages input (lines 186-195)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.limit_mm_per_prompt = {"image": 2, "video": 1}
            processor._check_mm_limits = Ernie4_5_VLProcessor._check_mm_limits.__get__(processor, Ernie4_5_VLProcessor)

            messages = [
                {"role": "user", "content": [{"type": "image", "data": "img1"}]},
                {"role": "user", "content": [{"type": "video", "data": "vid1"}]},
            ]
            # Should not raise
            processor._check_mm_limits(messages)

    def test_check_mm_limits_exceeded(self):
        """Test _check_mm_limits when limit is exceeded (line 201)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.limit_mm_per_prompt = {"image": 1, "video": 1}
            processor._check_mm_limits = Ernie4_5_VLProcessor._check_mm_limits.__get__(processor, Ernie4_5_VLProcessor)

            mm_data = {"image": [1, 2, 3], "video": []}  # 3 images, limit is 1
            with self.assertRaises(ValueError) as context:
                processor._check_mm_limits(mm_data)
            self.assertIn("Too many image items", str(context.exception))

    def test_apply_default_parameters_with_dict(self):
        """Test _apply_default_parameters with dict request (lines 102-116)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.generation_config = MagicMock()
            processor.generation_config.top_p = 0.8
            processor.generation_config.temperature = 0.9
            processor._apply_default_parameters = Ernie4_5_VLProcessor._apply_default_parameters.__get__(
                processor, Ernie4_5_VLProcessor
            )

            request = {}
            result = processor._apply_default_parameters(request)
            self.assertEqual(result["top_p"], 0.8)
            self.assertEqual(result["temperature"], 0.9)

    def test_apply_default_parameters_with_object(self):
        """Test _apply_default_parameters with object request (lines 108-109)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.generation_config = MagicMock()
            processor.generation_config.top_p = 0.8
            processor._apply_default_parameters = Ernie4_5_VLProcessor._apply_default_parameters.__get__(
                processor, Ernie4_5_VLProcessor
            )

            class MockRequest:
                def __init__(self):
                    self.top_p = None
                    self.temperature = None

                def get(self, key):
                    return getattr(self, key, None)

                def set(self, key, value):
                    setattr(self, key, value)

            request = MockRequest()
            result = processor._apply_default_parameters(request)
            self.assertEqual(result.top_p, 0.8)


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
            f"input_ids 不匹配：实际{outputs['input_ids']}，预期[{test_prompt_token_ids}]",
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


if __name__ == "__main__":
    unittest.main()
