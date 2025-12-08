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
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None) as mock_init:
            self.processor = Ernie4_5_VLProcessor("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

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

        self.mock_reasoning_parser = MagicMock()
        self.mock_reasoning_parser.__class__.__name__ = "ErnieX1ReasoningParser"
        # self.mock_reasoning_parser.extract_reasoning_content_streaming.return_value = ("reasoning", "text")
        self.processor.reasoning_parser = self.mock_reasoning_parser

        self.mock_tool_parser = MagicMock()
        self.mock_tool_parser.extract_tool_calls_streaming.return_value = None
        self.mock_tool_parser_obj = MagicMock()
        self.mock_tool_parser_obj.return_value = self.mock_tool_parser
        self.processor.tool_parser_obj = self.mock_tool_parser_obj

    def test_process_request_dict_with_options(self):
        # Test with prompt_token_ids - enable_thinking defaults to True
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        # Test with prompt_token_ids and chat_template_kwargs
        # Note: When prompt_token_ids is present, the code uses setdefault for enable_thinking
        # and doesn't process chat_template_kwargs for enable_thinking
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

        # When prompt_token_ids is present, chat_template_kwargs enable_thinking is NOT processed
        # The code uses setdefault which sets enable_thinking to True
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_token_ids": [1, 1, 1],
        }
        self.processor.process_request_dict(request_dict, 100)
        # Since prompt_token_ids branch uses setdefault, enable_thinking defaults to True
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

    def test_parse_processor_kwargs(self):
        """Test _parse_processor_kwargs with various inputs (lines 128-163)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._parse_processor_kwargs = Ernie4_5_VLProcessor._parse_processor_kwargs.__get__(
                processor, Ernie4_5_VLProcessor
            )

            # Test with valid kwargs
            valid_kwargs = {
                "spatial_conv_size": 14,
                "temporal_conv_size": 2,
                "image_min_pixels": 1000,
                "image_max_pixels": 10000,
            }
            result = processor._parse_processor_kwargs(valid_kwargs)
            self.assertEqual(result, valid_kwargs)

            # Test with invalid type (implementation catches exception and returns empty dict)
            invalid_kwargs = {"spatial_conv_size": "invalid"}  # Should be int
            result = Ernie4_5_VLProcessor._parse_processor_kwargs(processor, invalid_kwargs)
            self.assertEqual(result, {})

            # Test with non-dict input (implementation catches exception and returns empty dict)
            result = Ernie4_5_VLProcessor._parse_processor_kwargs(processor, "not a dict")
            self.assertEqual(result, {})

            # Test exception handling with None
            with patch("fastdeploy.input.ernie4_5_vl_processor.ernie4_5_vl_processor.data_processor_logger"):
                result = processor._parse_processor_kwargs(None)
                self.assertEqual(result, {})

    def test_parse_limits(self):
        """Test _parse_limits with various inputs (lines 165-179)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._parse_limits = Ernie4_5_VLProcessor._parse_limits.__get__(processor, Ernie4_5_VLProcessor)

            # Test with valid limits
            valid_limits = {"image": 5, "video": 3}
            result = processor._parse_limits(valid_limits)
            self.assertEqual(result["image"], 5)
            self.assertEqual(result["video"], 3)
            self.assertEqual(result["audio"], 1)  # Default value

            # Test with empty input (None)
            result = processor._parse_limits(None)
            self.assertEqual(result["image"], 1)
            self.assertEqual(result["video"], 1)
            self.assertEqual(result["audio"], 1)

            # Test with invalid type (implementation catches exception and returns default limits)
            result = Ernie4_5_VLProcessor._parse_limits(processor, "not a dict")
            self.assertEqual(result["image"], 1)
            self.assertEqual(result["video"], 1)
            self.assertEqual(result["audio"], 1)

    def test_check_mm_limits(self):
        """Test _check_mm_limits with various inputs (lines 182-201)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor._check_mm_limits = Ernie4_5_VLProcessor._check_mm_limits.__get__(processor, Ernie4_5_VLProcessor)

            # Test with dict input (should not raise)
            processor.limit_mm_per_prompt = {"image": 2, "video": 1}
            mm_data = {"image": [1, 2], "video": [1]}
            processor._check_mm_limits(mm_data)

            # Test with messages input (should not raise)
            messages = [
                {"role": "user", "content": [{"type": "image", "data": "img1"}]},
                {"role": "user", "content": [{"type": "video", "data": "vid1"}]},
            ]
            processor._check_mm_limits(messages)

            # Test when limit is exceeded (should raise ValueError)
            processor.limit_mm_per_prompt = {"image": 1, "video": 1}
            mm_data = {"image": [1, 2, 3], "video": []}  # 3 images, limit is 1
            with self.assertRaises(ValueError) as context:
                processor._check_mm_limits(mm_data)
            self.assertIn("Too many image items", str(context.exception))

    def test_process_request(self):
        """Test process_request method (lines 120-126)"""
        from fastdeploy.engine.request import Request

        # Mock the process_request_dict method
        self.processor.process_request_dict = MagicMock()

        # Create a mock Request object
        mock_request = MagicMock(spec=Request)
        mock_request.to_dict.return_value = {"messages": [{"role": "user", "content": "Hello"}]}

        # Mock Request.from_dict to return a mock request
        with patch.object(Request, "from_dict") as mock_from_dict:
            mock_result_request = MagicMock(spec=Request)
            mock_from_dict.return_value = mock_result_request

            self.processor.process_request(mock_request, max_model_len=100, chat_template_kwargs={"key": "value"})

            # Verify to_dict was called
            mock_request.to_dict.assert_called_once()

            # Verify process_request_dict was called
            self.processor.process_request_dict.assert_called_once()

            # Verify from_dict was called
            mock_from_dict.assert_called_once()

    def test_get_pad_id(self):
        """Test get_pad_id method (line 86)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.tokenizer = MagicMock()
            processor.tokenizer.pad_token_id = 100
            processor.get_pad_id = Ernie4_5_VLProcessor.get_pad_id.__get__(processor, Ernie4_5_VLProcessor)

            result = processor.get_pad_id()
            self.assertEqual(result, 100)

    def test_load_tokenizer(self):
        """Test _load_tokenizer method (line 95)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            mock_tokenizer = MagicMock()
            processor.ernie4_5_processor = MagicMock()
            processor.ernie4_5_processor.tokenizer = mock_tokenizer
            processor._load_tokenizer = Ernie4_5_VLProcessor._load_tokenizer.__get__(processor, Ernie4_5_VLProcessor)

            processor._load_tokenizer()
            self.assertEqual(processor.tokenizer, mock_tokenizer)

    def test_process_request_dict_with_stop_sequences(self):
        """Test process_request_dict with stop sequences (lines 212-214)"""
        self.processor.update_stop_seq = MagicMock(return_value=([100, 101], [1, 1]))

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
            "stop": ["stop1", "stop2"],
        }
        self.processor.process_request_dict(request_dict, 100)

        self.processor.update_stop_seq.assert_called_once_with(["stop1", "stop2"])
        self.assertEqual(request_dict["stop_token_ids"], [100, 101])
        self.assertEqual(request_dict["stop_seqs_len"], [1, 1])

    def test_process_request_dict_with_bad_words(self):
        """Test process_request_dict with bad words (lines 219-220)"""
        self.processor.update_bad_words = MagicMock(return_value=[[200], [201]])

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
            "bad_words": ["bad1", "bad2"],
        }
        self.processor.process_request_dict(request_dict, 100)

        self.processor.update_bad_words.assert_called_once()
        self.assertEqual(request_dict["bad_words_token_ids"], [[200], [201]])

    def test_process_request_dict_with_prompt(self):
        """Test process_request_dict with prompt (lines 228-235)"""
        self.processor.ernie4_5_processor.text2ids = MagicMock(
            return_value={
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "images": [],
                "grid_thw": [],
                "image_type_ids": [],
                "cur_position": 3,
            }
        )

        # Test with multimodal_data
        request_dict = {
            "prompt": "Hello world",
            "multimodal_data": {"image": [], "video": []},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.processor.ernie4_5_processor.text2ids.assert_called_once()
        self.assertEqual(request_dict["prompt_tokens"], "Hello world")

        # Test without multimodal_data - should default to empty dict
        self.processor.ernie4_5_processor.text2ids.reset_mock()
        request_dict = {
            "prompt": "Hello world",
        }
        self.processor.process_request_dict(request_dict, 100)
        self.processor.ernie4_5_processor.text2ids.assert_called_once()
        self.assertEqual(request_dict["prompt_tokens"], "Hello world")

    def test_process_request_dict_with_messages_only(self):
        """Test process_request_dict with messages only (lines 236-259)"""
        self.processor.ernie4_5_processor.request2ids = MagicMock(
            return_value={
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "images": [],
                "grid_thw": [],
                "image_type_ids": [],
                "cur_position": 3,
            }
        )

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        self.processor.process_request_dict(request_dict, 100)

        self.processor.ernie4_5_processor.request2ids.assert_called_once()
        self.assertEqual(request_dict["enable_thinking"], True)

    def test_process_request_dict_messages_with_chat_template_kwargs(self):
        """Test process_request_dict with messages and chat_template_kwargs (lines 243-245)"""
        self.processor.ernie4_5_processor.request2ids = MagicMock(
            return_value={
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "images": [],
                "grid_thw": [],
                "image_type_ids": [],
                "cur_position": 3,
            }
        )

        # Test that chat_template_kwargs values are copied to request
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "custom_key": "custom_value",
                "enable_thinking": False,
            },
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["custom_key"], "custom_value")
        self.assertEqual(request_dict["enable_thinking"], False)

    def test_process_request_dict_messages_with_thinking_mode(self):
        """Test process_request_dict with messages and different thinking_mode values (lines 248-255)"""
        self.processor.ernie4_5_processor.request2ids = MagicMock(
            return_value={
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "images": [],
                "grid_thw": [],
                "image_type_ids": [],
                "cur_position": 3,
            }
        )

        # Test thinking_mode = "close" (should set enable_thinking to False)
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "close"}},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        # Test thinking_mode = "false" (should set enable_thinking to False)
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "false"}},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], False)

        # Test thinking_mode = "open" (should set enable_thinking to True)
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {"options": {"thinking_mode": "open"}},
        }
        self.processor.process_request_dict(request_dict, 100)
        self.assertEqual(request_dict["enable_thinking"], True)

    def test_process_request_dict_with_chat_template_kwargs_not_dict(self):
        """Test process_request_dict with invalid chat_template_kwargs (line 247)"""
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": "not_a_dict",
        }

        with self.assertRaises(ValueError) as context:
            self.processor.process_request_dict(request_dict, 100)
        self.assertIn("chat_template_kwargs must be a dict", str(context.exception))

    def test_process_request_dict_no_valid_input(self):
        """Test process_request_dict with no valid input (line 262)"""
        request_dict = {}

        with self.assertRaises(ValueError) as context:
            self.processor.process_request_dict(request_dict, 100)
        self.assertIn("Request must contain", str(context.exception))

    def test_process_request_dict_with_completion_token_ids(self):
        """Test process_request_dict with completion_token_ids (line 264)"""
        self.processor.append_completion_tokens = MagicMock()

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
            "completion_token_ids": [10, 11, 12],
        }
        self.processor.process_request_dict(request_dict, 100)

        self.processor.append_completion_tokens.assert_called_once()

    def test_process_request_dict_prompt_truncation(self):
        """Test process_request_dict prompt truncation (line 275)"""
        mock_outputs = MagicMock()
        mock_outputs.__getitem__ = MagicMock(
            side_effect=lambda k: {
                "input_ids": np.array([1] * 150),
                "token_type_ids": np.array([0] * 150),
                "position_ids": np.array([[i, i, i] for i in range(150)]),
                "images": None,
                "grid_thw": None,
                "image_type_ids": None,
                "image_patch_id": 1001,
            }.get(k)
        )
        self.processor.pack_outputs = MagicMock(return_value=mock_outputs)

        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1] * 150,  # 150 tokens
        }
        self.processor.process_request_dict(request_dict, 100)  # max_model_len=100

        # Should be truncated to max_model_len - 1 = 99
        self.assertEqual(len(request_dict["prompt_token_ids"]), 99)

    def test_process_request_dict_max_tokens_calculation(self):
        """Test process_request_dict max_tokens calculation (lines 280, 286)"""
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
            "max_tokens": 200,  # Larger than available
        }
        self.processor.process_request_dict(request_dict, 100)

        # max_tokens should be min(max_model_len - prompt_len, max_tokens)
        self.assertLessEqual(request_dict["max_tokens"], 100 - 3)

    def test_process_request_dict_top_p_adjustment(self):
        """Test process_request_dict top_p adjustment (line 288)"""
        request_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_token_ids": [1, 1, 1],
            "top_p": 1e-10,  # Very small value
        }
        self.processor.process_request_dict(request_dict, 100)

        # top_p should be adjusted to _SAMPLING_EPS
        self.assertGreaterEqual(request_dict["top_p"], 1e-5)

    def test_append_completion_tokens(self):
        """Test append_completion_tokens method (lines 293-300)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.append_completion_tokens = Ernie4_5_VLProcessor.append_completion_tokens.__get__(
                processor, Ernie4_5_VLProcessor
            )

            multimodal_inputs = {
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "cur_position": 3,
            }
            completion_token_ids = [10, 11, 12]

            processor.append_completion_tokens(multimodal_inputs, completion_token_ids)

            self.assertEqual(multimodal_inputs["input_ids"], [1, 2, 3, 10, 11, 12])
            self.assertEqual(multimodal_inputs["token_type_ids"], [0, 0, 0, 0, 0, 0])
            self.assertEqual(len(multimodal_inputs["position_ids"]), 6)
            self.assertEqual(multimodal_inputs["cur_position"], 6)

    def test_pack_outputs(self):
        """Test pack_outputs with and without images (lines 304-319)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.image_patch_id = 1001
            processor.pack_outputs = Ernie4_5_VLProcessor.pack_outputs.__get__(processor, Ernie4_5_VLProcessor)

            # Test with images
            outs_with_images = {
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "images": [np.array([[1, 2], [3, 4]])],
                "grid_thw": [np.array([[1, 2, 2]])],
                "image_type_ids": [0],
            }

            result = processor.pack_outputs(outs_with_images)
            self.assertIsNotNone(result["images"])
            self.assertIsNotNone(result["grid_thw"])
            self.assertIsNotNone(result["image_type_ids"])
            self.assertEqual(result["image_patch_id"], 1001)
            self.assertIsInstance(result["input_ids"], np.ndarray)
            self.assertIsInstance(result["token_type_ids"], np.ndarray)
            self.assertIsInstance(result["position_ids"], np.ndarray)

            # Test without images
            outs_without_images = {
                "input_ids": [1, 2, 3],
                "token_type_ids": [0, 0, 0],
                "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                "images": [],
                "grid_thw": [],
                "image_type_ids": [],
            }

            result = processor.pack_outputs(outs_without_images)
            self.assertIsNone(result["images"])
            self.assertIsNone(result["grid_thw"])
            self.assertIsNone(result["image_type_ids"])

    def test_process_response_dict(self):
        """Test process_response_dict with different parameters (lines 331-336)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.process_response_dict = Ernie4_5_VLProcessor.process_response_dict.__get__(
                processor, Ernie4_5_VLProcessor
            )

            # Test with stream=True
            processor.process_response_dict_streaming = MagicMock(return_value={"text": "response"})
            response_dict = {"ids": [1, 2, 3]}
            result = processor.process_response_dict(response_dict, stream=True)
            processor.process_response_dict_streaming.assert_called_once()
            self.assertEqual(result, {"text": "response"})

            # Test with stream=False
            processor.process_response_dict_normal = MagicMock(return_value={"text": "response"})
            response_dict = {"ids": [1, 2, 3]}
            result = processor.process_response_dict(response_dict, stream=False)
            processor.process_response_dict_normal.assert_called_once()
            self.assertEqual(result, {"text": "response"})

            # Test with enable_thinking=None (should default to True)
            processor.process_response_dict_streaming = MagicMock(return_value={"text": "response"})
            response_dict = {"ids": [1, 2, 3]}
            processor.process_response_dict(response_dict, stream=True, enable_thinking=None)
            processor.process_response_dict_streaming.assert_called_once_with(response_dict, enable_thinking=True)

    def test_apply_default_parameters(self):
        """Test _apply_default_parameters with dict and object request (lines 102-116)"""
        with patch.object(Ernie4_5_VLProcessor, "__init__", return_value=None):
            processor = Ernie4_5_VLProcessor("model_path")
            processor.generation_config = MagicMock()
            processor.generation_config.top_p = 0.8
            processor.generation_config.temperature = 0.9
            processor._apply_default_parameters = Ernie4_5_VLProcessor._apply_default_parameters.__get__(
                processor, Ernie4_5_VLProcessor
            )

            # Test with dict request
            request = {}
            result = processor._apply_default_parameters(request)
            self.assertEqual(result["top_p"], 0.8)
            self.assertEqual(result["temperature"], 0.9)

            # Test with object request
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

        with patch.object(
            DataProcessor,
            "_load_tokenizer",
            side_effect=mock_load_tokenizer,
            autospec=True,
        ):
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
        self.data_processor.role_prefixes = {
            "user": "User: ",
            "assistant": "Assistant: ",
        }
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

        self.data_processor.extract_mm_items.return_value = (
            [],
            [],
            [],
            [],
            None,
            [],
            [],
        )

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
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": mock_img, "uuid": "img_uuid"}],
                }
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
        mock_preprocess = {
            "pixel_values": np.random.randn(1, 16, 16, 3),
            "image_grid_thw": np.array([[2, 4]]),
        }
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
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": mock_img_cache,
                            "uuid": "img_uuid",
                        }
                    ],
                }
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
                {
                    "role": "user",
                    "content": [{"type": "video_url", "video_url": frames, "uuid": "vid_uuid"}],
                }
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
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (
            None,
            (patches_h, patches_w),
        )
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
                {
                    "role": "user",
                    "content": [{"type": "video", "data": mock_frames_cache, "uuid": "vid_uuid"}],
                }
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
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": mock_img, "uuid": "img_uuid"}],
                }
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
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (
            None,
            (patches_h, patches_w),
        )
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
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": mock_img_cache,
                            "uuid": "img_uuid",
                        }
                    ],
                }
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
                {
                    "role": "user",
                    "content": [{"type": "video_url", "video_url": frames, "uuid": "vid_uuid"}],
                }
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
        self.data_processor.image_preprocessor.get_smarted_resize.return_value = (
            None,
            (patches_h, patches_w),
        )
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
                {
                    "role": "user",
                    "content": [{"type": "video", "data": mock_frames_cache, "uuid": "vid_uuid"}],
                }
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
