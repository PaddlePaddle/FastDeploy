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

from fastdeploy.input.encodings import ErnieEncoding, PaddleOCREncoding, QwenEncoding
from fastdeploy.input.mm_model_config import (
    ERNIE4_5_VL,
    MODEL_CONFIGS,
    PADDLEOCR_VL,
    QWEN3_VL,
    QWEN_VL,
)
from fastdeploy.input.multimodal_processor import (
    _DEFAULT_MM_LIMITS,
    _SAMPLING_EPS,
    MultiModalProcessor,
)
from fastdeploy.input.utils import IDS_TYPE_FLAG


def _make_processor(model_type, **overrides):
    """Create a MultiModalProcessor instance with __init__ bypassed.

    Manually sets the minimum attributes required by the methods under test.
    Uses the real MMModelConfig from MODEL_CONFIGS so that cfg-based
    dispatch works correctly without hardcoded model_type checks.
    """
    with patch.object(MultiModalProcessor, "__init__", return_value=None):
        proc = MultiModalProcessor.__new__(MultiModalProcessor)
    proc.model_type = model_type
    proc.cfg = MODEL_CONFIGS[model_type]
    proc.config = MagicMock()
    proc.enable_processor_cache = False
    proc.model_name_or_path = "/mock/model"
    proc.tokenizer_type = proc.cfg.tokenizer_type
    proc.limit_mm_per_prompt = dict(_DEFAULT_MM_LIMITS)
    proc.eos_token_ids = [2]
    proc.eos_token_id_len = 1
    proc.pad_token_id = 0
    proc.reasoning_parser = None
    proc.tool_parser_obj = None
    proc.model_status_dict = {}
    proc.decode_status = {}
    proc.tool_parser_dict = {}
    proc.generation_config = MagicMock()
    proc.generation_config.top_p = 0.7
    proc.generation_config.temperature = 1.0
    proc.generation_config.repetition_penalty = 1.0
    proc.generation_config.frequency_penalty = 0.0
    proc.generation_config.presence_penalty = 0.0

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token_id = 1
    tokenizer.bos_token = "<s>"
    tokenizer.pad_token_id = 0
    tokenizer.vocab_size = 32000
    tokenizer.chat_template = "dummy"
    tokenizer.tokenize.return_value = ["hello"]
    tokenizer.convert_tokens_to_ids.return_value = [100]
    tokenizer.decode.return_value = "hello"
    proc.tokenizer = tokenizer

    # Mock encoding strategy — use spec so hasattr checks work correctly
    if model_type == ERNIE4_5_VL:
        proc.enc = MagicMock(spec=ErnieEncoding)
    elif model_type == PADDLEOCR_VL:
        proc.enc = MagicMock(spec=PaddleOCREncoding)
    else:
        proc.enc = MagicMock(spec=QwenEncoding)

    # Mock image processor
    proc.image_processor = MagicMock()
    proc.image_processor.merge_size = 2
    proc.image_processor.temporal_patch_size = 2

    # Apply any overrides
    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


# ===================================================================
# __init__ validation
# ===================================================================
class TestMultiModalProcessorInitValidation(unittest.TestCase):

    def test_unsupported_model_type_raises(self):
        """Unsupported model_type should raise ValueError."""
        with self.assertRaises(ValueError):
            MultiModalProcessor("/mock", model_type="unsupported_type")


# ===================================================================
# _parse_processor_kwargs
# ===================================================================
class TestParseProcessorKwargs(unittest.TestCase):

    def test_empty_kwargs_returns_empty(self):
        proc = _make_processor(QWEN_VL)
        self.assertEqual(proc._parse_processor_kwargs(None), {})
        self.assertEqual(proc._parse_processor_kwargs({}), {})

    def test_valid_qwen_kwargs(self):
        """Valid kwargs for qwen model type."""
        proc = _make_processor(QWEN_VL)
        kwargs = {"video_max_frames": 10, "video_min_frames": 1}
        result = proc._parse_processor_kwargs(kwargs)
        self.assertEqual(result, kwargs)

    def test_valid_ernie_kwargs(self):
        """Valid kwargs for ernie model type."""
        proc = _make_processor(ERNIE4_5_VL)
        kwargs = {"spatial_conv_size": 2, "temporal_conv_size": 1, "video_max_frames": 32}
        result = proc._parse_processor_kwargs(kwargs)
        self.assertEqual(result, kwargs)

    def test_invalid_type_not_dict(self):
        """Non-dict kwargs should return empty."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_processor_kwargs("invalid")
        self.assertEqual(result, {})

    def test_invalid_value_type(self):
        """Wrong value type should return empty."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_processor_kwargs({"video_max_frames": "ten"})
        self.assertEqual(result, {})

    def test_mixed_valid_invalid_value_types(self):
        proc = _make_processor(ERNIE4_5_VL)
        result = proc._parse_processor_kwargs({"spatial_conv_size": 2, "image_min_pixels": "bad"})
        self.assertEqual(result, {})

    def test_unknown_keys_pass_through(self):
        """Keys not in expected_types are not validated, just passed through."""
        proc = _make_processor(QWEN_VL)
        kwargs = {"unknown_key": "any_value"}
        result = proc._parse_processor_kwargs(kwargs)
        self.assertEqual(result, kwargs)


# ===================================================================
# _parse_limits
# ===================================================================
class TestParseLimits(unittest.TestCase):

    def test_none_returns_defaults(self):
        proc = _make_processor(QWEN_VL)
        self.assertEqual(proc._parse_limits(None), dict(_DEFAULT_MM_LIMITS))

    def test_valid_limits_merged(self):
        """Valid limits merged with defaults."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_limits({"image": 5, "video": 3})
        self.assertEqual(result, {"image": 5, "video": 3, "audio": 1})

    def test_partial_limits(self):
        proc = _make_processor(QWEN_VL)
        result = proc._parse_limits({"image": 10})
        self.assertEqual(result, {"image": 10, "video": 1, "audio": 1})

    def test_invalid_type_returns_defaults(self):
        """Non-dict returns defaults."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_limits("invalid")
        self.assertEqual(result, dict(_DEFAULT_MM_LIMITS))


# ===================================================================
# _check_mm_limits
# ===================================================================
class TestCheckMMLimits(unittest.TestCase):

    def test_dict_input_within_limits(self):
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 2, "video": 1, "audio": 1}
        mm_data = {"image": ["img1"], "video": ["vid1"]}
        proc._check_mm_limits(mm_data)  # should not raise

    def test_dict_input_exceeds_limit(self):
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        mm_data = {"image": ["img1", "img2"]}
        with self.assertRaises(ValueError) as ctx:
            proc._check_mm_limits(mm_data)
        self.assertIn("Too many image items", str(ctx.exception))

    def test_messages_input_qwen_vl_accepts_url_suffix(self):
        """Messages with image_url/video_url for qwen_vl."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://img.jpg"}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        proc._check_mm_limits(messages)  # should not raise

    def test_messages_input_qwen_vl_image_type(self):
        """'image' type also accepted."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": [{"type": "image", "image": "data"}]},
        ]
        proc._check_mm_limits(messages)

    def test_messages_input_qwen_vl_video_url_type(self):
        """video_url type."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "file://vid.mp4"}}]},
        ]
        proc._check_mm_limits(messages)

    def test_messages_input_ernie_only_accepts_plain_types(self):
        """ernie4_5_vl only accepts 'image'/'video' types, not *_url."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        # image_url should NOT be counted for ernie
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://img.jpg"}}]},
        ]
        proc._check_mm_limits(messages)  # no exception since image_url not counted

    def test_messages_input_ernie_image_type(self):
        """ernie 'image' type is counted."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data1"},
                    {"type": "image", "image": "data2"},
                ],
            }
        ]
        with self.assertRaises(ValueError):
            proc._check_mm_limits(messages)

    def test_messages_input_ernie_video_type(self):
        """ernie 'video' type is counted."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": [{"type": "video", "video": "data"}]},
        ]
        proc._check_mm_limits(messages)  # within limit

    def test_messages_exceed_video_limit(self):
        """Video exceeding limit raises ValueError."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "file://v1.mp4"}},
                    {"type": "video_url", "video_url": {"url": "file://v2.mp4"}},
                ],
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            proc._check_mm_limits(messages)
        self.assertIn("Too many video items", str(ctx.exception))

    def test_messages_with_string_content_skipped(self):
        """Messages with string content (not list) should be skipped."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": "just text"},
        ]
        proc._check_mm_limits(messages)  # should not raise


# ===================================================================
# get_mm_max_tokens_per_item
# ===================================================================
class TestGetMmMaxTokensPerItem(unittest.TestCase):

    def test_ernie_returns_enc_result(self):
        """ErnieEncoding has get_mm_max_tokens_per_item, delegates to enc."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.enc.get_mm_max_tokens_per_item.return_value = {"image": 512}
        result = proc.get_mm_max_tokens_per_item(1024)
        self.assertEqual(result, {"image": 512})

    def test_non_ernie_returns_none(self):
        """QwenEncoding inherits default get_mm_max_tokens_per_item returning None."""
        proc = _make_processor(QWEN_VL)
        proc.enc.get_mm_max_tokens_per_item.return_value = None
        self.assertIsNone(proc.get_mm_max_tokens_per_item(1024))

        proc2 = _make_processor(QWEN3_VL)
        proc2.enc.get_mm_max_tokens_per_item.return_value = None
        self.assertIsNone(proc2.get_mm_max_tokens_per_item(1024))


# ===================================================================
# _tokenize_request
# ===================================================================
class TestTokenizeRequest(unittest.TestCase):

    def test_prompt_token_ids_qwen3_vl(self):
        """prompt_token_ids path for qwen3_vl delegates to enc."""
        proc = _make_processor(QWEN3_VL)
        expected = {"input_ids": [1, 2, 3]}
        proc.enc.prompt_token_ids2outputs.return_value = expected
        proc._extract_mm_items = MagicMock(return_value=([], [], [], [], None, [], []))

        request = {"prompt_token_ids": [1, 2, 3], "messages": [{"role": "user", "content": "hi"}]}
        result = proc._tokenize_request(request)
        self.assertEqual(result, expected)
        self.assertFalse(request.get("enable_thinking", True))  # default_thinking=False for qwen3_vl

    def test_prompt_token_ids_ernie(self):
        """prompt_token_ids path for ernie delegates to enc."""
        proc = _make_processor(ERNIE4_5_VL)
        expected = {"input_ids": [1, 2, 3]}
        proc.enc.prompt_token_ids2outputs.return_value = expected

        request = {"prompt_token_ids": [1, 2, 3]}
        result = proc._tokenize_request(request)
        self.assertEqual(result, expected)
        self.assertTrue(request.get("enable_thinking"))  # default_thinking=True for ernie

    def test_prompt_path(self):
        """prompt text path calls proc.text2ids."""
        proc = _make_processor(QWEN_VL)
        expected = {"input_ids": [10, 20]}
        proc.text2ids = MagicMock(return_value=expected)

        request = {"prompt": "hello", "multimodal_data": {"image": [], "video": []}}
        result = proc._tokenize_request(request)
        proc.text2ids.assert_called_once_with("hello", [], [])
        self.assertEqual(result, expected)

    def test_prompt_path_ernie_sets_prompt_tokens(self):
        """ernie sets prompt_tokens from prompt (cfg.sets_prompt_tokens)."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.text2ids = MagicMock(return_value={"input_ids": [1]})

        request = {"prompt": "test prompt"}
        proc._tokenize_request(request)
        self.assertEqual(request["prompt_tokens"], "test prompt")

    def test_messages_path(self):
        """messages path calls proc.request2ids."""
        proc = _make_processor(QWEN_VL)
        expected = {"input_ids": [5, 6]}
        proc.request2ids = MagicMock(return_value=expected)

        request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        result = proc._tokenize_request(request)
        proc.request2ids.assert_called_once()
        self.assertEqual(result, expected)

    def test_messages_path_with_chat_template_kwargs(self):
        """chat_template_kwargs are merged into request."""
        proc = _make_processor(QWEN_VL)
        proc.request2ids = MagicMock(return_value={"input_ids": [1]})

        request = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
        proc._tokenize_request(request)
        self.assertTrue(request.get("enable_thinking"))

    def test_messages_path_chat_template_kwargs_no_overwrite(self):
        """Existing request keys are not overwritten by chat_template_kwargs."""
        proc = _make_processor(QWEN_VL)
        proc.request2ids = MagicMock(return_value={"input_ids": [1]})

        request = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "chat_template_kwargs": {"enable_thinking": True},
            "enable_thinking": False,
        }
        proc._tokenize_request(request)
        self.assertFalse(request["enable_thinking"])

    def test_messages_path_invalid_chat_template_kwargs(self):
        """Non-dict chat_template_kwargs raises."""
        proc = _make_processor(QWEN_VL)
        request = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "chat_template_kwargs": "invalid",
        }
        with self.assertRaises(ValueError) as ctx:
            proc._tokenize_request(request)
        self.assertIn("must be a dict", str(ctx.exception))

    def test_no_input_raises(self):
        """No prompt/messages/prompt_token_ids raises."""
        proc = _make_processor(QWEN_VL)
        with self.assertRaises(ValueError) as ctx:
            proc._tokenize_request({"request_id": "test"})
        self.assertIn("must contain", str(ctx.exception))

    def test_prompt_path_no_multimodal_data(self):
        """Prompt with no multimodal_data passes None for images/videos."""
        proc = _make_processor(QWEN_VL)
        proc.text2ids = MagicMock(return_value={"input_ids": [1]})

        request = {"prompt": "hello"}
        proc._tokenize_request(request)
        proc.text2ids.assert_called_once_with("hello", None, None)


# ===================================================================
# _process_post_tokens
# ===================================================================
class TestProcessPostTokens(unittest.TestCase):

    def test_paddleocr_with_metadata_generated_tokens(self):
        """Fallback: generated_token_ids used when completion_token_ids absent."""
        proc = _make_processor(PADDLEOCR_VL)
        outputs = {"input_ids": [1, 2]}
        request = {"generated_token_ids": [10, 11]}
        proc._process_post_tokens(request, outputs)
        proc.enc.append_completion_tokens.assert_called_once_with(outputs, [10, 11])

    def test_paddleocr_without_metadata(self):
        """PaddleOCR with no metadata does nothing."""
        proc = _make_processor(PADDLEOCR_VL)
        outputs = {"input_ids": [1]}
        proc._process_post_tokens({}, outputs)
        proc.enc.append_completion_tokens.assert_not_called()

    def test_paddleocr_metadata_without_generated_tokens(self):
        """PaddleOCR with metadata but no generated_token_ids does nothing."""
        proc = _make_processor(PADDLEOCR_VL)
        outputs = {"input_ids": [1]}
        request = {"metadata": {"other_key": 123}}
        proc._process_post_tokens(request, outputs)
        proc.enc.append_completion_tokens.assert_not_called()

    def test_non_paddleocr_with_completion_tokens(self):
        """Non-paddleocr uses completion_token_source='completion_token_ids'."""
        proc = _make_processor(QWEN_VL)
        outputs = {"input_ids": [1]}
        request = {"completion_token_ids": [5, 6]}
        proc._process_post_tokens(request, outputs)
        proc.enc.append_completion_tokens.assert_called_once_with(outputs, [5, 6])

    def test_non_paddleocr_without_completion_tokens(self):
        """No completion_token_ids does nothing."""
        proc = _make_processor(QWEN_VL)
        outputs = {"input_ids": [1]}
        proc._process_post_tokens({}, outputs)
        proc.enc.append_completion_tokens.assert_not_called()


# ===================================================================
# _apply_reasoning_parser
# ===================================================================
class TestApplyReasoningParser(unittest.TestCase):

    def test_basic_request_id(self):
        """Basic request_id (no underscore split)."""
        proc = _make_processor(QWEN_VL)
        proc.reasoning_parser = MagicMock()
        proc.reasoning_parser.get_model_status.return_value = "think_start"
        proc.model_status_dict = {}

        request = {"request_id": "req1", "prompt_token_ids": [1, 2, 3]}
        proc._apply_reasoning_parser(request)

        self.assertEqual(proc.model_status_dict["req1"], "think_start")
        self.assertTrue(request["enable_thinking"])

    def test_compound_request_id(self):
        """request_id with underscore is split."""
        proc = _make_processor(QWEN_VL)
        proc.reasoning_parser = MagicMock()
        proc.reasoning_parser.get_model_status.return_value = "think_end"
        proc.model_status_dict = {}

        request = {"request_id": "req1_2", "prompt_token_ids": [1, 2], "n": 3}
        proc._apply_reasoning_parser(request)

        # index=2, n=3 → range(6, 9)
        for idx in [6, 7, 8]:
            self.assertEqual(proc.model_status_dict[f"req1_{idx}"], "think_end")
        self.assertFalse(request["enable_thinking"])

    def test_compound_request_id_default_n(self):
        """Default n=1."""
        proc = _make_processor(QWEN_VL)
        proc.reasoning_parser = MagicMock()
        proc.reasoning_parser.get_model_status.return_value = "think_start"
        proc.model_status_dict = {}

        request = {"request_id": "req1_0", "prompt_token_ids": [1]}
        proc._apply_reasoning_parser(request)

        self.assertIn("req1_0", proc.model_status_dict)
        self.assertTrue(request["enable_thinking"])


# ===================================================================
# append_completion_tokens
# ===================================================================
class TestAppendCompletionTokens(unittest.TestCase):

    def test_delegates_to_enc(self):
        """append_completion_tokens delegates to enc.append_completion_tokens."""
        proc = _make_processor(QWEN_VL)
        inputs = {"input_ids": [1]}
        proc.append_completion_tokens(inputs, [2, 3])
        proc.enc.append_completion_tokens.assert_called_once_with(inputs, [2, 3])

    def test_delegates_to_enc_ernie(self):
        """Same delegation for ernie."""
        proc = _make_processor(ERNIE4_5_VL)
        inputs = {"input_ids": [1]}
        proc.append_completion_tokens(inputs, [2, 3])
        proc.enc.append_completion_tokens.assert_called_once_with(inputs, [2, 3])


# ===================================================================
# pack_outputs
# ===================================================================
class TestPackOutputs(unittest.TestCase):

    def test_qwen_with_images(self):
        """Qwen pack_outputs with image data delegates position packing to enc."""
        proc = _make_processor(QWEN_VL)
        outputs = {
            "images": [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
            "grid_thw": [np.array([2, 2, 1]), np.array([2, 2, 1])],
            "image_type_ids": [0, 1],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }
        result = proc.pack_outputs(outputs)

        self.assertIsNotNone(result["images"])
        self.assertEqual(result["images"].shape[0], 4)
        self.assertIsNotNone(result["grid_thw"])
        self.assertEqual(result["input_ids"].dtype, np.int64)
        self.assertEqual(result["token_type_ids"].dtype, np.int64)
        self.assertEqual(result["mm_num_token_func"], proc.enc.mm_num_tokens)
        proc.enc.pack_position_ids.assert_called_once_with(outputs)

    def test_qwen_without_images(self):
        """Empty images set to None."""
        proc = _make_processor(QWEN_VL)
        outputs = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2],
            "token_type_ids": [0, 0],
            "position_ids": [np.array([[0, 1], [0, 1], [0, 1]])],
        }
        result = proc.pack_outputs(outputs)

        self.assertIsNone(result["images"])
        self.assertIsNone(result["grid_thw"])
        self.assertIsNone(result["image_type_ids"])

    def test_ernie_pack_outputs(self):
        """Ernie pack_outputs delegates position packing to enc."""
        proc = _make_processor(ERNIE4_5_VL)
        outputs = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2],
            "token_type_ids": [0, 0],
            "position_ids": [[0, 0, 0], [1, 1, 1]],
        }
        result = proc.pack_outputs(outputs)

        self.assertIsNone(result["images"])
        self.assertEqual(result["input_ids"].dtype, np.int64)
        self.assertEqual(result["token_type_ids"].dtype, np.int64)
        proc.enc.pack_position_ids.assert_called_once_with(outputs)


# ===================================================================
# process_request_dict (integration-level tests for flow coverage)
# ===================================================================
class TestProcessRequestDict(unittest.TestCase):

    def _make_mock_outputs(self, model_type=QWEN_VL):
        """Return mock outputs appropriate for the model type."""
        base = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3, 4, 5],
            "token_type_ids": [0, 0, 0, 0, 0],
        }
        if model_type == ERNIE4_5_VL:
            base["position_ids"] = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
        else:
            base["position_ids"] = [np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])]
        return base

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen_vl_messages_flow(self, mock_stop):
        """Full flow for qwen_vl with messages."""
        proc = _make_processor(QWEN_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(QWEN_VL))

        request = {
            "request_id": "test1",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertIn("prompt_token_ids", result)
        self.assertIn("multimodal_inputs", result)
        self.assertEqual(result["prompt_token_ids_len"], len(result["prompt_token_ids"]))
        self.assertFalse(result.get("enable_thinking"))  # qwen_vl force_disable_thinking

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen3_vl_with_prompt_token_ids(self, mock_stop):
        """Qwen3_vl with existing prompt_token_ids preserved."""
        proc = _make_processor(QWEN3_VL)
        outputs = self._make_mock_outputs(QWEN3_VL)
        proc.enc.prompt_token_ids2outputs.return_value = outputs
        proc._extract_mm_items = MagicMock(return_value=([], [], [], [], None, [], []))

        request = {
            "request_id": "test2",
            "prompt_token_ids": [10, 20, 30],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        # prompt_token_ids should be preserved (not overwritten)
        self.assertEqual(result["prompt_token_ids"], [10, 20, 30])

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_ernie_flow(self, mock_stop):
        """Ernie-specific branches in process_request_dict."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(ERNIE4_5_VL))

        request = {
            "request_id": "test3",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertIn("prompt_token_ids", result)
        self.assertIn("logits_processors_args", result)
        # ernie sets default reasoning_max_tokens when None
        self.assertIn("reasoning_max_tokens", result)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_ernie_low_top_p(self, mock_stop):
        """Ernie with top_p below _SAMPLING_EPS is clamped."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(ERNIE4_5_VL))

        request = {
            "request_id": "test4",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "top_p": 0.0,
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertAlmostEqual(result["top_p"], _SAMPLING_EPS)
        self.assertEqual(result["top_k"], 1)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_paddleocr_low_top_p(self, mock_stop):
        """PaddleOCR with top_p below _SAMPLING_EPS is clamped."""
        proc = _make_processor(PADDLEOCR_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(PADDLEOCR_VL))

        request = {
            "request_id": "test5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "top_p": 0.0,
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertAlmostEqual(result["top_p"], _SAMPLING_EPS)
        self.assertEqual(result["top_k"], 1)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen_vl_with_reasoning_parser(self, mock_stop):
        """Qwen_vl with reasoning parser (not skipped)."""
        proc = _make_processor(QWEN_VL)
        mock_parser = MagicMock()
        mock_parser.get_model_status.return_value = "think_start"
        proc.reasoning_parser = mock_parser
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(QWEN_VL))

        request = {
            "request_id": "test6",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertTrue(result["enable_thinking"])
        self.assertIn("test6", proc.model_status_dict)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_ernie_response_max_tokens_with_thinking_disabled(self, mock_stop):
        """Ernie with response_max_tokens and enable_thinking=False."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(ERNIE4_5_VL))

        request = {
            "request_id": "test8",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "response_max_tokens": 10,
            "enable_thinking": False,
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertLessEqual(result["max_tokens"], 10)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_prompt_truncation(self, mock_stop):
        """Prompt exceeding max_model_len is truncated."""
        proc = _make_processor(QWEN_VL)
        long_ids = list(range(200))
        outputs = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": long_ids,
            "token_type_ids": [0] * 200,
            "position_ids": [np.array([list(range(200))] * 3)],
        }
        proc.text2ids = MagicMock(return_value=outputs)

        request = {"request_id": "test9", "prompt": "hello " * 100}
        result = proc.process_request_dict(request, max_model_len=50)

        self.assertLessEqual(len(result["prompt_token_ids"]), 49)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_max_tokens_default(self, mock_stop):
        """max_tokens defaults to remaining model len."""
        proc = _make_processor(QWEN_VL)
        proc.text2ids = MagicMock(return_value=self._make_mock_outputs(QWEN_VL))

        request = {"request_id": "test10", "prompt": "hello"}
        result = proc.process_request_dict(request, max_model_len=100)

        expected_max = 100 - len(result["prompt_token_ids"])
        self.assertEqual(result["max_tokens"], max(1, expected_max))

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_max_tokens_capped(self, mock_stop):
        """User max_tokens capped by remaining model len."""
        proc = _make_processor(QWEN_VL)
        proc.text2ids = MagicMock(return_value=self._make_mock_outputs(QWEN_VL))

        request = {"request_id": "test11", "prompt": "hello", "max_tokens": 5000}
        result = proc.process_request_dict(request, max_model_len=100)

        remaining = 100 - len(result["prompt_token_ids"])
        self.assertEqual(result["max_tokens"], remaining)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_paddleocr_skips_bad_words(self, mock_stop):
        """PaddleOCR skips bad_words processing (cfg.has_bad_words=False)."""
        proc = _make_processor(PADDLEOCR_VL)
        proc.update_bad_words = MagicMock()
        proc.text2ids = MagicMock(return_value=self._make_mock_outputs(PADDLEOCR_VL))

        request = {"request_id": "test12", "prompt": "hi", "bad_words": ["test"]}
        proc.process_request_dict(request, max_model_len=100)

        proc.update_bad_words.assert_not_called()

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_eos_token_ids_not_overwritten(self, mock_stop):
        """Existing eos_token_ids preserved."""
        proc = _make_processor(QWEN_VL)
        proc.text2ids = MagicMock(return_value=self._make_mock_outputs(QWEN_VL))

        request = {"request_id": "test13", "prompt": "hi", "eos_token_ids": [99]}
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertEqual(result["eos_token_ids"], [99])

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_ernie_reasoning_max_tokens_default(self, mock_stop):
        """Ernie sets default reasoning_max_tokens."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(ERNIE4_5_VL))

        request = {
            "request_id": "test14",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertIn("reasoning_max_tokens", result)
        self.assertEqual(result["reasoning_max_tokens"], max(int(result["max_tokens"] * 0.8), 1))

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_prompt_path_flow(self, mock_stop):
        """Prompt path flow."""
        proc = _make_processor(QWEN_VL)
        proc.text2ids = MagicMock(return_value=self._make_mock_outputs(QWEN_VL))

        request = {
            "request_id": "test15",
            "prompt": "hello world",
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertEqual(result["prompt_token_ids"], list(np.array([1, 2, 3, 4, 5], dtype=np.int64)))
        self.assertIn("multimodal_inputs", result)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen3_stop_tokens_variant(self, mock_stop):
        """Qwen3 uses stop_tokens_variant='qwen3' with update_stop_seq."""
        proc = _make_processor(QWEN3_VL)
        proc.request2ids = MagicMock(return_value=self._make_mock_outputs(QWEN3_VL))
        proc.update_stop_seq = MagicMock(return_value=([[100]], [1]))

        request = {
            "request_id": "test16",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "stop": ["<stop>"],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        proc.update_stop_seq.assert_called_once_with(["<stop>"])
        self.assertEqual(result["stop_token_ids"], [[100]])
        self.assertEqual(result["stop_seqs_len"], [1])
        # process_stop_token_ids should NOT be called for qwen3
        mock_stop.assert_not_called()


# ===================================================================
# _load_tokenizer (just the branch coverage, actual loading is mocked)
# ===================================================================
class TestLoadTokenizer(unittest.TestCase):

    def test_auto_tokenizer_path(self):
        """Non-ernie path loads AutoTokenizer via paddleformers."""
        proc = _make_processor(QWEN_VL)
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"paddleformers.transformers": MagicMock(AutoTokenizer=mock_auto_tokenizer)}):
            result = proc._load_tokenizer()

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("/mock/model", padding_side="left", use_fast=True)
        self.assertEqual(result, mock_tokenizer)


# ===================================================================
# MultiModalProcessor — text2ids and _add_text tests
# ===================================================================
class TestText2ids(unittest.TestCase):
    """Tests for MultiModalProcessor.text2ids and _add_text."""

    def test_text_only(self):
        """Text with no placeholders."""
        proc = _make_processor(QWEN_VL)
        proc.enc = MagicMock(spec=QwenEncoding)
        outputs_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict

        # _add_text will be called; let it work by using the real _add_text
        proc.tokenizer.tokenize.return_value = ["hello", "world"]
        proc.tokenizer.convert_tokens_to_ids.return_value = [10, 20]

        proc.text2ids("hello world")
        proc.enc._make_outputs.assert_called_once()
        # _add_text should have been invoked (via text2ids → _add_text)
        # Since enc is mocked, add_text_positions is called
        proc.enc.add_text_positions.assert_called()

    def test_text_with_image_placeholder(self):
        """Text with image placeholder dispatches to enc.add_image."""
        proc = _make_processor(QWEN_VL)
        proc.enc = MagicMock(spec=QwenEncoding)
        outputs_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict

        mock_img = MagicMock()
        proc.tokenizer.tokenize.return_value = ["hi"]
        proc.tokenizer.convert_tokens_to_ids.return_value = [10]

        proc.text2ids(
            "hi<|image_pad|>",
            images=[mock_img],
            image_uuid=["img1"],
        )
        proc.enc.add_image.assert_called_once_with(mock_img, outputs_dict, "img1")

    def test_text_with_video_placeholder(self):
        """Text with video placeholder dispatches to enc.load_video + add_video."""
        proc = _make_processor(QWEN_VL)
        proc.enc = MagicMock(spec=QwenEncoding)
        outputs_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict

        mock_frames = MagicMock()
        mock_meta = {"fps": 2}
        proc.enc.load_video.return_value = (mock_frames, mock_meta)
        proc.tokenizer.tokenize.return_value = ["hi"]
        proc.tokenizer.convert_tokens_to_ids.return_value = [10]

        proc.text2ids(
            "hi<|video_pad|>",
            videos=["http://video.mp4"],
            video_uuid=["vid1"],
        )
        proc.enc.load_video.assert_called_once_with("http://video.mp4", {})
        proc.enc.add_video.assert_called_once_with(mock_frames, outputs_dict, "vid1", meta=mock_meta)

    def test_text_with_video_dict(self):
        """Video item as dict with 'video' key."""
        proc = _make_processor(QWEN_VL)
        proc.enc = MagicMock(spec=QwenEncoding)
        outputs_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict

        mock_frames = MagicMock()
        mock_meta = {"fps": 2}
        proc.enc.load_video.return_value = (mock_frames, mock_meta)
        proc.tokenizer.tokenize.return_value = []
        proc.tokenizer.convert_tokens_to_ids.return_value = []

        video_item = {"video": "http://video.mp4", "fps": 5}
        proc.text2ids(
            "<|video_pad|>",
            videos=[video_item],
        )
        proc.enc.load_video.assert_called_once_with("http://video.mp4", video_item)

    def test_text_with_processed_image(self):
        """Processed image (tuple) dispatches to enc.add_processed_image."""
        proc = _make_processor(QWEN_VL)
        proc.enc = MagicMock(spec=QwenEncoding)
        outputs_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict
        proc.tokenizer.tokenize.return_value = []
        proc.tokenizer.convert_tokens_to_ids.return_value = []

        cached_img = (np.zeros((4,)), {"thw": (1, 2, 2)})
        proc.text2ids(
            "<|image_pad|>",
            images=[cached_img],
            image_uuid=["cached_uuid"],
        )
        proc.enc.add_processed_image.assert_called_once_with(cached_img, outputs_dict, "cached_uuid")

    def test_text_with_processed_video(self):
        """Processed video (tuple) dispatches to enc.add_processed_video."""
        proc = _make_processor(QWEN_VL)
        proc.enc = MagicMock(spec=QwenEncoding)
        outputs_dict = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict
        proc.tokenizer.tokenize.return_value = []
        proc.tokenizer.convert_tokens_to_ids.return_value = []

        cached_vid = (np.zeros((8,)), {"thw": (2, 2, 2), "fps": 4})
        proc.text2ids(
            "<|video_pad|>",
            videos=[cached_vid],
            video_uuid=["cached_vid_uuid"],
        )
        proc.enc.add_processed_video.assert_called_once_with(cached_vid, outputs_dict, "cached_vid_uuid")


class TestAddText(unittest.TestCase):
    """Tests for MultiModalProcessor._add_text."""

    def test_empty_string_noop(self):
        proc = _make_processor(QWEN_VL)
        outputs = {"input_ids": [], "token_type_ids": []}
        proc._add_text("", outputs)
        self.assertEqual(outputs["input_ids"], [])

    def test_string_tokenization(self):
        proc = _make_processor(QWEN_VL)
        proc.tokenizer.tokenize.return_value = ["hello"]
        proc.tokenizer.convert_tokens_to_ids.return_value = [42]
        outputs = {"input_ids": [], "token_type_ids": []}
        proc._add_text("hello", outputs)
        self.assertEqual(outputs["input_ids"], [42])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]])
        proc.enc.add_text_positions.assert_called_once_with(outputs, 1)

    def test_list_tokens(self):
        proc = _make_processor(QWEN_VL)
        outputs = {"input_ids": [], "token_type_ids": []}
        proc._add_text([10, 20, 30], outputs)
        self.assertEqual(outputs["input_ids"], [10, 20, 30])
        self.assertEqual(outputs["token_type_ids"], [0, 0, 0])
        proc.enc.add_text_positions.assert_called_once_with(outputs, 3)


# ===================================================================
# MultiModalProcessor — _extract_mm_items tests
# ===================================================================
class TestExtractMmItems(unittest.TestCase):
    """Tests for MultiModalProcessor._extract_mm_items."""

    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_image_and_video_extraction(self, mock_parse):
        """Extract images and videos from parsed messages."""
        mock_parse.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": "img_data", "uuid": "img_uuid"},
                    {"type": "video", "data": "vid_data", "uuid": "vid_uuid"},
                ],
            }
        ]
        proc = _make_processor(QWEN_VL)
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}
        request = {"messages": [{"role": "user", "content": "test"}]}

        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = proc._extract_mm_items(request)
        self.assertEqual(images, ["img_data"])
        self.assertEqual(videos, ["vid_data"])
        self.assertEqual(image_uuid, ["img_uuid"])
        self.assertEqual(video_uuid, ["vid_uuid"])
        self.assertEqual(len(mm_items), 2)

    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_missing_data_without_cache_raises(self, mock_parse):
        """Missing data without processor cache raises ValueError."""
        mock_parse.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": None, "uuid": "missing_uuid"},
                ],
            }
        ]
        proc = _make_processor(QWEN_VL)
        proc.enable_processor_cache = False
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}
        request = {"messages": [{"role": "user", "content": "test"}]}

        with self.assertRaises(ValueError, msg="Missing items cannot be retrieved"):
            proc._extract_mm_items(request)

    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_audio_type_silently_skipped(self, mock_parse):
        """Audio type is not in ['image','video'] so it's silently skipped."""
        mock_parse.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "data": "audio_data", "uuid": "audio_uuid"},
                ],
            }
        ]
        proc = _make_processor(QWEN_VL)
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}
        request = {"messages": [{"role": "user", "content": "test"}]}

        images, videos, image_uuid, video_uuid, dealer, missing_idx, mm_items = proc._extract_mm_items(request)
        self.assertEqual(images, [])
        self.assertEqual(videos, [])
        self.assertEqual(mm_items, [])

    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_text_only_content_dict(self, mock_parse):
        """Text-only content dicts are skipped (no image/video type)."""
        mock_parse.return_value = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "just text"}],
            },
        ]
        proc = _make_processor(QWEN_VL)
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}
        request = {"messages": [{"role": "user", "content": "just text"}]}

        images, videos, *_ = proc._extract_mm_items(request)
        self.assertEqual(images, [])
        self.assertEqual(videos, [])


# ===================================================================
# MultiModalProcessor — request2ids tests
# ===================================================================
class TestRequest2ids(unittest.TestCase):
    """Tests for MultiModalProcessor.request2ids."""

    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_request2ids_basic(self, mock_parse):
        """Basic request2ids flow without cache."""
        mock_parse.return_value = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]
        proc = _make_processor(QWEN_VL)
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}

        proc.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        proc.tokenizer.tokenize.return_value = ["formatted", "prompt"]
        proc.tokenizer.convert_tokens_to_ids.return_value = [10, 20]

        outputs_dict = {
            "input_ids": [10, 20],
            "token_type_ids": [0, 0],
            "position_ids": [np.array([[0, 1], [0, 1], [0, 1]])],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 2,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
            "fps": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict
        proc.text2ids = MagicMock(return_value=outputs_dict)

        request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
        proc.request2ids(request)

        self.assertEqual(request["prompt_tokens"], "formatted prompt")
        proc.text2ids.assert_called_once()

    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_request2ids_ernie_passes_request(self, mock_parse):
        """Ernie passes full request to apply_chat_template."""
        mock_parse.return_value = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        proc = _make_processor(ERNIE4_5_VL)
        proc.role_prefixes = {"user": "", "assistant": "", "system": "", "tool": ""}

        proc.tokenizer.apply_chat_template = MagicMock(return_value="ernie prompt")
        outputs_dict = {
            "input_ids": [10],
            "token_type_ids": [0],
            "position_ids": [[0, 0, 0]],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 1,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "mm_positions": [],
            "mm_hashes": [],
        }
        proc.enc._make_outputs.return_value = outputs_dict
        proc.text2ids = MagicMock(return_value=outputs_dict)

        request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        proc.request2ids(request)

        # For ernie, the full request (not parsed messages) is passed
        call_args = proc.tokenizer.apply_chat_template.call_args
        self.assertIs(call_args[0][0], request)


# =====================================================================
# process_response_dict – dispatch
# =====================================================================


class TestProcessResponseDictDispatch(unittest.TestCase):
    """Test process_response_dict dispatches to streaming / normal."""

    def test_stream_true_dispatches_to_streaming(self):
        proc = _make_processor(QWEN_VL)
        proc.process_response_dict_streaming = MagicMock(return_value={"text": "ok"})
        result = proc.process_response_dict({"data": 1, "outputs": {"token_ids": [1]}}, stream=True)
        proc.process_response_dict_streaming.assert_called_once()
        self.assertEqual(result, {"text": "ok"})

    def test_stream_false_dispatches_to_normal(self):
        proc = _make_processor(ERNIE4_5_VL)
        proc.process_response_dict_normal = MagicMock(return_value={"text": "ok"})
        result = proc.process_response_dict({"data": 1, "outputs": {"token_ids": [1]}}, stream=False)
        proc.process_response_dict_normal.assert_called_once()
        self.assertEqual(result, {"text": "ok"})

    def test_default_stream_is_true(self):
        proc = _make_processor(QWEN3_VL)
        proc.process_response_dict_streaming = MagicMock(return_value={})
        proc.process_response_dict({"outputs": {"token_ids": [1]}})
        proc.process_response_dict_streaming.assert_called_once()


# =====================================================================
# process_response_dict_normal
# =====================================================================


class TestProcessResponseDictNormal(unittest.TestCase):
    """Tests for the non-streaming response handler."""

    def _make(self, reasoning=False, tool=False):
        proc = _make_processor(ERNIE4_5_VL)
        # stub ids2tokens to return deterministic values
        proc.ids2tokens = lambda token_ids, task_id: ("decoded", [], "prev")
        proc.model_status_dict = {"req-1": "think_start"}
        proc.decode_status = {"req-1": "some_state"}

        if reasoning:
            mock_rp = MagicMock()
            mock_rp.extract_reasoning_content.return_value = ("reasoning text", "final text")
            proc.reasoning_parser = mock_rp
            proc.tokenizer.tokenize.return_value = ["r1", "r2", "r3"]

        if tool:

            class _ToolInfo:
                tools_called = True
                tool_calls = [{"name": "my_tool"}]

            mock_tp_instance = MagicMock()
            mock_tp_instance.extract_tool_calls.return_value = _ToolInfo()
            mock_tp_cls = MagicMock(return_value=mock_tp_instance)
            proc.tool_parser_obj = mock_tp_cls

        return proc

    def test_basic_finished(self):
        """Finished response builds full_text and cleans decode_status."""
        proc = self._make()
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, 11]},
        }
        result = proc.process_response_dict_normal(resp)
        self.assertEqual(result["outputs"]["text"], "prevdecoded")
        self.assertEqual(result["outputs"]["completion_tokens"], "prevdecoded")
        self.assertNotIn("req-1", proc.decode_status)
        self.assertNotIn("req-1", proc.model_status_dict)

    def test_eos_stripped_when_finished(self):
        """EOS token at the end is removed when finished and include_stop_str_in_output is False."""
        proc = self._make()
        eos = proc.eos_token_ids[0]  # 2
        called_with = {}

        def mock_ids2tokens(token_ids, task_id):
            called_with["token_ids"] = list(token_ids)
            return ("decoded", [], "prev")

        proc.ids2tokens = mock_ids2tokens
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, eos]},
        }
        proc.process_response_dict_normal(resp, include_stop_str_in_output=False)
        # EOS should have been stripped before ids2tokens was called
        self.assertEqual(called_with["token_ids"], [10])

    def test_eos_kept_when_include_stop_str(self):
        """EOS is kept when include_stop_str_in_output=True."""
        proc = self._make()
        eos = proc.eos_token_ids[0]
        called_with = {}

        def mock_ids2tokens(token_ids, task_id):
            called_with["token_ids"] = list(token_ids)
            return ("decoded", [], "prev")

        proc.ids2tokens = mock_ids2tokens
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, eos]},
        }
        proc.process_response_dict_normal(resp, include_stop_str_in_output=True)
        self.assertEqual(called_with["token_ids"], [10, eos])

    def test_not_finished_returns_without_text(self):
        """Non-finished response does not set completion_tokens / text."""
        proc = self._make()
        resp = {
            "finished": False,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_normal(resp)
        self.assertNotIn("text", result["outputs"])
        # decode_status should NOT be cleaned up
        self.assertIn("req-1", proc.decode_status)

    def test_direct_decode_uses_tokenizer_decode(self):
        """direct_decode=True uses tokenizer.decode instead of ids2tokens."""
        proc = self._make()
        proc.tokenizer.decode.return_value = "direct_result"
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, 11]},
        }
        result = proc.process_response_dict_normal(resp, direct_decode=True)
        proc.tokenizer.decode.assert_called_with([10, 11])
        self.assertEqual(result["outputs"]["text"], "direct_result")

    def test_with_reasoning_parser(self):
        """Reasoning parser extracts reasoning_content and adjusts text."""
        proc = self._make(reasoning=True)
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_normal(resp)
        self.assertEqual(result["outputs"]["text"], "final text")
        self.assertEqual(result["outputs"]["reasoning_content"], "reasoning text")
        self.assertEqual(result["outputs"]["reasoning_token_num"], 3)

    def test_with_tool_parser(self):
        """Tool parser extracts tool_calls from full_text."""
        proc = self._make(tool=True)
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_normal(resp)
        self.assertIn("tool_calls", result["outputs"])
        self.assertEqual(result["outputs"]["tool_calls"][0]["name"], "my_tool")


# =====================================================================
# process_response_dict_streaming
# =====================================================================


class TestProcessResponseDictStreaming(unittest.TestCase):
    """Tests for the streaming response handler."""

    def _make(self, reasoning=False, tool=False):
        proc = _make_processor(QWEN_VL)
        proc.ids2tokens = lambda token_ids, task_id: ("delta", [2, 3], "previous")
        proc.model_status_dict = {"req-1": "think_start"}
        proc.decode_status = {"req-1": "state"}

        if reasoning:
            mock_rp = MagicMock()

            class _RD:
                reasoning_content = "think"
                content = "answer"

            mock_rp.extract_reasoning_content_streaming.return_value = _RD()
            proc.reasoning_parser = mock_rp
            proc.tokenizer.tokenize.return_value = ["t1", "t2"]

        if tool:

            class _TD:
                tool_calls = [{"name": "stream_tool"}]
                content = "tool_text"

            mock_tp_instance = MagicMock()
            mock_tp_instance.extract_tool_calls_streaming.return_value = _TD()
            mock_tp_cls = MagicMock(return_value=mock_tp_instance)
            proc.tool_parser_obj = mock_tp_cls

        return proc

    def test_basic_non_finished(self):
        """Non-finished streaming sets delta text fields correctly."""
        proc = self._make()
        resp = {
            "finished": False,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, 11]},
        }
        result = proc.process_response_dict_streaming(resp)
        self.assertEqual(result["outputs"]["text"], "delta")
        self.assertEqual(result["outputs"]["completion_tokens"], "delta")
        self.assertFalse(result["outputs"]["skipped"])
        self.assertIsNone(result["outputs"]["tool_calls"])
        self.assertEqual(result["outputs"]["reasoning_content"], "")

    def test_finished_cleans_up_status(self):
        """Finished streaming cleans up decode_status and model_status_dict."""
        proc = self._make()
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        proc.process_response_dict_streaming(resp)
        self.assertNotIn("req-1", proc.decode_status)
        self.assertNotIn("req-1", proc.model_status_dict)

    def test_finished_cleans_up_tool_parser_dict(self):
        """Finished streaming cleans up tool_parser_dict entry."""
        proc = self._make(tool=True)
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        proc.process_response_dict_streaming(resp)
        self.assertNotIn("req-1", proc.tool_parser_dict)

    def test_eos_stripped_when_finished(self):
        """EOS token stripped on finished streaming response."""
        proc = self._make()
        eos = proc.eos_token_ids[0]
        called_with = {}

        def mock_ids2tokens(token_ids, task_id):
            called_with["token_ids"] = list(token_ids)
            return ("delta", [2, 3], "prev")

        proc.ids2tokens = mock_ids2tokens
        resp = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, eos]},
        }
        proc.process_response_dict_streaming(resp)
        self.assertEqual(called_with["token_ids"], [10])

    def test_with_reasoning_parser(self):
        """Reasoning parser populates reasoning_content and adjusts text."""
        proc = self._make(reasoning=True)
        resp = {
            "finished": False,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_streaming(resp)
        self.assertEqual(result["outputs"]["reasoning_content"], "think")
        self.assertEqual(result["outputs"]["text"], "answer")
        self.assertEqual(result["outputs"]["reasoning_token_num"], 2)

    def test_reasoning_parser_returns_none_sets_skipped(self):
        """When reasoning parser returns None on non-end, skipped=True."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.ids2tokens = lambda tids, tid: ("delta", [2], "prev")
        proc.model_status_dict = {"req-1": "s"}
        proc.decode_status = {"req-1": "st"}
        mock_rp = MagicMock()
        mock_rp.extract_reasoning_content_streaming.return_value = None
        proc.reasoning_parser = mock_rp

        resp = {
            "finished": False,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_streaming(resp)
        self.assertTrue(result["outputs"]["skipped"])

    def test_with_tool_parser(self):
        """Tool parser extracts tool_calls in streaming mode."""
        proc = self._make(tool=True)
        resp = {
            "finished": False,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_streaming(resp)
        self.assertEqual(result["outputs"]["tool_calls"][0]["name"], "stream_tool")
        self.assertEqual(result["outputs"]["text"], "tool_text")
        self.assertFalse(result["outputs"]["skipped"])

    def test_tool_parser_returns_none_sets_skipped(self):
        """When tool parser returns None on non-end, skipped=True."""
        proc = _make_processor(QWEN3_VL)
        proc.ids2tokens = lambda tids, tid: ("delta", [2], "prev")
        proc.model_status_dict = {"req-1": "s"}
        proc.decode_status = {"req-1": "st"}
        mock_tp_instance = MagicMock()
        mock_tp_instance.extract_tool_calls_streaming.return_value = None
        mock_tp_cls = MagicMock(return_value=mock_tp_instance)
        proc.tool_parser_obj = mock_tp_cls

        resp = {
            "finished": False,
            "request_id": "req-1",
            "outputs": {"token_ids": [10]},
        }
        result = proc.process_response_dict_streaming(resp)
        self.assertTrue(result["outputs"]["skipped"])


# =====================================================================
# Cache lifecycle – get_processor_cache / update_processor_cache / _update_mm_cache
# =====================================================================


class TestGetProcessorCache(unittest.TestCase):
    """Tests for MultiModalProcessor.get_processor_cache."""

    def test_sends_pickled_hashes_and_returns_items(self):
        import pickle

        proc = _make_processor(QWEN_VL)
        mock_socket = MagicMock()
        cached = [{"data": "img_pixels"}, {"data": "vid_pixels"}]
        mock_socket.recv_multipart.return_value = (b"", pickle.dumps(cached))

        result = proc.get_processor_cache(mock_socket, ["hash1", "hash2"])
        # Verify send
        call_args = mock_socket.send_multipart.call_args[0][0]
        self.assertEqual(call_args[0], b"")
        self.assertEqual(pickle.loads(call_args[1]), ["hash1", "hash2"])
        # Verify return
        self.assertEqual(result, cached)

    def test_empty_hashes(self):
        import pickle

        proc = _make_processor(ERNIE4_5_VL)
        mock_socket = MagicMock()
        mock_socket.recv_multipart.return_value = (b"", pickle.dumps([]))

        result = proc.get_processor_cache(mock_socket, [])
        self.assertEqual(result, [])


class TestUpdateProcessorCache(unittest.TestCase):
    """Tests for MultiModalProcessor.update_processor_cache."""

    def test_sends_pickled_hashes_and_items(self):
        import pickle

        proc = _make_processor(QWEN3_VL)
        mock_socket = MagicMock()
        hashes = ["h1", "h2"]
        items = [("pixels1", {"thw": (1, 8, 8)}), ("pixels2", {"thw": (2, 4, 4)})]

        proc.update_processor_cache(mock_socket, hashes, items)

        call_args = mock_socket.send_multipart.call_args[0][0]
        self.assertEqual(call_args[0], b"")
        sent_data = pickle.loads(call_args[1])
        self.assertEqual(sent_data[0], hashes)
        self.assertEqual(sent_data[1], items)


class TestUpdateMmCache(unittest.TestCase):
    """Tests for MultiModalProcessor._update_mm_cache."""

    def test_skips_missing_idx_and_caches_rest(self):
        proc = _make_processor(QWEN_VL)
        mock_dealer = MagicMock()

        mm_items = [
            {"type": "image", "uuid": "img0"},  # idx 0 – missing
            {"type": "image", "uuid": "img1"},  # idx 1 – to cache
            {"type": "video", "uuid": "vid0"},  # idx 2 – to cache
        ]
        outputs = {
            "images": [np.array([1]), np.array([2]), np.array([3])],
            "grid_thw": [
                np.array([[1, 8, 8]]),  # 2D
                np.array([1, 4, 4]),  # 1D
                np.array([[2, 6, 6]]),  # 2D
            ],
            "mm_hashes": ["img0_hash", "img1_hash", "vid0_hash"],
            "fps": [None, None, 6],
        }

        proc._update_mm_cache(mock_dealer, missing_idx=[0], mm_items=mm_items, outputs=outputs)

        proc.update_processor_cache = MagicMock()
        # Re-run with the mock to inspect args
        proc._update_mm_cache(mock_dealer, missing_idx=[0], mm_items=mm_items, outputs=outputs)

        call_args = proc.update_processor_cache.call_args
        cached_hashes = call_args[0][1]
        cached_items = call_args[0][2]
        self.assertEqual(cached_hashes, ["img1_hash", "vid0_hash"])
        self.assertEqual(cached_items[0][1]["thw"], (1, 4, 4))
        self.assertEqual(cached_items[1][1]["thw"], (2, 6, 6))
        self.assertEqual(cached_items[1][1]["fps"], 6)

    def test_no_items_to_cache_skips_update(self):
        """When all items are missing, update_processor_cache is not called."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.update_processor_cache = MagicMock()

        mm_items = [{"type": "image", "uuid": "img0"}]
        outputs = {
            "images": [np.array([1])],
            "grid_thw": [np.array([1, 8, 8])],
            "mm_hashes": ["h0"],
        }
        proc._update_mm_cache(MagicMock(), missing_idx=[0], mm_items=mm_items, outputs=outputs)
        proc.update_processor_cache.assert_not_called()

    def test_1d_grid_thw(self):
        """1D grid_thw array is handled correctly."""
        proc = _make_processor(QWEN3_VL)
        proc.update_processor_cache = MagicMock()

        mm_items = [{"type": "image", "uuid": "img0"}]
        outputs = {
            "images": [np.array([1])],
            "grid_thw": [np.array([1, 16, 16])],
            "mm_hashes": ["hash0"],
        }
        proc._update_mm_cache(MagicMock(), missing_idx=[], mm_items=mm_items, outputs=outputs)

        cached_items = proc.update_processor_cache.call_args[0][2]
        self.assertEqual(cached_items[0][1]["thw"], (1, 16, 16))

    def test_2d_grid_thw(self):
        """2D grid_thw array extracts first row."""
        proc = _make_processor(QWEN_VL)
        proc.update_processor_cache = MagicMock()

        mm_items = [{"type": "image", "uuid": "img0"}]
        outputs = {
            "images": [np.array([1])],
            "grid_thw": [np.array([[3, 12, 12]])],
            "mm_hashes": ["hash0"],
        }
        proc._update_mm_cache(MagicMock(), missing_idx=[], mm_items=mm_items, outputs=outputs)

        cached_items = proc.update_processor_cache.call_args[0][2]
        self.assertEqual(cached_items[0][1]["thw"], (3, 12, 12))


# =====================================================================
# _extract_mm_items with cache enabled
# =====================================================================


class TestExtractMmItemsWithCache(unittest.TestCase):
    """Tests for _extract_mm_items when processor cache is enabled."""

    @patch("fastdeploy.input.multimodal_processor.zmq")
    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_cache_hit_fills_missing_data(self, mock_parse, mock_zmq):
        """Missing data items are filled from processor cache."""
        mock_parse.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": None, "uuid": "missing_img"},
                    {"type": "image", "data": "present_data", "uuid": "present_img"},
                ],
            }
        ]
        proc = _make_processor(QWEN_VL)
        proc.enable_processor_cache = True
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}
        proc.get_processor_cache = MagicMock(return_value=["cached_data"])

        fake_socket = MagicMock()
        mock_zmq.Context.return_value.socket.return_value = fake_socket
        mock_zmq.DEALER = "DEALER"

        request = {"messages": [{"role": "user", "content": "test"}]}
        images, videos, img_uuid, vid_uuid, dealer, missing_idx, mm_items = proc._extract_mm_items(request)

        proc.get_processor_cache.assert_called_once_with(fake_socket, ["missing_img"])
        self.assertEqual(images, ["cached_data", "present_data"])
        self.assertEqual(missing_idx, [0])

    @patch("fastdeploy.input.multimodal_processor.zmq")
    @patch("fastdeploy.input.multimodal_processor.parse_chat_messages")
    def test_cache_miss_raises(self, mock_parse, mock_zmq):
        """Missing item not found in cache raises ValueError."""
        mock_parse.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "data": None, "uuid": "no_cache"},
                ],
            }
        ]
        proc = _make_processor(ERNIE4_5_VL)
        proc.enable_processor_cache = True
        proc.role_prefixes = {"user": "", "assistant": "", "system": ""}
        proc.get_processor_cache = MagicMock(return_value=[None])

        fake_socket = MagicMock()
        mock_zmq.Context.return_value.socket.return_value = fake_socket
        mock_zmq.DEALER = "DEALER"

        request = {"messages": [{"role": "user", "content": "test"}]}
        with self.assertRaisesRegex(ValueError, "Missing item 0 not found in processor cache"):
            proc._extract_mm_items(request)


if __name__ == "__main__":
    unittest.main()
