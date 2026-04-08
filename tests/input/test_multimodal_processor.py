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

from fastdeploy.input.multimodal_processor import (
    _DEFAULT_MM_LIMITS,
    _SAMPLING_EPS,
    ERNIE4_5_VL,
    PADDLEOCR_VL,
    QWEN3_VL,
    QWEN_VL,
    MultiModalProcessor,
)
from fastdeploy.input.utils import IDS_TYPE_FLAG


def _make_processor(model_type, **overrides):
    """Create a MultiModalProcessor instance with __init__ bypassed.

    Manually sets the minimum attributes required by the methods under test.
    """
    with patch.object(MultiModalProcessor, "__init__", return_value=None):
        proc = MultiModalProcessor.__new__(MultiModalProcessor)
    proc.model_type = model_type
    proc.config = MagicMock()
    proc.enable_processor_cache = False
    proc.model_name_or_path = "/mock/model"
    proc.tokenizer_type = "ernie4_5" if model_type == ERNIE4_5_VL else "auto"
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

    # Mock processor (the internal DataProcessor)
    processor = MagicMock()
    processor.image_token_id = 151655
    processor.video_token_id = 151656
    processor.image_patch_id = 151655
    processor.spatial_conv_size = 14
    processor.mm_num_tokens = MagicMock(return_value=1)
    processor._compute_text_positions.return_value = np.array([[3, 4], [3, 4], [3, 4]])
    proc.processor = processor

    # Set attributes normally set by _init_mm_config
    if model_type in (QWEN_VL, QWEN3_VL):
        proc.image_patch_id = processor.image_token_id
    elif model_type == PADDLEOCR_VL:
        proc.image_patch_id = processor.image_patch_id
    elif model_type == ERNIE4_5_VL:
        proc.image_patch_id = processor.image_patch_id
        proc.spatial_conv_size = processor.spatial_conv_size

    # Apply any overrides
    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


# ===================================================================
# __init__ validation
# ===================================================================
class TestMultiModalProcessorInitValidation(unittest.TestCase):

    def test_unsupported_model_type_raises(self):
        """Line 86: unsupported model_type should raise ValueError."""
        with self.assertRaises(ValueError):
            # Directly construct with unsupported model_type to trigger validation
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
        """Lines 196, 198-204: valid kwargs for qwen model type."""
        proc = _make_processor(QWEN_VL)
        kwargs = {"video_max_frames": 10, "video_min_frames": 1}
        result = proc._parse_processor_kwargs(kwargs)
        self.assertEqual(result, kwargs)

    def test_valid_ernie_kwargs(self):
        """Lines 193-194: valid kwargs for ernie model type."""
        proc = _make_processor(ERNIE4_5_VL)
        kwargs = {"spatial_conv_size": 2, "temporal_conv_size": 1, "video_max_frames": 32}
        result = proc._parse_processor_kwargs(kwargs)
        self.assertEqual(result, kwargs)

    def test_invalid_type_not_dict(self):
        """Lines 188-189: non-dict kwargs should return empty."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_processor_kwargs("invalid")
        self.assertEqual(result, {})

    def test_invalid_value_type(self):
        """Lines 199-200: wrong value type should return empty."""
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
        """Lines 219: valid limits merged with defaults."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_limits({"image": 5, "video": 3})
        self.assertEqual(result, {"image": 5, "video": 3, "audio": 1})

    def test_partial_limits(self):
        proc = _make_processor(QWEN_VL)
        result = proc._parse_limits({"image": 10})
        self.assertEqual(result, {"image": 10, "video": 1, "audio": 1})

    def test_invalid_type_returns_defaults(self):
        """Lines 216-217, 220-222: non-dict returns defaults."""
        proc = _make_processor(QWEN_VL)
        result = proc._parse_limits("invalid")
        self.assertEqual(result, dict(_DEFAULT_MM_LIMITS))


# ===================================================================
# _check_mm_limits
# ===================================================================
class TestCheckMMLimits(unittest.TestCase):

    def test_dict_input_within_limits(self):
        """Lines 226-227: dict input within limits passes."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 2, "video": 1, "audio": 1}
        mm_data = {"image": ["img1"], "video": ["vid1"]}
        proc._check_mm_limits(mm_data)  # should not raise

    def test_dict_input_exceeds_limit(self):
        """Lines 247-251: dict input exceeding limit raises ValueError."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        mm_data = {"image": ["img1", "img2"]}
        with self.assertRaises(ValueError) as ctx:
            proc._check_mm_limits(mm_data)
        self.assertIn("Too many image items", str(ctx.exception))

    def test_messages_input_qwen_vl_accepts_url_suffix(self):
        """Lines 229-240: messages with image_url/video_url for qwen_vl."""
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
        """Lines 237: 'image' type also accepted for url_suffix models."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": [{"type": "image", "image": "data"}]},
        ]
        proc._check_mm_limits(messages)

    def test_messages_input_qwen_vl_video_url_type(self):
        """Lines 239-240: video_url type for qwen_vl."""
        proc = _make_processor(QWEN_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": [{"type": "video_url", "video_url": {"url": "file://vid.mp4"}}]},
        ]
        proc._check_mm_limits(messages)

    def test_messages_input_ernie_only_accepts_plain_types(self):
        """Lines 241-245: ernie4_5_vl only accepts 'image'/'video' types, not *_url."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        # image_url should NOT be counted for ernie
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://img.jpg"}}]},
        ]
        proc._check_mm_limits(messages)  # no exception since image_url not counted

    def test_messages_input_ernie_image_type(self):
        """Lines 242-243: ernie 'image' type is counted."""
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
        """Lines 244-245: ernie 'video' type is counted."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.limit_mm_per_prompt = {"image": 1, "video": 1, "audio": 1}
        messages = [
            {"role": "user", "content": [{"type": "video", "video": "data"}]},
        ]
        proc._check_mm_limits(messages)  # within limit

    def test_messages_exceed_video_limit(self):
        """Lines 247-251: video exceeding limit raises ValueError."""
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

    def test_ernie_returns_processor_result(self):
        """Line 271: ernie delegates to processor."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.processor.get_mm_max_tokens_per_item.return_value = {"image": 512}
        result = proc.get_mm_max_tokens_per_item(1024)
        self.assertEqual(result, {"image": 512})

    def test_non_ernie_returns_none(self):
        """Line 272: non-ernie returns None."""
        proc = _make_processor(QWEN_VL)
        self.assertIsNone(proc.get_mm_max_tokens_per_item(1024))

        proc2 = _make_processor(QWEN3_VL)
        self.assertIsNone(proc2.get_mm_max_tokens_per_item(1024))


# ===================================================================
# _process_stop_tokens
# ===================================================================
class TestProcessStopTokens(unittest.TestCase):

    def test_qwen3_vl_stop_handling(self):
        """Lines 348-353: qwen3_vl uses update_stop_seq differently."""
        proc = _make_processor(QWEN3_VL)
        proc.update_stop_seq = MagicMock(return_value=([[100]], [1]))
        request = {"stop": ["<stop>"]}
        proc._process_stop_tokens(request)
        self.assertEqual(request["stop_token_ids"], [[100]])
        self.assertEqual(request["stop_seqs_len"], [1])

    def test_qwen3_vl_no_stop(self):
        """Lines 348-350: qwen3_vl with empty stop list."""
        proc = _make_processor(QWEN3_VL)
        proc.update_stop_seq = MagicMock()
        request = {"stop": []}
        proc._process_stop_tokens(request)
        proc.update_stop_seq.assert_not_called()

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_non_qwen3_uses_process_stop_token_ids(self, mock_process):
        """Lines 354-355: non-qwen3 uses process_stop_token_ids utility."""
        proc = _make_processor(QWEN_VL)
        proc.update_stop_seq = MagicMock()
        request = {}
        proc._process_stop_tokens(request)
        mock_process.assert_called_once_with(request, proc.update_stop_seq)


# ===================================================================
# _process_bad_words
# ===================================================================
class TestProcessBadWords(unittest.TestCase):

    def test_with_bad_words(self):
        """Lines 359-363: bad_words are processed."""
        proc = _make_processor(QWEN_VL)
        proc.update_bad_words = MagicMock(return_value=[100, 200])
        request = {"bad_words": ["bad", "word"], "bad_words_token_ids": [50]}
        proc._process_bad_words(request)
        proc.update_bad_words.assert_called_once_with(["bad", "word"], [50])
        self.assertEqual(request["bad_words_token_ids"], [100, 200])

    def test_without_bad_words(self):
        """Lines 361: no bad_words means no processing."""
        proc = _make_processor(QWEN_VL)
        proc.update_bad_words = MagicMock()
        request = {}
        proc._process_bad_words(request)
        proc.update_bad_words.assert_not_called()


# ===================================================================
# _tokenize_request
# ===================================================================
class TestTokenizeRequest(unittest.TestCase):

    def test_prompt_token_ids_qwen3_vl(self):
        """Lines 369-374: prompt_token_ids path for qwen3_vl."""
        proc = _make_processor(QWEN3_VL)
        expected = {"input_ids": [1, 2, 3]}
        proc.processor.prompt_token_ids2outputs.return_value = expected

        request = {"prompt_token_ids": [1, 2, 3], "messages": [{"role": "user", "content": "hi"}]}
        result = proc._tokenize_request(request)
        self.assertEqual(result, expected)
        self.assertFalse(request.get("enable_thinking", True))  # default_thinking=False for qwen3_vl

    def test_prompt_token_ids_ernie(self):
        """Lines 369-374: prompt_token_ids path for ernie."""
        proc = _make_processor(ERNIE4_5_VL)
        expected = {"input_ids": [1, 2, 3]}
        proc.processor.prompt_token_ids2outputs.return_value = expected

        request = {"prompt_token_ids": [1, 2, 3]}
        result = proc._tokenize_request(request)
        self.assertEqual(result, expected)
        self.assertTrue(request.get("enable_thinking"))  # default_thinking=True for ernie

    def test_prompt_path(self):
        """Lines 376-384: prompt text path."""
        proc = _make_processor(QWEN_VL)
        expected = {"input_ids": [10, 20]}
        proc.processor.text2ids.return_value = expected

        request = {"prompt": "hello", "multimodal_data": {"image": [], "video": []}}
        result = proc._tokenize_request(request)
        proc.processor.text2ids.assert_called_once_with("hello", [], [])
        self.assertEqual(result, expected)

    def test_prompt_path_ernie_sets_prompt_tokens(self):
        """Lines 381-382: ernie sets prompt_tokens from prompt."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.processor.text2ids.return_value = {"input_ids": [1]}

        request = {"prompt": "test prompt"}
        proc._tokenize_request(request)
        self.assertEqual(request["prompt_tokens"], "test prompt")

    def test_messages_path(self):
        """Lines 386-398: messages path."""
        proc = _make_processor(QWEN_VL)
        expected = {"input_ids": [5, 6]}
        proc.processor.request2ids.return_value = expected

        request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        result = proc._tokenize_request(request)
        proc.processor.request2ids.assert_called_once()
        self.assertEqual(result, expected)

    def test_messages_path_with_chat_template_kwargs(self):
        """Lines 389-394: chat_template_kwargs are merged into request."""
        proc = _make_processor(QWEN_VL)
        proc.processor.request2ids.return_value = {"input_ids": [1]}

        request = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
        proc._tokenize_request(request)
        self.assertTrue(request.get("enable_thinking"))

    def test_messages_path_chat_template_kwargs_no_overwrite(self):
        """Lines 393: existing request keys are not overwritten."""
        proc = _make_processor(QWEN_VL)
        proc.processor.request2ids.return_value = {"input_ids": [1]}

        request = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "chat_template_kwargs": {"enable_thinking": True},
            "enable_thinking": False,
        }
        proc._tokenize_request(request)
        self.assertFalse(request["enable_thinking"])

    def test_messages_path_invalid_chat_template_kwargs(self):
        """Lines 395-396: non-dict chat_template_kwargs raises."""
        proc = _make_processor(QWEN_VL)
        request = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "chat_template_kwargs": "invalid",
        }
        with self.assertRaises(ValueError) as ctx:
            proc._tokenize_request(request)
        self.assertIn("must be a dict", str(ctx.exception))

    def test_no_input_raises(self):
        """Lines 400-401: no prompt/messages/prompt_token_ids raises."""
        proc = _make_processor(QWEN_VL)
        with self.assertRaises(ValueError) as ctx:
            proc._tokenize_request({"request_id": "test"})
        self.assertIn("must contain", str(ctx.exception))

    def test_prompt_path_no_multimodal_data(self):
        """Lines 377: prompt with no multimodal_data passes None for images/videos."""
        proc = _make_processor(QWEN_VL)
        proc.processor.text2ids.return_value = {"input_ids": [1]}

        request = {"prompt": "hello"}
        proc._tokenize_request(request)
        proc.processor.text2ids.assert_called_once_with("hello", None, None)


# ===================================================================
# _process_post_tokens
# ===================================================================
class TestProcessPostTokens(unittest.TestCase):

    def test_paddleocr_with_metadata_generated_tokens(self):
        """Lines 405-408: paddleocr_vl appends via _append_completion_tokens_qwen."""
        proc = _make_processor(PADDLEOCR_VL)
        proc._append_completion_tokens_qwen = MagicMock()
        outputs = {"input_ids": [1, 2]}
        request = {"metadata": {"generated_token_ids": [10, 11]}}
        proc._process_post_tokens(request, outputs)
        proc._append_completion_tokens_qwen.assert_called_once_with(outputs, [10, 11])

    def test_paddleocr_without_metadata(self):
        """Lines 405-406: paddleocr_vl with no metadata does nothing."""
        proc = _make_processor(PADDLEOCR_VL)
        proc._append_completion_tokens_qwen = MagicMock()
        outputs = {"input_ids": [1]}
        proc._process_post_tokens({}, outputs)
        proc._append_completion_tokens_qwen.assert_not_called()

    def test_non_paddleocr_with_completion_tokens(self):
        """Lines 410-411: non-paddleocr uses append_completion_tokens."""
        proc = _make_processor(QWEN_VL)
        proc.append_completion_tokens = MagicMock()
        outputs = {"input_ids": [1]}
        request = {"completion_token_ids": [5, 6]}
        proc._process_post_tokens(request, outputs)
        proc.append_completion_tokens.assert_called_once_with(outputs, [5, 6])

    def test_non_paddleocr_without_completion_tokens(self):
        """Lines 410: no completion_token_ids does nothing."""
        proc = _make_processor(QWEN_VL)
        proc.append_completion_tokens = MagicMock()
        outputs = {"input_ids": [1]}
        proc._process_post_tokens({}, outputs)
        proc.append_completion_tokens.assert_not_called()


# ===================================================================
# _apply_reasoning_parser
# ===================================================================
class TestApplyReasoningParser(unittest.TestCase):

    def test_basic_request_id(self):
        """Lines 415-425: basic request_id (no underscore split)."""
        proc = _make_processor(QWEN_VL)
        proc.reasoning_parser = MagicMock()
        proc.reasoning_parser.get_model_status.return_value = "think_start"
        proc.model_status_dict = {}

        request = {"request_id": "req1", "prompt_token_ids": [1, 2, 3]}
        proc._apply_reasoning_parser(request)

        self.assertEqual(proc.model_status_dict["req1"], "think_start")
        self.assertTrue(request["enable_thinking"])

    def test_compound_request_id(self):
        """Lines 416-422: request_id with underscore is split."""
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
        """Lines 420: default n=1."""
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

    def test_ernie_dispatches_to_ernie_method(self):
        """Lines 429-430: ernie dispatches to _append_completion_tokens_ernie."""
        proc = _make_processor(ERNIE4_5_VL)
        proc._append_completion_tokens_ernie = MagicMock()
        inputs = {"input_ids": [1]}
        proc.append_completion_tokens(inputs, [2, 3])
        proc._append_completion_tokens_ernie.assert_called_once_with(inputs, [2, 3])

    def test_non_ernie_dispatches_to_qwen_method(self):
        """Lines 431-432: non-ernie dispatches to _append_completion_tokens_qwen."""
        proc = _make_processor(QWEN_VL)
        proc._append_completion_tokens_qwen = MagicMock()
        inputs = {"input_ids": [1]}
        proc.append_completion_tokens(inputs, [2, 3])
        proc._append_completion_tokens_qwen.assert_called_once_with(inputs, [2, 3])


class TestAppendCompletionTokensQwen(unittest.TestCase):

    def test_qwen_append(self):
        """Lines 436-442: appends tokens, token_type_ids, position_ids for qwen."""
        proc = _make_processor(QWEN_VL)
        multimodal_inputs = {
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
            "cur_position": 3,
        }
        proc._append_completion_tokens_qwen(multimodal_inputs, [4, 5])

        self.assertEqual(multimodal_inputs["input_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(multimodal_inputs["token_type_ids"], [0, 0, 0, 0, 0])
        self.assertEqual(multimodal_inputs["cur_position"], 5)
        self.assertEqual(len(multimodal_inputs["position_ids"]), 2)


class TestAppendCompletionTokensErnie(unittest.TestCase):

    def test_ernie_append(self):
        """Lines 446-453: appends tokens with IDS_TYPE_FLAG for ernie."""
        proc = _make_processor(ERNIE4_5_VL)
        multimodal_inputs = {
            "input_ids": [10, 20],
            "token_type_ids": [IDS_TYPE_FLAG["text"], IDS_TYPE_FLAG["text"]],
            "position_ids": [[0, 0, 0], [1, 1, 1]],
            "cur_position": 2,
        }
        proc._append_completion_tokens_ernie(multimodal_inputs, [30, 40, 50])

        self.assertEqual(multimodal_inputs["input_ids"], [10, 20, 30, 40, 50])
        self.assertEqual(len(multimodal_inputs["token_type_ids"]), 5)
        self.assertTrue(all(t == IDS_TYPE_FLAG["text"] for t in multimodal_inputs["token_type_ids"]))
        self.assertEqual(multimodal_inputs["position_ids"], [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        self.assertEqual(multimodal_inputs["cur_position"], 5)


# ===================================================================
# pack_outputs
# ===================================================================
class TestPackOutputs(unittest.TestCase):

    def test_qwen_with_images(self):
        """Lines 457-474: qwen pack_outputs with image data."""
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
        self.assertEqual(result["position_ids"].dtype, np.int64)
        self.assertEqual(result["image_patch_id"], proc.processor.image_token_id)
        self.assertEqual(result["video_patch_id"], proc.processor.video_token_id)

    def test_qwen_without_images(self):
        """Lines 457-460: empty images set to None."""
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
        """Lines 475-477: ernie uses different position_ids handling."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.image_patch_id = 9999
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
        self.assertEqual(result["position_ids"].dtype, np.int64)
        self.assertEqual(result["position_ids"].shape, (2, 3))
        self.assertEqual(result["image_patch_id"], 9999)
        self.assertNotIn("video_patch_id", result)

    def test_paddleocr_with_images(self):
        """Lines 470-474: paddleocr uses same path as qwen."""
        proc = _make_processor(PADDLEOCR_VL)
        outputs = {
            "images": [np.array([[1, 2]])],
            "grid_thw": [np.array([1, 1, 2])],
            "image_type_ids": [0],
            "input_ids": [1],
            "token_type_ids": [0],
            "position_ids": [np.array([[0], [0], [0]])],
        }
        result = proc.pack_outputs(outputs)

        self.assertIsNotNone(result["images"])
        self.assertEqual(result["image_patch_id"], proc.processor.image_token_id)
        self.assertEqual(result["video_patch_id"], proc.processor.video_token_id)


# ===================================================================
# process_request_dict (integration-level tests for flow coverage)
# ===================================================================
class TestProcessRequestDict(unittest.TestCase):

    def _make_mock_outputs(self):
        return {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3, 4, 5],
            "token_type_ids": [0, 0, 0, 0, 0],
            "position_ids": [np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])],
        }

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen_vl_messages_flow(self, mock_stop):
        """Lines 281-344: full flow for qwen_vl with messages."""
        proc = _make_processor(QWEN_VL)
        proc.processor.request2ids.return_value = self._make_mock_outputs()

        request = {
            "request_id": "test1",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertIn("prompt_token_ids", result)
        self.assertIn("multimodal_inputs", result)
        self.assertEqual(result["prompt_token_ids_len"], len(result["prompt_token_ids"]))
        self.assertFalse(result.get("enable_thinking"))  # qwen_vl sets False

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen3_vl_with_prompt_token_ids(self, mock_stop):
        """Lines 306-307: qwen3_vl with existing prompt_token_ids preserved."""
        proc = _make_processor(QWEN3_VL)
        outputs = self._make_mock_outputs()
        proc.processor.prompt_token_ids2outputs.return_value = outputs

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
        """Lines 291-295, 316-320, 328-329, 339-341: ernie-specific branches."""
        proc = _make_processor(ERNIE4_5_VL)
        outputs = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        }
        proc.processor.request2ids.return_value = outputs

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
        """Lines 331-334: ernie with top_p below _SAMPLING_EPS."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.processor.request2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        }

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
        """Lines 331-334: paddleocr with top_p below _SAMPLING_EPS."""
        proc = _make_processor(PADDLEOCR_VL)
        proc.processor.request2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }

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
        """Lines 336-337: qwen_vl with reasoning parser (not qwen3)."""
        proc = _make_processor(QWEN_VL)
        mock_parser = MagicMock()
        mock_parser.get_model_status.return_value = "think_start"
        proc.reasoning_parser = mock_parser
        proc.processor.request2ids.return_value = self._make_mock_outputs()

        request = {
            "request_id": "test6",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertTrue(result["enable_thinking"])
        self.assertIn("test6", proc.model_status_dict)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_qwen3_skips_reasoning_parser(self, mock_stop):
        """Lines 336: qwen3_vl does NOT apply reasoning parser."""
        proc = _make_processor(QWEN3_VL)
        mock_parser = MagicMock()
        proc.reasoning_parser = mock_parser
        proc.processor.request2ids.return_value = self._make_mock_outputs()

        request = {
            "request_id": "test7",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
        proc.process_request_dict(request, max_model_len=100)

        mock_parser.get_model_status.assert_not_called()

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_ernie_response_max_tokens_with_thinking_disabled(self, mock_stop):
        """Lines 339-341: ernie with response_max_tokens and enable_thinking=False."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.processor.request2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        }

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
        """Lines 313-314: prompt exceeding max_model_len is truncated."""
        proc = _make_processor(QWEN_VL)
        long_ids = list(range(200))
        proc.processor.text2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": long_ids,
            "token_type_ids": [0] * 200,
            "position_ids": [np.array([list(range(200))] * 3)],
        }

        request = {"request_id": "test9", "prompt": "hello " * 100}
        result = proc.process_request_dict(request, max_model_len=50)

        self.assertLessEqual(len(result["prompt_token_ids"]), 49)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_max_tokens_default(self, mock_stop):
        """Lines 322-324: max_tokens defaults to remaining model len."""
        proc = _make_processor(QWEN_VL)
        proc.processor.text2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }

        request = {"request_id": "test10", "prompt": "hello"}
        result = proc.process_request_dict(request, max_model_len=100)

        expected_max = 100 - len(result["prompt_token_ids"])
        self.assertEqual(result["max_tokens"], max(1, expected_max))

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_max_tokens_capped(self, mock_stop):
        """Lines 325-326: user max_tokens capped by remaining model len."""
        proc = _make_processor(QWEN_VL)
        proc.processor.text2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }

        request = {"request_id": "test11", "prompt": "hello", "max_tokens": 5000}
        result = proc.process_request_dict(request, max_model_len=100)

        remaining = 100 - len(result["prompt_token_ids"])
        self.assertEqual(result["max_tokens"], remaining)

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_paddleocr_skips_bad_words(self, mock_stop):
        """Lines 288-289: paddleocr skips _process_bad_words."""
        proc = _make_processor(PADDLEOCR_VL)
        proc.update_bad_words = MagicMock()
        proc.processor.text2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2],
            "token_type_ids": [0, 0],
            "position_ids": [np.array([[0, 1], [0, 1], [0, 1]])],
        }

        request = {"request_id": "test12", "prompt": "hi", "bad_words": ["test"]}
        proc.process_request_dict(request, max_model_len=100)

        proc.update_bad_words.assert_not_called()

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_eos_token_ids_not_overwritten(self, mock_stop):
        """Lines 283-284: existing eos_token_ids preserved."""
        proc = _make_processor(QWEN_VL)
        proc.processor.text2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2],
            "token_type_ids": [0, 0],
            "position_ids": [np.array([[0, 1], [0, 1], [0, 1]])],
        }

        request = {"request_id": "test13", "prompt": "hi", "eos_token_ids": [99]}
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertEqual(result["eos_token_ids"], [99])

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_ernie_reasoning_max_tokens_default(self, mock_stop):
        """Lines 328-329: ernie sets default reasoning_max_tokens."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.processor.request2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        }

        request = {
            "request_id": "test14",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertIn("reasoning_max_tokens", result)
        self.assertEqual(result["reasoning_max_tokens"], max(int(result["max_tokens"] * 0.8), 1))

    @patch("fastdeploy.input.multimodal_processor.process_stop_token_ids")
    def test_prompt_path_flow(self, mock_stop):
        """Lines 297-299, 304-310: prompt path flow."""
        proc = _make_processor(QWEN_VL)
        proc.processor.text2ids.return_value = {
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "input_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 0],
            "position_ids": [np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])],
        }

        request = {
            "request_id": "test15",
            "prompt": "hello world",
        }
        result = proc.process_request_dict(request, max_model_len=100)

        self.assertEqual(result["prompt_token_ids"], [1, 2, 3])
        self.assertIn("multimodal_inputs", result)


# ===================================================================
# _init_mm_config (via _make_processor + direct attribute check)
# ===================================================================
class TestInitMmConfig(unittest.TestCase):

    def test_qwen_vl_sets_image_patch_id(self):
        """Lines 174-175: qwen_vl/qwen3_vl sets image_patch_id from image_token_id."""
        proc = _make_processor(QWEN_VL)
        proc.processor.image_token_id = 12345
        proc._init_mm_config()
        self.assertEqual(proc.image_patch_id, 12345)

    def test_qwen3_vl_sets_image_patch_id(self):
        proc = _make_processor(QWEN3_VL)
        proc.processor.image_token_id = 67890
        proc._init_mm_config()
        self.assertEqual(proc.image_patch_id, 67890)

    def test_paddleocr_sets_image_patch_id(self):
        """Lines 176-177: paddleocr sets image_patch_id from processor."""
        proc = _make_processor(PADDLEOCR_VL)
        proc.processor.image_patch_id = 11111
        proc._init_mm_config()
        self.assertEqual(proc.image_patch_id, 11111)

    def test_ernie_sets_image_patch_id_and_spatial_conv(self):
        """Lines 178-180: ernie sets image_patch_id and spatial_conv_size."""
        proc = _make_processor(ERNIE4_5_VL)
        proc.processor.image_patch_id = 22222
        proc.processor.spatial_conv_size = 14
        proc._init_mm_config()
        self.assertEqual(proc.image_patch_id, 22222)
        self.assertEqual(proc.spatial_conv_size, 14)


# ===================================================================
# _load_tokenizer (just the branch coverage, actual loading is mocked)
# ===================================================================
class TestLoadTokenizer(unittest.TestCase):

    def test_auto_tokenizer_path(self):
        """Lines 123-125: non-ernie path loads AutoTokenizer via paddleformers."""
        proc = _make_processor(QWEN_VL)
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        with patch.dict("sys.modules", {"paddleformers.transformers": MagicMock(AutoTokenizer=mock_auto_tokenizer)}):
            result = proc._load_tokenizer()

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("/mock/model", padding_side="left", use_fast=True)
        self.assertEqual(result, mock_tokenizer)


if __name__ == "__main__":
    unittest.main()
