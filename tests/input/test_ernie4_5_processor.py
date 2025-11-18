import unittest
from unittest.mock import MagicMock, patch

import numpy as np

MODULE_PATH = "fastdeploy.input.ernie4_5_processor"

from fastdeploy.input.ernie4_5_processor import _SAMPLING_EPS, Ernie4_5Processor


class MockTokenizer:
    """Simple fake tokenizer for unit tests."""

    def __init__(self):
        self.bos_token = "<bos>"
        self.bos_token_id = 101
        self.eos_token = "<eos>"
        self.eos_token_id = 102
        self.pad_token_id = 0
        self.vocab_size = 200
        # Any non-None value means chat_template is supported
        self.chat_template = "dummy"

    def tokenize(self, text):
        """
        Make “multi” return multiple tokens to cover multi-token branch
        All other texts return single-token
        """
        if text.startswith("multi"):
            return ["multi", "word"]
        return [text]

    def convert_tokens_to_ids(self, tokens):
        """Token → ID mapping used for specific branch coverage."""
        mapping = {
            "bad": 5,
            " bad": 6,
            "multi": 7,
            "word": 8,
            "oov": 250,  # > vocab_size → out-of-range branch
            " oov": 251,
            "hello": 9,
            "REASON": 42,
        }
        return [mapping.get(t, 1) for t in tokens]

    def decode(self, token_ids, **kwargs):
        """Simple decode implementation."""
        return " ".join(str(t) for t in token_ids)

    def decode_token(self, token_ids, prefix_offset, read_offset):
        """
        Incremental decode:
        - Use read_offset to get new tokens
        - Return new string and updated read_offset
        """
        new_tokens = token_ids[read_offset:]
        decode_str = " ".join(str(t) for t in new_tokens)
        new_read_offset = len(token_ids)
        return decode_str, prefix_offset, new_read_offset

    def apply_chat_template(self, request_or_messages, tokenize, split_special_tokens, add_special_tokens, **kwargs):
        """Minimal chat template behavior."""
        if isinstance(request_or_messages, dict) and "messages" in request_or_messages:
            return " | ".join(m["content"] for m in request_or_messages["messages"])
        return str(request_or_messages)


class ErnieX1ReasoningParser:
    """Fake reasoning parser used to trigger reasoning branch in streaming mode."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_reasoning_content_streaming(
        self,
        previous_texts,
        full_text,
        delta_text,
        previous_token_ids,
        all_token_ids,
        delta_token_ids,
    ):
        """Return a minimal reasoning object."""

        class ReasoningDelta:
            def __init__(self, content):
                self.reasoning_content = content

        return ReasoningDelta("REASON")


class MockToolParser:
    """Fake tool parser used to trigger tool-calling branch."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_tool_calls_streaming(
        self,
        previous_texts,
        full_text,
        delta_text,
        previous_token_ids,
        all_token_ids,
        delta_token_ids,
        response_dict,
    ):
        """Return minimal tool-calling object."""

        class ToolDelta:
            def __init__(self):
                self.tool_calls = [{"name": "fake_tool"}]

        return ToolDelta()


class TestErnie4_5Processor(unittest.TestCase):
    def setUp(self):
        """Patch GenerationConfig, Tokenizer, and get_eos_token_id."""
        self.gen_patcher = patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained", return_value=MagicMock())
        self.tokenizer_patcher = patch(
            f"{MODULE_PATH}.Ernie4_5Tokenizer.from_pretrained", side_effect=lambda path: MockTokenizer()
        )
        self.eos_patcher = patch(
            "paddleformers.trl.llm_utils.get_eos_token_id",
            side_effect=lambda tokenizer, cfg: [tokenizer.eos_token_id],
        )

        self.gen_patcher.start()
        self.tokenizer_patcher.start()
        self.eos_patcher.start()

    def tearDown(self):
        self.gen_patcher.stop()
        self.tokenizer_patcher.stop()
        self.eos_patcher.stop()

    def _make_processor(self, reasoning=False, tool=False):
        """Helper to construct Ernie4_5Processor with mocked tokenizer."""
        reasoning_cls = ErnieX1ReasoningParser if reasoning else None
        tool_cls = MockToolParser if tool else None
        proc = Ernie4_5Processor("dummy-model", reasoning_parser_obj=reasoning_cls, tool_parser_obj=tool_cls)
        proc._apply_default_parameters = lambda req: req  # avoid dependency on parent class
        return proc

    # 1) update_bad_words
    def test_update_bad_words(self):
        proc = self._make_processor()

        bad_words = ["bad", "multi", "oov"]  # single → multi → OOV
        token_ids = proc.update_bad_words(bad_words, bad_words_token_ids=None)

        # Only “bad” and its prefixed-space version should remain
        self.assertEqual(token_ids, [5, 6, 1])

    # 2) process_request_dict → prompt branch
    def test_process_request_dict_with_prompt_string(self):
        proc = self._make_processor()
        req = {
            "prompt": "hello",
            "temperature": 0.0,
            "top_p": 0.0,
        }

        processed = proc.process_request_dict(req, max_model_len=10)

        self.assertIn("eos_token_ids", processed)
        self.assertEqual(processed["eos_token_ids"], [proc.tokenizer.eos_token_id])

        expected_ids = proc.tokenizer.convert_tokens_to_ids(proc.tokenizer.tokenize("hello"))
        self.assertEqual(processed["prompt_token_ids"], expected_ids)

        self.assertEqual(processed["max_tokens"], max(1, 10 - len(expected_ids)))
        self.assertEqual(processed["temperature"], 1)
        self.assertAlmostEqual(processed["top_p"], _SAMPLING_EPS)

        self.assertEqual(processed["prompt_tokens"], "hello")

    # 3) pad_batch_data
    def test_pad_batch_data_right_and_left_and_empty(self):
        proc = self._make_processor()

        insts = [[1, 2], [3]]

        # right pad
        padded, seq_len = proc.pad_batch_data(
            insts, pad_id=0, return_seq_len=True, return_array=True, pad_style="right"
        )
        np.testing.assert_array_equal(padded, np.array([[1, 2], [3, 0]], dtype=np.int64))
        np.testing.assert_array_equal(seq_len, np.array([[2], [1]], dtype=np.int64))

        # left pad
        padded_left, seq_len_left = proc.pad_batch_data(
            insts, pad_id=0, return_seq_len=True, return_array=True, pad_style="left"
        )
        np.testing.assert_array_equal(padded_left, np.array([[1, 2], [0, 3]], dtype=np.int64))
        np.testing.assert_array_equal(seq_len_left, np.array([[2], [1]], dtype=np.int64))

        # empty
        padded_empty, seq_len_empty = proc.pad_batch_data(
            [], pad_id=0, return_seq_len=True, return_array=True, pad_style="right"
        )
        np.testing.assert_array_equal(padded_empty, np.array([[]], dtype=np.int64))
        np.testing.assert_array_equal(seq_len_empty, np.array([], dtype=np.int64))

    # 4) process_response_dict_streaming reasoning + tool branches
    def test_process_response_dict_streaming_with_reasoning_and_tool(self):
        proc = self._make_processor(reasoning=True, tool=True)

        response = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {
                "token_ids": [10, 11],
            },
        }

        result = proc.process_response_dict_streaming(
            response,
            enable_thinking=False,  # reasoning still enabled because class name matches
            include_stop_str_in_output=False,
        )

        outputs = result["outputs"]

        self.assertIn("completion_tokens", outputs)
        self.assertIn("text", outputs)
        self.assertEqual(outputs["completion_tokens"], outputs["text"])

        self.assertIn("reasoning_token_num", outputs)
        self.assertGreaterEqual(outputs["reasoning_token_num"], 0)

        self.assertIn("delta_message", outputs)
        delta_msg = outputs["delta_message"]
        self.assertTrue(hasattr(delta_msg, "tool_calls"))
        self.assertEqual(delta_msg.tool_calls[0]["name"], "fake_tool")

        self.assertNotIn("req-1", proc.decode_status)
        self.assertNotIn("req-1", proc.tool_parser_dict)


if __name__ == "__main__":
    unittest.main()
