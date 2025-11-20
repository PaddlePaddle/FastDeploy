import unittest
from unittest.mock import MagicMock, patch

import numpy as np

MODULE_PATH = "fastdeploy.input.ernie4_5_processor"

from fastdeploy.input.ernie4_5_processor import _SAMPLING_EPS, Ernie4_5Processor


class MockTokenizer:
    """A simple mock tokenizer used to simulate tokenization behavior in unit tests."""

    def __init__(self):
        self.bos_token = "<bos>"
        self.bos_token_id = 101
        self.eos_token = "<eos>"
        self.eos_token_id = 102
        self.pad_token_id = 0
        self.vocab_size = 200
        # Non-None value indicates chat_template support
        self.chat_template = "dummy"

    def tokenize(self, text):
        """Return multi-token output for 'multi*' to test branching; otherwise return single-token."""
        if text.startswith("multi"):
            return ["multi", "word"]
        return [text]

    def convert_tokens_to_ids(self, tokens):
        """Map tokens to synthetic IDs for branch coverage."""
        mapping = {
            "bad": 5,
            " bad": 6,
            "multi": 7,
            "word": 8,
            "oov": 250,
            " oov": 251,
            "hello": 9,
            "REASON": 42,
        }
        return [mapping.get(t, 1) for t in tokens]

    def decode(self, token_ids, **kwargs):
        """Simple decode implementation returning a space-separated string."""
        return " ".join(str(t) for t in token_ids)

    def decode_token(self, token_ids, prefix_offset, read_offset):
        """Incremental decode used to test streaming behavior."""
        new_tokens = token_ids[read_offset:]
        decode_str = " ".join(str(t) for t in new_tokens)
        new_read_offset = len(token_ids)
        return decode_str, prefix_offset, new_read_offset

    def apply_chat_template(self, request_or_messages, tokenize, split_special_tokens, add_special_tokens, **kwargs):
        """Minimal chat template implementation used by messages2ids."""
        if isinstance(request_or_messages, dict) and "messages" in request_or_messages:
            return " | ".join(m["content"] for m in request_or_messages["messages"])
        return str(request_or_messages)


class ErnieX1ReasoningParser:
    """Mock reasoning parser to trigger reasoning-related branches during streaming."""

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
        """Return a simple object with reasoning_content to cover reasoning branch."""

        class ReasoningDelta:
            def __init__(self, content):
                self.reasoning_content = content

        return ReasoningDelta("REASON")


class MockToolParser:
    """Mock tool parser to cover tool-related branches in both normal and streaming responses."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    class ToolDelta:
        """Simple container representing detected tool calls."""

        def __init__(self):
            self.tool_calls = [{"name": "fake_tool"}]
            self.tools_called = True
            self.content = "tool_content"

    def extract_tool_calls(self, full_text, response_dict):
        """Used in process_response and process_response_dict_normal."""
        return MockToolParser.ToolDelta()

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
        """Used in process_response_dict_streaming."""
        return MockToolParser.ToolDelta()


class TestErnie4_5Processor(unittest.TestCase):
    """Unit tests for Ernie4_5Processor focusing on preprocessing and postprocessing logic."""

    def setUp(self):
        """Patch external dependencies: tokenizer, generation config, eos token resolution."""
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
        """Stop all patches after each test."""
        self.gen_patcher.stop()
        self.tokenizer_patcher.stop()
        self.eos_patcher.stop()

    def _make_processor(self, reasoning=False, tool=False):
        """Helper for creating a processor with optional reasoning/tool parser support."""
        reasoning_cls = ErnieX1ReasoningParser if reasoning else None
        tool_cls = MockToolParser if tool else None
        proc = Ernie4_5Processor("dummy-model", reasoning_parser_obj=reasoning_cls, tool_parser_obj=tool_cls)
        proc._apply_default_parameters = lambda req: req
        return proc

    def test_update_bad_words(self):
        """Verify filtering, multi-token skipping, and OOV behavior in update_bad_words."""
        proc = self._make_processor()

        bad_words = ["bad", "multi", "oov"]
        token_ids = proc.update_bad_words(bad_words, bad_words_token_ids=None)

        self.assertEqual(token_ids, [5, 6, 1])

    def test_process_request_dict_with_prompt_string(self):
        """Test prompt-based tokenization, truncation, and temperature/top_p correction."""
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

    def test_pad_batch_data_right_and_left_and_empty(self):
        """Test left/right padding and empty input behavior."""
        proc = self._make_processor()

        insts = [[1, 2], [3]]

        padded, seq_len = proc.pad_batch_data(
            insts, pad_id=0, return_seq_len=True, return_array=True, pad_style="right"
        )
        np.testing.assert_array_equal(padded, np.array([[1, 2], [3, 0]], dtype=np.int64))
        np.testing.assert_array_equal(seq_len, np.array([[2], [1]], dtype=np.int64))

        padded_left, seq_len_left = proc.pad_batch_data(
            insts, pad_id=0, return_seq_len=True, return_array=True, pad_style="left"
        )
        np.testing.assert_array_equal(padded_left, np.array([[1, 2], [0, 3]], dtype=np.int64))
        np.testing.assert_array_equal(seq_len_left, np.array([[2], [1]], dtype=np.int64))

        padded_empty, seq_len_empty = proc.pad_batch_data(
            [], pad_id=0, return_seq_len=True, return_array=True, pad_style="right"
        )
        np.testing.assert_array_equal(padded_empty, np.array([[]], dtype=np.int64))
        np.testing.assert_array_equal(seq_len_empty, np.array([], dtype=np.int64))

    def test_process_response_dict_streaming_with_reasoning_and_tool(self):
        """Ensure streaming mode handles reasoning and tool-call parsing correctly."""
        proc = self._make_processor(reasoning=True, tool=True)

        response = {
            "finished": True,
            "request_id": "req-1",
            "outputs": {"token_ids": [10, 11]},
        }

        result = proc.process_response_dict_streaming(
            response, enable_thinking=False, include_stop_str_in_output=False
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

        self.assertNotIn("req-1", proc.decode_status)
        self.assertNotIn("req-1", proc.tool_parser_dict)

    def test_update_stop_seq(self):
        """Test stop sequence tokenization and padding."""
        proc = self._make_processor()

        stop_seqs, stop_lens = proc.update_stop_seq("stop")
        self.assertIsInstance(stop_seqs, list)
        self.assertIsInstance(stop_lens, list)

        stop_seqs2, stop_lens2 = proc.update_stop_seq(["stop", "hello"])
        self.assertEqual(len(stop_seqs2), 2)
        self.assertEqual(len(stop_lens2), 2)

    def test_process_request_chat_template_kwargs(self):
        """Test chat_template_kwargs application inside process_request."""

        proc = self._make_processor()

        class ReqObj(dict):
            """Mock request object supporting attributes, set(), and to_dict()."""

            def set(self, k, v):
                self[k] = v

            def __getattr__(self, item):
                return self.get(item, None)

            def to_dict(self):
                return dict(self)

        request = ReqObj(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.5,
                "top_p": 0.5,
            }
        )

        processed = proc.process_request(request, max_model_len=20, chat_template_kwargs={"extra": "VALUE"})

        self.assertEqual(processed.eos_token_ids, [proc.tokenizer.eos_token_id])

        expected_ids = proc.tokenizer.convert_tokens_to_ids(proc.tokenizer.tokenize("hello"))
        self.assertIsNotNone(processed.prompt_token_ids)
        self.assertEqual(processed.prompt_token_ids, expected_ids)

        self.assertIn("max_tokens", processed)
        self.assertEqual(processed["max_tokens"], max(1, 20 - len(expected_ids)))

    def test_process_request_dict_chat_template_kwargs(self):
        """Test chat_template_kwargs insertion in process_request_dict."""
        proc = self._make_processor()

        req = {
            "messages": [{"role": "user", "content": "hey"}],
            "chat_template_kwargs": {"A": "B"},
            "temperature": 0.5,
            "top_p": 0.5,
        }

        result = proc.process_request_dict(req, max_model_len=30)

        self.assertIn("prompt_token_ids", result)
        self.assertEqual(result["A"], "B")

    def test_init_generation_config_exception(self):
        """Test fallback behavior when GenerationConfig loading fails."""
        with patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained", side_effect=Exception("fail")):
            proc = self._make_processor()
            self.assertIsNone(proc.generation_config)

    def test_process_response_with_tool_parser(self):
        """Verify tool_call extraction in process_response."""
        proc = self._make_processor(tool=True)

        class RespObj:
            """Mock response carrying token_ids and index for testing."""

            def __init__(self):
                self.request_id = "reqx"
                self.outputs = MagicMock()
                self.outputs.token_ids = [9, proc.tokenizer.eos_token_id]
                self.outputs.index = 0

        resp = RespObj()
        result = proc.process_response(resp)

        self.assertTrue(hasattr(result.outputs, "tool_calls"))
        self.assertEqual(result.outputs.tool_calls[0]["name"], "fake_tool")

    def test_process_response_dict_normal_with_tool(self):
        """Verify tool_call extraction in normal (non-streaming) response mode."""
        proc = self._make_processor(tool=True)

        resp = {
            "finished": True,
            "request_id": "task-99",
            "outputs": {"token_ids": [10, 11], "text": ""},
        }

        result = proc.process_response_dict_normal(resp, enable_thinking=False, include_stop_str_in_output=False)

        self.assertIn("tool_call", result["outputs"])
        self.assertEqual(result["outputs"]["tool_call"][0]["name"], "fake_tool")


if __name__ == "__main__":
    unittest.main()
