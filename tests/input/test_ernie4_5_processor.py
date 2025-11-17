import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# Fake opentelemetry to avoid import errors when importing fastdeploy
opentelemetry_mod = types.ModuleType("opentelemetry")
instrumentation_mod = types.ModuleType("opentelemetry.instrumentation")
logging_mod = types.ModuleType("opentelemetry.instrumentation.logging")


class DummyLoggingInstrumentor:
    @staticmethod
    def instrument(*args, **kwargs):
        pass


logging_mod.LoggingInstrumentor = DummyLoggingInstrumentor
sys.modules["opentelemetry"] = opentelemetry_mod
sys.modules["opentelemetry.instrumentation"] = instrumentation_mod
sys.modules["opentelemetry.instrumentation.logging"] = logging_mod

# Now import the module under test
from fastdeploy.input.ernie4_5_processor import (  # noqa: E402
    _SAMPLING_EPS,
    Ernie4_5Processor,
)

MODULE_PATH = "fastdeploy.input.ernie4_5_processor"


class DummyTokenizer:
    """Simple fake tokenizer used for unit tests."""

    def __init__(self):
        self.bos_token = "<bos>"
        self.bos_token_id = 1
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = 1000
        self.chat_template = "dummy_template"

    def tokenize(self, text):
        # Treat the whole string as a single token
        if text is None:
            return []
        return [text]

    def convert_tokens_to_ids(self, tokens):
        # Map token -> length-based id; special BIG token to be > vocab_size
        ids = []
        for t in tokens:
            if t == "BIG":
                ids.append(self.vocab_size + 1)
            else:
                ids.append(len(t) % self.vocab_size)
        return ids

    def decode(self, token_ids, **kwargs):
        # Join ids into a string so we can assert easily
        return "|".join(str(i) for i in token_ids)

    def decode_token(self, token_ids, prefix_offset, read_offset):
        # Streaming decode: return new part from read_offset onward
        new_text = "|".join(str(i) for i in token_ids[read_offset:])
        new_read = len(token_ids)
        return new_text, prefix_offset, new_read

    def apply_chat_template(self, request_or_messages, **kwargs):
        # Join all message contents into a single text
        if isinstance(request_or_messages, dict):
            messages = request_or_messages.get("messages", [])
        else:
            messages = request_or_messages
        contents = []
        for m in messages:
            if isinstance(m, dict):
                contents.append(m.get("content", ""))
            else:
                contents.append(str(m))
        return "\n".join(contents)


class DummyRequest:
    """Simple request-like object with get/set methods."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        return self.__dict__


class DummyOutputs:
    """Simple outputs-like container for process_response."""

    def __init__(self, token_ids, index=0):
        self.token_ids = token_ids
        self.index = index
        self.text = ""
        self.reasoning_content = ""


class DummyReasoningParser:
    """Fake reasoning parser."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract_reasoning_content(self, full_text, response_dict):
        # Return dummy reasoning + plain text
        return "reasoning_part", "final_text"

    def extract_reasoning_content_streaming(
        self,
        previous_texts,
        full_text,
        delta_text,
        previous_token_ids,
        full_token_ids,
        delta_token_ids,
    ):
        # Return an object with reasoning_content attribute
        msg = types.SimpleNamespace()
        msg.reasoning_content = "stream_reasoning"
        return msg


class DummyToolParser:
    """Fake tool parser."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.called_times = 0

    def extract_tool_calls(self, full_text, response_dict):
        # When called first time, pretend we found a tool call
        self.called_times += 1
        if self.called_times == 1:
            msg = types.SimpleNamespace()
            msg.tools_called = True
            msg.tool_calls = [{"name": "tool1"}]
            msg.content = "tool_content"
            return msg
        msg = types.SimpleNamespace()
        msg.tools_called = False
        msg.tool_calls = []
        msg.content = full_text
        return msg

    def extract_tool_calls_streaming(
        self,
        previous_texts,
        full_text,
        delta_text,
        previous_token_ids,
        full_token_ids,
        delta_token_ids,
        response_dict,
    ):
        msg = types.SimpleNamespace()
        msg.tools_called = True
        msg.tool_calls = [{"name": "stream_tool"}]
        msg.content = full_text
        return msg


class TestErnie45ProcessorInit(unittest.TestCase):
    """Tests for __init__ and tokenizer loading."""

    @patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained")
    @patch(f"{MODULE_PATH}.Ernie4_5Tokenizer.from_pretrained")
    @patch(f"{MODULE_PATH}.get_eos_token_id")
    def test_init_with_generation_config(
        self,
        mock_get_eos_token_id,
        mock_from_pretrained_tokenizer,
        mock_from_pretrained_config,
    ):
        dummy_tokenizer = DummyTokenizer()
        mock_from_pretrained_tokenizer.return_value = dummy_tokenizer
        mock_get_eos_token_id.return_value = [2, 3]
        mock_from_pretrained_config.return_value = object()

        processor = Ernie4_5Processor("dummy_model_path")

        self.assertIsNotNone(processor.generation_config)
        self.assertEqual(processor.tokenizer, dummy_tokenizer)
        self.assertEqual(processor.eos_token_ids, [2, 3])
        self.assertEqual(processor.pad_token_id, dummy_tokenizer.pad_token_id)

    @patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained")
    @patch(f"{MODULE_PATH}.Ernie4_5Tokenizer.from_pretrained")
    @patch(f"{MODULE_PATH}.get_eos_token_id")
    def test_init_without_generation_config(
        self,
        mock_get_eos_token_id,
        mock_from_pretrained_tokenizer,
        mock_from_pretrained_config,
    ):
        dummy_tokenizer = DummyTokenizer()
        mock_from_pretrained_tokenizer.return_value = dummy_tokenizer
        mock_get_eos_token_id.return_value = [2]
        mock_from_pretrained_config.side_effect = Exception("no config")

        processor = Ernie4_5Processor("dummy_model_path")

        self.assertIsNone(processor.generation_config)
        self.assertEqual(processor.eos_token_ids, [2])
        self.assertEqual(processor.pad_token_id, dummy_tokenizer.pad_token_id)


class TestErnie45ProcessorRequest(unittest.TestCase):
    """Tests for process_request and process_request_dict."""

    def setUp(self):
        patcher_tok = patch(f"{MODULE_PATH}.Ernie4_5Tokenizer.from_pretrained", return_value=DummyTokenizer())
        patcher_cfg = patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained", return_value=object())
        patcher_eos = patch(f"{MODULE_PATH}.get_eos_token_id", return_value=[2])

        self.addCleanup(patcher_tok.stop)
        self.addCleanup(patcher_cfg.stop)
        self.addCleanup(patcher_eos.stop)

        self.mock_tok = patcher_tok.start()
        self.mock_cfg = patcher_cfg.start()
        self.mock_eos = patcher_eos.start()

        self.processor = Ernie4_5Processor("dummy_model_path")

    @patch.object(Ernie4_5Processor, "_apply_default_parameters", side_effect=lambda self, r: r)
    def test_process_request_with_prompt_string(self, _mock_apply_default):
        req = DummyRequest(
            request_id="req1",
            prompt="hello",
            prompt_token_ids=[],
            messages=None,
            eos_token_ids=None,
            stop=[],
            bad_words=None,
            bad_words_token_ids=None,
            max_tokens=None,
            temperature=0.0,
            top_p=0.0,
        )

        processed = self.processor.process_request(req, max_model_len=10)

        self.assertEqual(processed.eos_token_ids, [2])
        self.assertTrue(len(processed.prompt_token_ids) > 0)
        self.assertEqual(processed.temperature, 1)
        self.assertEqual(processed.top_p, _SAMPLING_EPS)
        self.assertEqual(processed.max_tokens, 10 - len(processed.prompt_token_ids))

    @patch.object(Ernie4_5Processor, "_apply_default_parameters", side_effect=lambda self, r: r)
    def test_process_request_with_messages(self, _mock_apply_default):
        messages = [{"role": "user", "content": "hi"}]
        req = DummyRequest(
            request_id="req2",
            prompt=None,
            prompt_token_ids=[],
            messages=messages,
            eos_token_ids=None,
            stop=[],
            bad_words=None,
            bad_words_token_ids=None,
            max_tokens=None,
            temperature=1.0,
            top_p=1.0,
        )

        processed = self.processor.process_request(
            req,
            max_model_len=50,
            chat_template_kwargs={"extra": "value"},
        )

        self.assertTrue(len(processed.prompt_token_ids) > 0)
        self.assertEqual(processed.max_tokens, 50 - len(processed.prompt_token_ids))

    @patch.object(Ernie4_5Processor, "_apply_default_parameters", side_effect=lambda self, r: r)
    def test_process_request_missing_all_prompt_sources_raises(self, _mock_apply_default):
        req = DummyRequest(
            request_id="req3",
            prompt=None,
            prompt_token_ids=[],
            messages=None,
            eos_token_ids=None,
            stop=[],
            bad_words=None,
            bad_words_token_ids=None,
            max_tokens=None,
            temperature=1.0,
            top_p=1.0,
        )

        with self.assertRaises(ValueError):
            self.processor.process_request(req, max_model_len=10)

    @patch.object(Ernie4_5Processor, "_apply_default_parameters", side_effect=lambda self, r: r)
    def test_process_request_empty_prompt_token_ids_raises(self, _mock_apply_default):
        req = DummyRequest(
            request_id="req4",
            prompt="",
            prompt_token_ids=[],
            messages=None,
            eos_token_ids=None,
            stop=[],
            bad_words=None,
            bad_words_token_ids=None,
            max_tokens=None,
            temperature=1.0,
            top_p=1.0,
        )

        with self.assertRaises(ValueError):
            self.processor.process_request(req, max_model_len=10)

    @patch.object(Ernie4_5Processor, "_apply_default_parameters", side_effect=lambda self, r: r)
    def test_process_request_dict_with_prompt_list(self, _mock_apply_default):
        req = {
            "request_id": "req5",
            "prompt_token_ids": [],
            "prompt": [1, 2, 3],
            "messages": None,
            "eos_token_ids": None,
            "stop": [],
            "bad_words": None,
            "bad_words_token_ids": None,
            "max_tokens": None,
            "temperature": 0.0,
            "top_p": 0.0,
        }

        processed = self.processor.process_request_dict(req, max_model_len=20)

        self.assertEqual(processed["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(processed["temperature"], 1)
        self.assertEqual(processed["top_p"], _SAMPLING_EPS)
        self.assertEqual(processed["max_tokens"], 20 - len(processed["prompt_token_ids"]))


class TestErnie45ProcessorResponse(unittest.TestCase):
    """Tests for process_response and response_dict helpers."""

    def setUp(self):
        patcher_tok = patch(f"{MODULE_PATH}.Ernie4_5Tokenizer.from_pretrained", return_value=DummyTokenizer())
        patcher_cfg = patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained", return_value=object())
        patcher_eos = patch(f"{MODULE_PATH}.get_eos_token_id", return_value=[2])

        self.addCleanup(patcher_tok.stop)
        self.addCleanup(patcher_cfg.stop)
        self.addCleanup(patcher_eos.stop)

        self.mock_tok = patcher_tok.start()
        self.mock_cfg = patcher_cfg.start()
        self.mock_eos = patcher_eos.start()

    def test_process_response_basic(self):
        processor = Ernie4_5Processor("dummy_model_path")
        outputs = DummyOutputs(token_ids=[10, 11, 2], index=2)
        response = types.SimpleNamespace(request_id="req1", outputs=outputs)

        result = processor.process_response(response)

        self.assertIsNotNone(result)
        # eos token should be stripped
        self.assertEqual(result.outputs.text, "10|11")
        self.assertEqual(result.usage["completion_tokens"], 3)

    def test_process_response_with_reasoning_and_tool(self):
        processor = Ernie4_5Processor(
            "dummy_model_path",
            reasoning_parser_obj=DummyReasoningParser,
            tool_parser_obj=DummyToolParser,
        )
        outputs = DummyOutputs(token_ids=[10, 11, 2], index=1)
        response = types.SimpleNamespace(request_id="req2", outputs=outputs)

        result = processor.process_response(response)

        self.assertEqual(result.outputs.reasoning_content, "reasoning_part")
        self.assertEqual(result.outputs.text, "tool_content")
        self.assertTrue(hasattr(result.outputs, "tool_calls"))

    def test_process_response_dict_normal_end(self):
        processor = Ernie4_5Processor("dummy_model_path")
        resp = {
            "request_id": "req3",
            "finished": True,
            "outputs": {"token_ids": [5, 6, 7]},
        }

        result = processor.process_response_dict_normal(resp)

        self.assertEqual(result["outputs"]["text"], "5|6|7")
        self.assertNotIn("req3", processor.decode_status)

    def test_process_response_dict_streaming_end(self):
        processor = Ernie4_5Processor("dummy_model_path")
        resp = {
            "request_id": "req4",
            "finished": True,
            "outputs": {"token_ids": [8, 9]},
        }

        result = processor.process_response_dict_streaming(resp)

        self.assertEqual(result["outputs"]["text"], "8|9")
        self.assertNotIn("req4", processor.decode_status)


class TestErnie45ProcessorUtilities(unittest.TestCase):
    """Tests for pad_batch_data, update_stop_seq, process_logprob_response, update_bad_words."""

    def setUp(self):
        patcher_tok = patch(f"{MODULE_PATH}.Ernie4_5Tokenizer.from_pretrained", return_value=DummyTokenizer())
        patcher_cfg = patch(f"{MODULE_PATH}.GenerationConfig.from_pretrained", return_value=object())
        patcher_eos = patch(f"{MODULE_PATH}.get_eos_token_id", return_value=[2])

        self.addCleanup(patcher_tok.stop)
        self.addCleanup(patcher_cfg.stop)
        self.addCleanup(patcher_eos.stop)

        self.mock_tok = patcher_tok.start()
        self.mock_cfg = patcher_cfg.start()
        self.mock_eos = patcher_eos.start()

        self.processor = Ernie4_5Processor("dummy_model_path")

    def test_pad_batch_data_right_and_left(self):
        insts = [[1, 2], [3]]
        padded_right, seq_len = self.processor.pad_batch_data(
            insts,
            pad_id=0,
            return_seq_len=True,
            return_array=True,
            pad_style="right",
        )
        padded_left = self.processor.pad_batch_data(
            insts,
            pad_id=0,
            return_seq_len=False,
            return_array=True,
            pad_style="left",
        )

        self.assertEqual(padded_right.shape, (2, 2))
        self.assertTrue(np.array_equal(seq_len.reshape(-1).tolist(), [2, 1]))
        self.assertEqual(padded_left.shape, (2, 2))
        self.assertTrue((padded_left[1, 0] == 0) and (padded_left[1, 1] == 3))

    def test_update_stop_seq(self):
        stop_seqs, stop_lens = self.processor.update_stop_seq(["stop1", "stop2"])

        self.assertEqual(len(stop_seqs), 2)
        self.assertEqual(len(stop_lens), 2)
        self.assertTrue(all(l > 0 for l in stop_lens))

    def test_process_logprob_response(self):
        token_ids = [1, 2, 3]
        text = self.processor.process_logprob_response(token_ids)

        self.assertEqual(text, "1|2|3")

    def test_update_bad_words_valid_and_invalid(self):
        token_ids = self.processor.update_bad_words(["bad", "BIG"], bad_words_token_ids=None)

        self.assertTrue(len(token_ids) > 0)
        # "BIG" maps to id > vocab_size and should be skipped, so only "bad" is counted once
        self.assertEqual(len(token_ids), 1)


if __name__ == "__main__":
    unittest.main()
