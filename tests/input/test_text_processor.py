"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


class DummyTokenizer:
    bos_token = "<s>"
    cls_token = "<cls>"
    sep_token = "</s>"
    eos_token = "</eos>"
    mask_token = "<mask>"
    chat_template = "dummy"

    def __init__(self):
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.eos_token = 2
        self.vocab_size = 256
        self.bos_token_id = self._convert_token_to_id(self.bos_token)
        self.cls_token_id = self._convert_token_to_id(self.cls_token)
        self.sep_token_id = self._convert_token_to_id(self.sep_token)
        self.mask_token_id = self._convert_token_to_id(self.mask_token)

    def _convert_token_to_id(self, token):
        return len(str(token))

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            values = [self._value(item) for item in text]
        else:
            values = [self._value(text)]
        max_length = kwargs.get("max_length")
        if max_length is not None:
            values = values[:max_length]
        return {"input_ids": np.array([values], dtype=np.int64)}

    def _value(self, item):
        if isinstance(item, str):
            return len(item)
        return int(item)

    def tokenize(self, text):
        if isinstance(text, str):
            return [text]
        return [str(text)]

    def convert_tokens_to_ids(self, tokens):
        return [self._value(token) for token in tokens]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(t) for t in token_ids)

    def decode_token(self, token_ids, prefix_offset, read_offset):
        start = read_offset
        delta_tokens = token_ids[start:]
        delta = "".join(str(t) for t in delta_tokens)
        prefix_offset += len(token_ids)
        read_offset += len(delta_tokens)
        return delta, prefix_offset, read_offset

    def batch_decode(self, batch, **kwargs):
        return [self.decode(seq) for seq in batch]

    def apply_chat_template(self, request, **kwargs):
        if isinstance(request, dict):
            system = request.get("system")
            messages = request.get("messages", [])
        else:
            system = getattr(request, "system", None)
            messages = getattr(request, "messages", [])
        parts = [system] if system else []
        parts.extend(msg.get("content", "") for msg in messages)
        return " ".join(part for part in parts if part)


class DummyLlamaTokenizer(DummyTokenizer):
    pass


class DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyTokenizer()


class DummyHFTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyTokenizer()


def _create_dummy_modules():
    """Create all dummy modules needed for testing fastdeploy.input.text_processor."""
    repo_root = Path(__file__).resolve().parents[2]

    dummy_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )

    utils_module = types.ModuleType("fastdeploy.utils")
    utils_module.data_processor_logger = dummy_logger

    envs_module = types.ModuleType("fastdeploy.envs")
    envs_module.FD_USE_HF_TOKENIZER = False

    generation_module = types.ModuleType("paddleformers.generation")

    class DummyGenerationConfig:
        def __init__(self):
            self.top_p = 0.8
            self.temperature = 0.9
            self.repetition_penalty = 1.1
            self.frequency_penalty = 0.2
            self.presence_penalty = 0.1

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    generation_module.GenerationConfig = DummyGenerationConfig

    transformers_module = types.ModuleType("paddleformers.transformers")
    transformers_module.AutoTokenizer = DummyAutoTokenizer
    transformers_module.LlamaTokenizer = DummyLlamaTokenizer
    transformers_module.Llama3Tokenizer = DummyLlamaTokenizer

    hf_transformers_module = types.ModuleType("transformers")
    hf_transformers_module.AutoTokenizer = DummyHFTokenizer

    llm_utils_module = types.ModuleType("paddleformers.trl.llm_utils")
    llm_utils_module.get_eos_token_id = lambda tokenizer, config: [tokenizer.eos_token_id]

    fastdeploy_module = types.ModuleType("fastdeploy")
    fastdeploy_module.__path__ = [str(repo_root / "fastdeploy")]
    fastdeploy_module.utils = utils_module
    fastdeploy_module.envs = envs_module

    return {
        "fastdeploy": fastdeploy_module,
        "fastdeploy.utils": utils_module,
        "fastdeploy.envs": envs_module,
        "paddleformers.generation": generation_module,
        "paddleformers.transformers": transformers_module,
        "transformers": hf_transformers_module,
        "paddleformers.trl.llm_utils": llm_utils_module,
    }


def _import_text_processor(use_hf_tokenizer=False):
    modules = _create_dummy_modules()

    modules["fastdeploy.envs"].FD_USE_HF_TOKENIZER = use_hf_tokenizer

    previous_modules = {}
    for name, module in modules.items():
        previous_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    try:
        text_processor_module = importlib.import_module("fastdeploy.input.text_processor")
        importlib.reload(text_processor_module)
    except Exception:
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        raise

    def cleanup():
        sys.modules.pop("fastdeploy.input.text_processor", None)
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return text_processor_module, cleanup


class DummyRequest:
    def __init__(self, **kwargs):
        self.request_id = kwargs.get("request_id", "req")
        self.prompt = kwargs.get("prompt")
        self.prompt_token_ids = kwargs.get("prompt_token_ids")
        self.messages = kwargs.get("messages")
        self.eos_token_ids = kwargs.get("eos_token_ids")
        self.chat_template = kwargs.get("chat_template")
        self.enable_thinking = kwargs.get("enable_thinking")
        self.history = kwargs.get("history")
        self.tools = kwargs.get("tools")
        self.system = kwargs.get("system")
        self.sampling_params = SimpleNamespace(
            top_p=kwargs.get("top_p"),
            temperature=kwargs.get("temperature"),
            repetition_penalty=kwargs.get("repetition_penalty"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            stop=kwargs.get("stop"),
            stop_token_ids=kwargs.get("stop_token_ids"),
            stop_seqs_len=kwargs.get("stop_seqs_len"),
            bad_words=kwargs.get("bad_words"),
            bad_words_token_ids=kwargs.get("bad_words_token_ids"),
            max_tokens=kwargs.get("max_tokens"),
        )

    def get(self, key, default=None):
        if hasattr(self, key) and getattr(self, key) is not None:
            return getattr(self, key)
        return getattr(self.sampling_params, key, default)

    def set(self, key, value):
        if hasattr(self.sampling_params, key):
            setattr(self.sampling_params, key, value)
        else:
            setattr(self, key, value)

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "messages": self.messages,
            "prompt": self.prompt,
            "system": self.system,
            "history": self.history,
            "tools": self.tools,
            "chat_template": self.chat_template,
            "enable_thinking": self.enable_thinking,
        }

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)


class DataProcessorTestCase(unittest.TestCase):
    @staticmethod
    def create_dummy_reasoning(tokenizer, reasoning_content="think"):
        class DummyReasoning:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def extract_reasoning_content(self, full_text, response_dict):
                return reasoning_content, f"{full_text}!"

        return DummyReasoning(tokenizer)

    @staticmethod
    def create_dummy_tool_parser(tokenizer, content="tool-text"):
        class DummyToolParser:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def extract_tool_calls(self, full_text, response_dict):
                return SimpleNamespace(tools_called=True, tool_calls=["tool"], content=content)

        return DummyToolParser

    def setUp(self):
        module, cleanup = _import_text_processor()
        self.text_processor_module = module
        self.addCleanup(cleanup)
        self.processor = self.text_processor_module.DataProcessor("stub-model")

    def test_base_data_processor_contract(self):
        text_processor_module = self.text_processor_module

        class MinimalProcessor(text_processor_module.BaseDataProcessor):
            def __init__(self):
                self.generation_config = SimpleNamespace(
                    top_p=0.5,
                    temperature=0.6,
                    repetition_penalty=1.1,
                    frequency_penalty=0.2,
                    presence_penalty=0.3,
                )
                super().__init__()

            def _load_tokenizer(self):
                return DummyTokenizer()

            def process_request(self, request, **kwargs):
                return super().process_request(request, **kwargs)

            def process_response(self, response_dict):
                return super().process_response(response_dict)

        processor = MinimalProcessor()
        defaults = processor._apply_default_parameters({})
        self.assertAlmostEqual(defaults["top_p"], 0.5)
        with self.assertRaises(NotImplementedError):
            processor.process_request({}, max_model_len=None)
        with self.assertRaises(NotImplementedError):
            processor.process_response({})
        with self.assertRaises(NotImplementedError):
            processor.text2ids("text")
        with self.assertRaises(NotImplementedError):
            processor.messages2ids([])
        with self.assertRaises(NotImplementedError):
            processor.ids2tokens([1], "task")

    def test_process_request_dict_prompt_defaults(self):
        request = {"prompt": "hi", "temperature": 0, "top_p": 0, "stop": ["stop"]}
        processed = self.processor.process_request_dict(request, max_model_len=5)

        self.assertEqual(processed["prompt_token_ids"], [2])
        self.assertEqual(processed["stop_token_ids"], [[4]])
        self.assertEqual(processed["stop_seqs_len"], [1])
        self.assertEqual(processed["temperature"], 1)
        self.assertAlmostEqual(processed["top_p"], 1e-5)
        self.assertEqual(processed["max_tokens"], 4)

    def test_process_request_dict_messages_template(self):
        request = {
            "request_id": "chat",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"system": "system prompt"},
        }
        processed = self.processor.process_request_dict(request, max_model_len=6)

        self.assertEqual(processed["prompt_token_ids"], [len("system prompt hello")])
        self.assertEqual(processed["system"], "system prompt")
        self.assertTrue(processed["enable_thinking"])
        self.assertEqual(processed["prompt_tokens"], "system prompt hello")

    def test_process_request_object_handles_sequences(self):
        request = DummyRequest(
            prompt=[1, 2, 3, 4, 5, 6],
            stop=["stop"],
            bad_words=["zz"],
            temperature=0,
            top_p=0,
        )
        processed = self.processor.process_request(request, max_model_len=5)

        self.assertEqual(processed.prompt_token_ids, [1, 2, 3, 4])
        self.assertEqual(processed.sampling_params.max_tokens, 1)
        self.assertEqual(processed.sampling_params.stop_token_ids, [[4]])
        self.assertEqual(set(processed.sampling_params.bad_words_token_ids), {2, 3})
        self.assertEqual(processed.sampling_params.temperature, 1)
        self.assertAlmostEqual(processed.sampling_params.top_p, 1e-5)

    def test_process_request_requires_prompt_or_messages(self):
        request = DummyRequest(prompt=None, messages=None, prompt_token_ids=None)
        with self.assertRaisesRegex(ValueError, "should have `input_ids`, `text` or `messages`"):
            self.processor.process_request(request, max_model_len=5)

    def test_process_request_dict_rejects_bad_kwargs(self):
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": "invalid",
        }
        with self.assertRaisesRegex(ValueError, "chat_template_kwargs must be a dict"):
            self.processor.process_request_dict(request)

    def test_ids2tokens_and_clear_request_status(self):
        delta, _, _ = self.processor.ids2tokens([3], "task-1")
        self.assertEqual(delta, "3")
        delta, _, _ = self.processor.ids2tokens([4], "task-1")
        self.assertEqual(delta, "4")

        combined = self.processor.clear_request_status("task-1")
        self.assertEqual(combined, "34")
        self.assertNotIn("task-1", self.processor.decode_status)

    def test_clear_request_status_hf_branch(self):
        module, cleanup = _import_text_processor(use_hf_tokenizer=True)
        self.addCleanup(cleanup)
        processor = module.DataProcessor("stub-model")
        processor.decode_status = {"task": [[], [], "transcript"]}

        self.assertEqual(processor.clear_request_status("task"), "transcript")
        self.assertNotIn("task", processor.decode_status)

    def test_data_processor_init_handles_missing_generation_config(self):
        with mock.patch.object(
            self.text_processor_module.GenerationConfig,
            "from_pretrained",
            side_effect=OSError("missing"),
        ):
            processor = self.text_processor_module.DataProcessor("stub-model")
        self.assertIsNone(processor.generation_config)

    def test_process_response_with_reasoning_and_tools(self):
        processor = self.processor

        processor.reasoning_parser = self.create_dummy_reasoning(processor.tokenizer)
        processor.tool_parser_obj = self.create_dummy_tool_parser(processor.tokenizer, content="tool-only")

        response = SimpleNamespace(
            request_id="resp",
            outputs=SimpleNamespace(token_ids=[1, processor.tokenizer.eos_token_id]),
        )

        processed = processor.process_response(response)
        self.assertEqual(processed.outputs.text, "tool-only")
        self.assertEqual(processed.outputs.reasoning_content, "think")
        self.assertEqual(processed.outputs.tool_calls, ["tool"])

    def test_process_response_streaming_clears_state(self):
        processor = self.processor
        req_id = "stream"
        processor.decode_status[req_id] = [0, 0, [], ""]
        response = {"finished": True, "request_id": req_id, "outputs": {"token_ids": [7]}}

        result = processor.process_response_dict_streaming(response, enable_thinking=False)
        self.assertEqual(result["outputs"]["text"], "7")
        self.assertNotIn(req_id, processor.decode_status)

    def test_process_response_dict_normal_with_reasoning(self):
        processor = self.processor

        processor.reasoning_parser = self.create_dummy_reasoning(processor.tokenizer, reasoning_content="because")
        processor.tool_parser_obj = self.create_dummy_tool_parser(processor.tokenizer, content="tool-text")

        response = {
            "finished": True,
            "request_id": "normal",
            "outputs": {"token_ids": [7, processor.tokenizer.eos_token_id]},
        }

        result = processor.process_response_dict_normal(response, enable_thinking=True)
        self.assertEqual(result["outputs"]["completion_tokens"], "7")
        self.assertEqual(result["outputs"]["text"], "tool-text")
        self.assertEqual(result["outputs"]["reasoning_content"], "because")
        self.assertEqual(result["outputs"]["reasoning_token_num"], 1)

    def test_process_response_dict_dispatch(self):
        processor = self.processor
        calls = {}

        def fake_stream(response_dict, **kwargs):
            calls["stream"] = kwargs
            return "stream"

        def fake_normal(response_dict, **kwargs):
            calls["normal"] = kwargs
            return "normal"

        original_stream = processor.process_response_dict_streaming
        original_normal = processor.process_response_dict_normal
        processor.process_response_dict_streaming = fake_stream
        processor.process_response_dict_normal = fake_normal
        self.addCleanup(lambda: setattr(processor, "process_response_dict_streaming", original_stream))
        self.addCleanup(lambda: setattr(processor, "process_response_dict_normal", original_normal))

        response = {"outputs": {}, "finished": False, "request_id": "req"}
        self.assertEqual(processor.process_response_dict(response), "stream")
        self.assertTrue(calls["stream"]["enable_thinking"])
        self.assertEqual(
            processor.process_response_dict(response, stream=False, enable_thinking=None),
            "normal",
        )
        self.assertTrue(calls["normal"]["enable_thinking"])

    def test_update_stop_seq_excludes_eos(self):
        stop_seqs, stop_len = self.processor.update_stop_seq(["stop", self.processor.tokenizer.eos_token_id])
        self.assertEqual(stop_seqs, [[4]])
        self.assertEqual(stop_len, [1])

    def test_pad_batch_data_left_padding(self):
        padded, lengths = self.processor.pad_batch_data(
            [[1], [2, 3]],
            pad_id=-1,
            return_seq_len=True,
            return_array=False,
            pad_style="left",
        )
        self.assertEqual(padded, [[-1, 1], [2, 3]])
        self.assertEqual(lengths, [1, 2])

    def test_pad_batch_data_empty_returns_array(self):
        padded, lengths = self.processor.pad_batch_data([], return_seq_len=True)
        self.assertEqual(padded.shape, (1, 0))
        self.assertEqual(lengths.shape, (0,))

    def test_get_pad_id_prefers_eos_when_missing(self):
        processor = self.text_processor_module.DataProcessor("stub-model")
        llama_tokenizer = DummyLlamaTokenizer()
        llama_tokenizer.pad_token_id = None
        llama_tokenizer.eos_token = 99
        processor.tokenizer = llama_tokenizer

        self.assertEqual(processor.get_pad_id(), 99)

    def test_load_tokenizer_hf_branch(self):
        module, cleanup = _import_text_processor(use_hf_tokenizer=True)
        self.addCleanup(cleanup)
        processor = module.DataProcessor("stub-model")
        self.assertIsInstance(processor.tokenizer, DummyTokenizer)

    def test_text2ids_hf_branch(self):
        module, cleanup = _import_text_processor(use_hf_tokenizer=True)
        self.addCleanup(cleanup)
        processor = module.DataProcessor("stub-model")
        ids = processor.text2ids("hi", max_model_len=5)
        self.assertEqual(ids.tolist(), [2, 0, 0, 0, 0][: len(ids)])

    def test_process_logprob_response(self):
        self.assertEqual(self.processor.process_logprob_response([1, 2]), "1 2")

    def test_process_request_dict_uses_existing_ids(self):
        request = {"prompt_token_ids": [1, 2, 3], "max_tokens": 5}
        processed = self.processor.process_request_dict(request, max_model_len=6)
        self.assertEqual(processed["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(processed["max_tokens"], 5)

    def test_process_request_dict_requires_chat_template(self):
        original_template = self.processor.tokenizer.chat_template
        self.processor.tokenizer.chat_template = None
        self.addCleanup(lambda: setattr(self.processor.tokenizer, "chat_template", original_template))
        with self.assertRaisesRegex(ValueError, "chat_template"):
            self.processor.process_request_dict({"messages": [{"role": "user", "content": "hi"}]})

    def test_update_bad_words_with_warnings(self):
        processor = self.processor

        def custom_tokenize(text):
            base = text.strip()
            if base == "combo":
                return ["co", "mbo"]
            if base == "oversize":
                return [base]
            return [base]

        def custom_convert(tokens):
            if tokens == ["co", "mbo"]:
                return [1, 2]
            if tokens == ["oversize"]:
                return [processor.tokenizer.vocab_size + 1]
            return [len(tokens[0])]

        original_tokenize = processor.tokenizer.tokenize
        original_convert = processor.tokenizer.convert_tokens_to_ids
        processor.tokenizer.tokenize = custom_tokenize
        processor.tokenizer.convert_tokens_to_ids = custom_convert
        self.addCleanup(lambda: setattr(processor.tokenizer, "tokenize", original_tokenize))
        self.addCleanup(lambda: setattr(processor.tokenizer, "convert_tokens_to_ids", original_convert))

        self.assertEqual(processor.update_bad_words(["combo", "oversize"], []), [])


if __name__ == "__main__":
    unittest.main()
