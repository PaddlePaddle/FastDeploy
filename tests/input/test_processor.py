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

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class DummyTokenizer:
    bos_token = "<s>"
    cls_token = "<cls>"
    sep_token = "</s>"
    eos_token = "</eos>"
    mask_token = "<mask>"
    chat_template = "dummy"

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.vocab_size = 256
        self.bos_token_id = self._convert_token_to_id(self.bos_token)
        self.cls_token_id = self._convert_token_to_id(self.cls_token)
        self.sep_token_id = self._convert_token_to_id(self.sep_token)
        self.mask_token_id = self._convert_token_to_id(self.mask_token)

    def _convert_token_to_id(self, token):
        if token == "<think>":
            return 11
        if token == "</think>":
            return 12
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

    def get_vocab(self):
        return {"<think>": 11, "</think>": 12}

    def tokenize(self, text):
        if isinstance(text, str):
            if text in ("<think>", "</think>"):
                return [text]
            return [text]
        return [str(text)]

    def convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]

    def encode(self, text, add_special_tokens=True, **kwargs):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, token_ids, **kwargs):
        return " ".join(str(t) for t in token_ids)

    def decode_token(self, token_ids, prefix_offset, read_offset):
        delta_tokens = token_ids[read_offset:]
        delta = "".join(str(t) for t in delta_tokens)
        prefix_offset += len(token_ids)
        read_offset += len(delta_tokens)
        return delta, prefix_offset, read_offset

    def batch_decode(self, batch, **kwargs):
        return [self.decode(seq) for seq in batch]

    def apply_chat_template(self, request, **kwargs):
        if isinstance(request, dict):
            messages = request.get("messages", [])
            system = request.get("system") or kwargs.get("system")
        else:
            messages = request
            system = kwargs.get("system")

        parts = [system] if system else []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") in ("image", "video"):
                        parts.append(f"<{item['type']}>")
            else:
                parts.append(content)
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


def _process_stop_token_ids(request, update_stop_seq_fn):
    stop_token_ids_final = []
    stop_seqs_len_final = []

    if request.get("stop_token_ids") is not None:
        stop_token_ids = request.get("stop_token_ids")
        if isinstance(stop_token_ids, list) and len(stop_token_ids) > 0:
            if isinstance(stop_token_ids[0], int):
                stop_token_ids_final.extend([[t] for t in stop_token_ids])
                stop_seqs_len_final.extend([1] * len(stop_token_ids))
            elif isinstance(stop_token_ids[0], list):
                stop_token_ids_final.extend(stop_token_ids)
                stop_seqs_len_final.extend([len(seq) for seq in stop_token_ids])

    stop_sequences = request.get("stop", [])
    if stop_sequences:
        stop_seqs, stop_seqs_actual_lens = update_stop_seq_fn(stop_sequences)
        stop_token_ids_final.extend(stop_seqs)
        stop_seqs_len_final.extend(stop_seqs_actual_lens)

    if stop_token_ids_final:
        request["stop_token_ids"] = stop_token_ids_final
        request["stop_seqs_len"] = stop_seqs_len_final


def _create_dummy_modules():
    repo_root = Path(__file__).resolve().parents[2]

    dummy_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )

    fastdeploy_module = types.ModuleType("fastdeploy")
    fastdeploy_module.__path__ = [str(repo_root / "fastdeploy")]

    input_module = types.ModuleType("fastdeploy.input")
    input_module.__path__ = [str(repo_root / "fastdeploy" / "input")]

    envs_module = types.ModuleType("fastdeploy.envs")
    envs_module.FD_USE_HF_TOKENIZER = False
    envs_module.FD_LOG_REQUESTS = 0
    envs_module.FD_LOG_REQUESTS_LEVEL = 0

    utils_module = types.ModuleType("fastdeploy.utils")
    utils_module.data_processor_logger = dummy_logger

    input_utils_module = types.ModuleType("fastdeploy.input.utils")
    input_utils_module.process_stop_token_ids = _process_stop_token_ids

    logger_module = types.ModuleType("fastdeploy.logger")
    request_logger_module = types.ModuleType("fastdeploy.logger.request_logger")

    class DummyRequestLogLevel:
        LIFECYCLE = 0
        STAGES = 1
        CONTENT = 2
        FULL = 3

    request_logger_module.RequestLogLevel = DummyRequestLogLevel
    request_logger_module.log_request = lambda *args, **kwargs: None

    chat_utils_module = types.ModuleType("fastdeploy.entrypoints.chat_utils")
    chat_utils_module.parse_chat_messages = lambda messages: messages

    entrypoints_module = types.ModuleType("fastdeploy.entrypoints")
    entrypoints_module.__path__ = [str(repo_root / "fastdeploy" / "entrypoints")]

    paddleformers_module = types.ModuleType("paddleformers")
    generation_module = types.ModuleType("paddleformers.generation")
    generation_module.GenerationConfig = DummyGenerationConfig

    transformers_module = types.ModuleType("paddleformers.transformers")
    transformers_module.AutoTokenizer = DummyAutoTokenizer
    transformers_module.LlamaTokenizer = DummyLlamaTokenizer
    transformers_module.Llama3Tokenizer = DummyLlamaTokenizer

    cli_module = types.ModuleType("paddleformers.cli")
    cli_utils_module = types.ModuleType("paddleformers.cli.utils")
    llm_utils_module = types.ModuleType("paddleformers.cli.utils.llm_utils")
    llm_utils_module.get_eos_token_id = lambda tokenizer, config: [tokenizer.eos_token_id]

    hf_transformers_module = types.ModuleType("transformers")
    hf_transformers_module.AutoTokenizer = DummyHFTokenizer

    return {
        "fastdeploy": fastdeploy_module,
        "fastdeploy.input": input_module,
        "fastdeploy.envs": envs_module,
        "fastdeploy.utils": utils_module,
        "fastdeploy.input.utils": input_utils_module,
        "fastdeploy.logger": logger_module,
        "fastdeploy.logger.request_logger": request_logger_module,
        "fastdeploy.entrypoints": entrypoints_module,
        "fastdeploy.entrypoints.chat_utils": chat_utils_module,
        "paddleformers": paddleformers_module,
        "paddleformers.generation": generation_module,
        "paddleformers.transformers": transformers_module,
        "paddleformers.cli": cli_module,
        "paddleformers.cli.utils": cli_utils_module,
        "paddleformers.cli.utils.llm_utils": llm_utils_module,
        "transformers": hf_transformers_module,
    }


def _import_processor(use_hf_tokenizer=False):
    modules = _create_dummy_modules()
    modules["fastdeploy.envs"].FD_USE_HF_TOKENIZER = use_hf_tokenizer

    module_names = set(modules) | {"fastdeploy.input.processor"}
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    sys.modules.pop("fastdeploy.input.processor", None)
    for name, module in modules.items():
        sys.modules[name] = module

    try:
        processor_module = importlib.import_module("fastdeploy.input.processor")
    except Exception:
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        raise

    def cleanup():
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return processor_module, cleanup


class DummyReasoningParser:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def get_model_status(self, prompt_token_ids):
        return "think_start" if 11 in prompt_token_ids else "normal"

    def extract_reasoning_content(self, full_text, request, model_status):
        return "because", f"{full_text}!"

    def extract_reasoning_content_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        model_status,
    ):
        return SimpleNamespace(reasoning_content="because", content="visible")


class DummyToolParser:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def extract_tool_calls(self, full_text, request):
        return SimpleNamespace(tools_called=True, tool_calls=["tool-call"])

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        return SimpleNamespace(tool_calls=["tool-delta"], content="tool-visible")


class DummyMMProcessor:
    def __init__(self):
        self.process_calls = []
        self.append_calls = []

    def process(self, request):
        self.process_calls.append(request)
        request["prompt_token_ids"] = [10, 20, 30]
        request["multimodal_inputs"] = {"image": ["encoded"]}

    def get_mm_max_tokens_per_item(self, seq_len):
        return {"image": seq_len + 1}

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        self.append_calls.append((multimodal_inputs, completion_token_ids))
        multimodal_inputs["completion_token_ids"] = completion_token_ids


class BatchEncodingLike:
    def __init__(self, ids):
        self.input_ids = ids

    def __getitem__(self, key):
        return getattr(self, key)


class ProcessorTestCase(unittest.TestCase):
    def setUp(self):
        module, cleanup = _import_processor()
        self.processor_module = module
        self.addCleanup(cleanup)
        self.processor = module.Processor("stub-model")

    def test_init_loads_tokenizer_config_and_parsers(self):
        processor = self.processor_module.Processor(
            "stub-model",
            reasoning_parser_obj=DummyReasoningParser,
            tool_parser_obj=DummyToolParser,
        )

        self.assertEqual(processor.model_name_or_path, "stub-model")
        self.assertIsInstance(processor.tokenizer, DummyTokenizer)
        self.assertEqual(processor.eos_token_ids, [2])
        self.assertEqual(processor.eos_token_id_len, 1)
        self.assertEqual(processor.pad_token_id, 0)
        self.assertIsInstance(processor.reasoning_parser, DummyReasoningParser)
        self.assertIs(processor.tool_parser_obj, DummyToolParser)

    def test_process_request_dict_prompt_defaults_and_stop_words(self):
        request = {
            "prompt": "hi",
            "temperature": 0,
            "top_p": 0,
            "stop": ["stop"],
            "bad_words": ["bad"],
        }

        processed = self.processor.process_request_dict(request, max_model_len=5)

        self.assertEqual(processed["prompt_token_ids"], [2])
        self.assertEqual(processed["prompt_token_ids_len"], 1)
        self.assertEqual(processed["eos_token_ids"], [2])
        self.assertEqual(processed["stop_token_ids"], [[4]])
        self.assertEqual(processed["stop_seqs_len"], [1])
        self.assertEqual(set(processed["bad_words_token_ids"]), {3, 4})
        self.assertEqual(processed["temperature"], 1)
        self.assertEqual(processed["top_k"], 1)
        self.assertAlmostEqual(processed["top_p"], 1e-5)
        self.assertEqual(processed["max_tokens"], 4)

    def test_process_request_dict_uses_existing_ids_and_thinking_args(self):
        request = {
            "request_id": "think",
            "prompt_token_ids": [11, 42],
            "logits_processors_args": {"thinking_budget": 8, "think_stop_sentence": "done"},
        }

        processed = self.processor.process_request_dict(request, max_model_len=8)
        logits_args = processed["logits_processors_args"]

        self.assertEqual(processed["prompt_token_ids"], [11, 42])
        self.assertEqual(logits_args["think_stop_sentence_token_ids"], [4])
        self.assertNotIn("think_stop_sentence", logits_args)
        self.assertTrue(logits_args["think_prompt_checked"])
        self.assertTrue(logits_args["think_prompt_started"])
        self.assertFalse(logits_args["think_prompt_ended"])
        self.assertEqual(logits_args["think_prompt_last_token_id"], 42)
        self.assertEqual(processed["max_tokens"], 6)

    def test_process_request_dict_messages_extracts_multimodal_data(self):
        request = {
            "request_id": "chat",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image", "image": "image-data"},
                        {"type": "video", "video": "video-data"},
                    ],
                }
            ],
            "chat_template_kwargs": {"system": "system prompt"},
        }

        processed = self.processor.process_request_dict(request, max_model_len=10)

        self.assertEqual(processed["system"], "system prompt")
        self.assertTrue(processed["enable_thinking"])
        self.assertEqual(processed["prompt"], "system prompt describe <image> <video>")
        self.assertEqual(processed["prompt_tokens"], processed["prompt"])
        self.assertEqual(processed["prompt_token_ids"], [len(processed["prompt"])])
        self.assertEqual(processed["multimodal_data"]["mm_order"], ["image", "video"])
        self.assertEqual(len(processed["multimodal_data"]["image"]), 1)
        self.assertEqual(len(processed["multimodal_data"]["video"]), 1)

    def test_process_request_dict_delegates_to_mm_processor(self):
        mm_processor = DummyMMProcessor()
        processor = self.processor_module.Processor("stub-model", mm_processor=mm_processor)
        processor.text2ids = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("text path should not run"))
        request = {"prompt": "ignored", "max_tokens": 4}

        processed = processor.process_request_dict(request, max_model_len=10)

        self.assertIs(mm_processor.process_calls[0], request)
        self.assertEqual(processed["prompt_token_ids"], [10, 20, 30])
        self.assertEqual(processed["prompt_token_ids_len"], 3)
        self.assertEqual(processed["max_tokens"], 4)
        self.assertEqual(processed["multimodal_inputs"], {"image": ["encoded"]})
        self.assertEqual(processor.get_mm_max_tokens_per_item(6), {"image": 7})

        multimodal_inputs = {}
        processor.append_completion_tokens(multimodal_inputs, [1, 2])
        self.assertEqual(mm_processor.append_calls, [(multimodal_inputs, [1, 2])])
        self.assertEqual(multimodal_inputs["completion_token_ids"], [1, 2])

    def test_process_request_dict_force_disable_and_default_reasoning_tokens(self):
        processor = self.processor_module.Processor(
            "stub-model",
            force_disable_thinking=True,
            set_default_reasoning_max_tokens=True,
        )
        request = {"prompt_token_ids": [1, 2], "max_tokens": 5}

        processed = processor.process_request_dict(request, max_model_len=10)

        self.assertFalse(processed["enable_thinking"])
        self.assertEqual(processed["max_tokens"], 5)
        self.assertEqual(processed["reasoning_max_tokens"], 4)

    def test_messages2ids_normalizes_batch_encoding_and_plain_dict(self):
        class BatchEncodingTokenizer(DummyTokenizer):
            def encode(self, text, add_special_tokens=True, **kwargs):
                return BatchEncodingLike([len(text)])

        self.processor.tokenizer = BatchEncodingTokenizer()
        request = {"request_id": "chat", "messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(self.processor.messages2ids(request), [5])

        class PlainDictTokenizer(DummyTokenizer):
            def encode(self, text, add_special_tokens=True, **kwargs):
                return {"input_ids": np.array([[len(text)]], dtype=np.int64), "attention_mask": [1]}

        self.processor.tokenizer = PlainDictTokenizer()
        request = {"request_id": "chat", "messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(self.processor.messages2ids(request), [5])

    def test_ids2tokens_and_clear_request_status(self):
        delta, previous_token_ids, previous_texts = self.processor.ids2tokens([3], "task")
        self.assertEqual(delta, "3")
        self.assertEqual(previous_token_ids, [3])
        self.assertEqual(previous_texts, "")

        delta, previous_token_ids, previous_texts = self.processor.ids2tokens([4], "task")
        self.assertEqual(delta, "4")
        self.assertEqual(previous_token_ids, [3, 4])
        self.assertEqual(previous_texts, "3")
        self.assertEqual(self.processor.clear_request_status("task"), "34")
        self.assertNotIn("task", self.processor.decode_status)

    def test_hf_tokenizer_branch_ids2tokens_and_text2ids(self):
        module, cleanup = _import_processor(use_hf_tokenizer=True)
        self.addCleanup(cleanup)
        processor = module.Processor("stub-model")

        ids = processor.text2ids("hi", max_model_len=5)
        self.assertEqual(ids.tolist(), [2])

        delta, previous_token_ids, previous_texts = processor.ids2tokens([3], "task")
        self.assertEqual(delta, "3")
        self.assertEqual(previous_token_ids, [])
        self.assertEqual(previous_texts, "3")
        self.assertEqual(processor.clear_request_status("task"), "3")

    def test_process_response_dict_normal_with_reasoning_and_tools(self):
        processor = self.processor
        processor.reasoning_parser = DummyReasoningParser(processor.tokenizer)
        processor.tool_parser_obj = DummyToolParser
        processor.model_status_dict["resp"] = "normal"
        response = {
            "finished": True,
            "request_id": "resp",
            "outputs": {"token_ids": [7, processor.tokenizer.eos_token_id]},
        }

        result = processor.process_response_dict(response, stream=False, request={"id": "resp"})

        self.assertEqual(result["outputs"]["completion_tokens"], "7")
        self.assertEqual(result["outputs"]["text"], "7!")
        self.assertEqual(result["outputs"]["reasoning_content"], "because")
        self.assertEqual(result["outputs"]["reasoning_token_num"], 1)
        self.assertEqual(result["outputs"]["tool_calls"], ["tool-call"])
        self.assertNotIn("resp", processor.decode_status)
        self.assertNotIn("resp", processor.model_status_dict)

    def test_process_response_dict_streaming_with_reasoning_and_tools(self):
        processor = self.processor
        processor.reasoning_parser = DummyReasoningParser(processor.tokenizer)
        processor.tool_parser_obj = DummyToolParser
        processor.model_status_dict["stream"] = "think_start"
        response = {
            "finished": True,
            "request_id": "stream",
            "outputs": {"token_ids": [7, processor.tokenizer.eos_token_id]},
        }

        result = processor.process_response_dict_streaming(response, request={"id": "stream"})

        self.assertEqual(result["outputs"]["completion_tokens"], "7")
        self.assertEqual(result["outputs"]["text"], "tool-visible")
        self.assertEqual(result["outputs"]["reasoning_content"], "because")
        self.assertEqual(result["outputs"]["reasoning_token_num"], 1)
        self.assertEqual(result["outputs"]["tool_calls"], ["tool-delta"])
        self.assertFalse(result["outputs"]["skipped"])
        self.assertNotIn("stream", processor.decode_status)
        self.assertNotIn("stream", processor.tool_parser_dict)
        self.assertNotIn("stream", processor.model_status_dict)

    def test_has_multimodal_content(self):
        self.assertFalse(self.processor._has_multimodal_content({"prompt": "text"}))
        self.assertTrue(self.processor._has_multimodal_content({"multimodal_data": {"image": ["img"]}}))
        self.assertTrue(
            self.processor._has_multimodal_content(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://img"}}]}
                    ]
                }
            )
        )

    def test_get_pad_id_prefers_llama_eos_when_pad_missing(self):
        llama_tokenizer = DummyLlamaTokenizer()
        llama_tokenizer.pad_token_id = None
        llama_tokenizer.eos_token = 99
        self.processor.tokenizer = llama_tokenizer

        self.assertEqual(self.processor.get_pad_id(), 99)


if __name__ == "__main__":
    unittest.main()
