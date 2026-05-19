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

"""Unit tests for fastdeploy.input.processor.Processor — server-level length control logic."""

import importlib
import sys
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Dummy stubs
# ---------------------------------------------------------------------------


class DummyTokenizer:
    bos_token = "<s>"
    eos_token = "</eos>"

    def __init__(self):
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.vocab_size = 256
        self.bos_token_id = 3

    def __call__(self, text, **kwargs):
        # Simple: each character is one token
        ids = list(range(len(text)))
        return {"input_ids": [ids]}

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, tokens):
        return list(range(len(tokens)))

    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(range(len(text)))

    def decode(self, token_ids, **kwargs):
        return "".join(str(t) for t in token_ids)


class DummyAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return DummyTokenizer()


class DummyGenerationConfig:
    top_p = 0.8
    temperature = 0.9
    repetition_penalty = 1.1
    frequency_penalty = 0.2
    presence_penalty = 0.1

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def _setup_modules():
    """Inject dummy modules so we can import processor without real dependencies."""
    repo_root = Path(__file__).resolve().parents[2]

    dummy_logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )

    utils_module = types.ModuleType("fastdeploy.utils")
    utils_module.data_processor_logger = dummy_logger
    utils_module.CHOICE_SEPARATOR = "::n::"

    envs_module = types.ModuleType("fastdeploy.envs")
    envs_module.FD_USE_HF_TOKENIZER = False

    generation_module = types.ModuleType("paddleformers.generation")
    generation_module.GenerationConfig = DummyGenerationConfig

    transformers_module = types.ModuleType("paddleformers.transformers")
    transformers_module.AutoTokenizer = DummyAutoTokenizer
    transformers_module.LlamaTokenizer = type("LlamaTokenizer", (), {})
    transformers_module.Llama3Tokenizer = type("Llama3Tokenizer", (), {})

    llm_utils_module = types.ModuleType("paddleformers.cli.utils.llm_utils")
    llm_utils_module.get_eos_token_id = lambda tokenizer, config: [tokenizer.eos_token_id]

    trl_utils_module = types.ModuleType("paddleformers.trl.llm_utils")
    trl_utils_module.get_eos_token_id = lambda tokenizer, config: [tokenizer.eos_token_id]

    fastdeploy_module = types.ModuleType("fastdeploy")
    fastdeploy_module.__path__ = [str(repo_root / "fastdeploy")]
    fastdeploy_module.utils = utils_module
    fastdeploy_module.envs = envs_module

    logger_module = types.ModuleType("fastdeploy.logger")
    logger_module.__path__ = [str(repo_root / "fastdeploy" / "logger")]

    request_logger_module = types.ModuleType("fastdeploy.logger.request_logger")
    request_logger_module.RequestLogLevel = SimpleNamespace(CONTENT="CONTENT")
    request_logger_module.log_request = lambda *a, **kw: None

    input_utils_module = types.ModuleType("fastdeploy.input.utils")
    input_utils_module.process_stop_token_ids = lambda req, fn: None

    input_module = types.ModuleType("fastdeploy.input")
    input_module.__path__ = [str(repo_root / "fastdeploy" / "input")]

    modules = {
        "fastdeploy": fastdeploy_module,
        "fastdeploy.utils": utils_module,
        "fastdeploy.envs": envs_module,
        "fastdeploy.logger": logger_module,
        "fastdeploy.logger.request_logger": request_logger_module,
        "fastdeploy.input": input_module,
        "fastdeploy.input.utils": input_utils_module,
        "paddleformers.generation": generation_module,
        "paddleformers.transformers": transformers_module,
        "paddleformers.cli.utils.llm_utils": llm_utils_module,
        "paddleformers.trl.llm_utils": trl_utils_module,
    }
    return modules


def _import_processor():
    modules = _setup_modules()
    saved = {}
    for name, mod in modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        processor_module = importlib.import_module("fastdeploy.input.processor")
        importlib.reload(processor_module)
    except Exception:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        raise

    return processor_module, saved


# Import once at module level
_processor_module, _saved_modules = _import_processor()
Processor = _processor_module.Processor


def _make_processor(**overrides):
    """Create a Processor instance bypassing heavy __init__."""
    proc = object.__new__(Processor)
    proc.model_name_or_path = "/tmp/fake_model"
    proc.tokenizer_type = "auto"
    proc.mm_processor = None
    proc.force_disable_thinking = False
    proc.set_default_reasoning_max_tokens = False
    proc.max_completion_tokens = None
    proc.reasoning_max_tokens = None
    proc.response_max_tokens = None
    proc.min_completion_tokens = None
    proc.input_max_tokens = None
    proc.truncate_prompt_tokens = True
    proc.tokenizer = DummyTokenizer()
    proc.eos_token_ids = [2]
    proc.eos_token_id_len = 1
    proc.pad_token_id = 1
    proc.generation_config = DummyGenerationConfig()
    proc.reasoning_parser = None
    proc.tool_parser_obj = None
    proc.decode_status = {}
    proc.model_status_dict = {}
    proc.tool_parser_dict = {}
    proc._tokenize_cache = OrderedDict()
    proc._tokenize_cache_capacity = 128

    for key, val in overrides.items():
        setattr(proc, key, val)
    return proc


class TestProcessRequestDictLengthControl(unittest.TestCase):
    """Tests for server-level length control parameters in Processor.process_request_dict."""

    def _process(self, proc, request, max_model_len=300):
        """Helper to call process_request_dict with a prompt_token_ids-ready request."""
        # Provide prompt_token_ids directly to skip tokenization complexity
        if "prompt_token_ids" not in request:
            request["prompt_token_ids"] = [1, 2, 3, 4, 5]  # length 5
        return proc.process_request_dict(request, max_model_len=max_model_len)

    # ==================================================================
    # max_completion_tokens
    # ==================================================================

    def test_max_completion_tokens_server_default(self):
        """Server max_completion_tokens used when request omits max_tokens."""
        proc = _make_processor(max_completion_tokens=50)
        result = self._process(proc, {})
        # context_remaining = 300 - 5 = 295; min(295, 50) = 50
        self.assertEqual(result["max_tokens"], 50)

    def test_max_completion_tokens_server_larger_than_context(self):
        """Server max_completion_tokens > context_remaining, clamped."""
        proc = _make_processor(max_completion_tokens=5000)
        result = self._process(proc, {})
        # context_remaining = 295; min(295, 5000) = 295
        self.assertEqual(result["max_tokens"], 295)

    def test_max_completion_tokens_request_smaller(self):
        """Request max_tokens < server, request wins."""
        proc = _make_processor(max_completion_tokens=50)
        result = self._process(proc, {"max_tokens": 30})
        # min(295, 50, 30) = 30
        self.assertEqual(result["max_tokens"], 30)

    def test_max_completion_tokens_request_larger_clamped(self):
        """Request max_tokens > server, server clamps."""
        proc = _make_processor(max_completion_tokens=50)
        result = self._process(proc, {"max_tokens": 200})
        # min(295, 50, 200) = 50
        self.assertEqual(result["max_tokens"], 50)

    def test_max_completion_tokens_none_uses_context(self):
        """Server=None, no request → uses context_remaining."""
        proc = _make_processor(max_completion_tokens=None)
        result = self._process(proc, {})
        self.assertEqual(result["max_tokens"], 295)

    # ==================================================================
    # reasoning_max_tokens
    # ==================================================================

    def test_reasoning_max_tokens_server_only(self):
        """Only server reasoning_max_tokens set."""
        proc = _make_processor(reasoning_max_tokens=100)
        result = self._process(proc, {})
        # min(max_tokens=295, server=100) = 100
        self.assertEqual(result["reasoning_max_tokens"], 100)

    def test_reasoning_max_tokens_request_only(self):
        """Only request reasoning_max_tokens set."""
        proc = _make_processor(reasoning_max_tokens=None)
        result = self._process(proc, {"reasoning_max_tokens": 150})
        # min(295, 150) = 150
        self.assertEqual(result["reasoning_max_tokens"], 150)

    def test_reasoning_max_tokens_both_take_min(self):
        """Both set, take min."""
        proc = _make_processor(reasoning_max_tokens=100)
        result = self._process(proc, {"reasoning_max_tokens": 150})
        # min(295, 100, 150) = 100
        self.assertEqual(result["reasoning_max_tokens"], 100)

    def test_reasoning_max_tokens_request_smaller(self):
        """Both set, request is smaller."""
        proc = _make_processor(reasoning_max_tokens=200)
        result = self._process(proc, {"reasoning_max_tokens": 80})
        # min(295, 200, 80) = 80
        self.assertEqual(result["reasoning_max_tokens"], 80)

    def test_reasoning_max_tokens_clamped_by_max_tokens(self):
        """Clamped by max_tokens."""
        proc = _make_processor(max_completion_tokens=50, reasoning_max_tokens=100)
        result = self._process(proc, {})
        # max_tokens=50, min(50, 100) = 50
        self.assertEqual(result["reasoning_max_tokens"], 50)

    def test_reasoning_max_tokens_both_none_not_set(self):
        """Both None → key not set."""
        proc = _make_processor(reasoning_max_tokens=None)
        result = self._process(proc, {})
        self.assertNotIn("reasoning_max_tokens", result)

    # ==================================================================
    # response_max_tokens
    # ==================================================================

    def test_response_max_tokens_server_only(self):
        """Only server response_max_tokens set."""
        proc = _make_processor(response_max_tokens=100)
        result = self._process(proc, {})
        self.assertEqual(result["response_max_tokens"], 100)

    def test_response_max_tokens_request_only(self):
        """Only request response_max_tokens set."""
        proc = _make_processor(response_max_tokens=None)
        result = self._process(proc, {"response_max_tokens": 150})
        self.assertEqual(result["response_max_tokens"], 150)

    def test_response_max_tokens_both_take_min(self):
        """Both set, take min."""
        proc = _make_processor(response_max_tokens=100)
        result = self._process(proc, {"response_max_tokens": 150})
        self.assertEqual(result["response_max_tokens"], 100)

    def test_response_max_tokens_request_smaller(self):
        """Both set, request is smaller."""
        proc = _make_processor(response_max_tokens=200)
        result = self._process(proc, {"response_max_tokens": 80})
        self.assertEqual(result["response_max_tokens"], 80)

    def test_response_max_tokens_clamped_by_max_tokens(self):
        """Clamped by max_tokens."""
        proc = _make_processor(max_completion_tokens=50, response_max_tokens=100)
        result = self._process(proc, {})
        self.assertEqual(result["response_max_tokens"], 50)

    def test_response_max_tokens_both_none_not_set(self):
        """Both None → key not set."""
        proc = _make_processor(response_max_tokens=None)
        result = self._process(proc, {})
        self.assertNotIn("response_max_tokens", result)

    # ==================================================================
    # min_completion_tokens
    # ==================================================================

    def test_min_completion_tokens_server_only(self):
        """Only server min_completion_tokens set."""
        proc = _make_processor(min_completion_tokens=20)
        result = self._process(proc, {})
        self.assertEqual(result["min_tokens"], 20)

    def test_min_completion_tokens_request_only(self):
        """Only request min_tokens set."""
        proc = _make_processor(min_completion_tokens=None)
        result = self._process(proc, {"min_tokens": 30})
        self.assertEqual(result["min_tokens"], 30)

    def test_min_completion_tokens_take_max_user_larger(self):
        """max(server, user), user is larger."""
        proc = _make_processor(min_completion_tokens=20)
        result = self._process(proc, {"min_tokens": 50})
        # max(20, 50) = 50
        self.assertEqual(result["min_tokens"], 50)

    def test_min_completion_tokens_take_max_server_larger(self):
        """max(server, user), server is larger."""
        proc = _make_processor(min_completion_tokens=80)
        result = self._process(proc, {"min_tokens": 30})
        # max(80, 30) = 80
        self.assertEqual(result["min_tokens"], 80)

    def test_min_completion_tokens_server_exceeds_max_tokens_raises(self):
        """Server min_completion_tokens > max_tokens raises ValueError."""
        proc = _make_processor(min_completion_tokens=300, max_completion_tokens=50)
        with self.assertRaises(ValueError):
            self._process(proc, {})

    def test_min_completion_tokens_request_exceeds_max_tokens_raises(self):
        """Request min_tokens > max_tokens raises ValueError."""
        proc = _make_processor(min_completion_tokens=None, max_completion_tokens=50)
        with self.assertRaises(ValueError):
            self._process(proc, {"min_tokens": 500})

    def test_min_completion_tokens_both_none_not_set(self):
        """Both None → min_tokens key not set."""
        proc = _make_processor(min_completion_tokens=None)
        result = self._process(proc, {})
        self.assertNotIn("min_tokens", result)

    # ==================================================================
    # input_max_tokens & truncate_prompt_tokens
    # ==================================================================

    def test_input_max_tokens_reject(self):
        """Prompt exceeding input_max_tokens raises ValueError."""
        proc = _make_processor(input_max_tokens=3)
        with self.assertRaises(ValueError):
            self._process(proc, {"prompt_token_ids": [1, 2, 3, 4, 5]})

    def test_truncate_prompt_tokens_enabled(self):
        """Prompt exceeding max_model_len is truncated when truncate_prompt_tokens=True."""
        proc = _make_processor(truncate_prompt_tokens=True)
        result = self._process(proc, {"prompt_token_ids": list(range(50))}, max_model_len=10)
        # Truncated to max_model_len - 1 = 9
        self.assertEqual(len(result["prompt_token_ids"]), 9)

    def test_truncate_prompt_tokens_disabled_raises(self):
        """Prompt exceeding max_model_len raises when truncate_prompt_tokens=False."""
        proc = _make_processor(truncate_prompt_tokens=False)
        with self.assertRaises(ValueError):
            self._process(proc, {"prompt_token_ids": list(range(50))}, max_model_len=10)

    # ==================================================================
    # set_server_defaults
    # ==================================================================

    def test_set_server_defaults(self):
        """set_server_defaults copies all fields from model_config."""
        proc = _make_processor()
        mc = SimpleNamespace(
            max_completion_tokens=100,
            reasoning_max_tokens=80,
            response_max_tokens=60,
            min_completion_tokens=10,
            input_max_tokens=512,
            truncate_prompt_tokens=False,
        )
        proc.set_server_defaults(mc)
        self.assertEqual(proc.max_completion_tokens, 100)
        self.assertEqual(proc.reasoning_max_tokens, 80)
        self.assertEqual(proc.response_max_tokens, 60)
        self.assertEqual(proc.min_completion_tokens, 10)
        self.assertEqual(proc.input_max_tokens, 512)
        self.assertFalse(proc.truncate_prompt_tokens)


if __name__ == "__main__":
    unittest.main()
