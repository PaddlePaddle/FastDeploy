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

"""Abstract base class for all data processors.

Provides unified response-processing logic (ids2tokens, process_response_dict*,
update_stop_seq, update_bad_words, pad_batch_data, …) extracted from the two
existing concrete processors:

    DataProcessor      (fastdeploy/input/text_processor.py)
    Ernie4_5Processor  (fastdeploy/input/ernie4_5_processor.py)

Key design decisions
--------------------
* ``__init__`` only initialises response-handling state (decode_status,
  model_status_dict, tool_parser_dict).  Tokeniser setup is the responsibility
  of each subclass.  Subclasses that do not call ``super().__init__()`` must
  initialise those three attributes themselves.

* ``process_response_dict`` reads ``stream`` from ``kwargs`` (DataProcessor
  convention).  Callers that previously passed ``stream`` as a positional
  argument (ERNIE convention) must be updated to use ``stream=`` keyword.

* EOS removal uses ``in self.eos_token_ids`` (list membership).  ERNIE's
  ``eos_token_ids`` contains exactly one element, so this is equivalent to the
  ``==`` check it currently uses.

* tool_parser result never updates ``outputs["text"]``; only ``tool_calls`` is
  set.  This matches DataProcessor behaviour.

* ``ids2tokens`` always returns a three-tuple
  ``(delta_text, previous_token_ids, previous_texts)``.  The HF-tokeniser
  branch previously returned a bare string; the base class fixes that
  inconsistency.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import numpy as np
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import Llama3Tokenizer, LlamaTokenizer

from fastdeploy import envs
from fastdeploy.input.utils import process_stop_token_ids
from fastdeploy.logger.request_logger import RequestLogLevel, log_request
from fastdeploy.utils import data_processor_logger

_SAMPLING_EPS = 1e-5


class BaseTextProcessor(ABC):
    """Abstract base class shared by all text / VL processors.

    Handles the full initialisation sequence: generation config, tokeniser
    loading (via the abstract ``_load_tokenizer`` hook), EOS / pad token
    setup, and parser initialisation.  Concrete subclasses only need to
    implement ``_load_tokenizer`` and ``text2ids``.
    """

    def __init__(self, model_name_or_path, tokenizer_type="auto", reasoning_parser_obj=None, tool_parser_obj=None):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_type = tokenizer_type

        # Response-handling state.
        self.decode_status: Dict[str, list] = {}
        self.model_status_dict: Dict[str, dict] = {}
        self.tool_parser_dict: Dict = {}
        # Token-encode cache shared by all subclasses.
        self._tokenize_cache: OrderedDict = OrderedDict()
        self._tokenize_cache_capacity: int = 128

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config: {e}, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        # Tokenizer (delegated to concrete subclass via @abstractmethod)
        self.tokenizer = self._load_tokenizer()
        data_processor_logger.info(
            f"tokenizer information: bos_token is {self.tokenizer.bos_token}, "
            f"{self.tokenizer.bos_token_id}, "
            f"eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}"
        )

        # EOS tokens
        try:
            from paddleformers.trl.llm_utils import get_eos_token_id
        except Exception:
            from paddleformers.cli.utils.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        data_processor_logger.info(
            f"The eos_token_ids obtained by merging tokenizer and generation_config is {self.eos_token_ids}"
        )
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.tokenizer.pad_token_id = self.pad_token_id
        self._init_parsers(reasoning_parser_obj, tool_parser_obj)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_tokenizer(self): ...  # noqa: E704

    def text2ids(self, text, max_model_len=None, **kwargs):
        """Convert text to token IDs (auto tokenizer path).

        Subclasses with non-standard tokenizers (e.g. ernie4_5, multimodal)
        should override this method.
        """
        add_special_tokens = kwargs.get("add_special_tokens", False)
        if envs.FD_USE_HF_TOKENIZER:
            tokens = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        else:
            text_input = [text] if isinstance(text, str) else text
            tokens = self.tokenizer(
                text_input,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_model_len,
                add_special_tokens=add_special_tokens,
            )
        return tokens["input_ids"][0]

    def messages2ids(self, request, **kwargs):
        """Convert a chat-template request into a token-ID list.

        Works for both ``auto`` and ``ernie4_5`` tokeniser types.
        The ``add_generation_prompt`` kwarg is only injected for non-ernie4_5
        types because that tokeniser does not recognise the argument.
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")
        if self.tokenizer_type != "ernie4_5":
            if "add_generation_prompt" not in kwargs:
                kwargs["add_generation_prompt"] = request.get("add_generation_prompt", True)
        spliced_message = self.tokenizer.apply_chat_template(
            request,
            tokenize=False,
            split_special_tokens=False,
            add_special_tokens=False,
            **kwargs,
        )
        request["prompt_tokens"] = spliced_message
        req_id = request.get("request_id", None) if isinstance(request, dict) else None
        if self.tokenizer_type == "ernie4_5":
            # NOTE: ernie4_5 tokenizer will hang when meet long input when use .encode()
            token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(spliced_message))
        else:
            token_ids = self.tokenizer.encode(spliced_message, add_special_tokens=False)
            if hasattr(token_ids, "input_ids") or (isinstance(token_ids, dict) and "input_ids" in token_ids):
                token_ids = token_ids["input_ids"]
                if hasattr(token_ids, "ndim") and token_ids.ndim > 1:
                    token_ids = token_ids[0]
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if not isinstance(token_ids, list):
                token_ids = list(token_ids)
        log_request(
            level=1,
            message="req_id:{req_id}, token_ids: {token_ids}",
            req_id=req_id,
            token_ids=token_ids,
        )
        return token_ids

    # ------------------------------------------------------------------
    # Parser initialisation helper
    # ------------------------------------------------------------------

    def _init_parsers(self, reasoning_parser_obj, tool_parser_obj):
        """Initialise reasoning / tool parser attributes.

        Must be called *after* ``self.tokenizer`` has been set by the subclass.
        """
        self.reasoning_parser = None
        self.tool_parser_obj = tool_parser_obj
        if reasoning_parser_obj:
            self.reasoning_parser = reasoning_parser_obj(self.tokenizer)

    # ------------------------------------------------------------------
    # ids2tokens
    # ------------------------------------------------------------------

    def ids2tokens(self, token_id, task_id):
        """Incrementally decode *token_id* and return a three-tuple.

        Returns:
            (delta_text, previous_token_ids, previous_texts)

        Both the HF and the PaddleFormers/ERNIE tokeniser paths return the
        same tuple shape.  The HF path sets ``previous_token_ids`` to ``[]``
        since it does not expose per-token ids during batch-decode.
        """
        if envs.FD_USE_HF_TOKENIZER:
            if task_id not in self.decode_status:
                # [all_token_ids, list_of_deltas, full_accumulated_string]
                self.decode_status[task_id] = [[], [], ""]
            status = self.decode_status[task_id]
            status[0].extend(token_id)
            decode_str = self.tokenizer.batch_decode(
                [status[0]],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if isinstance(decode_str, list) and len(decode_str):
                new_str = decode_str[0].replace(status[2], "", 1)
                status[1].append(new_str)
                status[2] = decode_str[0]
            else:
                new_str = ""
            # Return consistent three-tuple; previous_token_ids not available.
            return new_str, [], status[2]
        else:
            if task_id not in self.decode_status:
                # [prefix_offset, read_offset, all_token_ids, accumulated_text]
                self.decode_status[task_id] = [0, 0, [], ""]
            status = self.decode_status[task_id]
            previous_texts = status[3]
            status[2].extend(token_id)
            decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(status[2], status[0], status[1])
            status[0] = prefix_offset
            status[1] = read_offset
            status[3] += decode_str
            return decode_str, status[2], previous_texts

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def process_response_dict(self, response_dict, **kwargs):
        """Dispatch to streaming or non-streaming handler.

        ``stream`` is read from ``kwargs`` (default: True).
        """
        stream = kwargs.get("stream", True)
        if stream:
            return self.process_response_dict_streaming(response_dict, **kwargs)
        else:
            return self.process_response_dict_normal(response_dict, **kwargs)

    def process_response_dict_normal(self, response_dict, **kwargs):
        """Accumulate tokens and build the full completion text (non-streaming)."""
        token_ids = response_dict["outputs"]["token_ids"]
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        request = kwargs.get("request", None)
        direct_decode = kwargs.get("direct_decode", False)

        if is_end and len(token_ids) > 0 and not kwargs.get("include_stop_str_in_output"):
            if token_ids[-1] in self.eos_token_ids:
                token_ids = token_ids[:-1]

        if direct_decode:
            delta_text = self.tokenizer.decode(token_ids)
            previous_texts = ""
        else:
            delta_text, _, previous_texts = self.ids2tokens(token_ids, req_id)

        if is_end:
            full_text = previous_texts + delta_text
            response_dict["outputs"]["completion_tokens"] = full_text
            response_dict["outputs"]["text"] = full_text

            if self.reasoning_parser:
                reasoning_content, text = self.reasoning_parser.extract_reasoning_content(
                    full_text, request, self.model_status_dict[req_id]
                )
                response_dict["outputs"]["text"] = text
                response_dict["outputs"]["reasoning_content"] = reasoning_content
                reasoning_tokens = self.tokenizer.tokenize(reasoning_content)
                response_dict["outputs"]["reasoning_token_num"] = len(reasoning_tokens)

            if self.tool_parser_obj:
                tool_parser = self.tool_parser_obj(self.tokenizer)
                tool_call_info = tool_parser.extract_tool_calls(full_text, request)
                if tool_call_info.tools_called:
                    response_dict["outputs"]["tool_calls"] = tool_call_info.tool_calls

            if req_id in self.decode_status:
                del self.decode_status[req_id]
            if req_id in self.model_status_dict:
                del self.model_status_dict[req_id]

        return response_dict

    def process_response_dict_streaming(self, response_dict, **kwargs):
        """Incrementally decode and populate streaming output fields."""
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        token_ids = response_dict["outputs"]["token_ids"]
        request = kwargs.get("request", None)

        if is_end and len(token_ids) > 0 and not kwargs.get("include_stop_str_in_output"):
            if token_ids[-1] in self.eos_token_ids:
                token_ids = token_ids[:-1]

        delta_text, previous_token_ids, previous_texts = self.ids2tokens(token_ids, req_id)

        response_dict["outputs"]["text"] = delta_text
        response_dict["outputs"]["completion_tokens"] = delta_text
        response_dict["outputs"]["skipped"] = False
        response_dict["outputs"]["tool_calls"] = None
        response_dict["outputs"]["reasoning_content"] = ""

        if self.reasoning_parser:
            reasoning_delta_message = self.reasoning_parser.extract_reasoning_content_streaming(
                previous_texts,
                previous_texts + delta_text,
                delta_text,
                previous_token_ids,
                previous_token_ids + token_ids,
                token_ids,
                self.model_status_dict[req_id],
            )
            if reasoning_delta_message:
                reasoning_content = reasoning_delta_message.reasoning_content
                reasoning_tokens = self.tokenizer.tokenize(reasoning_content) if reasoning_content else []
                response_dict["outputs"]["reasoning_token_num"] = len(reasoning_tokens)
                response_dict["outputs"]["reasoning_content"] = reasoning_content or ""
                response_dict["outputs"]["text"] = reasoning_delta_message.content or ""
            else:
                if not is_end:
                    response_dict["outputs"]["skipped"] = True

        if self.tool_parser_obj:
            if req_id not in self.tool_parser_dict:
                self.tool_parser_dict[req_id] = self.tool_parser_obj(self.tokenizer)
            tool_parser = self.tool_parser_dict[req_id]
            tool_call_delta_message = tool_parser.extract_tool_calls_streaming(
                previous_texts,
                previous_texts + delta_text,
                delta_text,
                previous_token_ids,
                previous_token_ids + token_ids,
                token_ids,
                request,
            )
            if tool_call_delta_message:
                if tool_call_delta_message.tool_calls:
                    response_dict["outputs"]["text"] = tool_call_delta_message.content
                    response_dict["outputs"]["tool_calls"] = tool_call_delta_message.tool_calls
                    response_dict["outputs"]["skipped"] = False
            else:
                if not is_end:
                    response_dict["outputs"]["skipped"] = True

        if is_end:
            del self.decode_status[req_id]
            if req_id in self.tool_parser_dict:
                del self.tool_parser_dict[req_id]
            if req_id in self.model_status_dict:
                del self.model_status_dict[req_id]

        return response_dict

    def process_request_dict(self, request, max_model_len=None, **kwargs):
        """Unified request pre-processing shared by all processors."""
        log_request(RequestLogLevel.CONTENT, message="Start processing request dict: {request}", request=request)
        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids

        # processing stop_sequences and stop_token_ids
        process_stop_token_ids(request, self.update_stop_seq)

        # processing bad_words
        bad_words = request.get("bad_words")
        bad_words_token_ids = request.get("bad_words_token_ids")
        if bad_words:
            bad_words_token_ids = self.update_bad_words(bad_words, bad_words_token_ids)
            request["bad_words_token_ids"] = bad_words_token_ids

        logits_processors_args = self._prepare_think_stop_sentence(
            request.get("logits_processors_args") or {}, max_model_len
        )
        request["logits_processors_args"] = logits_processors_args

        # processing prompt_token_ids
        if not request.get("prompt_token_ids"):
            if request.get("prompt"):
                prompt = request.get("prompt")
                assert isinstance(prompt, str) or (
                    isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)
                ), f"prompt must be a string or a list of integers, but got {type(prompt)}"
                if isinstance(prompt, list):
                    request["prompt_token_ids"] = prompt
                else:
                    request["prompt_tokens"] = prompt
                    add_special_tokens = request.get("add_special_tokens", False)
                    token_ids = self.text2ids(prompt, max_model_len, add_special_tokens=add_special_tokens)
                    if hasattr(token_ids, "tolist"):
                        token_ids = token_ids.tolist()
                    request["prompt_token_ids"] = token_ids
            elif request.get("messages"):
                chat_template_kwargs = request.get("chat_template_kwargs", {})
                if chat_template_kwargs:
                    if isinstance(chat_template_kwargs, dict):
                        for k, v in chat_template_kwargs.items():
                            if k not in request:
                                request[k] = v
                    else:
                        raise ValueError("Invalid input: chat_template_kwargs must be a dict")
                request.setdefault("enable_thinking", True)
                request["prompt_token_ids"] = self.messages2ids(request, **chat_template_kwargs)
            else:
                raise ValueError(f"Request must contain 'prompt_token_ids', 'prompt', or 'messages': {request}")

        if len(request["prompt_token_ids"]) == 0:
            raise ValueError("Invalid input: prompt_token_ids must be a non-empty sequence of token IDs")

        if request.get("completion_token_ids"):
            request["prompt_token_ids"].extend(request["completion_token_ids"])

        # truncate prompts that exceed the length limit
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]

        logits_processors_args = self._update_thinking_prompt_state(
            request["prompt_token_ids"], request.get("logits_processors_args") or {}
        )
        request["logits_processors_args"] = logits_processors_args

        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])
        if request.get("temperature") < _SAMPLING_EPS:
            # zero temperature means greedy decoding: set top_k=1 to force argmax
            request["temperature"] = 1
            request["top_k"] = 1
        if request.get("top_p") < _SAMPLING_EPS:
            request["top_p"] = _SAMPLING_EPS
            request["top_k"] = 1

        if self.reasoning_parser:
            self._apply_reasoning_parser(request)

        if request.get("response_max_tokens") is not None and request.get("enable_thinking") is False:
            request["max_tokens"] = min(request["response_max_tokens"], request["max_tokens"])

        log_request(RequestLogLevel.CONTENT, message="Processed request dict: {request}", request=request)
        return request

    def _apply_reasoning_parser(self, request):
        """Apply reasoning parser to determine model thinking status."""
        model_status = self.reasoning_parser.get_model_status(request["prompt_token_ids"])
        parts = request["request_id"].split("_")
        if len(parts) > 1:
            real_req_id = parts[0]
            index = int(parts[1])
            n = request.get("n", 1)
            for idx in range(index * n, (index + 1) * n):
                self.model_status_dict[f"{real_req_id}_{idx}"] = model_status
        else:
            self.model_status_dict[request["request_id"]] = model_status
        request["enable_thinking"] = model_status == "think_start"

    def clear_request_status(self, task_id):
        """Clear all per-request decode state and return the accumulated text."""
        results_all = ""
        if task_id in self.decode_status:
            if envs.FD_USE_HF_TOKENIZER:
                results_all = self.decode_status[task_id][2]
            else:
                results_all = "".join(self.decode_status[task_id][3])
            del self.decode_status[task_id]
        return results_all

    # ------------------------------------------------------------------
    # Common utility methods
    # ------------------------------------------------------------------

    def update_stop_seq(self, stop_sequences):
        """Convert stop strings to padded token-id sequences."""
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        stop_seqs = []
        for seq in stop_sequences:
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(stop_seqs, pad_id=-1, return_seq_len=True, return_array=False)
        log_request(
            level=3,
            message="processed stop_seqs: {stop_seqs}, {stop_seqs_len}",
            stop_seqs=stop_seqs,
            stop_seqs_len=stop_seqs_len,
        )
        return stop_seqs, stop_seqs_len

    # ------------------------------------------------------------------
    # Request pre-processing helpers (shared with process_request_dict)
    # ------------------------------------------------------------------

    def _apply_default_parameters(self, request):
        """Apply default values for sampling parameters in request."""

        def set_value(req, key, value):
            value = getattr(self.generation_config, key, value)
            if isinstance(req, dict):
                if key not in req or req[key] is None:
                    req[key] = value
            else:
                if req.get(key) is None:
                    req.set(key, value)

        set_value(request, "top_p", 0.7)
        set_value(request, "temperature", 1.0)
        set_value(request, "repetition_penalty", 1.0)
        set_value(request, "frequency_penalty", 0.0)
        set_value(request, "presence_penalty", 0.0)
        return request

    def _encode_literal_text_with_cache(self, text):
        if not hasattr(self, "_tokenize_cache"):
            self._tokenize_cache = OrderedDict()
            self._tokenize_cache_capacity = 128
        key = ("literal_text", text)
        cached = self._tokenize_cache.get(key)
        if cached is not None:
            self._tokenize_cache.move_to_end(key)
            return cached
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            token_ids = list(token_ids)
        self._tokenize_cache[key] = token_ids
        if len(self._tokenize_cache) > self._tokenize_cache_capacity:
            self._tokenize_cache.popitem(last=False)
        return token_ids

    def _get_think_token_ids(self):
        think_token_ids = getattr(self, "_think_token_ids", None)
        if think_token_ids is not None:
            return think_token_ids
        tokenizer = getattr(self, "tokenizer", None)
        vocab = tokenizer.get_vocab() if tokenizer is not None else {}
        think_start_id = vocab.get("<think>", -1)
        think_end_id = vocab.get("</think>", -1)
        self._think_token_ids = (think_start_id, think_end_id)
        return self._think_token_ids

    def _prepare_think_stop_sentence(self, logits_processors_args, max_model_len=None):
        if not isinstance(logits_processors_args, dict):
            return logits_processors_args
        think_stop_sentence = logits_processors_args.get("think_stop_sentence")
        if isinstance(think_stop_sentence, str) and think_stop_sentence:
            sentence_token_ids = self._encode_literal_text_with_cache(think_stop_sentence)
            logits_processors_args["think_stop_sentence_token_ids"] = sentence_token_ids
            logits_processors_args.pop("think_stop_sentence", None)
        return logits_processors_args

    def _update_thinking_prompt_state(self, prompt_token_ids, logits_processors_args):
        if not isinstance(logits_processors_args, dict):
            return logits_processors_args
        thinking_budget = logits_processors_args.get("thinking_budget")
        if thinking_budget is None or not isinstance(thinking_budget, int) or thinking_budget < 0:
            return logits_processors_args
        if logits_processors_args.get("think_prompt_checked"):
            return logits_processors_args
        if prompt_token_ids is None:
            return logits_processors_args
        token_len = getattr(prompt_token_ids, "size", None) or len(prompt_token_ids)
        if token_len == 0:
            return logits_processors_args
        think_start_id, think_end_id = self._get_think_token_ids()
        if think_start_id < 0 or think_end_id < 0:
            return logits_processors_args

        if hasattr(prompt_token_ids, "tolist"):
            token_list = prompt_token_ids.tolist()
        else:
            token_list = list(prompt_token_ids)

        started = False
        ended = False
        tokens_after_start = 0
        last_token_id = None
        in_thinking = False
        for token_id in token_list:
            if token_id == think_start_id:
                started = True
                ended = False
                in_thinking = True
            elif token_id == think_end_id and in_thinking:
                ended = True
                in_thinking = False
        if started and token_list:
            last_token_id = int(token_list[-1])

        logits_processors_args["think_prompt_checked"] = True
        logits_processors_args["think_prompt_started"] = started
        logits_processors_args["think_prompt_ended"] = ended
        logits_processors_args["think_prompt_tokens_after_start"] = tokens_after_start
        if last_token_id is not None:
            logits_processors_args["think_prompt_last_token_id"] = last_token_id
        else:
            logits_processors_args.pop("think_prompt_last_token_id", None)
        return logits_processors_args

    def update_bad_words(self, bad_words, bad_words_token_ids):
        """Tokenize bad-word strings and merge with existing bad-word token ids."""
        token_ids = bad_words_token_ids
        if token_ids is None:
            token_ids = []
        for bad_word in bad_words:
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
                if len(prompt_token_ids) != 1:
                    if not add_prefix_space:
                        log_request(
                            level=1,
                            message="bad_words: '{prompt}' tokenises to {num_tokens} tokens, skipping",
                            prompt=prompt,
                            num_tokens=len(prompt_token_ids),
                        )
                    continue
                if prompt_token_ids[0] > self.tokenizer.vocab_size:
                    if not add_prefix_space:
                        log_request(
                            level=1,
                            message="bad_words: '{prompt}' token id {token_id} > vocab_size, skipping",
                            prompt=prompt,
                            token_id=prompt_token_ids[0],
                        )
                    continue
                if prompt_token_ids not in token_ids:
                    token_ids.extend(prompt_token_ids)
        return token_ids

    def get_pad_id(self):
        """Return the padding token id, with LlamaTokenizer fallback."""
        if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
            return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id

    def pad_batch_data(self, insts, pad_id=0, return_seq_len=False, return_array=True, pad_style="right"):
        """Pad a list of variable-length lists to a rectangular array."""
        if len(insts) == 0:
            padded_insts = np.array([[]], dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts
        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts]
        else:
            padded_insts = [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts]
        if return_array:
            padded_insts = np.array(padded_insts, dtype=np.int64).reshape([-1, max_len])
        if return_seq_len:
            seq_len = [len(inst) for inst in insts]
            if return_array:
                seq_len = np.array(seq_len, dtype=np.int64).reshape(-1, 1)
            return padded_insts, seq_len
        return padded_insts

    def get_mm_max_tokens_per_item(self, seq_len: int):
        """Return the maximum number of tokens per item for each modality.

        Text-only processors return None; multimodal processors override this.
        """
        return None
