"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import warnings

from fastdeploy.input.base_processor import (  # backward compat  # noqa: F401
    _SAMPLING_EPS,
)
from fastdeploy.input.text_processor import (  # backward compat  # noqa: F401
    BaseDataProcessor,
    TextProcessor,
)


class Ernie4_5Processor(TextProcessor):
    """Deprecated. Use ``TextProcessor(tokenizer_type='ernie4_5')`` instead."""

    def __init__(self, model_name_or_path, reasoning_parser_obj=None, tool_parser_obj=None):
        warnings.warn(
            "Ernie4_5Processor is deprecated. " "Use TextProcessor(tokenizer_type='ernie4_5') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from paddleformers.trl.llm_utils import get_eos_token_id
        except Exception:
            from paddleformers.cli.utils.llm_utils import get_eos_token_id

        self.eos_token_ids = get_eos_token_id(self.tokenizer, self.generation_config)
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.reasoning_parser = None
        self.tool_parser_obj = tool_parser_obj
        if reasoning_parser_obj:
            self.reasoning_parser = reasoning_parser_obj(self.tokenizer)

    def process_request(self, request, max_model_len=None, **kwargs):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        data_processor_logger.info(f"Start processing request: {request}")
        request = self._apply_default_parameters(request)
        if request.get("eos_token_ids") is None or len(request.eos_token_ids) == 0:
            request.eos_token_ids = self.eos_token_ids

        # processing stop_sequences and stop_token_ids
        process_stop_token_ids(request, self.update_stop_seq)

        # processing bad_words
        bad_words = request.get("bad_words")
        bad_words_token_ids = request.get("bad_words_token_ids")
        if bad_words:
            bad_words_token_ids = self.update_bad_words(bad_words, bad_words_token_ids)
            request["bad_words_token_ids"] = bad_words_token_ids

        # processing prompt_token_ids
        if request.prompt_token_ids is None or len(request.prompt_token_ids) == 0:
            if request.prompt is not None:
                prompt = request.prompt
                tokens = self.tokenizer.tokenize(prompt)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                request.prompt_token_ids = token_ids
                data_processor_logger.debug(
                    f"request_ids: {request.request_id}, prompt: {prompt}, tokens: {tokens}, token_ids: {token_ids}"
                )
            elif request.messages is not None:
                task = request.to_dict()
                chat_template_kwargs = kwargs.get("chat_template_kwargs", {})
                if chat_template_kwargs:
                    if isinstance(chat_template_kwargs, dict):
                        for k, v in chat_template_kwargs.items():
                            if k not in task or task[k] is None:
                                task[k] = v
                    else:
                        raise ValueError("Invalid input: chat_template_kwargs must be a dict")
                request.prompt_token_ids = self.messages2ids(task, **chat_template_kwargs)
            else:
                raise ValueError(f"The request should have `prompt_token_ids`, `prompt` or `messages`: {request}.")

        if len(request.prompt_token_ids) == 0:
            raise ValueError("Invalid input: prompt_token_ids must be a non-empty sequence of token IDs")

        # truncate prompts that exceed the length limit
        if max_model_len is not None and len(request.prompt_token_ids) > max_model_len:
            request.prompt_token_ids = request.prompt_token_ids[: max_model_len - 1]
        max_tokens = max_model_len - len(request.prompt_token_ids)
        if request.get("max_tokens") is None:
            request.set("max_tokens", max(1, max_tokens))
        else:
            request.set("max_tokens", min(max_tokens, request.get("max_tokens")))
        if request.get("temperature") < _SAMPLING_EPS:
            # zero temperature is equivalent to greedy sampling:
            # use temperature=1 (no scaling) with tiny top_p to force argmax
            request.set("temperature", 1)
            request.set("top_p", _SAMPLING_EPS)
        if request.get("top_p") < _SAMPLING_EPS:
            request.set("top_p", _SAMPLING_EPS)
        if self.reasoning_parser:
            model_status = self.reasoning_parser.get_model_status(request.prompt_token_ids)
            parts = request.request_id.split("_")
            if len(parts) > 1:
                real_req_id = parts[0]
                index = int(parts[1])
                n = request.get("n", 1)
                for idx in range(index * n, (index + 1) * n):
                    self.model_status_dict[f"{real_req_id}_{idx}"] = model_status
            else:
                self.model_status_dict[request.request_id] = model_status
            request.enable_thinking = model_status == "think_start"

        data_processor_logger.info(f"Processed request: {request}")
        return request

    def process_request_dict(self, request, max_model_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        data_processor_logger.info(f"Start processing request dict: {request}")
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

        # processing prompt_token_ids
        if not request.get("prompt_token_ids"):
            if request.get("prompt"):
                prompt = request.get("prompt")
                assert isinstance(prompt, str) or (
                    isinstance(prompt, list) and all([isinstance(t, int) for t in prompt])
                ), f"prompt must be a string or a list of integers, but got {type(prompt)}"
                if isinstance(prompt, list):  # if prompt is a token id list
                    request["prompt_token_ids"] = prompt
                else:
                    request["prompt_tokens"] = prompt
                    tokens = self.tokenizer.tokenize(prompt)
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    request["prompt_token_ids"] = token_ids
                    req_id = request.get("request_id", None)
                    data_processor_logger.info(f"req_id:{req_id}, tokens:{tokens}, token_ids: {token_ids}")
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

        # truncate prompts that exceed the length limit
        if max_model_len is not None and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][: max_model_len - 1]
        max_tokens = max_model_len - len(request["prompt_token_ids"])
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_tokens)
        else:
            request["max_tokens"] = min(max_tokens, request["max_tokens"])
        if request.get("temperature") < _SAMPLING_EPS:
            # zero temperature is equivalent to greedy sampling:
            # use temperature=1 (no scaling) with tiny top_p to force argmax
            request["temperature"] = 1
            request["top_p"] = _SAMPLING_EPS
        if request.get("top_p") < _SAMPLING_EPS:
            request["top_p"] = _SAMPLING_EPS

        if self.reasoning_parser:
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
        if request.get("response_max_tokens") is not None and request.get("enable_thinking") is False:
            request["max_tokens"] = min(request["response_max_tokens"], request["max_tokens"])
        data_processor_logger.info(f"Processed request dict: {request}")
        return request

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        req_id = response_dict.request_id
        token_ids = response_dict.outputs.token_ids

        response_dict.usage = {"completion_tokens": response_dict.outputs.index + 1}
        if token_ids[-1] == self.tokenizer.eos_token_id:
            token_ids = token_ids[:-1]
        full_text = self.tokenizer.decode(token_ids)
        if self.reasoning_parser:
            reasoning_content, text = self.reasoning_parser.extract_reasoning_content(
                full_text,
                response_dict,
                self.model_status_dict[req_id],
            )
            response_dict.outputs.text = text
            response_dict.outputs.reasoning_content = reasoning_content
        else:
            response_dict.outputs.text = full_text
        if self.tool_parser_obj:
            tool_parser = self.tool_parser_obj(self.tokenizer)
            tool_call_info = tool_parser.extract_tool_calls(full_text, response_dict)
            if tool_call_info.tools_called:
                response_dict.outputs.tool_calls = tool_call_info.tool_calls
                response_dict.outputs.text = tool_call_info.content
        if req_id in self.model_status_dict:
            del self.model_status_dict[req_id]
        data_processor_logger.info(f"req_id:{req_id}, token_ids: {token_ids}")
        if response_dict.outputs.text == "" and response_dict.outputs.reasoning_content == "":
            return None
        return response_dict

    def process_response_dict(self, response_dict, stream, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        if stream:
            return self.process_response_dict_streaming(response_dict, **kwargs)
        else:
            return self.process_response_dict_normal(response_dict, **kwargs)

    def process_response_dict_normal(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        token_ids = response_dict["outputs"]["token_ids"]
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        if is_end and len(token_ids) > 0 and not kwargs.get("include_stop_str_in_output"):
            if token_ids[-1] == self.tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
        delta_text, _, previous_texts = self.ids2tokens(token_ids, req_id)
        response_dict["outputs"]["enable_parser"] = False
        if is_end:
            full_text = previous_texts + delta_text
            response_dict["outputs"]["text"] = full_text
            if self.reasoning_parser:
                response_dict["outputs"]["enable_parser"] = True
                reasoning_content, text = self.reasoning_parser.extract_reasoning_content(
                    full_text,
                    response_dict,
                    self.model_status_dict[req_id],
                )
                response_dict["outputs"]["text"] = text
                response_dict["outputs"]["reasoning_content"] = reasoning_content
                reasoning_tokens = self.tokenizer.tokenize(reasoning_content)
                response_dict["outputs"]["reasoning_token_num"] = len(reasoning_tokens)
            if self.tool_parser_obj:
                response_dict["outputs"]["enable_parser"] = True
                tool_parser = self.tool_parser_obj(self.tokenizer)
                tool_call_info = tool_parser.extract_tool_calls(full_text, response_dict)
                if tool_call_info.tools_called:
                    response_dict["outputs"]["tool_calls"] = tool_call_info.tool_calls
                    response_dict["outputs"]["text"] = tool_call_info.content
            response_dict["outputs"]["completion_tokens"] = full_text
            data_processor_logger.info(f"req_id:{req_id}, decode_status: {self.decode_status[req_id]}")
            del self.decode_status[req_id]
            if req_id in self.model_status_dict:
                del self.model_status_dict[req_id]
        return response_dict

    def process_response_dict_streaming(self, response_dict, **kwargs):
        """
        Preprocess the response streaming

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        token_ids = response_dict["outputs"]["token_ids"]
        response_dict["outputs"]["enable_parser"] = False

        if is_end and len(token_ids) > 0 and not kwargs.get("include_stop_str_in_output"):
            if token_ids[-1] == self.tokenizer.eos_token_id:
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
            response_dict["outputs"]["enable_parser"] = True
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
                response_dict,
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
            data_processor_logger.info(f"req_id:{req_id}, decode_status: {self.decode_status[req_id]}")
            del self.decode_status[req_id]
            if req_id in self.tool_parser_dict:
                del self.tool_parser_dict[req_id]
            if req_id in self.model_status_dict:
                del self.model_status_dict[req_id]
        return response_dict

    def messages2ids(self, request_or_messages, **kwargs):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            request_or_messages: Either a request dict containing 'messages' field,
                                or a list of message dicts directly

        Returns:
            List of token IDs as strings (converted from token objects)
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")
        spliced_message = self.tokenizer.apply_chat_template(
            request_or_messages,
            tokenize=False,
            split_special_tokens=False,
            add_special_tokens=False,
            **kwargs,
        )
