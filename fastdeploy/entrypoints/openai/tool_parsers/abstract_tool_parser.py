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

import os
from collections.abc import Sequence
from functools import cached_property
from typing import Callable, Optional, Union

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
)
from fastdeploy.utils import data_processor_logger, import_from_path, is_list_of


class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    # Subclasses should override these with the literal tool-call sentinel
    # tokens they recognize (e.g. ``"<tool_call>"`` / ``"</tool_call>"``).
    # Used by :meth:`detect_tool_prefix` to support ``tool_choice=required``
    # style prompt-prefix injection. Empty defaults make the detection a no-op
    # for parsers that have not opted in.
    tool_call_start_token: str = ""
    tool_call_end_token: str = ""

    def __init__(self, tokenizer):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

        # Per-request tool-prefix state populated by the serving layer when
        # the chat template injects a forced tool-call prefix into the prompt.
        self._tool_prefix: str = ""
        self._tool_prefix_token_ids: list[int] = []
        # Set after the prefix is computed once for this request.
        self._tool_prefix_computed: bool = False
        # Set after the prefix has been spliced into the streaming delta
        # (only the first chunk needs it).
        self._tool_prefix_injected_to_delta: bool = False

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Static method that used to adjust the request parameters.
        """
        return request

    def detect_tool_prefix(self, prompt: str) -> str:
        """Detect a tool-call prefix that the chat template injected at the tail
        of the rendered prompt to force tool output (``tool_choice=required``).

        The check is generic: find the **last** occurrence of
        :attr:`tool_call_start_token` in ``prompt`` and, if it is **not** closed
        by a subsequent :attr:`tool_call_end_token`, treat the substring from
        that position to the end of the prompt as the injected prefix. The
        injected prefix must reach the very end of the prompt (modulo trailing
        whitespace) — anything else is treated as historical / unrelated and
        we conservatively return an empty string.

        Returns ``""`` for parsers that have not declared their sentinel tokens
        or for prompts where no such prefix is detected.

        Subclasses with non-paired tag formats (e.g. a single sentinel without
        a closing counterpart) may override this method.
        """
        start = self.tool_call_start_token
        if not start or not prompt:
            return ""

        last_start = prompt.rfind(start)
        if last_start == -1:
            return ""

        end = self.tool_call_end_token
        if end and prompt.find(end, last_start + len(start)) != -1:
            # The last start token is closed — this is a historical, completed
            # tool-call (e.g. from a previous assistant turn), not an injected
            # forced prefix.
            return ""

        # By construction, ``prompt[last_start:]`` reaches the end of the
        # prompt. We treat the whole tail as the injected prefix. Subclasses
        # whose chat templates place additional content after the prefix can
        # override this method to apply stricter validation.
        return prompt[last_start:]

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError("AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError("AbstractToolParser.extract_tool_calls_streaming has not been " "implemented!")


class ToolParserManager:
    tool_parsers: dict[str, type] = {}

    @classmethod
    def get_tool_parser(cls, name) -> type:
        """
        Get tool parser by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        name = name.replace("_", "-")
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        raise KeyError(f"tool helper: '{name}' not found in tool_parsers")

    @classmethod
    def _register_module(
        cls, module: type, module_name: Optional[Union[str, list[str]]] = None, force: bool = True
    ) -> None:
        if not issubclass(module, ToolParser):
            raise TypeError(f"module must be subclass of ToolParser, but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.tool_parsers:
                existed_module = cls.tool_parsers[name]
                raise KeyError(f"{name} is already registered " f"at {existed_module.__module__}")
            cls.tool_parsers[name] = module

    @classmethod
    def register_module(
        cls, name: Optional[Union[str, list[str]]] = None, force: bool = True, module: Union[type, None] = None
    ) -> Union[type, Callable]:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_list_of(name, str)):
            raise TypeError("name must be None, an instance of str, or a sequence of str, " f"but got {type(name)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined tool parser by the path of the tool parser define
        file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            data_processor_logger.exception("Failed to load module '%s' from %s.", module_name, plugin_path)
            return
