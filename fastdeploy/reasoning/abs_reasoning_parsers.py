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

from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Callable, Optional, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.utils import is_list_of


class ReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly.
    Provided and methods should be used in derived classes.

    It is used to extract reasoning content from the model output.
    """

    def __init__(self, tokenizer):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        """
        Get the vocabulary of the model tokenizer.
        """
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        It is used in structured engines like `xgrammar` to check if the
        reasoning content ends in the model output.

        Parameters:
        input_ids: list[int]
            The input_ids of the model output.

        Returns:
        bool
            True if the reasoning content ends in the input_ids.
        """

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        Parameters:
        input_ids: list[int]
            The input_ids of the model output.
        Returns:
        list[int]
            The extracted content from the input_ids.
        """

    @abstractmethod
    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        """

    @abstractmethod
    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """


class ReasoningParserManager:
    """
    ReasoningParserManager
    """

    reasoning_parsers: dict[str, type] = {}

    @classmethod
    def get_reasoning_parser(cls, name: Optional[str]) -> type[ReasoningParser]:
        """
        Get reasoning parser by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        name = name.replace("_", "-")
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]

        raise KeyError(f"reasoning helper: '{name}' not found in reasoning_parsers")

    @classmethod
    def _register_module(
        cls,
        module: type,
        module_name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
    ) -> None:
        if not issubclass(module, ReasoningParser):
            raise TypeError("module must be subclass of ReasoningParser, " f"but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.reasoning_parsers:
                existed_module = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered " f"at {existed_module.__module__}")
            cls.reasoning_parsers[name] = module

    @classmethod
    def register_module(
        cls,
        name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
        module: Union[type, None] = None,
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
