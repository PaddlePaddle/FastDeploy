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

import multiprocessing
import os
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from fastdeploy.config import ErnieArchitectures, FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.reasoning import ReasoningParserManager
from fastdeploy.utils import llm_logger


class LogitsProcessorBase:
    """
    Abstract base class for logits processors in guided decoding.

    This class defines the interface for logits processors that modify token probabilities
    during generation to enforce schema constraints. Subclasses should implement all
    abstract methods to provide specific constraint enforcement logic.

    Attributes:
        None (all state should be managed by subclasses)
    """

    def __init__(self, enable_reasoning):
        self.reasoning_ended = False
        self.enable_reasoning = enable_reasoning

    def fill_token_bitmask(self, token_bitmask, idx):
        """
        Fill the vocabulary mask.

        Args:
            token_bitmask (tensor): The vocabulary mask tensor.
            idx (tensor): The tensor index.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def apply_token_mask(self, logits, token_bitmask):
        """
        Apply the vocabulary mask to logits.

        Args:
            logits (tensor): The logits tensor.
            token_bitmask (tensor): The vocabulary mask tensor.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def allocate_token_bitmask(self, batch_size, vocab_size):
        """
        Allocate a token bitmask for the given batch size and vocabulary size.

        Args:
            batch_size (int): The batch size.
            vocab_size (int): The vocabulary size.

        Returns:
            tensor: The allocated token bitmask.
        """
        raise NotImplementedError

    def accept_token(self, token):
        """
        Accept tokens based on the token bitmask

        Args:
            token (int): The token id.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def is_terminated(self):
        """
        Check if the processor has been terminated.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the matcher state.
        """
        raise NotImplementedError

    def copy(self):
        """
        Create a copy of the backend instance.

        Returns:
            BackendBase: A copy of the backend instance.
        """
        raise NotImplementedError


class BackendBase:
    """
    Abstract base class for guided decoding backends.

    This class provides the core infrastructure for managing schema processors and
    their caching. It handles:
    - Processor creation and caching
    - Tokenizer initialization
    - Thread pool management for async operations

    Attributes:
        cache (dict): Cache of schema processors
        fd_config (FDConfig): FastDeploy configuration
        executor (ThreadPoolExecutor): Thread pool for async operations
        max_cache_size (int): Maximum number of processors to cache
        hf_tokenizer: HuggingFace tokenizer instance
    """

    def __init__(self, fd_config: FDConfig):
        self.fd_config = fd_config
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_cache_size = 2048
        self.reasoning_parser = None

        self.hf_tokenizer = self._get_tokenizer_hf()
        if self.fd_config.structured_outputs_config.reasoning_parser:
            reasoning_parser_obj = ReasoningParserManager.get_reasoning_parser(
                self.fd_config.structured_outputs_config.reasoning_parser
            )
            self.reasoning_parser = reasoning_parser_obj(self.hf_tokenizer)

    def _create_processor(self):
        """
        Create a specific logits processor instance.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def _json_processor(self, schemata, enable_thinking=False):
        """
        Process JSON schemata.

        Args:
            schemata (str): The schemata string.
            enable_thinking (bool): Whether to enable thinking mode.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def _regex_processor(self, schemata, enable_thinking=False):
        """
        Process regular expression schemata.

        Args:
            schemata (str): The schemata string.
            enable_thinking (bool): Whether to enable thinking mode.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def _grammar_processor(self, schemata, enable_thinking=False):
        """
        Process grammar schemata.

        Args:
            schemata (str): The schemata string.
            enable_thinking (bool): Whether to enable thinking mode.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def _structural_tag_processor(self, schemata, enable_thinking=False):
        """
        Process structural tag schemata.

        Args:
            schemata (str): The schemata string.
            enable_thinking (bool): Whether to enable thinking mode.

        Raises:
            NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

    def _unsupported_processor_type(self, key_type, schemata, enable_thinking=False):
        """
        Process unsupported type.

        Args:
            key_type (str): The key type string.
            schemata (str): The schemata string.
            enable_thinking (bool): Whether to enable thinking mode.
        """
        raise Exception(f"Unsupported processor type {key_type}.")

    def get_reasoning_parser(self):
        """
        Get reasoning parser object.
        Returns:
            ReasoningParser: Reasoning parser object or None
        """
        return self.reasoning_parser

    def _init_logits_processor(
        self,
        schemata_key: tuple[str, str],
        enable_thinking: bool = False,
    ) -> LogitsProcessorBase:
        """
        init logits processor by type and schemata.

        Args:
            schemata_key (tuple[str, str]): Tuple containing processor type and schema string
            enable_thinking (bool): Whether to enable thinking step

        Returns:
            LogitsProcessorBase: Initialized logits processor instance

        Raises:
            ValueError: If processor type is not supported
        """
        key_type, schemata = schemata_key
        if key_type == "json":
            return self._json_processor(schemata, enable_thinking)
        elif key_type == "regex":
            return self._regex_processor(schemata, enable_thinking)
        elif key_type == "grammar":
            return self._grammar_processor(schemata, enable_thinking)
        elif key_type == "structural_tag":
            return self._structural_tag_processor(schemata, enable_thinking)
        else:
            llm_logger.error(f"Unsupported processor type {key_type}.")
            return None

    def get_logits_processor(
        self,
        schemata_key: tuple[str, str],
        enable_thinking: bool = False,
    ) -> Future[LogitsProcessorBase]:
        """
        get logits processor by key from cache or create new one.

        Args:
            schemata_key (tuple[str, str]): Tuple containing processor type and schema string

        Returns:
            tuple[LogitsProcessorBase, bool]: Tuple containing:
                - LogitsProcessorBase: The logits processor instance
                - bool: True if processor was from cache, False if newly created
        """
        value = self.executor.submit(self._init_logits_processor, schemata_key, enable_thinking)
        return value

    def _get_tokenizer_hf(self):
        """
        Initialize and return a HuggingFace tokenizer instance.

        This method handles special cases for Ernie models and falls back to standard
        AutoTokenizer for other models. It also ensures fast tokenizer is used when possible.

        Returns:
            Tokenizer: Initialized HuggingFace tokenizer instance

        Raises:
            Exception: If tokenizer initialization fails
        """
        try:
            architectures = self.fd_config.model_config.architectures
            if not ErnieArchitectures.contains_ernie_arch(architectures):
                from transformers import AutoTokenizer, PreTrainedTokenizerFast

                tokenizer = AutoTokenizer.from_pretrained(
                    self.fd_config.model_config.model,
                    use_fast=True,
                )

                if not isinstance(tokenizer, PreTrainedTokenizerFast):
                    tokenizer = PreTrainedTokenizerFast(__slow_tokenizer=tokenizer)
            else:
                from fastdeploy.model_executor.guided_decoding.ernie_tokenizer import (
                    Ernie4_5Tokenizer,
                )

                vocab_file_names = [
                    "tokenizer.model",
                    "spm.model",
                    "ernie_token_100k.model",
                ]
                for i in range(len(vocab_file_names)):
                    if os.path.exists(
                        os.path.join(
                            self.fd_config.model_config.model,
                            vocab_file_names[i],
                        )
                    ):
                        Ernie4_5Tokenizer.vocab_files_names["vocab_file"] = vocab_file_names[i]
                        break

                tokenizer = Ernie4_5Tokenizer.from_pretrained(self.fd_config.model_config.model)

            return tokenizer
        except Exception as e:
            raise Exception(f"Fail to initialize hf tokenizer: {e}, {str(traceback.format_exc())}")


class BaseChecker:
    """
    Abstract base class for schema checkers.

    This class defines the interface for validating and formatting schemas
    before they are used by logits processors. Subclasses should implement
    schema-specific validation and formatting logic.

    Attributes:
        None (all state should be managed by subclasses)
    """

    def __init__(self):
        pass

    def schema_format(self, request: Request):
        """
        format schema to backend specific format.
        Args:
            request (Request): request object.

        Returns:
            request (Request): request object with formatted schema.
        """
        raise NotImplementedError
