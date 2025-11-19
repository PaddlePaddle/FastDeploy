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

import json
import re
import traceback
from typing import Any, List, Optional

import paddle
import torch

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.model_executor.guided_decoding import (
    BackendBase,
    BaseChecker,
    LogitsProcessorBase,
)
from fastdeploy.utils import llm_logger

try:
    from xgrammar import (
        CompiledGrammar,
        Grammar,
        GrammarCompiler,
        GrammarMatcher,
        StructuralTagItem,
        TokenizerInfo,
        allocate_token_bitmask,
        apply_token_bitmask_inplace,
    )
except Exception as e:
    raise Exception(f"import XGrammar failed, please check your environment:\n\t {e}")


class XGrammarProcessor(LogitsProcessorBase):
    """
    XGrammar-specific implementation of LogitsProcessorBase.

    This processor enforces grammar constraints during token generation using XGrammar.
    It manages the grammar matching state and applies token masks to logits.

    Attributes:
        max_rollback_tokens (int): Maximum number of tokens to rollback on mismatch
        vocab_size (int): Size of the vocabulary
        batch_size (int): Batch size for processing
        compiled_grammar (CompiledGrammar): Compiled grammar rules
        terminate_without_stop_token (bool): Whether to terminate without stop token
        override_stop_tokens (Optional[List[int]]): Custom stop tokens
        matcher (GrammarMatcher): Grammar matching engine
    """

    def __init__(
        self,
        compiled_grammar: CompiledGrammar,
        terminate_without_stop_token: bool = False,
        override_stop_tokens: Optional[List[int]] = None,
        vocab_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_thinking: bool = False,
    ):
        super().__init__(enable_reasoning=enable_thinking)
        self.max_rollback_tokens = 200
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.compiled_grammar = compiled_grammar
        self.terminate_without_stop_token = terminate_without_stop_token
        self.override_stop_tokens = override_stop_tokens

        self.matcher = GrammarMatcher(
            compiled_grammar=compiled_grammar,
            max_rollback_tokens=self.max_rollback_tokens,
            terminate_without_stop_token=terminate_without_stop_token,
            override_stop_tokens=override_stop_tokens,
        )
        # when matcher accept eos_token_id, is_terminated = True
        self.is_terminated: bool = False

    def allocate_token_bitmask(self) -> torch.Tensor:
        """
        Allocate a token bitmask tensor for grammar constraints.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, vocab_size) initialized to 0
        """
        return allocate_token_bitmask(self.batch_size, self.vocab_size)

    def fill_token_bitmask(self, token_bitmask: torch.Tensor, idx: int) -> None:
        """
        Fill the token bitmask with allowed tokens for the given index.

        Args:
            token_bitmask (torch.Tensor): The token bitmask tensor to fill
            idx (int): The batch index to fill the mask for

        Returns:
            None: Modifies the token_bitmask in-place
        """
        self.matcher.fill_next_token_bitmask(token_bitmask, idx)

    def reset(self) -> None:
        """
        Reset the grammar matcher state to initial conditions.

        Returns:
            None: No return value
        """
        self.matcher.reset()

    def accept_token(self, token: int) -> None:
        """
        Validate and accept a generated token against the grammar constraints.
        when accept eos_token, is_terminated = True

        Args:
            token (int): The token ID to validate

        """
        if self.is_terminated or self.matcher.is_terminated():
            self.is_terminated = True
            return False
        if not self.matcher.accept_token(token):
            self.matcher.reset()
            return False
        if self.matcher.is_terminated():
            self.is_terminated = True
        return True

    def copy(self) -> "XGrammarProcessor":
        """
        Create a deep copy of this processor instance.

        Returns:
            XGrammarProcessor: A new processor instance with identical state
        """
        return XGrammarProcessor(
            compiled_grammar=self.compiled_grammar,
            terminate_without_stop_token=self.terminate_without_stop_token,
            override_stop_tokens=self.override_stop_tokens,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
        )


class XGrammarBackend(BackendBase):
    """
    XGrammar-specific implementation of BackendBase.

    This backend handles compilation of various schema types (JSON, regex, grammar)
    into XGrammar processors. It manages the grammar compiler and tokenizer info.

    Attributes:
        vocab_size (int): Size of the vocabulary from config
        batch_size (int): Maximum batch size from config
        any_whitespace (bool): Whether to allow any whitespace in JSON
        grammar_compiler (GrammarCompiler): Grammar compilation engine
    """

    def __init__(
        self,
        fd_config: FDConfig,
        **kwargs,
    ):
        super().__init__(fd_config=fd_config)
        self.vocab_size = fd_config.model_config.vocab_size
        self.batch_size = fd_config.scheduler_config.max_num_seqs

        self.any_whitespace = not fd_config.structured_outputs_config.disable_any_whitespace

        try:
            tokenizer_info = TokenizerInfo.from_huggingface(self.hf_tokenizer, vocab_size=self.vocab_size)
            llm_logger.info(f"xgrammar_backend.py tokenizer_info={tokenizer_info.dump_metadata()}")
            # Read configuration values, fallback to defaults if not set
            xgrammar_cfg = getattr(fd_config, "xgrammar_config", {})
            max_threads = getattr(xgrammar_cfg, "max_threads", 8)
            cache_enabled = getattr(xgrammar_cfg, "cache_enabled", True)
            cache_limit_bytes = getattr(xgrammar_cfg, "cache_limit_bytes", 4 * 1024 * 1024)
            self.grammar_compiler = GrammarCompiler(
                tokenizer_info=tokenizer_info,
                max_threads=max_threads,
                cache_enabled=cache_enabled,
                cache_limit_bytes=cache_limit_bytes,
            )
        except Exception as e:
            raise Exception(f"Failed to load XGrammar tokenizer: {e}")

    def _create_processor(
        self,
        compiled_grammar: CompiledGrammar,
        terminate_without_stop_token: bool = False,
        override_stop_tokens: Optional[List[int]] = None,
        enable_thinking: bool = False,
    ) -> XGrammarProcessor:
        """
        Create a logits processor instance for the given compiled grammar.

        Args:
            compiled_grammar (CompiledGrammar): Compiled grammar rules
            terminate_without_stop_token (bool): Whether to terminate without stop token
            override_stop_tokens (Optional[List[int]]): Custom stop tokens to override defaults
            enable_thinking (bool): Whether to enable thinking mode

        Returns:
            XGrammarProcessor: Configured grammar processor instance
        """
        return XGrammarProcessor(
            compiled_grammar=compiled_grammar,
            terminate_without_stop_token=terminate_without_stop_token,
            override_stop_tokens=override_stop_tokens,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            enable_thinking=enable_thinking,
        )

    def _json_processor(self, schemata: str, enable_thinking: bool = False) -> Optional[XGrammarProcessor]:
        """
        Compile JSON schema into a grammar processor.

        Args:
            schemata (str): JSON schema string to compile
            enable_thinking (bool): Whether to enable thinking mode

        Returns:
            Optional[XGrammarProcessor]: Configured processor if successful, None on failure
        """
        try:
            compiled_grammar = self.grammar_compiler.compile_json_schema(schemata, any_whitespace=self.any_whitespace)
        except Exception as e:
            llm_logger.error(f"Failed to compile json schema: {e}, {str(traceback.format_exc())}")
            return None
        return self._create_processor(compiled_grammar, enable_thinking=enable_thinking)

    def _regex_processor(self, schemata: str, enable_thinking: bool = False) -> Optional[XGrammarProcessor]:
        """
        Compile regex pattern into a grammar processor.

        Args:
            schemata (str): Regex pattern string to compile
            enable_thinking (bool): Whether to enable thinking mode

        Returns:
            Optional[XGrammarProcessor]: Configured processor if successful, None on failure
        """
        try:
            compiled_grammar = self.grammar_compiler.compile_regex(schemata)
        except Exception as e:
            llm_logger.error(f"Failed to compile regex schema: {e}, {str(traceback.format_exc())}")
            return None
        return self._create_processor(compiled_grammar, enable_thinking=enable_thinking)

    def _grammar_processor(self, schemata: str, enable_thinking: bool = False) -> Optional[XGrammarProcessor]:
        """
        Compile grammar (EBNF) into a grammar processor.

        Args:
            schemata (str): Grammar string in EBNF format
            enable_thinking (bool): Whether to enable thinking mode

        Returns:
            Optional[XGrammarProcessor]: Configured processor if successful, None on failure
        """
        try:
            compiled_grammar = self.grammar_compiler.compile_grammar(schemata)
        except Exception as e:
            llm_logger.error(f"Failed to compile ebnf schema: {e}, {str(traceback.format_exc())}")
            return None
        return self._create_processor(compiled_grammar, enable_thinking=enable_thinking)

    def _structural_tag_processor(self, schemata: str, enable_thinking: bool = False) -> Optional[XGrammarProcessor]:
        """
        Compile structural tags into a grammar processor.

        Args:
            schemata (str): JSON string containing structural tag definitions

        Returns:
            Optional[XGrammarProcessor]: Configured processor if successful, None on failure
        """
        try:
            structural_tag = json.loads(schemata)
            tags = [
                StructuralTagItem(
                    begin=structure["begin"],
                    schema=json.dumps(structure["schema"]),
                    end=structure["end"],
                )
                for structure in structural_tag["structures"]
            ]

            compiled_grammar = self.grammar_compiler.compile_structural_tag(tags, structural_tag["triggers"])
        except Exception as e:
            llm_logger.error(f"Failed to compile structural tags schema: {e}, {str(traceback.format_exc())}")
            return None
        return self._create_processor(compiled_grammar, enable_thinking=enable_thinking)


class XGrammarChecker(BaseChecker):
    """
    XGrammar-specific implementation of BaseChecker.

    This validator checks and formats various schema types (JSON, regex, grammar)
    for compatibility with XGrammar before processing.

    Attributes:
        any_whitespace (bool): Whether to allow any whitespace in JSON
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.any_whitespace = not kwargs.get("disable_any_whitespace", True)

    def _unsupported_json_schema(self, schema: dict[str, Any]) -> bool:
        """
        Check if JSON schema contains unsupported features.

        Args:
            schema (dict[str, Any]): JSON schema to validate

        Returns:
            bool: True if schema contains unsupported features, False otherwise
        """

        def check_object(obj: dict[str, Any]) -> bool:
            if not isinstance(obj, dict):
                return False

            if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
                return True

            if obj.get("type") == "array" and any(
                key in obj
                for key in (
                    "uniqueItems",
                    "contains",
                    "minContains",
                    "maxContains",
                )
            ):
                return True

            if obj.get("type") == "string" and "format" in obj:
                return True

            if obj.get("type") == "object" and any(
                key in obj
                for key in (
                    "minProperties",
                    "maxProperties",
                    "propertyNames",
                    "patternProperties",
                )
            ):
                return True

            for value in obj.values():
                if isinstance(value, dict):
                    if check_object(value):
                        return True
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and check_object(item):
                            return True
            return False

        return check_object(schema)

    def schema_format(self, request: Request):
        """
        format schema to backend specific format.
        """
        if request.guided_json:
            try:
                if not isinstance(request.guided_json, str):
                    guided_json = json.dumps(request.guided_json)
                else:
                    guided_json = request.guided_json

                Grammar.from_json_schema(guided_json, any_whitespace=self.any_whitespace)
            except RuntimeError as e:
                err_msg = f"Invalid JSON format: {guided_json}, error message: {e!s}"
                return request, err_msg

            if self._unsupported_json_schema(guided_json):
                err_msg = f"unsupported JSON schema: {guided_json}"
                return request, err_msg

            request.guided_json = guided_json
            return request, None
        elif request.guided_grammar:
            # TODO: XGrammar only supports GBNF grammars, convert Lark to GBNF
            guided_grammar = request.guided_grammar
            try:
                Grammar.from_ebnf(guided_grammar)
            except RuntimeError as e:
                err_msg = f"Invalid grammar format: {guided_grammar}, error message: {e!s}"
                return request, err_msg
            request.guided_grammar = guided_grammar
            return request, None
        elif request.guided_json_object:
            request.guided_json = '{"type": "object"}'
            return request, None
        elif request.guided_choice:
            try:
                escaped_choices = (re.sub(r'(["\\])', r"\\\1", c) for c in request.guided_choice)
                guided_choice = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)

                Grammar.from_ebnf(guided_choice)
            except RuntimeError as e:
                err_msg = f"Invalid choice format: {guided_choice}, error message: {e!s}"
                return request, err_msg

            request.guided_grammar = guided_choice
            return request, None
        elif request.structural_tag:
            try:
                structural_tag = json.loads(request.structural_tag)
                tags = [
                    StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in structural_tag["structures"]
                ]
                Grammar.from_structural_tag(tags, structural_tag["triggers"])
            except RuntimeError as e:
                err_msg = f"Invalid structural_tag format: {structural_tag}, error message: {e!s}"
                return request, err_msg
            return request, None
        else:
            # regex is not format
            return request, None


def apply_token_mask(
    logits: paddle.Tensor,
    token_bitmask: torch.Tensor,
    indices: Optional[List[int]] = None,
    is_cuda_platform: bool = True,
) -> paddle.Tensor:
    """
    Apply the token mask to the logits, modifying probabilities of invalid tokens.

    Args:
        logits (paddle.Tensor): The logits tensor to modify
        token_bitmask (torch.Tensor): The token bitmask indicating allowed tokens
        indices (Optional[List[int]]): Optional list of batch indices to apply mask to

    Returns:
        paddle.Tensor: The modified logits tensor
    """
    skip_out_indices = len(indices) == logits.shape[0]
    if is_cuda_platform:
        dlpack = paddle.utils.dlpack.to_dlpack(logits)
        t_logits = torch.from_dlpack(dlpack)
        apply_token_bitmask_inplace(
            logits=t_logits,
            bitmask=token_bitmask.to(t_logits.device, non_blocking=True),
            indices=indices if not skip_out_indices else None,
        )
        return logits
    else:
        origin_place = logits.place
        origin_dtype = logits.dtype
        logits = torch.from_numpy(logits.numpy())

        logits = logits.float()  # cpu
        apply_token_bitmask_inplace(
            logits=logits,
            bitmask=token_bitmask.to(logits.device, non_blocking=True),
            indices=indices if not skip_out_indices else None,
        )

        return paddle.to_tensor(
            logits.numpy(),
            dtype=origin_dtype,
            place=origin_place,
        )
