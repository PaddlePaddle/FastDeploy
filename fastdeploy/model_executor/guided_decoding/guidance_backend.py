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

import copy
import json
import traceback
from typing import Any, Optional, Tuple, Union

import llguidance
import llguidance.hf
import llguidance.torch
import torch

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.envs import FD_GUIDANCE_DISABLE_ADDITIONAL, FD_LLGUIDANCE_LOG_LEVEL
from fastdeploy.model_executor.guided_decoding import (
    BackendBase,
    BaseChecker,
    LogitsProcessorBase,
)
from fastdeploy.utils import llm_logger


class LLGuidanceProcessor(LogitsProcessorBase):
    """
    LLGuidance-specific implementation of LogitsProcessorBase.

    This processor enforces grammar constraints during token generation using llguidance.
    It manages the grammar matching state and applies token masks to logits.
    """

    def __init__(
        self,
        ll_matcher: llguidance.LLMatcher,
        ll_tokenizer: llguidance.LLTokenizer,
        serialized_grammar: str,
        vocab_size: int,
        batch_size: int,
        enable_thinking: bool = False,
    ):
        super().__init__(enable_reasoning=enable_thinking)
        self.matcher = ll_matcher
        self.ll_tokenizer = ll_tokenizer
        self.serialized_grammar = serialized_grammar
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.is_terminated: bool = False
        self._printed_error: bool = False

    def _check_error(self):
        """Checks for and logs any errors from the LLMatcher."""
        if not self._printed_error:
            err = self.matcher.get_error()
            if err:
                self._printed_error = True
                llm_logger.warning(f"LLGuidance Matcher error: {err}")

    def allocate_token_bitmask(self) -> torch.Tensor:
        """
        Allocate a token bitmask tensor for grammar constraints.
        """
        return llguidance.torch.allocate_token_bitmask(self.batch_size, self.vocab_size)

    def fill_token_bitmask(self, token_bitmask: torch.Tensor, idx: int) -> None:
        """
        Fill the token bitmask with allowed tokens for the given index.
        This will automatically provide an EOS mask if the matcher is stopped.
        """
        llguidance.torch.fill_next_token_bitmask(self.matcher, token_bitmask, idx)
        self._check_error()

    def reset(self) -> None:
        """
        Reset the grammar matcher state to initial conditions.
        """
        self.matcher.reset()
        self.is_terminated = False
        self._printed_error = False
        self._check_error()

    def accept_token(self, token: int) -> bool:
        """
        Validate and accept a generated token against the grammar constraints.
        Returns True if the token is accepted, False otherwise.
        """
        if self.is_terminated:
            return False
        if self.ll_tokenizer.eos_token == token:
            self.is_terminated = True
            return True

        result = self.matcher.consume_tokens([token])
        self._check_error()

        return result


class LLGuidanceBackend(BackendBase):
    """
    LLGuidance-specific implementation of BackendBase.

    This backend handles the compilation of various schema types (JSON, regex, etc.)
    into LLGuidance processors.
    """

    def __init__(self, fd_config: FDConfig, **kwargs):
        super().__init__(fd_config=fd_config)
        self.vocab_size = fd_config.model_config.vocab_size
        self.batch_size = fd_config.scheduler_config.max_num_seqs
        self.any_whitespace = not fd_config.structured_outputs_config.disable_any_whitespace

        llm_logger.info(f"LLGuidanceBackend vocab_size={self.vocab_size} batch_size={self.batch_size}")
        try:
            self.ll_tokenizer = llguidance.hf.from_tokenizer(self.hf_tokenizer, self.vocab_size)
        except Exception as e:
            import traceback

            raise RuntimeError(
                f"Failed to initialize llguidance tokenizer from HuggingFace tokenizer: {e} {traceback.format_exc()}"
            )

    def _create_processor(
        self,
        compiled_grammar: str,
        enable_thinking: bool = False,
    ) -> Optional[LLGuidanceProcessor]:
        """
        Create a logits processor instance for the given grammar schemata.
        """
        try:

            ll_matcher = llguidance.LLMatcher(
                self.ll_tokenizer,
                compiled_grammar,
                log_level=FD_LLGUIDANCE_LOG_LEVEL,
            )

            return LLGuidanceProcessor(
                ll_matcher=ll_matcher,
                ll_tokenizer=self.ll_tokenizer,
                serialized_grammar=compiled_grammar,
                vocab_size=self.vocab_size,
                batch_size=self.batch_size,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            llm_logger.error(f"Failed to create llguidance processor: {e}, {str(traceback.format_exc())}")
            return None

    def _json_processor(self, compiled_grammar: str, enable_thinking: bool = False) -> Optional[LLGuidanceProcessor]:
        return self._create_processor(compiled_grammar, enable_thinking)

    def _regex_processor(self, compiled_grammar: str, enable_thinking: bool = False) -> Optional[LLGuidanceProcessor]:
        return self._create_processor(compiled_grammar, enable_thinking)

    def _grammar_processor(
        self, compiled_grammar: str, enable_thinking: bool = False
    ) -> Optional[LLGuidanceProcessor]:
        return self._create_processor(compiled_grammar, enable_thinking)

    def _structural_tag_processor(
        self, compiled_grammar: str, enable_thinking: bool = False
    ) -> Optional[LLGuidanceProcessor]:
        return self._create_processor(compiled_grammar, enable_thinking)


def _walk_json_for_additional_properties(data: object):
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if "additionalProperties" not in data and ("properties" in data or "patternProperties" in data):
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def process_for_additional_properties(guide_json: Union[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


class LLGuidanceChecker(BaseChecker):
    """
    LLGuidance-specific implementation of BaseChecker.

    This checker validates various schema types for compatibility with the
    llguidance library before processing.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Although the backend handles serialization, we can perform a quick
        # static check here without a full tokenizer.
        self.any_whitespace = not kwargs.get("disable_any_whitespace", False)
        self.disable_additional_properties = FD_GUIDANCE_DISABLE_ADDITIONAL
        """If `True`, the `guidance` backend will not use `additionalProperties`
        in the JSON schema. This is only supported for the `guidance` backend and
        is used to better align its behaviour with `outlines` and `xgrammar`."""

    def serialize_guidance_grammar(self, request: Request):
        def _process_schema(
            grammar_spec: Union[str, dict[str, Any]],
        ) -> str:
            if self.disable_additional_properties:
                grammar_spec = process_for_additional_properties(grammar_spec)
            return llguidance.LLMatcher.grammar_from_json_schema(
                grammar_spec,
                defaults={
                    "whitespace_flexible": self.any_whitespace,
                },
            )

        if request.guided_json:
            if isinstance(request.guided_json, dict):
                guided_json = json.dumps(request.guided_json)
            else:
                guided_json = request.guided_json
            return _process_schema(guided_json)
        elif request.guided_json_object:
            return llguidance.LLMatcher.grammar_from_json_schema(
                '{"type": "object"}',
                defaults={
                    "whitespace_flexible": self.any_whitespace,
                },
            )

        if request.structural_tag:
            if isinstance(request.structural_tag, str):
                s_tag = json.loads(request.structural_tag)
            else:
                s_tag = request.structural_tag
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise ValueError(f"Trigger {begin} not found in triggers {triggers}")
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        grammar=_process_schema(s["schema"]),
                        end=s["end"],
                    )
                )
            if not tags:
                raise ValueError("No structural tags found in the grammar spec.")
            return llguidance.StructTag.to_grammar(tags)

        if request.guided_regex:
            tp = "regex"
            grammar_spec = request.guided_regex
        elif request.guided_choice:
            tp = "choice"
            grammar_spec = request.guided_choice
        elif request.guided_grammar:
            tp = "grammar"
            grammar_spec = request.guided_grammar
        else:
            llm_logger.error("Validation should have already occurred. " "Please file an issue.")
            raise ValueError("grammar is not of valid supported types. ")
        return llguidance.grammar_from(tp, grammar_spec)

    def schema_format(self, request: Request) -> Tuple[Request, Optional[str]]:
        """
        Validates and formats the schema for the LLGuidance backend.
        """
        try:
            guidance_grm = self.serialize_guidance_grammar(request)
            err = llguidance.LLMatcher.validate_grammar(guidance_grm, None)
            if err:
                raise ValueError(f"Grammar error: {err}")
            else:
                llm_logger.info(f"valid schema_format {guidance_grm} {request}")
            if request.guided_regex:
                request.guided_regex = guidance_grm
            elif request.guided_choice:
                request.guided_grammar = guidance_grm
                request.guided_choice = None
            elif request.guided_grammar:
                request.guided_grammar = guidance_grm
            elif request.guided_json:
                request.guided_json = guidance_grm

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            err_msg = f"Invalid format for guided decoding: {e!s} request={request}"
            return request, err_msg

        except Exception as e:
            err_msg = f"An unexpected error occurred during schema validation: {e!s}"
            return request, err_msg

        return request, None
