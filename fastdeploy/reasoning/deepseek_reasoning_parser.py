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

from collections.abc import Sequence
from typing import Optional, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(["deepseek", "deepseek-r1", "deepseek-v3.1", "deepseek-v3-0324"])
class DeepSeekReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek models (V3.1, V3-0324, R1).
    Extracts reasoning content and response content from model output.
    """

    def __init__(self, tokenizer, model_name=None):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser " "constructor during construction."
            )

        # Get special token IDs
        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)

        if self.think_end_token_id is None:
            raise RuntimeError("DeepSeek reasoning parser could not locate think end " "tokens in the tokenizer!")

        # Detect model version to determine if reasoning toggle is supported
        self.model_name = model_name or ""
        self.supports_reasoning_toggle = "v3.1" in self.model_name.lower()

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Check if reasoning content has ended (check for </think> token)."""
        return self.think_end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract content token IDs after </think>."""
        if self.think_end_token_id not in input_ids:
            return input_ids

        # Find position of </think>
        end_index = input_ids.index(self.think_end_token_id)
        # Return all token IDs after the end token
        return input_ids[end_index + 1 :]

    def detect_output_stage(self, prompt_token_ids: Sequence[int]) -> str:
        """Detect output stage based on prompt token IDs."""
        # Check if prompt contains <think> start token
        if self.think_start_token_id is not None and self.think_start_token_id in prompt_token_ids:
            # Check if thinking stage has ended
            if self.think_end_token_id is not None and self.think_end_token_id in prompt_token_ids:
                # Thinking ended, enter content stage
                return "CONTENT_STAGE"
            else:
                # Still in thinking stage
                return "REASONING_STAGE"
        else:
            # No thinking tokens, possibly reasoning toggle is off
            # Default to content stage
            return "CONTENT_STAGE"

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest, output_stage: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content and response content from complete model output (non-streaming).
        Supports formats: <think>abc</think>xyz, abc</think>xyz, or xyz.
        """
        # Check for start token
        if self.think_start_token in model_output:
            # Standard format: <think>content</think>answer
            # Remove start token
            model_output_parts = model_output.partition(self.think_start_token)
            model_output = model_output_parts[2] if model_output_parts[1] else model_output_parts[0]

            # Check for end token
            if self.think_end_token not in model_output:
                # Only start token, no end token: treat entire content as reasoning
                return model_output, None

            # Extract reasoning and response content
            reasoning_content, _, content = model_output.partition(self.think_end_token)

            # Strip whitespace but preserve newlines
            final_content = content.strip() if content.strip() else None
            return reasoning_content, final_content

        # Check for end token (but no start token)
        if self.think_end_token in model_output:
            # Missing start token format: content</think>answer
            parts = model_output.split(self.think_end_token, 1)

            if len(parts) == 2:
                reasoning_content = parts[0].strip()
                final_content = parts[1].strip() if parts[1].strip() else None
                return reasoning_content, final_content

        # No thinking tokens mode
        if output_stage == "REASONING_STAGE":
            # If detected as reasoning stage but no end token, treat as protocol error
            # Return entire output as reasoning_content
            return model_output, None
        else:
            # Reasoning toggle off or in content stage: return entire output as content
            return None, model_output

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        output_stage: Optional[str] = None,
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from incremental messages (streaming).
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        """
        # Ignore single </think> token
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None

        # If delta contains </think>
        if self.think_end_token_id in delta_token_ids:
            # If delta contains both <think> and </think>
            if self.think_start_token_id and self.think_start_token_id in delta_token_ids:
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                if start_index != -1 and end_index != -1:
                    reasoning_content = delta_text[start_index + len(self.think_start_token) : end_index]
                    content = delta_text[end_index + len(self.think_end_token) :]
                    return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            # If </think> in delta but <think> in previous
            else:
                end_index = delta_text.find(self.think_end_token)
                if end_index != -1:
                    reasoning_content = delta_text[:end_index]
                    content = delta_text[end_index + len(self.think_end_token) :]
                    # Strip whitespace but preserve newlines
                    content = content if content.strip() else None
                    return DeltaMessage(reasoning_content=reasoning_content, content=content)

        # If </think> in previous, already in content stage
        if self.think_end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        # If <think> in previous, still in thinking stage
        if self.think_start_token_id and self.think_start_token_id in previous_token_ids:
            return DeltaMessage(reasoning_content=delta_text)

        # If <think> in delta
        if self.think_start_token_id and self.think_start_token_id in delta_token_ids:
            start_index = delta_text.find(self.think_start_token)
            if start_index != -1:
                reasoning_content = delta_text[start_index + len(self.think_start_token) :]
                return DeltaMessage(reasoning_content=reasoning_content, content=None)

        # Default: determine based on output_stage
        # If no tokens seen, possibly reasoning toggle is off
        if output_stage == "CONTENT_STAGE":
            # In content stage, return delta as content
            return DeltaMessage(content=delta_text)
        else:
            # In thinking stage or unknown, return delta as reasoning_content
            # Will be handled correctly if </think> appears later
            return DeltaMessage(reasoning_content=delta_text)
