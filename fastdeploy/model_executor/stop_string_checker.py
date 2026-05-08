# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
Stop string checker for string-level stop sequence matching.

This module provides string-level stop sequence checking which is more robust
than token-level matching as it avoids BPE boundary issues.

Reference:
- vLLM: vllm/v1/engine/detokenizer.py - check_stop_strings
- SGLang: python/sglang/srt/sampling/sampling_params.py - stop strings handling
"""

from typing import List, Optional, Tuple


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: List[str],
    include_in_output: bool,
) -> Tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence output text accordingly.

    Args:
        output_text: The full output text generated so far.
        new_char_count: Number of new characters added since last check.
        stop: List of stop strings to check against.
        include_in_output: Whether to include the stop string in the output.

    Returns:
        Tuple of (stop_string, truncate_offset) if matched, or None.
        truncate_offset is the position to truncate output_text to, or -1 for no truncation.
    """
    if not stop or not output_text:
        return None

    # Only check the newly added text for efficiency
    # But we also need to check for stop strings that might span the boundary
    # Get the text to check (including a buffer for stop strings that might cross boundaries)
    check_start = max(0, len(output_text) - new_char_count - max(len(s) for s in stop))
    text_to_check = output_text[check_start:]

    for stop_str in stop:
        stop_pos = text_to_check.rfind(stop_str)
        if stop_pos != -1:
            # Calculate the actual position in output_text
            actual_pos = check_start + stop_pos

            if include_in_output:
                # Include the stop string in output, truncate after it
                truncate_to = actual_pos + len(stop_str)
                return stop_str, truncate_to
            else:
                # Exclude the stop string from output
                return stop_str, actual_pos

    return None


class StopStringChecker:
    """Handles string-level stop sequence checking during generation.

    This class provides incremental stop string checking by maintaining
    the output text and checking for stop string matches after each token generation.

    Attributes:
        stop: List of stop strings.
        include_stop_str_in_output: Whether to include stop strings in output.
        min_tokens: Minimum number of tokens before stop checking is enabled.
    """

    def __init__(
        self,
        stop: Optional[List[str]] = None,
        include_stop_str_in_output: bool = False,
        min_tokens: int = 1,
    ):
        """Initialize the StopStringChecker.

        Args:
            stop: List of stop strings. Can be a single string or list of strings.
            include_stop_str_in_output: Whether to include stop strings in output.
            min_tokens: Minimum number of tokens before stop checking is enabled.
        """
        # Normalize stop to a list
        if stop is None:
            self.stop: List[str] = []
        elif isinstance(stop, str):
            self.stop = [stop]
        else:
            self.stop = list(stop)

        self.include_stop_str_in_output = include_stop_str_in_output
        self.min_tokens = min_tokens

        # Internal state
        self._output_text: str = ""
        self._token_count: int = 0

    def reset(self):
        """Reset the internal state for a new generation."""
        self._output_text = ""
        self._token_count = 0

    def update(
        self,
        new_token_ids: List[int],
        tokenizer,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> Tuple[Optional[str], Optional[int]]:
        """Update with new tokens and check for stop string matches.

        Args:
            new_token_ids: List of newly generated token IDs.
            tokenizer: The tokenizer to use for decoding.
            skip_special_tokens: Whether to skip special tokens during decoding.
            spaces_between_special_tokens: Whether to add spaces between special tokens.

        Returns:
            Tuple of (matched_stop_string, truncate_offset) if stop matched, else (None, None).
            truncate_offset is the position to truncate output_text to, or None for no truncation.
        """
        if not new_token_ids:
            return None, None

        # Increment token count
        self._token_count += len(new_token_ids)

        # Skip stop string check if we haven't generated enough tokens
        if self._token_count <= self.min_tokens:
            # Still need to decode the tokens for future checks
            if tokenizer is not None and self.stop:
                try:
                    new_text = tokenizer.decode(
                        new_token_ids,
                        skip_special_tokens=skip_special_tokens,
                        spaces_between_special_tokens=spaces_between_special_tokens,
                    )
                    self._output_text += new_text
                except Exception:
                    pass
            return None, None

        # If no stop strings configured, just update text and return
        if not self.stop:
            if tokenizer is not None:
                try:
                    new_text = tokenizer.decode(
                        new_token_ids,
                        skip_special_tokens=skip_special_tokens,
                        spaces_between_special_tokens=spaces_between_special_tokens,
                    )
                    self._output_text += new_text
                except Exception:
                    pass
            return None, None

        # Decode new tokens
        new_char_count = 0
        if tokenizer is not None:
            try:
                new_text = tokenizer.decode(
                    new_token_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )
                self._output_text += new_text
                new_char_count = len(new_text)
            except Exception:
                return None, None

        # Check for stop string matches
        result = check_stop_strings(
            output_text=self._output_text,
            new_char_count=new_char_count,
            stop=self.stop,
            include_in_output=self.include_stop_str_in_output,
        )

        if result is not None:
            stop_string, truncate_to = result
            if truncate_to != -1:
                self._output_text = self._output_text[:truncate_to]
            return stop_string, truncate_to

        return None, None

    @property
    def output_text(self) -> str:
        """Get the current output text."""
        return self._output_text

    @property
    def token_count(self) -> int:
        """Get the current token count."""
        return self._token_count
