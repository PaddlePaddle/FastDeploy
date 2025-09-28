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
from collections.abc import Sequence
from typing import Union

from fastdeploy.entrypoints.chat_utils import random_tool_call_id
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from fastdeploy.utils import data_processor_logger


@ToolParserManager.register_module("ernie_x1")
class ErnieX1ToolParser(ToolParser):
    """
    Tool parser for Ernie model version 4.5.1.
    This parser handles tool calls with newline formats.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []  # map what has been streamed for each tool so far to a list
        self.buffer: str = ""  # buffer for accumulating unprocessed streaming content
        self.bracket_counts: dict = {"total_l": 0, "total_r": 0}  # track bracket counts in streamed deltas
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError("Ernie x1 Tool parser could not locate tool call start/end tokens in the tokenizer!")

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolCallParser constructor during construction."
            )

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        Supports XML-style formats with newlines:
        - XML format: <think>\n...\n</think>\n\n\n<tool_call>\n{...}\n</tool_call>\n...

        Handles boundary cases:
        1. Only name and partial arguments: {"name": "get_weather", "arguments": {"location": "北京"
        2. Only partial name: {"name": "get_we
        3. Only name and arguments field without content: {"name": "get_weather", "argume
        """

        try:
            if self.tool_call_start_token not in model_output:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)
            function_call_tuples = self.tool_call_regex.findall(model_output)

            raw_function_calls = [json.loads(match[0] if match[0] else match[1]) for match in function_call_tuples]

            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        # function call args are JSON but as a string
                        arguments=json.dumps(function_call["arguments"], ensure_ascii=False),
                    ),
                )
                for function_call in raw_function_calls
            ]
            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content="")
        except Exception:
            data_processor_logger.error("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: dict,
    ) -> Union[DeltaMessage, None]:

        if self.tool_call_start_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)
        # Skip empty chunks
        if len(delta_text.strip()) == 0:
            return None

        try:
            delta = None
            # Use buffer to accumulate delta_text content
            self.buffer += delta_text

            # Process the buffer content
            if "<tool_call>" in delta_text:
                self.current_tool_id = (
                    max(self.current_tool_id, 0) if self.current_tool_id == -1 else self.current_tool_id + 1
                )
                self.current_tool_name_sent = False
                if len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                data_processor_logger.debug(f"New tool call started with ID: {self.current_tool_id}")

            # 1. Try to parse the name field
            if not self.current_tool_name_sent and '"name"' in self.buffer:
                name_match = re.search(r'"name"\s*:\s*"([^"]*)"', self.buffer)
                if name_match:
                    name = name_match.group(1)
                    if name:
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    type="function",
                                    id=random_tool_call_id(),
                                    function=DeltaFunctionCall(name=name).model_dump(exclude_none=True),
                                )
                            ]
                        )
                        # Delete the processed name part from the buffer
                        self.buffer = self.buffer[name_match.end() :]
                        self.current_tool_name_sent = True
                        return delta
            # 2. Processing arguments field
            if '"arguments"' in self.buffer:
                args_match = re.search(r'"arguments"\s*:\s*(\{.*)', self.buffer)
                if args_match:
                    args_content = args_match.group(1)
                    try:
                        # Check if arguments field is complete by bracket matching
                        if "}}" in args_content:
                            matched_pos = -1
                            for i, ch in enumerate(delta_text):
                                if ch == "{":
                                    self.bracket_counts["total_l"] += 1
                                elif ch == "}":
                                    self.bracket_counts["total_r"] += 1

                                if self.bracket_counts["total_l"] == self.bracket_counts["total_r"]:
                                    matched_pos = i
                                    break

                            if matched_pos >= 0:
                                # Clean up bracket counts for next tool call
                                truncate_text = delta_text[: matched_pos + 1]
                                delta = DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=self.current_tool_id,
                                            function=DeltaFunctionCall(arguments=truncate_text).model_dump(
                                                exclude_none=True
                                            ),
                                        )
                                    ]
                                )
                                self.buffer = self.buffer[args_match.end() :]
                                return delta
                            else:
                                # No complete match yet
                                return None
                        else:
                            # Return partial arguments
                            for ch in delta_text:
                                if ch == "{":
                                    self.bracket_counts["total_l"] += 1
                                elif ch == "}":
                                    self.bracket_counts["total_r"] += 1
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(arguments=delta_text).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            return delta
                    except Exception as e:
                        data_processor_logger.error(f"Error in streaming tool call extraction: {str(e)}")
                        return None
            if "</tool_call>" in self.buffer:
                end_pos = self.buffer.find("</tool_call>")
                self.buffer = self.buffer[end_pos + len("</tool_call>") :]

                self.streamed_args_for_tool.append("")

            return delta

        except Exception as e:
            data_processor_logger.error(f"Error in streaming tool call extraction: {str(e)}")
            return None
