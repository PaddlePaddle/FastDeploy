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
import uuid
from collections.abc import Sequence
from typing import Union

import partial_json_parser


def random_tool_call_id() -> str:
    """Generate a random tool call ID"""
    return f"chatcmpl-tool-{str(uuid.uuid4().hex)}"


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


@ToolParserManager.register_module("ernie-45-vl-thinking")
class Ernie45VLThinkingToolParser(ToolParser):
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
        self.valid = None

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if self.tool_call_start_token_id is None:
            self.tool_call_start_token_id = -1

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
            tool_calls = []

            function_call_arr = []
            remaining_text = model_output

            think_end = remaining_text.find("</think>")
            think_end = think_end + len("</think>") if think_end != -1 else 0
            tool_begin = remaining_text.find("<tool_call>")
            if tool_begin != -1:
                middle_str = remaining_text[think_end:tool_begin]
                if len(middle_str.strip("\n")) > 0:
                    return ExtractedToolCallInformation(tools_called=False, content=model_output)

            while True:
                # Find the next <tool_call>
                tool_call_pos = remaining_text.find("<tool_call>")
                if tool_call_pos == -1:
                    break

                # Extract content after <tool_call>
                tool_content_start = tool_call_pos + len("<tool_call>")
                tool_content_end = remaining_text.find("</tool_call>", tool_content_start)

                tool_json = ""
                if tool_content_end == -1:
                    # Processing unclosed tool_call block (truncated case)
                    tool_json = remaining_text[tool_content_start:].strip()
                    remaining_text = ""  # No more content to process
                else:
                    # Processing closed </tool_call> block
                    tool_json = remaining_text[tool_content_start:tool_content_end].strip()
                    remaining_text = remaining_text[tool_content_end + len("</tool_call>") :]

                if not tool_json:
                    continue

                # Process tool_json
                tool_json = tool_json.strip()
                if not tool_json.startswith("{"):
                    tool_json = "{" + tool_json
                if not tool_json.endswith("}"):
                    tool_json = tool_json + "}"

                try:
                    # Parsing strategy: First try standard json.loads
                    try:
                        tool_data = json.loads(tool_json)

                        if isinstance(tool_data, dict) and "name" in tool_data and "arguments" in tool_data:
                            function_call_arr.append(
                                {
                                    "name": tool_data["name"],
                                    "arguments": tool_data["arguments"],
                                    "_is_complete": True,  # Mark as complete
                                }
                            )
                            continue
                    except json.JSONDecodeError:
                        pass

                    # Try partial_json_parser when standard parsing fails
                    from partial_json_parser.core.options import Allow

                    try:
                        tool_data = {}
                        flags = Allow.ALL & ~Allow.STR

                        # Parse the name field
                        name_match = re.search(r'"name"\s*:\s*"([^"]*)"', tool_json)
                        if name_match:
                            tool_data["name"] = name_match.group(1)

                        # Parse the arguments field
                        args_match = re.search(r'"arguments"\s*:\s*(\{.*)', tool_json)
                        if args_match:
                            try:
                                tool_data["arguments"] = partial_json_parser.loads(args_match.group(1), flags=flags)
                            except:
                                tool_data["arguments"] = None

                        if isinstance(tool_data, dict):
                            function_call_arr.append(
                                {
                                    "name": tool_data.get("name", ""),
                                    "arguments": tool_data.get("arguments", {}),
                                    "_is_partial": True,  # Mark as partial
                                }
                            )
                    except Exception as e:
                        data_processor_logger.debug(f"Failed to parse tool call: {str(e)}")
                        continue
                except Exception as e:
                    data_processor_logger.debug(f"Failed to parse tool call: {str(e)}")
                    continue

            if not function_call_arr:
                data_processor_logger.error("No valid tool calls found")
                return ExtractedToolCallInformation(tools_called=False, content=model_output)

            tool_calls = []
            all_complete = True  # Initialize as all complete

            for tool_call in function_call_arr:
                # Set flags
                is_complete = tool_call.get("_is_complete", False)
                is_partial = tool_call.get("_is_partial", False)

                # If any tool call is incomplete or partial, mark all_complete as False
                if not is_complete or is_partial:
                    all_complete = False

                # Process arguments
                tool_args = tool_call.get("arguments", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                try:
                    args_str = json.dumps(tool_args, ensure_ascii=False) if tool_args else "{}"
                except:
                    args_str = "{}"

                tool_calls.append(
                    ToolCall(
                        type="function",
                        id=random_tool_call_id(),
                        function=FunctionCall(
                            name=tool_call.get("name", ""),
                            arguments=args_str,
                        ),
                    )
                )

            # Only return tools_called=True if all tool calls are complete
            return ExtractedToolCallInformation(
                tools_called=all_complete, tool_calls=tool_calls if tool_calls else None, content=""
            )

        except Exception as e:
            data_processor_logger.error(f"Error in extracting tool call from response: {str(e)}")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=None, content=model_output)

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

        if self.valid is not None and not self.valid:
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
                if self.valid is None:
                    tool_call_begin = current_text.find(self.tool_call_start_token)
                    prefix = current_text[:tool_call_begin]
                    prefix = prefix.strip("\n")
                    if len(prefix) > 0 and not prefix.endswith("</think>"):
                        self.valid = False
                        return DeltaMessage(content=delta_text)
                    self.valid = True
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
