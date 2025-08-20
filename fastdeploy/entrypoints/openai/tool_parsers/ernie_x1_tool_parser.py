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

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Hermes 2 Pro Tool parser could not locate tool call start/end " "tokens in the tokenizer!"
            )

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

            # Check for invalid <response> tags before tool calls
            if re.search(r"<response>[\s\S]*?</response>\s*(?=<tool_call>)", model_output):
                data_processor_logger.error("Invalid format: <response> tags found before <tool_call>")
                return ExtractedToolCallInformation(tools_called=False, content=model_output)

            function_call_arr = []
            remaining_text = model_output

            while True:
                # 查找下一个tool_call块
                tool_call_pos = remaining_text.find("<tool_call>")
                if tool_call_pos == -1:
                    break

                # 提取tool_call开始位置后的内容
                tool_content_start = tool_call_pos + len("<tool_call>")
                tool_content_end = remaining_text.find("</tool_call>", tool_content_start)

                tool_json = ""
                if tool_content_end == -1:
                    # 处理未闭合的tool_call块（截断情况）
                    tool_json = remaining_text[tool_content_start:].strip()
                    remaining_text = ""  # 没有更多内容需要处理
                else:
                    # 处理完整的tool_call块
                    tool_json = remaining_text[tool_content_start:tool_content_end].strip()
                    remaining_text = remaining_text[tool_content_end + len("</tool_call>") :]

                if not tool_json:
                    continue

                # 处理JSON内容
                tool_json = tool_json.strip()
                if not tool_json.startswith("{"):
                    tool_json = "{" + tool_json
                if not tool_json.endswith("}"):
                    tool_json = tool_json + "}"

                try:
                    # 首先尝试标准JSON解析
                    try:
                        tool_data = json.loads(tool_json)

                        if isinstance(tool_data, dict) and "name" in tool_data and "arguments" in tool_data:
                            function_call_arr.append(
                                {
                                    "name": tool_data["name"],
                                    "arguments": tool_data["arguments"],
                                    "_is_complete": True,  # 明确标记为完整解析
                                }
                            )
                            continue
                    except json.JSONDecodeError:
                        pass

                    # 标准解析失败时尝试partial_json_parser
                    from partial_json_parser.core.options import Allow

                    try:
                        tool_data = {}
                        flags = Allow.ALL & ~Allow.STR

                        # 解析name字段
                        name_match = re.search(r'"name"\s*:\s*"([^"]*)"', tool_json)
                        if name_match:
                            tool_data["name"] = name_match.group(1)

                        # 解析arguments字段
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
                                    "_is_partial": True,  # 标记为部分解析
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
            all_complete = True  # 初始设为True，只要有一个不完整就变为False

            for tool_call in function_call_arr:
                # 记录工具调用解析状态
                is_complete = tool_call.get("_is_complete", False)
                is_partial = tool_call.get("_is_partial", False)

                # 只要有一个不完整就认为整体不完整
                if not is_complete or is_partial:
                    all_complete = False

                # 处理参数序列化
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

            # 只有当所有工具调用都明确标记为complete时才返回tools_called=True
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
        # 忽略空chunk
        if len(delta_text.strip()) == 0:
            return None

        try:
            delta = None
            # 使用buffer累积delta_text内容
            self.buffer += delta_text

            # 处理增量中的新tool_call开始
            if "<tool_call>" in delta_text:
                self.current_tool_id = (
                    max(self.current_tool_id, 0) if self.current_tool_id == -1 else self.current_tool_id + 1
                )
                self.current_tool_name_sent = False
                if len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                data_processor_logger.debug(f"New tool call started with ID: {self.current_tool_id}")

            # 1. 尝试解析name字段
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
                        # 删除已处理的name部分
                        self.buffer = self.buffer[name_match.end() :]
                        self.current_tool_name_sent = True
                        return delta
            # 2. 尝试解析arguments字段
            if '"arguments"' in self.buffer:
                args_match = re.search(r'"arguments"\s*:\s*(\{.*)', self.buffer)
                if args_match:
                    args_content = args_match.group(1)
                    try:
                        # 检查是否到达arguments结尾(括号完全匹配)
                        if "}}" in args_content:
                            # 逐个字符检查括号匹配状态
                            matched_pos = -1
                            for i, ch in enumerate(delta_text):
                                if ch == "{":
                                    self.bracket_counts["total_l"] += 1
                                elif ch == "}":
                                    self.bracket_counts["total_r"] += 1

                                if self.bracket_counts["total_l"] == self.bracket_counts["total_r"]:  # 括号完全匹配
                                    matched_pos = i
                                    break

                            if matched_pos >= 0:
                                # 找到匹配点，清理buffer并返回
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
                                # 没有完全匹配，继续累积
                                return None
                        else:
                            # 增量返回当前可解析的部分
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

                # 完成当前工具调用处理
                self.streamed_args_for_tool.append("")

            return delta

        except Exception as e:
            data_processor_logger.error(f"Error in streaming tool call extraction: {str(e)}")
            return None
