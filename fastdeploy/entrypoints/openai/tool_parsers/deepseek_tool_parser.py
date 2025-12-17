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
from typing import Optional, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

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


def random_tool_call_id() -> str:
    """Generate a random tool call ID"""
    return f"chatcmpl-tool-{str(uuid.uuid4().hex)}"


@ToolParserManager.register_module(["deepseek", "deepseek-r1", "deepseek-v3.1", "deepseek-v3-0324"])
class DeepSeekToolParser(ToolParser):
    """
    DeepSeek 系列模型的工具调用解析器（支持 V3.1、V3-0324、R1 三种模型）

    支持的格式：
    - V3.1: <｜tool▁call▁begin｜>function_name<｜tool▁sep｜>{"arg": "value"}<｜tool▁call▁end｜>
    - V3-0324/R1: <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name\n```json\n{"arg": "value"}\n```<｜tool▁call▁end｜>
    """

    def __init__(self, tokenizer, model_name=None):
        super().__init__(tokenizer)

        self.model_name = model_name or ""
        self.buffer: str = ""
        # Streaming state
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        # 特殊标记
        self.tool_calls_begin_token = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token = "<｜tool▁calls▁end｜>"
        self.tool_call_begin_token = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token = "<｜tool▁call▁end｜>"
        self.tool_sep_token = "<｜tool▁sep｜>"

        # 获取 token IDs
        self.tool_calls_begin_token_id = self.vocab.get(self.tool_calls_begin_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)
        self.tool_call_begin_token_id = self.vocab.get(self.tool_call_begin_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self.tool_sep_token_id = self.vocab.get(self.tool_sep_token)

        if self.tool_calls_begin_token_id is None or self.tool_call_begin_token_id is None:
            raise RuntimeError("DeepSeek Tool parser could not locate tool call tokens in the tokenizer!")

        # 检测模型版本
        self.is_v31 = self._detect_model_version()

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

    def _detect_model_version(self) -> bool:
        """检测模型版本：V3.1 还是 V3-0324/R1"""
        if "v3.1" in self.model_name.lower():
            return True
        elif "v3-0324" in self.model_name.lower() or "r1" in self.model_name.lower():
            return False
        # 默认使用 V3.1 格式
        return True

    def detect_output_stage(self, prompt_token_ids: Sequence[int]) -> str:
        """
        根据进入模型的 prompt_token_ids，判断接下来模型输出是否处于工具调用阶段
        """
        if self.tool_calls_begin_token_id in prompt_token_ids:
            return "TOOL_CALL_STAGE"
        return "CONTENT_STAGE"

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest, output_stage: Optional[str] = None
    ) -> ExtractedToolCallInformation:
        """
        从完整的模型输出中提取工具调用（非流式场景）
        """
        try:
            # 检查是否有工具调用标记
            if self.tool_calls_begin_token not in model_output:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=None, content=model_output)

            # 检查 </think> 与工具调用之间是否有非空白字符
            reasoning_end = "</think>"
            if reasoning_end in model_output:
                reasoning_end_pos = model_output.find(reasoning_end)
                after_reasoning = model_output[reasoning_end_pos + len(reasoning_end) :]
                tool_calls_begin_pos = after_reasoning.find(self.tool_calls_begin_token)
                if tool_calls_begin_pos > 0:
                    # 检查中间是否有非空白字符
                    between_text = after_reasoning[:tool_calls_begin_pos]
                    if between_text.strip() and not between_text.strip().isspace():
                        # 有非空白字符，协议不规范，不解析工具调用
                        return ExtractedToolCallInformation(tools_called=False, tool_calls=None, content=model_output)

            tool_calls = []

            if self.is_v31:
                # V3.1 格式：<｜tool▁call▁begin｜>function_name<｜tool▁sep｜>{"arg": "value"}<｜tool▁call▁end｜>
                # 转义特殊标记中的 | 字符（在正则表达式中 | 是特殊字符）
                begin_escaped = self.tool_call_begin_token.replace("|", r"\|")
                sep_escaped = self.tool_sep_token.replace("|", r"\|")
                end_escaped = self.tool_call_end_token.replace("|", r"\|")
                pattern = (
                    f"{begin_escaped}(?P<function_name>[^<]+?){sep_escaped}(?P<function_arguments>.*?){end_escaped}"
                )
            else:
                # V3-0324/R1 格式：<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name\n```json\n{"arg": "value"}\n```<｜tool▁call▁end｜>
                begin_escaped = self.tool_call_begin_token.replace("|", r"\|")
                sep_escaped = self.tool_sep_token.replace("|", r"\|")
                end_escaped = self.tool_call_end_token.replace("|", r"\|")
                # 注意：代码块标记 ``` 需要转义
                pattern = f"{begin_escaped}(?P<tool_type>[^<]+?){sep_escaped}(?P<function_name>[^\\n]+?)\\n```json\\n(?P<function_arguments>.*?)\\n```\\n{end_escaped}"

            matches = re.finditer(pattern, model_output, re.DOTALL)

            for match in matches:
                function_name = match.group("function_name").strip()
                function_arguments = match.group("function_arguments").strip()

                # 解析参数
                try:
                    if function_arguments:
                        args_dict = json.loads(function_arguments)
                    else:
                        args_dict = {}
                except json.JSONDecodeError:
                    # 尝试使用 partial_json_parser
                    try:
                        args_dict = partial_json_parser.loads(function_arguments, flags=Allow.ALL)
                    except:
                        args_dict = {}

                args_str = json.dumps(args_dict, ensure_ascii=False) if args_dict else "{}"

                tool_calls.append(
                    ToolCall(
                        type="function",
                        id=random_tool_call_id(),
                        function=FunctionCall(
                            name=function_name,
                            arguments=args_str,
                        ),
                    )
                )

            if tool_calls:
                return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content="")
            else:
                return ExtractedToolCallInformation(tools_called=False, tool_calls=None, content=model_output)

        except Exception as e:
            data_processor_logger.error(f"Error in extracting tool calls from response: {str(e)}")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=None, content=model_output)

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
        从增量消息中提取工具调用（流式场景）
        """
        try:
            # 如果没有工具调用标记，返回 None
            if self.tool_calls_begin_token_id not in current_token_ids:
                return None

            # 累积到 buffer
            self.buffer += delta_text

            # 检测新的工具调用开始
            if self.tool_call_begin_token_id in delta_token_ids:
                self.current_tool_id = (
                    max(self.current_tool_id, 0) if self.current_tool_id == -1 else self.current_tool_id + 1
                )
                self.current_tool_name_sent = False
                if len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                data_processor_logger.debug(f"New tool call started with ID: {self.current_tool_id}")

            # 1. 尝试提取工具名称
            if not self.current_tool_name_sent:
                # 查找工具名称：在 <｜tool▁call▁begin｜> 和 <｜tool▁sep｜> 之间
                begin_pos = self.buffer.find(self.tool_call_begin_token)
                if begin_pos != -1:
                    after_begin = self.buffer[begin_pos + len(self.tool_call_begin_token) :]
                    sep_pos = after_begin.find(self.tool_sep_token)
                    if sep_pos != -1:
                        # 提取分隔符之前的内容
                        tool_type_or_name = after_begin[:sep_pos].strip()
                        after_sep = after_begin[sep_pos + len(self.tool_sep_token) :]

                        # 判断格式：如果是 V3-0324/R1 且提取到的是 "function"，则从分隔符后提取函数名
                        if not self.is_v31 and tool_type_or_name == "function":
                            # V3-0324/R1 格式：提取分隔符后、换行符前的内容
                            newline_pos = after_sep.find("\n")
                            if newline_pos != -1:
                                function_name = after_sep[:newline_pos].strip()
                            else:
                                # 如果还没有换行符，暂时返回 None，等待更多数据
                                return None
                        else:
                            # V3.1 格式：分隔符前的内容就是函数名
                            function_name = tool_type_or_name

                        if function_name:
                            # 创建 DeltaMessage
                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        type="function",
                                        id=random_tool_call_id(),
                                        function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            # 从 buffer 中移除已处理的部分
                            if not self.is_v31 and tool_type_or_name == "function":
                                # V3-0324/R1：需要移除到换行符之后（包括换行符）
                                processed_end = (
                                    begin_pos
                                    + len(self.tool_call_begin_token)
                                    + sep_pos
                                    + len(self.tool_sep_token)
                                    + newline_pos
                                    + 1
                                )
                            else:
                                # V3.1：移除到分隔符之后
                                processed_end = (
                                    begin_pos + len(self.tool_call_begin_token) + sep_pos + len(self.tool_sep_token)
                                )
                            self.buffer = self.buffer[processed_end:]
                            self.current_tool_name_sent = True
                            return delta

            # 2. 处理参数部分
            if self.current_tool_name_sent:
                # 检查是否到达工具调用结束标记
                if self.tool_call_end_token_id in delta_token_ids:
                    # 工具调用结束，提取完整参数
                    end_pos = self.buffer.find(self.tool_call_end_token)
                    if end_pos != -1:
                        args_text = self.buffer[:end_pos].strip()

                        # 对于 V3-0324/R1，需要从代码块中提取 JSON
                        if not self.is_v31:
                            # 仅提取最后一个 code block，避免历史 buffer 累积造成重复片段
                            code_blocks = re.findall(r"```json\s*(.*?)\s*```", args_text, flags=re.DOTALL)
                            if code_blocks:
                                args_text = code_blocks[-1].strip()
                            else:
                                # 如果未匹配到 code block，回退到去除标记的方式
                                args_text = re.sub(r"^```json\s*", "", args_text, flags=re.MULTILINE)
                                args_text = re.sub(r"\s*```\s*$", "", args_text, flags=re.MULTILINE)
                                args_text = args_text.strip()

                        if args_text:
                            try:
                                # 尝试解析完整 JSON
                                args_dict = json.loads(args_text)
                                args_str = json.dumps(args_dict, ensure_ascii=False)
                            except json.JSONDecodeError:
                                # 使用 partial_json_parser
                                try:
                                    args_dict = partial_json_parser.loads(args_text, flags=Allow.ALL)
                                    args_str = json.dumps(args_dict, ensure_ascii=False)
                                except:
                                    args_str = args_text

                            delta = DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(arguments=args_str).model_dump(exclude_none=True),
                                    )
                                ]
                            )
                            # 清理 buffer
                            self.buffer = self.buffer[end_pos + len(self.tool_call_end_token) :]
                            return delta
                else:
                    # 流式输出参数
                    # 对于 V3-0324/R1，需要跳过代码块标记
                    args_text = self.buffer
                    if not self.is_v31:
                        # 移除开头的 ```json 标记（如果存在）
                        args_text = re.sub(r"^```json\s*", "", args_text, flags=re.MULTILINE)

                    if args_text.strip():
                        # 尝试解析部分 JSON
                        try:
                            # 使用 partial_json_parser 解析部分 JSON
                            args_dict = partial_json_parser.loads(args_text, flags=Allow.ALL)
                            args_str = json.dumps(args_dict, ensure_ascii=False)
                        except:
                            # 如果解析失败，直接使用原始文本
                            args_str = args_text

                        # 计算增量部分（只返回新增的部分）
                        if len(self.streamed_args_for_tool) > self.current_tool_id:
                            prev_args = self.streamed_args_for_tool[self.current_tool_id]
                            if args_str.startswith(prev_args):
                                new_args = args_str[len(prev_args) :]
                                # 如果没有新增内容（或只是重复的整段），直接返回 None，避免重复推送
                                if not new_args or new_args.strip() == "" or new_args.strip() == prev_args.strip():
                                    return None
                                self.streamed_args_for_tool[self.current_tool_id] = args_str
                                return DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=self.current_tool_id,
                                            function=DeltaFunctionCall(arguments=new_args).model_dump(
                                                exclude_none=True
                                            ),
                                        )
                                    ]
                                )
                        else:
                            # 第一次收到参数
                            self.streamed_args_for_tool.append(args_str)
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_id,
                                        function=DeltaFunctionCall(arguments=args_str).model_dump(exclude_none=True),
                                    )
                                ]
                            )

            return None

        except Exception as e:
            data_processor_logger.error(f"Error in streaming tool call extraction: {str(e)}")
            return None
