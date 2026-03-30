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

import partial_json_parser


def random_tool_call_id() -> str:
    """Generate a random tool call ID"""
    return f"chatcmpl-tool-{str(uuid.uuid4().hex)}"


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
from fastdeploy.utils import data_processor_logger as logger


@ToolParserManager.register_module("ernie-x1")
class ErnieX1ToolParser(ToolParser):
    """
    This parser handles tool calls with newline formats.
    """

    def __init__(self, tokenizer):
        """
        Ernie thinking model format:
        abc\n</think>\n\n\n<tool_call>\ndef\n</tool_call>\n
        """
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []
        self.think_end_token = "</think>"
        self.response_start_token: str = "<response>"
        self.response_end_token: str = "</response>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_call_regex = re.compile(r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>", re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser " "constructor during construction."
            )

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.response_start_token_id = self.vocab.get(self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

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
            tool_call_json_list = self.tool_call_regex.findall(model_output)
            tool_calls = []
            for tool_call_json in tool_call_json_list:
                tool_call_dict = json.loads(tool_call_json)
                args_str = json.dumps(tool_call_dict.get("arguments", {}), ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        id=random_tool_call_id(),
                        function=FunctionCall(
                            name=tool_call_dict.get("name", ""),
                            arguments=args_str,
                        ),
                    )
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
            )
        except Exception:
            logger.warning("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self.tool_call_start_token not in current_text:
            logger.debug("No tool call tokens found!")
            return DeltaMessage(content=delta_text)

        try:
            prev_tool_start_count = previous_text.count(self.tool_call_start_token)
            prev_tool_end_count = previous_text.count(self.tool_call_end_token)
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)
            tool_call_portion = None
            text_portion = None

            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1].split(self.tool_call_end_token)[0].rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            if cur_tool_start_count > cur_tool_end_count and cur_tool_start_count > prev_tool_start_count:
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            elif cur_tool_start_count > cur_tool_end_count and cur_tool_start_count == prev_tool_start_count:
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            elif cur_tool_start_count == cur_tool_end_count and cur_tool_end_count >= prev_tool_end_count:
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    logger.debug("attempting to close tool call, but no tool call")
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    if '"}' not in delta_text:
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug(
                        "Finishing tool and found diff that had not " "been streamed yet: %s",
                        diff,
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(exclude_none=True),
                            )
                        ]
                    )

            else:
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                return delta

            try:
                current_tool_call = (
                    partial_json_parser.loads(tool_call_portion or "{}", flags) if tool_call_portion else None
                )
                logger.debug("Parsed tool call %s", current_tool_call)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None
            except json.decoder.JSONDecodeError:
                logger.debug("unable to parse JSON")
                return None

            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=random_tool_call_id(),
                                function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None

            if tool_call_portion is None:
                delta = DeltaMessage(content=delta_text) if text_portion is not None else None
                return delta

            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
            cur_arguments = current_tool_call.get("arguments")

            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            elif not cur_arguments and prev_arguments:
                logger.error("should be impossible to have arguments reset " "mid-call. skipping streaming anything.")
                delta = None

            elif cur_arguments and not prev_arguments:
                function_name = current_tool_call.get("name")
                match = re.search(
                    r'\{"name":\s*"' + re.escape(function_name) + r'"\s*,\s*"arguments":\s*(.*)',
                    tool_call_portion.strip(),
                    re.DOTALL,
                )
                if match:
                    cur_arguments_json = match.group(1)
                else:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                logger.debug("finding %s in %s", delta_text, cur_arguments_json)

                if delta_text not in cur_arguments_json:
                    return None
                args_delta_start_loc = cur_arguments_json.rindex(delta_text) + len(delta_text)

                arguments_delta = cur_arguments_json[:args_delta_start_loc]
                logger.debug("First tokens in arguments received: %s", arguments_delta)

                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=arguments_delta).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += arguments_delta

            elif cur_arguments and prev_arguments:
                try:
                    json.loads(tool_call_portion)
                    is_complete_json = True
                except Exception:
                    is_complete_json = False

                if (
                    isinstance(delta_text, str)
                    and len(delta_text.rstrip()) >= 1
                    and delta_text.rstrip()[-1] == "}"
                    and is_complete_json
                ):
                    delta_text = delta_text.rstrip()[:-1]

                logger.debug("got diff %s", delta_text)
                if is_complete_json and delta_text.strip() == "":
                    return None
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(arguments=delta_text).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] += delta_text

            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            return delta

        except Exception:
            logger.warning("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
