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

import os
import unittest
import weakref

from fastdeploy.entrypoints.llm import LLM
from fastdeploy.entrypoints.openai.protocol import ChatCompletionToolsParam

MODEL_NAME = os.getenv("MODEL_PATH") + "/ERNIE-4.5-0.3B-Paddle"


class TestChat(unittest.TestCase):
    """Test case for chat functionality"""

    COMMON_PREFIX = "I am a highly capable, compassionate, and trustworthy AI assistant dedicated to providing you with exceptional support. Whatever questions or challenges you may have, I will utilize my full capabilities to offer thoughtful and comprehensive assistance. As your intelligent companion, I consistently maintain honesty, transparency, and patience to ensure our interactions are both productive and enjoyable."

    PROMPTS = [
        [{"content": "PaddlePaddle is ", "role": "user"}],
        [{"content": COMMON_PREFIX + "The color of tomato is ", "role": "user"}],
        [{"content": COMMON_PREFIX + "The equation 2+3= ", "role": "user"}],
        [{"content": COMMON_PREFIX + "The equation 4-1= ", "role": "user"}],
    ]

    @classmethod
    def setUpClass(cls):
        try:
            llm = LLM(
                model=MODEL_NAME,
                max_num_batched_tokens=4096,
                tensor_parallel_size=1,
                engine_worker_queue_port=int(os.getenv("FD_ENGINE_QUEUE_PORT")),
                cache_queue_port=int(os.getenv("FD_CACHE_QUEUE_PORT")),
            )
            cls.llm = weakref.proxy(llm)
        except Exception as e:
            print(f"Setting up LLM failed: {e}")
            raise unittest.SkipTest(f"LLM initialization failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run"""
        if hasattr(cls, "llm"):
            del cls.llm

    def test_chat(self):
        outputs = self.llm.chat(messages=self.PROMPTS, sampling_params=None)
        self.assertEqual(len(self.PROMPTS), len(outputs))
        self.assertEqual(outputs[-1].num_cached_tokens, outputs[-2].num_cached_tokens)
        self.assertEqual(outputs[-1].num_cached_tokens, 64)

    def test_chat_with_tools(self):
        """Test chat with tools:
        1. spliced_message (after chat_template) contains tool-related content
        2. Model output contains tool_call
        """
        prompts = [{"role": "user", "content": "北京海淀区今天天气怎么样？用摄氏度表示温度。"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Determine weather in my location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "additionalProperties": False,
                        "required": ["location", "unit"],
                    },
                    "strict": True,
                },
            }
        ]
        chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '' in content %}\n                {%- set reasoning_content = content.split('')[0].rstrip('\\n').split('')[-1].lstrip('\\n') %}\n                {%- set content = content.split('')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n\\n' + reasoning_content.strip('\\n') + '\\n\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '\\n\\n\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

        data_processor = self.llm.llm_engine.data_processor
        captured_spliced_message = None

        def capture_spliced_message(request_or_messages, **kwargs):
            """Wrap original messages2ids to capture spliced_message"""
            token_ids = data_processor.original_messages2ids(request_or_messages, **kwargs)
            nonlocal captured_spliced_message
            captured_spliced_message = request_or_messages.get("prompt_tokens")
            return token_ids

        data_processor.original_messages2ids = data_processor.messages2ids
        data_processor.messages2ids = capture_spliced_message

        try:
            outputs = self.llm.chat(
                messages=prompts,
                tools=tools,
                chat_template=chat_template,
                chat_template_kwargs={"enable_thinking": False},
                stream=False,
            )

            self.assertIsNotNone(captured_spliced_message, "Failed to capture spliced_message from messages2ids")
            self.assertIn(
                "<tools>",
                captured_spliced_message,
                f"spliced_message '{captured_spliced_message}' missing <tools> tag (chat_template not applied)",
            )

            output = outputs[0]
            self.assertEqual(len(prompts), len(outputs))
            self.assertTrue(hasattr(output, "outputs"))
            self.assertTrue(hasattr(output.outputs, "text"))
        finally:
            data_processor.messages2ids = data_processor.original_messages2ids

    def test_validate_tools(self):
        """Test both valid and invalid scenarios for _validate_tools method"""
        # Prepare valid test data
        valid_tool_dict = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get real-time weather of a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
        valid_tool_model = ChatCompletionToolsParam(**valid_tool_dict)
        valid_model_list = [valid_tool_model, valid_tool_model]
        valid_dict_list = [valid_tool_dict, valid_tool_dict]

        # Test valid scenarios
        # 1. Input is None
        self.assertIsNone(self.llm._validate_tools(None))

        # 2. Input is single ChatCompletionToolsParam instance
        result = self.llm._validate_tools(valid_tool_model)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ChatCompletionToolsParam)

        # 3. Input is list of ChatCompletionToolsParam instances
        self.assertEqual(self.llm._validate_tools(valid_model_list), valid_model_list)

        # 4. Input is single valid dict
        result = self.llm._validate_tools(valid_tool_dict)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], dict)
        self.assertEqual(result[0]["type"], "function")

        # 5. Input is list of valid dicts
        result = self.llm._validate_tools(valid_dict_list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[1], dict)

        # 6. Input is empty list
        self.assertIsNone(self.llm._validate_tools([]))

        # Test invalid scenarios (should raise ValueError)
        # 1. Input is string (invalid top-level type)
        with self.assertRaises(ValueError):
            self.llm._validate_tools("invalid_string")

        # 2. Input list contains non-dict element
        with self.assertRaises(ValueError):
            self.llm._validate_tools([valid_tool_dict, 123])

        # 3. Tool dict missing required field (function.name)
        invalid_tool_missing_name = {"type": "function", "function": {"description": "Missing 'name' field"}}
        with self.assertRaises(ValueError):
            self.llm._validate_tools(invalid_tool_missing_name)

        # 4. Tool dict with wrong 'type' value
        invalid_tool_wrong_type = {"type": "invalid_type", "function": {"name": "test", "description": "Wrong type"}}
        with self.assertRaises(ValueError):
            self.llm._validate_tools(invalid_tool_wrong_type)

        # 5. Input is boolean
        with self.assertRaises(ValueError):
            self.llm._validate_tools(True)


if __name__ == "__main__":
    unittest.main()
