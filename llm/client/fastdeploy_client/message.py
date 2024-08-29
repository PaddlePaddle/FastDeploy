# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

class ChatMessage(object):
    """
    多轮对话数据结构，当使用这个与ChatBot对话时
    会将对话记录存储在此结构体内，支持多轮
    """
    def __init__(self, prompt=None):
        if prompt is not None:
            self.message = [{"role": "user", "content": prompt}]
        else:
            self.message = []

    def add_user_message(self, content):
        """
        添加一个用户消息
        """
        if len(self.message) > 0 and self.message[-1]["role"] != "assistant":
            raise Exception("Cannot add user message, because the role of the "
                            f"last message is not assistant. The message is {self.message}")
        self.message.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """
        添加一个assistant消息
        """
        if len(self.message) > 0 and self.message[-1]["role"] != "user":
            raise Exception("Cannot add user message, because the role of the "
                            f"last message is not user. The message is {self.message}")
        self.message.append({"role": "assistant", "content": content})

    def next_prompt(self, content):
        """
        添加一个新的对话，保留用于兼容。
        """
        self.add_user_message(content)

    def __str__(self):
        return str(self.message)
