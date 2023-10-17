# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import openai
from fastdeploy_llm.Client import grpcClient

model = "llama-ptuning"
port = 2001
url = "0.0.0.0:8812"

client = grpcClient(base_url=url, model_name=model, openai_port=port)

# Modify OpenAI's API key and API base.
openai.api_key = "EMPTY"
openai.api_base = "http://0.0.0.0:" + str(port) + "/v1"

# Completion API
completion = openai.Completion.create(
    model=model, prompt="A robot may not injure a human being")

print("Completion results:")
print(completion)

# ChatCompletion API
chat_completion = openai.ChatCompletion.create(
    model=model,
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }])
print("Chat completion results:")
print(chat_completion)
