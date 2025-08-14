# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

ip = "0.0.0.0"
service_http_port = "8188"  # 服务配置的
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

# 非流式对话
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "The largest ocean is"},
    ],
    temperature=1,
    top_p=0,
    max_tokens=64,
    stream=False,
)
print(f"response is: {response}", flush=True)

generate_context = response.choices[0].message.content
print(f"\ngenerate_context is: {generate_context}", flush=True)

assert "pacific ocean" in generate_context.lower(), "The answer was incorrect!"

print("Test successfully!", flush=True)
