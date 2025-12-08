import json
import openai

# # === 参数 ===
# host = "0.0.0.0"
# port = "8623"
# api_key = "null"  # 本地部署通常不验证 key
# line_number = 2   # 读取 test_data 中的第 N 行（从 1 开始）

# # # === 初始化客户端 ===
# client = openai.Client(
#     base_url=f"http://{host}:{port}/v1",
#     api_key=api_key,
# )
# # for x in range(1):
# #     line_number = x + 25
# # === 从 test_data 中读取第 N 行的 user content ===
# with open("mingming_0410_eb_2k_ds_108_32_128k_part1_fd", "r", encoding="utf-8") as f:
#     for i, line in enumerate(f, start=1):
#         if i == line_number:
#             print(line)
#             json_obj = json.loads(line)
#             break
#     else:
#         raise ValueError(f"文件中没有第 {line_number} 行")

# print(json_obj["messages"],len(json_obj["messages"]))

# # === 发起非流式请求（添加参数） ===
# response = client.chat.completions.create(
#     model="null",  # 本地部署模型名可根据需要替换
#     messages=json_obj["messages"][0],
#     top_p=0.0,
#     temperature=0.8,
#     max_tokens=10,
#     frequency_penalty=0,
#     presence_penalty=0,
#     logit_bias={},  # optional
#     extra_body={
#         "metadata": {
#             "min_tokens": 1
#         },
#         "repetition_penalty": 1.0
#     },
#     stream=True,
# )

# # # === 打印完整响应 ===
# print(response.choices[0].message.content)

import openai
host = "0.0.0.0"
port = "8188"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

# response = client.completions.create(
#     model="null",
#     prompt="Where is the capital of China?",
#     stream=True,
# )
# for chunk in response:
#     print(chunk.choices[0].text, end='')
# print('\n')

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "北京天安门在哪里？"},
    ],
    top_p=0.0,
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')