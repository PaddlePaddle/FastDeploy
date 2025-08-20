import openai

ip = "0.0.0.0"
service_http_port = "9809"  # 服务配置的

client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

# 非流式返回
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
    max_tokens=64,
    stream=False,
)

print(response.choices[0].text)
print("\n")

# 流式返回
response = client.completions.create(
    model="default",
    prompt="Hello, how are you?",
    max_tokens=100,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].text, end="")
print("\n")

# Chat completion
# 非流式返回
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello, who are you"},
        {"role": "assistant", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=1,
    max_tokens=64,
    stream=False,
)

print(response.choices[0].message.content)
print("\n")


# # 流式返回
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello, who are you"},
        {"role": "assistant", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=1,
    max_tokens=64,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta is not None:
        print(chunk.choices[0].delta.content, end="")
print("\n")
