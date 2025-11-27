import openai
host = "127.0.0.1"
port = "8188"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "今天去哪里好"},
    ],
    stream=False,
    max_tokens=100,
    seed=0,
    top_p=1e-5,
)

print(response)