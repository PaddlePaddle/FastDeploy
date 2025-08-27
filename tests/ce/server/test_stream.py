import json

from core import TEMPLATE, URL, build_request_payload, send_request


def test_stream_and_non_stream():
    """
    测试接口在 stream 模式和非 stream 模式下返回的内容是否一致。
    """

    # 构造 stream=True 的请求数据
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 100,
    }

    # 构建请求 payload 并发送流式请求
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

    # 按行解析流式响应
    resp_chunks = []
    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            decoded = decoded[len("data: ") :]

        if decoded == "[DONE]":
            break

        resp_chunks.append(json.loads(decoded))

    # 拼接模型最终输出内容
    final_content = "".join(
        chunk["choices"][0]["delta"]["content"]
        for chunk in resp_chunks
        if "choices" in chunk and "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]
    )
    print(final_content)

    # 修改为 stream=False，发送普通请求
    data["stream"] = False
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

    # 打印格式化后的完整响应
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    response_json = response.json()

    # 对比两种模式下输出是否一致
    assert final_content == response_json["choices"][0]["message"]["content"]


if __name__ == "__main__":
    test_stream_and_non_stream()
