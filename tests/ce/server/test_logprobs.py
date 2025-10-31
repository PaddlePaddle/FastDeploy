import json

from core import TEMPLATE, URL, build_request_payload, send_request


def test_unstream_with_logprobs():
    """
    测试非流式响应开启 logprobs 后，返回的 token 概率信息是否正确。
    """
    data = {
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与概率信息
    assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["choices"][0]["logprobs"]["content"][0]["token"] == "牛顿"
    assert resp_json["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.031025361269712448
    assert resp_json["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "牛顿",
        "logprob": -0.031025361269712448,
        "bytes": [231, 137, 155, 233, 161, 191],
        "top_logprobs": None,
    }

    assert resp_json["usage"]["prompt_tokens"] == 22
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["usage"]["total_tokens"] == 25


def test_unstream_without_logprobs():
    """
    测试非流式响应关闭 logprobs 后，返回结果中不包含 logprobs 字段。
    """
    data = {
        "stream": False,
        "logprobs": False,
        "top_logprobs": None,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }

    # 构建请求并发送
    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    resp_json = response.json()

    # 校验返回内容与 logprobs 字段
    assert resp_json["choices"][0]["message"]["content"] == "牛顿的"
    assert resp_json["choices"][0]["logprobs"] is None
    assert resp_json["usage"]["prompt_tokens"] == 22
    assert resp_json["usage"]["completion_tokens"] == 3
    assert resp_json["usage"]["total_tokens"] == 25


def test_stream_with_logprobs():
    """
    测试流式响应开启 logprobs 后，首个 token 的概率信息是否正确。
    """
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

    # 解析首个包含 content 的流式 chunk
    result_chunk = {}
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        chunk = json.loads(decoded)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            result_chunk = chunk
            print(json.dumps(result_chunk, indent=2, ensure_ascii=False))
            break

    # 校验概率字段
    assert result_chunk["choices"][0]["delta"]["content"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["token"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.031025361269712448
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "牛顿",
        "logprob": -0.031025361269712448,
        "bytes": [231, 137, 155, 233, 161, 191],
    }


def test_stream_without_logprobs():
    """
    测试流式响应关闭 logprobs 后，确认响应中不包含 logprobs 字段。
    """
    data = {
        "stream": True,
        "logprobs": False,
        "top_logprobs": None,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

    # 解析首个包含 content 的流式 chunk
    result_chunk = {}
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        chunk = json.loads(decoded)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            result_chunk = chunk
            print(json.dumps(result_chunk, indent=2, ensure_ascii=False))
            break

    # 校验 logprobs 字段不存在
    assert result_chunk["choices"][0]["delta"]["content"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"] is None


def test_stream_with_temp_scaled_logprobs():
    """
    测试流式响应开启 temp_scaled_logprobs 后，首个 token 的概率信息是否正确。
    """
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "temperature": 0.8,
        "top_p": 0,
        "temp_scaled_logprobs": True,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

    # 解析首个包含 content 的流式 chunk
    result_chunk = {}
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        chunk = json.loads(decoded)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            result_chunk = chunk
            print(json.dumps(result_chunk, indent=2, ensure_ascii=False))
            break

    # 校验概率字段
    assert result_chunk["choices"][0]["delta"]["content"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["token"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.006811376195400953
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "牛顿",
        "logprob": -0.006811376195400953,
        "bytes": [231, 137, 155, 233, 161, 191],
    }


def test_stream_with_top_p_normalized_logprobs():
    """
    测试流式响应开启 top_p_normalized_logprobs 后，首个 token 的概率信息是否正确。
    """
    data = {
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "牛顿的三大运动定律是什么？"},
        ],
        "max_tokens": 3,
        "top_p": 0,
        "top_p_normalized_logprobs": True,
    }

    payload = build_request_payload(TEMPLATE, data)
    response = send_request(URL, payload)

    # 解析首个包含 content 的流式 chunk
    result_chunk = {}
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").removeprefix("data: ")
        if decoded == "[DONE]":
            break

        chunk = json.loads(decoded)
        content = chunk["choices"][0]["delta"].get("content")
        if content:
            result_chunk = chunk
            print(json.dumps(result_chunk, indent=2, ensure_ascii=False))
            break

    # 校验概率字段
    assert result_chunk["choices"][0]["delta"]["content"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["token"] == "牛顿"
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["logprob"] == 0.0
    assert result_chunk["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0] == {
        "token": "牛顿",
        "logprob": 0.0,
        "bytes": [231, 137, 155, 233, 161, 191],
    }


if __name__ == "__main__":
    test_unstream_with_logprobs()
    test_unstream_without_logprobs()
    test_stream_with_logprobs()
    test_stream_without_logprobs()
    test_stream_with_temp_scaled_logprobs()
    test_stream_with_top_p_normalized_logprobs()
