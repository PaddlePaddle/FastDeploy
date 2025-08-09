import os

import openai
import pytest

# Read ports from environment variables; use default values if not set
FD_API_PORT = int(os.getenv("FD_API_PORT", 8188))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", 8133))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", 8233))

# List of ports to clean before and after tests
PORTS_TO_CLEAN = [FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT]

# ==========================
# OpenAI Client chat.completions Test
# ==========================


@pytest.fixture
def openai_client():
    ip = "0.0.0.0"
    service_http_port = str(FD_API_PORT)
    client = openai.Client(
        base_url=f"http://{ip}:{service_http_port}/v1",
        api_key="EMPTY_API_KEY",
    )
    return client


def test_non_streaming_prompt_echo_response(openai_client, capsys):
    """
    测试非流式 completion 中的 echo 选项功能。
    测试以下行号相关的功能：
    - 310, 311: 与 prompt 处理和 echo 标志相关的逻辑
    - 313, 314: 与多个 prompt 处理相关的逻辑
    - 316: 与文本拼接相关的逻辑
    - 318, 319: 与 token IDs 和文本处理相关的逻辑
    """
    # 测试单个 prompt 的 echo 功能
    response_0 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=10,
        stream=False,
        echo=True,
    )
    assert response_0.choices[0].text.startswith("Hello, how are you?")

    # 验证 token IDs 是否正确返回（测试 318, 319 相关功能）
    assert hasattr(response_0.choices[0], "prompt_token_ids") or hasattr(response_0.choices[0], "token_ids")

    # 测试多个 prompts 的 echo 功能
    prompts = ["Hello, how are you?", "What is your name?"]
    response_1 = openai_client.completions.create(
        model="default",
        prompt=prompts,
        temperature=1,
        max_tokens=10,
        stream=False,
        echo=True,
    )
    for i in range(len(response_1.choices)):
        assert response_1.choices[i].text.startswith(prompts[i])
        # 验证 token IDs 是否正确返回
        assert hasattr(response_1.choices[i], "prompt_token_ids") or hasattr(response_1.choices[i], "token_ids")


def test_streaming_prompt_echo_response(openai_client, capsys):
    """
    测试流式 completion 中的 echo 选项功能。
    测试以下行号相关的功能：
    - 409, 410: 与流式响应中的第一个 token 处理相关的逻辑
    - 433: 与流式响应中的文本拼接相关的逻辑
    - 435, 436: 与多个流式 prompt 处理相关的逻辑
    - 438: 与流式响应中的 token IDs 处理相关的逻辑
    - 440: 与流式响应中的 finish_reason 处理相关的逻辑
    """
    # 测试单个 prompt 的流式 echo 功能
    response_0 = openai_client.completions.create(
        model="default",
        prompt="Hello, how are you?",
        temperature=1,
        max_tokens=10,
        stream=True,
        echo=True,
    )
    output = []
    for chunk in response_0:
        if hasattr(chunk.choices[0], "text"):
            output.append(chunk.choices[0].text)
        # 测试 438 相关的 token IDs 处理
        if hasattr(chunk.choices[0], "prompt_token_ids"):
            assert isinstance(chunk.choices[0].prompt_token_ids, list)
        if hasattr(chunk.choices[0], "completion_token_ids"):
            assert isinstance(chunk.choices[0].completion_token_ids, list)

    assert "".join(output).startswith("Hello, how are you?")

    # 测试多个 prompts 的流式 echo 功能
    prompts = ["Hello, how are you?", "What is your name?"]
    response_1 = openai_client.completions.create(
        model="default",
        prompt=prompts,
        temperature=1,
        max_tokens=10,
        stream=True,
        echo=True,
    )
    outputs = {i: [] for i in range(len(prompts))}
    first_responses = [False] * len(prompts)  # 标记是否已经收到每个prompt的第一个响应

    for chunk in response_1:
        if chunk == "data: [DONE]":
            break

        for choice in chunk.choices:
            index = choice.index
            text = choice.text
            if text is not None:
                outputs[index].append(text)

            # 测试 409, 410 相关的第一个响应处理
            if not first_responses[index] and text is not None:
                assert text.startswith(
                    prompts[index]
                ), f"Prompt {index} first response '{text}' doesn't match prompt '{prompts[index]}'"
                first_responses[index] = True

            # 测试 438 相关的 token IDs 处理
            if hasattr(choice, "prompt_token_ids"):
                assert isinstance(choice.prompt_token_ids, list)
            if hasattr(choice, "completion_token_ids"):
                assert isinstance(choice.completion_token_ids, list)

    # 验证所有 prompts 都收到了响应
    assert all(first_responses)
    for i in range(len(prompts)):
        assert "".join(outputs[i]).startswith(prompts[i])
