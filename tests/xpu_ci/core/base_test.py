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

"""Base test class for XPU CI tests."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import openai
import pytest

from .config import TestConfig
from .server_manager import ServerManager


class BaseXPUTest(ABC):
    """Base class for XPU CI tests.

    This class provides common functionality for all XPU tests including
    server management, OpenAI client setup, and common test methods.

    Subclasses should implement get_test_config() to provide the specific
    test configuration.
    """

    @classmethod
    @abstractmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get the test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID (0 or 1).

        Returns:
            TestConfig instance for this test.
        """
        pass

    @pytest.fixture(scope="class")
    def server_manager(self, request, model_path, xpu_id) -> ServerManager:
        """Fixture to manage server lifecycle for the test class.

        Args:
            request: pytest request fixture.
            model_path: Path to model files.
            xpu_id: XPU device ID.

        Yields:
            ServerManager instance with server running.
        """
        config = self.get_test_config(model_path, xpu_id)
        manager = ServerManager(config)

        if not manager.start():
            pytest.fail(f"服务启动失败: {config.description}")

        yield manager

        manager.stop()

    @pytest.fixture(scope="class")
    def openai_client(self, server_manager: ServerManager) -> openai.Client:
        """Fixture to create OpenAI client.

        Args:
            server_manager: Server manager with running server.

        Returns:
            OpenAI client configured to connect to the test server.
        """
        return openai.Client(
            base_url=server_manager.api_base_url,
            api_key="EMPTY_API_KEY",
        )

    @pytest.fixture(scope="class")
    def test_config(self, model_path, xpu_id) -> TestConfig:
        """Fixture to get test configuration.

        Args:
            model_path: Path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig instance.
        """
        return self.get_test_config(model_path, xpu_id)

    def _chat_completion(
        self,
        client: openai.Client,
        messages: list,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.0,
        stream: bool = False,
    ) -> Any:
        """Make a chat completion request.

        Args:
            client: OpenAI client.
            messages: List of message dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            stream: Whether to stream response.

        Returns:
            Chat completion response.
        """
        return client.chat.completions.create(
            model="default",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

    def _assert_response_contains_keywords(
        self,
        response_content: str,
        keywords: list,
        message: Optional[str] = None,
    ):
        """Assert that response contains at least one of the expected keywords.

        Args:
            response_content: The response content to check.
            keywords: List of keywords to look for.
            message: Optional custom assertion message.
        """
        if message is None:
            message = f"Response should contain one of: {keywords}"

        assert any(keyword in response_content for keyword in keywords), (
            f"{message}\nActual response: {response_content}"
        )


class TextModelTest(BaseXPUTest):
    """Base class for text-only model tests."""

    def test_non_streaming_chat(
        self,
        openai_client: openai.Client,
        test_config: TestConfig,
    ):
        """Test non-streaming chat completion.

        Args:
            openai_client: OpenAI client fixture.
            test_config: Test configuration fixture.
        """
        response = self._chat_completion(
            openai_client,
            messages=[{"role": "user", "content": "你好，你是谁？"}],
            max_tokens=64,
        )

        print(f"Response: {response.choices[0].message.content}")

        self._assert_response_contains_keywords(
            response.choices[0].message.content,
            test_config.expected_keywords,
        )

    def test_streaming_chat(
        self,
        openai_client: openai.Client,
        test_config: TestConfig,
    ):
        """Test streaming chat completion.

        Args:
            openai_client: OpenAI client fixture.
            test_config: Test configuration fixture.
        """
        response = self._chat_completion(
            openai_client,
            messages=[{"role": "user", "content": "你好，你是谁？"}],
            max_tokens=64,
            stream=True,
        )

        content_parts = []
        for chunk in response:
            if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        full_content = "".join(content_parts)
        print(f"Streaming Response: {full_content}")

        self._assert_response_contains_keywords(
            full_content,
            test_config.expected_keywords,
        )


class VLModelTest(BaseXPUTest):
    """Base class for vision-language model tests."""

    def test_image_understanding(
        self,
        openai_client: openai.Client,
        test_config: TestConfig,
    ):
        """Test image understanding capability.

        Args:
            openai_client: OpenAI client fixture.
            test_config: Test configuration fixture.
        """
        response = self._chat_completion(
            openai_client,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                            },
                        },
                        {"type": "text", "text": "图片中的文物来自哪个时代？"},
                    ],
                },
            ],
            max_tokens=70,
        )

        print(f"Response: {response.choices[0].message.content}")

        self._assert_response_contains_keywords(
            response.choices[0].message.content,
            test_config.expected_keywords,
        )
