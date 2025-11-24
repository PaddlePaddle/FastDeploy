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

"""Expert Parallel Tests for XPU CI.

This module contains tests for Expert Parallel inference with different
tensor parallel and data parallel configurations.
"""

import pytest

from ..core import TestConfig
from ..core.base_test import TextModelTest


@pytest.mark.expert_parallel
@pytest.mark.ep4tp4
class TestEP4TP4(TextModelTest):
    """EP4TP4 Expert Parallel test class.

    Tests Expert Parallel inference with:
    - Tensor Parallel Size = 4
    - Data Parallel Size = 1
    - Expert Parallel enabled
    - BKCL environment configured for RDMA
    """

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get EP4TP4 test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig for EP4TP4 test.
        """
        return TestConfig.create_ep4tp4_test(model_path, xpu_id)

    def test_ep4tp4_inference(
        self,
        openai_client,
        test_config: TestConfig,
    ):
        """Test EP4TP4 inference.

        Args:
            openai_client: OpenAI client fixture.
            test_config: Test configuration fixture.
        """
        response = self._chat_completion(
            openai_client,
            messages=[{"role": "user", "content": "你好，你是谁？"}],
            max_tokens=64,
        )

        content = response.choices[0].message.content
        print(f"EP4TP4 Response: {content}")

        self._assert_response_contains_keywords(
            content,
            test_config.expected_keywords,
            "EP4TP4推理结果不符合预期",
        )


@pytest.mark.expert_parallel
@pytest.mark.ep4tp1
class TestEP4TP1(TextModelTest):
    """EP4TP1 Expert Parallel test class.

    Tests Expert Parallel inference with:
    - Tensor Parallel Size = 1
    - Data Parallel Size = 4
    - Expert Parallel enabled
    - BKCL environment configured for RDMA
    """

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get EP4TP1 test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig for EP4TP1 test.
        """
        return TestConfig.create_ep4tp1_test(model_path, xpu_id)

    def test_ep4tp1_inference(
        self,
        openai_client,
        test_config: TestConfig,
    ):
        """Test EP4TP1 inference.

        Args:
            openai_client: OpenAI client fixture.
            test_config: Test configuration fixture.
        """
        response = self._chat_completion(
            openai_client,
            messages=[{"role": "user", "content": "你好，你是谁？"}],
            max_tokens=64,
        )

        content = response.choices[0].message.content
        print(f"EP4TP1 Response: {content}")

        self._assert_response_contains_keywords(
            content,
            test_config.expected_keywords,
            "EP4TP1推理结果不符合预期",
        )


@pytest.mark.expert_parallel
@pytest.mark.all2all
class TestEP4TP4All2All(TextModelTest):
    """EP4TP4 all2all test class.

    Tests Expert Parallel inference with all2all communication:
    - Tensor Parallel Size = 4
    - Data Parallel Size = 1
    - Expert Parallel enabled
    - all2all communication pattern
    """

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get EP4TP4 all2all test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig for EP4TP4 all2all test.
        """
        return TestConfig.create_ep4tp4_all2all_test(model_path, xpu_id)

    def test_ep4tp4_all2all_inference(
        self,
        openai_client,
        test_config: TestConfig,
    ):
        """Test EP4TP4 all2all inference.

        Args:
            openai_client: OpenAI client fixture.
            test_config: Test configuration fixture.
        """
        response = self._chat_completion(
            openai_client,
            messages=[{"role": "user", "content": "你好，你是谁？"}],
            max_tokens=64,
        )

        content = response.choices[0].message.content
        print(f"EP4TP4 all2all Response: {content}")

        self._assert_response_contains_keywords(
            content,
            test_config.expected_keywords,
            "EP4TP4 all2all推理结果不符合预期",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
