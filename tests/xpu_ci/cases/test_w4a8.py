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

"""W4A8 Quantization Test for XPU CI.

This test validates the W4A8 quantization inference with ERNIE-4.5-300B model.
"""

import pytest

from ..core import TestConfig
from ..core.base_test import TextModelTest


@pytest.mark.w4a8
class TestW4A8(TextModelTest):
    """W4A8 quantization test class.

    Tests the W4A8 quantization inference functionality with:
    - ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle model
    - W4A8 quantization
    """

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get W4A8 test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig for W4A8 test.
        """
        return TestConfig.create_w4a8_test(model_path, xpu_id)

    def test_w4a8_inference(
        self,
        openai_client,
        test_config: TestConfig,
    ):
        """Test W4A8 quantization inference.

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
        print(f"W4A8 Response: {content}")

        self._assert_response_contains_keywords(
            content,
            test_config.expected_keywords,
            "W4A8量化推理结果不符合预期",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
