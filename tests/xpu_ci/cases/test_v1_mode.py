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

"""V1 Mode Test for XPU CI.

This test validates the V1 mode inference with ERNIE-4.5-300B model
using wint4 quantization.
"""

import pytest

from ..core import TestConfig
from ..core.base_test import TextModelTest


@pytest.mark.v1_mode
class TestV1Mode(TextModelTest):
    """V1 Mode test class.

    Tests the V1 mode inference functionality with:
    - ERNIE-4.5-300B-A47B-Paddle model
    - wint4 quantization
    - Prefix caching enabled
    - Chunked prefill enabled
    """

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get V1 mode test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig for V1 mode test.
        """
        return TestConfig.create_v1_test(model_path, xpu_id)

    def test_basic_inference(
        self,
        openai_client,
        test_config: TestConfig,
    ):
        """Test basic V1 mode inference.

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
        print(f"V1 Mode Response: {content}")

        self._assert_response_contains_keywords(
            content,
            test_config.expected_keywords,
            "V1模式推理结果不符合预期",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
