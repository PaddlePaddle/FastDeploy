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

"""VL Model Test for XPU CI.

This test validates the VL (Vision-Language) model inference with
ERNIE-4.5-VL-28B model.
"""

import pytest

from ..core import TestConfig
from ..core.base_test import VLModelTest


@pytest.mark.vl_model
class TestVLModel(VLModelTest):
    """VL model test class.

    Tests the VL model inference functionality with:
    - ERNIE-4.5-VL-28B-A3B-Paddle model
    - wint8 quantization
    - Multi-modal enabled
    """

    @classmethod
    def get_test_config(cls, model_path: str, xpu_id: int) -> TestConfig:
        """Get VL model test configuration.

        Args:
            model_path: Base path to model files.
            xpu_id: XPU device ID.

        Returns:
            TestConfig for VL model test.
        """
        return TestConfig.create_vl_test(model_path, xpu_id)

    def test_vl_image_understanding(
        self,
        openai_client,
        test_config: TestConfig,
    ):
        """Test VL model image understanding.

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

        content = response.choices[0].message.content
        print(f"VL Model Response: {content}")

        self._assert_response_contains_keywords(
            content,
            test_config.expected_keywords,
            "VL模型推理结果不符合预期",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
