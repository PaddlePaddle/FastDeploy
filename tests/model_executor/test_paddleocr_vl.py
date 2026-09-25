"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import unittest
from unittest.mock import MagicMock

import paddle

from fastdeploy.model_executor.models.paddleocr_vl.paddleocr_vl import (
    PaddleOCRVLForConditionalGeneration,
)


IMAGE_TOKEN_ID = 101304


class TestPaddleOCRVLForConditionalGeneration(unittest.TestCase):
    """Test PaddleOCRVLForConditionalGeneration class."""

    def _make_model(self):
        model = PaddleOCRVLForConditionalGeneration.__new__(PaddleOCRVLForConditionalGeneration)
        model.model = MagicMock()
        model.model.config.image_token_id = IMAGE_TOKEN_ID
        model.model.get_input_embeddings = MagicMock(
            return_value=paddle.zeros([5, 768], dtype=paddle.float32)
        )
        model._dtype = paddle.float32
        return model

    def test_get_input_embeddings_no_crash_when_image_features_none_without_image_tokens(self):
        model = self._make_model()
        ids = paddle.to_tensor([[1, 2, 3, 4, 5]])
        result = model.get_input_embeddings(
            ids_remove_padding=ids, image_features=None
        )
        self.assertIsNotNone(result)

    def test_get_input_embeddings_no_crash_when_image_features_none_with_image_tokens(self):
        model = self._make_model()
        ids = paddle.to_tensor([[1, IMAGE_TOKEN_ID, 3, 4, 5]])
        result = model.get_input_embeddings(
            ids_remove_padding=ids, image_features=None
        )
        self.assertIsNotNone(result)

    def test_get_input_embeddings_with_image_features_replaces_image_tokens(self):
        model = self._make_model()
        ids = paddle.to_tensor([[1, IMAGE_TOKEN_ID, 3, 4, 5]])
        image_features = paddle.ones([1, 768], dtype=paddle.float32)
        result = model.get_input_embeddings(
            ids_remove_padding=ids, image_features=image_features
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, [5, 768])
        self.assertTrue(paddle.allclose(result[1], paddle.ones([768])))


if __name__ == "__main__":
    unittest.main()
