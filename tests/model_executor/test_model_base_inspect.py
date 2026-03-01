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

"""
Tests for ModelRegistry.inspect_model_cls error handling.

Covers:
- model_base.py line 248: ValueError for empty architectures
"""

import unittest

from fastdeploy.model_executor.models.model_base import ModelRegistry


class TestInspectModelClsEmptyArchitectures(unittest.TestCase):
    """Test inspect_model_cls raises ValueError for empty architectures (L248)."""

    def test_empty_list_raises_valueerror(self):
        """Empty architectures list triggers ValueError (L248)."""
        registry = object.__new__(ModelRegistry)
        with self.assertRaises(ValueError) as ctx:
            registry.inspect_model_cls([], model_config=None)
        self.assertIn("No model architectures are specified", str(ctx.exception))
        self.assertIn("config.json", str(ctx.exception))

    def test_none_coerced_to_empty_list(self):
        """Falsy architectures also triggers ValueError."""
        registry = object.__new__(ModelRegistry)
        with self.assertRaises(ValueError) as ctx:
            registry.inspect_model_cls([], model_config=None)
        self.assertIn("No model architectures are specified", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
