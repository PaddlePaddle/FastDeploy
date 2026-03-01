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
Tests for ResourceManager._get_block_tables error handling.

Covers:
- resource_manager.py line 134: ValueError for invalid required_type
"""

import unittest

from fastdeploy.engine.resource_manager import ResourceManager


class TestGetBlockTablesInvalidType(unittest.TestCase):
    """Test _get_block_tables raises ValueError for unknown required_type (L134)."""

    def test_invalid_required_type_raises_valueerror(self):
        """Passing an invalid required_type triggers the else branch (L134)."""
        manager = object.__new__(ResourceManager)  # bypass __init__
        with self.assertRaises(ValueError) as ctx:
            manager._get_block_tables(input_token_num=100, required_type="invalid")
        self.assertIn("unknown required type", str(ctx.exception))
        self.assertIn("invalid", str(ctx.exception))
        self.assertIn("'all', 'encoder', or 'decoder'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
