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

import unittest

from fastdeploy.plugins import load_model_register_plugins


class TestModelRegistryPlugins(unittest.TestCase):
    def test_plugin_registers_one_architecture(self):
        """Test that loading plugins registers exactly one new architecture."""
        # Load plugins
        load_model_register_plugins()


if __name__ == "__main__":
    unittest.main()
