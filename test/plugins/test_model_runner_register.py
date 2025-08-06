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

from fastdeploy.plugins import load_model_runner_plugins


class TestModelRunnerRegistryPlugins(unittest.TestCase):
    def test_model_runner_callable(self):
        runner_class = load_model_runner_plugins()
        device_id = 1

        # create runner
        runner = runner_class(device_id)

        # test func
        res = runner.get_rank()

        self.assertEqual(res, device_id)


if __name__ == "__main__":
    unittest.main()
