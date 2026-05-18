"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from utils import FakeModelConfig, get_default_test_fd_config

from fastdeploy.config import SpeculativeConfig
from fastdeploy.spec_decode.types import SpecMethod


class TestNaiveCreateProposer(unittest.TestCase):
    def setUp(self):
        self.fd_config = get_default_test_fd_config()
        self.fd_config.model_config = FakeModelConfig()
        self.fd_config.speculative_config = SpeculativeConfig({"method": "naive", "num_speculative_tokens": 1})

    def test_create_proposer_returns_none(self):
        proposer = SpecMethod.NAIVE.create_proposer(self.fd_config)
        self.assertIsNone(proposer)

    def test_needs_proposer_returns_false(self):
        self.assertFalse(SpecMethod.NAIVE.needs_proposer)


if __name__ == "__main__":
    unittest.main()
