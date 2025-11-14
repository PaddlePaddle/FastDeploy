"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import patch

import fastdeploy
from fastdeploy.utils import current_package_version


class TestVersion(unittest.TestCase):
    def test_get_version(self):
        ver = fastdeploy.version()
        assert ver.count("COMMIT") > 0

    @patch("fastdeploy.utils.version")
    def test_normal_version(self, mock_version):
        """测试正常版本号解析"""
        mock_version.return_value = "fastdeploy version: 1.0.0\nother info"
        self.assertEqual(current_package_version(), "1.0.0")

    @patch("fastdeploy.utils.version")
    def test_unknown_version(self, mock_version):
        """测试version返回Unknown的情况"""
        mock_version.return_value = "Unknown"
        self.assertEqual(current_package_version(), "Unknown")

    @patch("fastdeploy.utils.version")
    def test_no_version_line(self, mock_version):
        """测试找不到版本行的情况"""
        mock_version.return_value = "some other content"
        self.assertEqual(current_package_version(), "Unknown")


if __name__ == "__main__":
    unittest.main()
