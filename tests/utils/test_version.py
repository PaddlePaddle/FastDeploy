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
from unittest.mock import mock_open, patch

import fastdeploy
from fastdeploy.utils import current_package_version, get_version_info


class TestVersion(unittest.TestCase):
    def test_get_version(self):
        ver = fastdeploy.version()
        assert ver.count("COMMIT") > 0

    @patch("builtins.open", new_callable=mock_open, read_data="fastdeploy version: 1.0.0\nother info")
    def test_normal_version(self, mock_file):
        """测试正常版本号解析"""
        self.assertEqual(current_package_version(), "1.0.0")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_file_not_found(self, mock_file):
        """测试文件不存在的情况"""
        self.assertEqual(current_package_version(), "Unknown")

    @patch("builtins.open", new_callable=mock_open, read_data="some other content")
    def test_no_version_line(self, mock_file):
        """测试找不到版本行的情况"""
        self.assertEqual(current_package_version(), "Unknown")

    @patch("builtins.open", new_callable=mock_open, read_data="""fastdeploy GIT COMMIT ID: 23d488c488779fdda73b427b49f6be40cf4408ba
Paddle version: 3.3.0.dev20251222
Paddle GIT COMMIT ID: f68bb752a51aacd333d74336e6ee62b7b3b21231
CUDA version: 12.6
CXX compiler version: 11.2.1
fastdeploy version: 2.4.0.dev20251223""")
    def test_get_version_info(self, mock_file):
        """测试get_version_info函数"""
        version_info = get_version_info()
        self.assertIsNotNone(version_info)
        self.assertEqual(version_info["fastdeploy_commit"], "23d488c488779fdda73b427b49f6be40cf4408ba")
        self.assertEqual(version_info["paddle_version"], "3.3.0.dev20251222")
        self.assertEqual(version_info["paddle_commit"], "f68bb752a51aacd333d74336e6ee62b7b3b21231")
        self.assertEqual(version_info["cuda_version"], "12.6")
        self.assertEqual(version_info["cxx_version"], "11.2.1")
        self.assertEqual(version_info["fastdeploy_version"], "2.4.0.dev20251223")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_get_version_info_file_not_found(self, mock_file):
        """测试get_version_info在文件不存在时返回None"""
        version_info = get_version_info()
        self.assertIsNone(version_info)

    @patch("builtins.open", new_callable=mock_open, read_data="invalid content")
    def test_get_version_info_empty_dict(self, mock_file):
        """测试get_version_info在内容无效时返回None"""
        version_info = get_version_info()
        self.assertIsNone(version_info)


if __name__ == "__main__":
    unittest.main()
