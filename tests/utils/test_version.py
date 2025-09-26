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
