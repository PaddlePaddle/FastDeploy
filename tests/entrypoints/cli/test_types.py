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

import unittest
from unittest.mock import MagicMock

from fastdeploy.entrypoints.cli.types import CLISubcommand


class TestCLISubcommand(unittest.TestCase):
    """Test cases for CLISubcommand class."""

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CLISubcommand.cmd(None)

        with self.assertRaises(NotImplementedError):
            CLISubcommand().subparser_init(None)

    def test_validate_default_implementation(self):
        """Test the default validate implementation does nothing."""
        # Should not raise any exception
        CLISubcommand().validate(None)

    def test_name_attribute(self):
        """Test that name attribute is required."""

        class TestSubcommand(CLISubcommand):
            name = "test"

            @staticmethod
            def cmd(args):
                pass

            def subparser_init(self, subparsers):
                return MagicMock()

        # Should not raise AttributeError
        test_cmd = TestSubcommand()
        self.assertEqual(test_cmd.name, "test")


if __name__ == "__main__":
    unittest.main()
