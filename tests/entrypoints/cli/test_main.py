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
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.cli.main import main as cli_main


class TestCliMain(unittest.TestCase):
    @patch("fastdeploy.utils.FlexibleArgumentParser")
    def test_main_basic(self, mock_parser):
        # Setup mocks
        mock_args = MagicMock()
        mock_args.subparser = None
        mock_parser.return_value.parse_args.return_value = mock_args

        # Test basic call
        cli_main()

        # Verify version check
        mock_args.dispatch_function.assert_called_once()


if __name__ == "__main__":
    unittest.main()
