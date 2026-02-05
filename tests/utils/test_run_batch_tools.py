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

import argparse
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.utils import (
    FASTDEPLOY_SUBCMD_PARSER_EPILOG,
    show_filtered_argument_or_group_from_help,
)


class TestHelpFilter(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser(prog="fastdeploy", epilog=FASTDEPLOY_SUBCMD_PARSER_EPILOG)
        self.subcommand = ["bench"]
        self.mock_sys_argv = ["fastdeploy"] + self.subcommand

        # Add test groups and arguments
        self.model_group = self.parser.add_argument_group("ModelConfig", "Model configuration parameters")
        self.model_group.add_argument("--model-path", help="Path to model")
        self.model_group.add_argument("--max-num-seqs", help="Max sequences")

        self.train_group = self.parser.add_argument_group("TrainingConfig", "Training parameters")
        self.train_group.add_argument("--epochs", help="Training epochs")

    @patch("sys.argv", ["fastdeploy", "bench", "--help=page"])
    @patch("subprocess.Popen")
    def test_page_help(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (None, None)
        mock_popen.return_value = mock_proc

        # Expect SystemExit with code 0
        with self.assertRaises(SystemExit) as cm:
            show_filtered_argument_or_group_from_help(self.parser, self.subcommand)

        self.assertEqual(cm.exception.code, 0)
        mock_popen.assert_called_once()

    @patch("sys.argv", ["fastdeploy", "bench", "--help=listgroup"])
    @patch("fastdeploy.utils._output_with_pager")
    def test_list_groups(self, mock_output):
        # Expect SystemExit with code 0
        with self.assertRaises(SystemExit) as cm:
            show_filtered_argument_or_group_from_help(self.parser, self.subcommand)

        self.assertEqual(cm.exception.code, 0)
        # Verify that the output function was called
        mock_output.assert_called_once()
        # Check that the output contains expected groups
        output_text = mock_output.call_args[0][0]
        self.assertIn("ModelConfig", output_text)
        self.assertIn("TrainingConfig", output_text)

    @patch("sys.argv", ["fastdeploy", "bench", "--help=ModelConfig"])
    @patch("fastdeploy.utils._output_with_pager")
    def test_group_search(self, mock_output):
        # Expect SystemExit with code 0
        with self.assertRaises(SystemExit) as cm:
            show_filtered_argument_or_group_from_help(self.parser, self.subcommand)

        self.assertEqual(cm.exception.code, 0)
        # Verify that the output function was called
        mock_output.assert_called_once()
        # Check that the output contains expected content
        output_text = mock_output.call_args[0][0]
        self.assertIn("ModelConfig", output_text)
        self.assertIn("--model-path", output_text)

    @patch("sys.argv", ["fastdeploy", "bench", "--help=max"])
    @patch("fastdeploy.utils._output_with_pager")
    def test_arg_search(self, mock_output):
        # Expect SystemExit with code 0
        with self.assertRaises(SystemExit) as cm:
            show_filtered_argument_or_group_from_help(self.parser, self.subcommand)

        self.assertEqual(cm.exception.code, 0)
        # Verify that the output function was called
        mock_output.assert_called_once()
        # Check that the output contains expected content
        output_text = mock_output.call_args[0][0]
        self.assertIn("--max-num-seqs", output_text)
        self.assertNotIn("--epochs", output_text)

    @patch("sys.argv", ["fastdeploy", "bench", "--help=nonexistent"])
    @patch("builtins.print")
    def test_no_match(self, mock_print):
        # Expect SystemExit with code 1 (error case)
        with self.assertRaises(SystemExit) as cm:
            show_filtered_argument_or_group_from_help(self.parser, self.subcommand)

        self.assertEqual(cm.exception.code, 1)
        # Check that error message was printed
        mock_print.assert_called()
        call_args = [call.args[0] for call in mock_print.call_args_list]
        self.assertTrue(any("No group or parameter matching" in arg for arg in call_args))

    @patch("sys.argv", ["fastdeploy", "othercmd"])
    def test_wrong_subcommand(self):
        # This should not raise SystemExit, just return normally
        try:
            show_filtered_argument_or_group_from_help(self.parser, self.subcommand)
        except SystemExit:
            self.fail("Function should not exit when subcommand doesn't match")


if __name__ == "__main__":
    unittest.main()
