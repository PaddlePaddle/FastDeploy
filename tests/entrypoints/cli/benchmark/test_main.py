"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.cli.benchmark.main import (
    BenchmarkSubcommand,
    _output_with_pager,
    cmd_init,
    show_filtered_argument_or_group_from_help,
)


class TestOutputWithPager(unittest.TestCase):
    """Test _output_with_pager function."""

    @patch("fastdeploy.entrypoints.cli.benchmark.main.subprocess.Popen")
    def test_uses_less_pager(self, mock_popen):
        """Uses 'less -R' pager successfully."""
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        _output_with_pager("hello world")

        mock_popen.assert_called_once_with(["less", "-R"], stdin=subprocess.PIPE, text=True)
        mock_proc.communicate.assert_called_once_with(input="hello world")

    @patch("fastdeploy.entrypoints.cli.benchmark.main.subprocess.Popen")
    def test_falls_back_to_more(self, mock_popen):
        """Falls back to 'more' when 'less' fails."""
        mock_proc = MagicMock()
        mock_popen.side_effect = [FileNotFoundError("less not found"), mock_proc]

        _output_with_pager("text")

        self.assertEqual(mock_popen.call_count, 2)
        mock_popen.assert_any_call(["less", "-R"], stdin=subprocess.PIPE, text=True)
        mock_popen.assert_any_call(["more"], stdin=subprocess.PIPE, text=True)
        mock_proc.communicate.assert_called_once_with(input="text")

    @patch("builtins.print")
    @patch("fastdeploy.entrypoints.cli.benchmark.main.subprocess.Popen")
    def test_falls_back_to_print(self, mock_popen, mock_print):
        """Falls back to print when all pagers fail."""
        mock_popen.side_effect = OSError("no pager")

        _output_with_pager("fallback text")

        mock_print.assert_called_once_with("fallback text")

    @patch("fastdeploy.entrypoints.cli.benchmark.main.subprocess.Popen")
    def test_subprocess_error_tries_next(self, mock_popen):
        """SubprocessError on first pager tries next."""
        mock_proc = MagicMock()
        mock_popen.side_effect = [subprocess.SubprocessError("err"), mock_proc]

        _output_with_pager("data")

        self.assertEqual(mock_popen.call_count, 2)
        mock_proc.communicate.assert_called_once_with(input="data")


class TestShowFilteredArgumentOrGroupFromHelp(unittest.TestCase):
    """Test show_filtered_argument_or_group_from_help function."""

    def _make_parser(self):
        """Create a parser with groups and arguments for testing."""
        parser = argparse.ArgumentParser(prog="fastdeploy")
        group = parser.add_argument_group("ModelConfig", "Configuration for model loading")
        group.add_argument("--max-num-seqs", type=int, default=32, help="Max sequences")
        group.add_argument("--max-model-len", type=int, default=4096, help="Max model length")

        group2 = parser.add_argument_group("SchedulerConfig", "Scheduler settings")
        group2.add_argument("--scheduler-type", type=str, default="default", help="Scheduler type")
        return parser

    @patch("sys.argv", ["fastdeploy", "serve", "--help=page"])
    def test_skips_when_subcommand_not_in_argv(self):
        """Skips processing when subcommand doesn't match sys.argv."""
        parser = self._make_parser()
        # subcommand_name is ["bench", "latency"] but sys.argv has "serve"
        show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])
        # Should return without doing anything (no sys.exit)

    @patch("sys.argv", ["fastdeploy"])
    def test_skips_when_argv_too_short(self):
        """Skips processing when sys.argv is too short for subcommand."""
        parser = self._make_parser()
        show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])
        # Should return without error

    @patch("fastdeploy.entrypoints.cli.benchmark.main._output_with_pager")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=page"])
    def test_page_outputs_help_and_exits(self, mock_pager):
        """--help=page outputs full help and exits."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 0)
        mock_pager.assert_called_once()
        # The pager receives the full help text
        self.assertIn("fastdeploy", mock_pager.call_args[0][0])

    @patch("fastdeploy.entrypoints.cli.benchmark.main._output_with_pager")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=listgroup"])
    def test_listgroup_outputs_groups_and_exits(self, mock_pager):
        """--help=listgroup lists all argument groups and exits."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 0)
        output = mock_pager.call_args[0][0]
        self.assertIn("ModelConfig", output)
        self.assertIn("SchedulerConfig", output)
        self.assertIn("Configuration for model loading", output)

    @patch("fastdeploy.entrypoints.cli.benchmark.main._output_with_pager")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=ModelConfig"])
    def test_group_search_exact_match(self, mock_pager):
        """--help=ModelConfig shows matching group and exits."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 0)
        output = mock_pager.call_args[0][0]
        self.assertIn("max-num-seqs", output)

    @patch("fastdeploy.entrypoints.cli.benchmark.main._output_with_pager")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=modelconfig"])
    def test_group_search_case_insensitive(self, mock_pager):
        """Group search is case-insensitive."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 0)
        output = mock_pager.call_args[0][0]
        self.assertIn("max-num-seqs", output)

    @patch("fastdeploy.entrypoints.cli.benchmark.main._output_with_pager")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=max-num-seqs"])
    def test_single_arg_search(self, mock_pager):
        """--help=max-num-seqs finds matching argument."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 0)
        output = mock_pager.call_args[0][0]
        self.assertIn("max-num-seqs", output)
        self.assertIn("matching", output.lower())

    @patch("fastdeploy.entrypoints.cli.benchmark.main._output_with_pager")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=max"])
    def test_partial_arg_search_matches_multiple(self, mock_pager):
        """--help=max matches multiple arguments containing 'max'."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 0)
        output = mock_pager.call_args[0][0]
        self.assertIn("max-num-seqs", output)
        self.assertIn("max-model-len", output)

    @patch("builtins.print")
    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--help=nonexistent_xyz"])
    def test_no_match_prints_error_and_exits_1(self, mock_print):
        """No matching group or arg prints error and exits with code 1."""
        parser = self._make_parser()

        with self.assertRaises(SystemExit) as ctx:
            show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])

        self.assertEqual(ctx.exception.code, 1)
        # Check that error message was printed
        calls = [str(c) for c in mock_print.call_args_list]
        joined = " ".join(calls)
        self.assertIn("nonexistent_xyz", joined)

    @patch("sys.argv", ["fastdeploy", "bench", "latency", "--other-arg", "value"])
    def test_no_help_arg_does_nothing(self):
        """No --help= argument returns without action."""
        parser = self._make_parser()
        # Should return without SystemExit
        show_filtered_argument_or_group_from_help(parser, ["bench", "latency"])


class TestBenchmarkSubcommandCmd(unittest.TestCase):
    """Test BenchmarkSubcommand.cmd."""

    def test_cmd_calls_dispatch_function(self):
        """cmd() calls args.dispatch_function(args)."""
        args = MagicMock()
        BenchmarkSubcommand.cmd(args)
        args.dispatch_function.assert_called_once_with(args)


class TestBenchmarkSubcommandValidate(unittest.TestCase):
    """Test BenchmarkSubcommand.validate."""

    def test_validate_does_nothing(self):
        """validate() is a no-op."""
        subcmd = BenchmarkSubcommand()
        args = MagicMock()
        # Should not raise
        subcmd.validate(args)


class TestBenchmarkSubcommandSubparserInit(unittest.TestCase):
    """Test BenchmarkSubcommand.subparser_init."""

    @patch("fastdeploy.entrypoints.cli.benchmark.main.show_filtered_argument_or_group_from_help")
    @patch("fastdeploy.entrypoints.cli.benchmark.main.BenchmarkSubcommandBase.__subclasses__")
    def test_subparser_init_registers_subcommands(self, mock_subclasses, mock_show_help):
        """subparser_init registers benchmark subcommands."""
        # Create a mock subcommand class
        mock_cmd_cls = MagicMock()
        mock_cmd_cls.name = "latency"
        mock_cmd_cls.help = "Run latency benchmark"
        mock_subclasses.return_value = [mock_cmd_cls]

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        subcmd = BenchmarkSubcommand()
        result = subcmd.subparser_init(subparsers)

        self.assertIsNotNone(result)
        mock_cmd_cls.add_cli_args.assert_called_once()
        mock_show_help.assert_called_once()


class TestCmdInit(unittest.TestCase):
    """Test cmd_init function."""

    def test_returns_list_with_benchmark_subcommand(self):
        """cmd_init returns a list containing BenchmarkSubcommand."""
        result = cmd_init()
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], BenchmarkSubcommand)


if __name__ == "__main__":
    unittest.main()
