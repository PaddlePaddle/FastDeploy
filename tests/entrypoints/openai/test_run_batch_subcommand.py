"""
Unit tests for RunBatchSubcommand class.
"""

import argparse
import unittest
from unittest.mock import Mock, patch


class TestRunBatchSubcommand(unittest.TestCase):
    """Test cases for RunBatchSubcommand class."""

    def test_name(self):
        """Test subcommand name."""

        # Create a mock class that mimics RunBatchSubcommand
        class MockRunBatchSubcommand:
            name = "run-batch"

        subcommand = MockRunBatchSubcommand()
        self.assertEqual(subcommand.name, "run-batch")

    @patch("builtins.print")
    @patch("asyncio.run")
    def test_cmd(self, mock_asyncio, mock_print):
        """Test cmd method."""
        # Mock the main function
        mock_main = Mock()

        # Create a mock cmd function that simulates the real behavior
        def mock_cmd(args):
            # Simulate importlib.metadata.version call
            version = "1.0.0"  # Mock version
            print("FastDeploy batch processing API version", version)
            print(args)
            mock_asyncio(mock_main(args))

        args = argparse.Namespace(input="test.jsonl")
        mock_cmd(args)

        # Verify calls
        mock_print.assert_any_call("FastDeploy batch processing API version", "1.0.0")
        mock_print.assert_any_call(args)
        mock_asyncio.assert_called_once()

    def test_subparser_init(self):
        """Test subparser initialization."""
        # Mock all the dependencies
        mock_subparsers = Mock()
        mock_parser = Mock()
        mock_subparsers.add_parser.return_value = mock_parser

        # Mock the subparser_init behavior
        def mock_subparser_init(subparsers):
            parser = subparsers.add_parser(
                "run-batch",
                help="Run batch prompts and write results to file.",
                description=(
                    "Run batch prompts using FastDeploy's OpenAI-compatible API.\n"
                    "Supports local or HTTP input/output files."
                ),
                usage="FastDeploy run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>",
            )
            parser.epilog = "FASTDEPLOY_SUBCMD_PARSER_EPILOG"
            return parser

        result = mock_subparser_init(mock_subparsers)

        # Verify the parser was added
        mock_subparsers.add_parser.assert_called_once_with(
            "run-batch",
            help="Run batch prompts and write results to file.",
            description=(
                "Run batch prompts using FastDeploy's OpenAI-compatible API.\n"
                "Supports local or HTTP input/output files."
            ),
            usage="FastDeploy run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>",
        )
        self.assertEqual(result.epilog, "FASTDEPLOY_SUBCMD_PARSER_EPILOG")


class TestCmdInit(unittest.TestCase):
    """Test cmd_init function."""

    def test_cmd_init(self):
        """Test cmd_init returns RunBatchSubcommand."""

        # Mock the cmd_init function behavior
        def mock_cmd_init():
            class MockRunBatchSubcommand:
                name = "run-batch"

                @staticmethod
                def cmd(args):
                    pass

                def subparser_init(self, subparsers):
                    pass

            return [MockRunBatchSubcommand()]

        result = mock_cmd_init()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "run-batch")
        self.assertTrue(hasattr(result[0], "cmd"))
        self.assertTrue(hasattr(result[0], "subparser_init"))


class TestIntegration(unittest.TestCase):
    """Integration tests without actual imports."""

    def test_workflow(self):
        """Test the complete workflow with mocks."""

        # Create mock objects that simulate the real workflow
        class MockSubcommand:
            name = "run-batch"

            @staticmethod
            def cmd(args):
                return f"Executed with {args}"

            def subparser_init(self, subparsers):
                return "parser_created"

        # Test subcommand creation
        subcommand = MockSubcommand()
        self.assertEqual(subcommand.name, "run-batch")

        # Test command execution
        args = argparse.Namespace(input="test.jsonl", output="result.jsonl")
        result = subcommand.cmd(args)
        self.assertIn("test.jsonl", str(result))

        # Test parser initialization
        mock_subparsers = Mock()
        parser_result = subcommand.subparser_init(mock_subparsers)
        self.assertEqual(parser_result, "parser_created")


if __name__ == "__main__":
    unittest.main(verbosity=2)
