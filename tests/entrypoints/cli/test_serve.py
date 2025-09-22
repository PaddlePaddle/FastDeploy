import argparse
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.cli.serve import ServeSubcommand, cmd_init


class TestServeSubcommand(unittest.TestCase):
    """Tests for ServeSubcommand class."""

    def test_name_property(self):
        """Test the name property is correctly set."""
        self.assertEqual(ServeSubcommand.name, "serve")

    @patch("fastdeploy.entrypoints.cli.serve.api_server")
    def test_cmd_method(self, mock_api_server):
        """Test the cmd method calls the expected API server functions."""
        test_args = argparse.Namespace()
        ServeSubcommand.cmd(test_args)
        mock_api_server.main.assert_called_once_with(test_args)

    def test_validate_method(self):
        """Test the validate method does nothing (no-op)."""
        test_args = argparse.Namespace()
        instance = ServeSubcommand()
        instance.validate(test_args)  # Should not raise any exceptions

    @patch("argparse._SubParsersAction.add_parser")
    def test_subparser_init(self, mock_add_parser):
        """Test the subparser initialization."""
        mock_subparsers = MagicMock()
        instance = ServeSubcommand()
        result = instance.subparser_init(mock_subparsers)
        self.assertIsNotNone(result)

    def test_cmd_init_returns_list(self):
        """Test cmd_init returns a list of subcommands."""
        result = cmd_init()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ServeSubcommand)


if __name__ == "__main__":
    unittest.main()
