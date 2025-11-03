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
