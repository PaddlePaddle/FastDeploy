import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.cli.main import main as cli_main


class TestCliMain(unittest.TestCase):
    @patch("fastdeploy.utils.FlexibleArgumentParser")
    @patch("fastdeploy.entrypoints.cli.main.importlib.metadata")
    def test_main_basic(self, mock_metadata, mock_parser):
        # Setup mocks
        mock_metadata.version.return_value = "1.0.0"
        mock_args = MagicMock()
        mock_args.subparser = None
        mock_parser.return_value.parse_args.return_value = mock_args

        # Test basic call
        cli_main()

        # Verify version check
        mock_metadata.version.assert_called_once_with("fastdeploy-gpu")
        mock_args.dispatch_function.assert_called_once()


if __name__ == "__main__":
    unittest.main()
