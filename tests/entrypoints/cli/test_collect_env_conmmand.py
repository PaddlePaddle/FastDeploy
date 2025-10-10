import unittest
from argparse import Namespace, _SubParsersAction
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.cli.collect_env import CollectEnvSubcommand, cmd_init


class TestCollectEnvSubcommand(unittest.TestCase):
    def setUp(self):
        self.subcommand = CollectEnvSubcommand()

    def test_name_property(self):
        self.assertEqual(self.subcommand.name, "collect-env")

    @patch("fastdeploy.entrypoints.cli.collect_env.collect_env_main")
    def test_cmd(self, mock_collect_env_main):
        args = Namespace()
        self.subcommand.cmd(args)
        mock_collect_env_main.assert_called_once()

    def test_subparser_init(self):
        mock_subparsers = MagicMock(spec=_SubParsersAction)
        parser = self.subcommand.subparser_init(mock_subparsers)
        print(parser)
        mock_subparsers.add_parser.assert_called_once_with(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="fastdeploy collect-env",
        )


class TestCmdInit(unittest.TestCase):
    def test_cmd_init(self):
        subcommands = cmd_init()
        self.assertEqual(len(subcommands), 1)
        self.assertIsInstance(subcommands[0], CollectEnvSubcommand)


if __name__ == "__main__":
    unittest.main()
