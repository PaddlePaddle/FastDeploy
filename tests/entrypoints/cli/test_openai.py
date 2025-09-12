import argparse
import signal
import unittest
from unittest.mock import MagicMock, call, patch

from fastdeploy.entrypoints.cli.openai import (
    ChatCommand,
    CompleteCommand,
    _add_query_options,
    _interactive_cli,
    _register_signal_handlers,
    chat,
    cmd_init,
)


class TestOpenAICli(unittest.TestCase):

    @patch("fastdeploy.entrypoints.cli.openai.signal.signal")
    def test_register_signal_handlers(self, mock_signal):
        """测试信号处理器注册"""
        _register_signal_handlers()

        # 验证信号处理器被正确注册
        mock_signal.assert_has_calls([call(signal.SIGINT, unittest.mock.ANY), call(signal.SIGTSTP, unittest.mock.ANY)])

    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    @patch("fastdeploy.entrypoints.cli.openai.os.environ.get")
    @patch("fastdeploy.entrypoints.cli.openai._register_signal_handlers")
    def test_interactive_cli_with_model_name(self, mock_register, mock_environ, mock_openai):
        """测试交互式CLI初始化（指定模型名）"""
        # 设置mock
        mock_environ.return_value = "test_api_key"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # 测试参数
        args = argparse.Namespace()
        args.url = "http://localhost:9904/v1"
        args.api_key = None
        args.model_name = "test-model"

        # 执行测试
        model_name, client = _interactive_cli(args)

        # 验证结果
        self.assertEqual(model_name, "test-model")
        self.assertEqual(client, mock_client)
        mock_openai.assert_called_once_with(api_key="test_api_key", base_url="http://localhost:9904/v1")
        mock_register.assert_called_once()

    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    @patch("fastdeploy.entrypoints.cli.openai.os.environ.get")
    @patch("fastdeploy.entrypoints.cli.openai._register_signal_handlers")
    def test_interactive_cli_without_model_name(self, mock_register, mock_environ, mock_openai):
        """测试交互式CLI初始化（未指定模型名）"""
        # 设置mock
        mock_environ.return_value = "test_api_key"
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_models.data = [MagicMock(id="first-model")]
        mock_client.models.list.return_value = mock_models
        mock_openai.return_value = mock_client

        # 测试参数
        args = argparse.Namespace()
        args.url = "http://localhost:9904/v1"
        args.api_key = None
        args.model_name = None

        # 执行测试
        model_name, client = _interactive_cli(args)

        # 验证结果
        self.assertEqual(model_name, "first-model")
        self.assertEqual(client, mock_client)
        mock_client.models.list.assert_called_once()

    @patch("fastdeploy.entrypoints.cli.openai.input")
    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    def test_chat_function(self, mock_openai, mock_input):
        """测试chat函数"""
        # 设置mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock(content="Test response")
        mock_client.chat.completions.create.return_value = mock_completion

        # 模拟用户输入和EOF
        mock_input.side_effect = ["Hello", EOFError]

        # 执行测试
        chat("System prompt", "test-model", mock_client)

        # 验证API调用
        mock_client.chat.completions.create.assert_called_once()

    def test_add_query_options(self):
        """测试查询选项添加"""
        mock_parser = MagicMock()

        result = _add_query_options(mock_parser)

        # 验证parser方法被调用
        self.assertEqual(result, mock_parser)
        self.assertEqual(mock_parser.add_argument.call_count, 3)

    def test_cmd_init(self):
        """测试命令初始化"""
        commands = cmd_init()

        # 验证返回的命令列表
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0].name, "chat")
        self.assertEqual(commands[1].name, "complete")


class TestChatCommand(unittest.TestCase):

    @patch("fastdeploy.entrypoints.cli.openai._interactive_cli")
    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    def test_chat_command_quick_mode(self, mock_openai, mock_interactive):
        """测试ChatCommand快速模式"""
        # 设置mock
        mock_interactive.return_value = ("test-model", MagicMock())

        args = argparse.Namespace()
        args.quick = "Quick message"
        args.system_prompt = None

        # 执行测试
        ChatCommand.cmd(args)

        # 验证_interactive_cli被调用
        mock_interactive.assert_called_once_with(args)

    @patch("fastdeploy.entrypoints.cli.openai.input")
    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    def test_chat_empty_input(self, mock_openai, mock_input):
        """Test chat with empty input."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock empty input then EOF
        mock_input.side_effect = ["", EOFError()]

        args = argparse.Namespace()
        args.quick = None
        args.url = "http://test.com"
        args.api_key = None

        args.model_name = None
        args.system_prompt = "System prompt"

        ChatCommand().cmd(args)


class TestCompleteCommand(unittest.TestCase):

    @patch("fastdeploy.entrypoints.cli.openai._interactive_cli")
    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    def test_complete_command_quick_mode(self, mock_openai, mock_interactive):
        """测试CompleteCommand快速模式"""
        # 设置mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(text="Completion text")]
        mock_client.completions.create.return_value = mock_completion
        mock_interactive.return_value = ("test-model", mock_client)

        args = argparse.Namespace()
        args.quick = "Quick prompt"

        # 执行测试
        CompleteCommand.cmd(args)

        # 验证API调用
        mock_client.completions.create.assert_called_once_with(model="test-model", prompt="Quick prompt")

    @patch("fastdeploy.entrypoints.cli.openai.input")
    @patch("fastdeploy.entrypoints.cli.openai.OpenAI")
    def test_completion_empty_input(self, mock_openai, mock_input):
        """Test completion with empty input."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock empty input then EOF
        mock_input.side_effect = ["", EOFError()]

        args = argparse.Namespace()
        args.quick = None
        args.url = "http://test.com"
        args.api_key = None
        args.model_name = None

        CompleteCommand.cmd(args)


if __name__ == "__main__":
    unittest.main()
