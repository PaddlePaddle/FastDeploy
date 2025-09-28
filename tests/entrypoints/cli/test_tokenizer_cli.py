"""
Test cases for tokenizer CLI
"""

import argparse
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, PropertyMock, patch


class MockCLISubcommand:
    """模拟CLISubcommand基类"""

    pass


class MockInputPreprocessor:
    """模拟InputPreprocessor类"""

    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

    def create_processor(self):
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        return mock_processor


# 导入被测试代码，使用模拟的依赖
with patch("fastdeploy.entrypoints.cli.types.CLISubcommand", MockCLISubcommand):
    with patch("fastdeploy.input.preprocess.InputPreprocessor", MockInputPreprocessor):
        # 这里直接包含被测试的代码内容
        from fastdeploy.entrypoints.cli.tokenizer import (
            TokenizerSubcommand,
            cmd_init,
            export_vocabulary,
            get_tokenizer_info,
            get_vocab_dict,
            get_vocab_size,
            main,
        )


class TestTokenizerSubcommand(unittest.TestCase):
    """测试TokenizerSubcommand类"""

    def test_name_attribute(self):
        self.assertEqual(TokenizerSubcommand.name, "tokenizer")

    def test_subparser_init(self):
        subcommand = TokenizerSubcommand()
        mock_subparsers = MagicMock()
        mock_parser = MagicMock()
        mock_subparsers.add_parser.return_value = mock_parser

        parser = subcommand.subparser_init(mock_subparsers)

        # 验证解析器创建
        mock_subparsers.add_parser.assert_called_once_with(
            name="tokenizer",
            help="Start the FastDeploy Tokenizer Server.",
            description="Start the FastDeploy Tokenizer Server.",
            usage="fastdeploy tokenizer [--encode/-e TEXT] [--decode/-d TEXT]",
        )
        self.assertEqual(parser, mock_parser)

        # 验证参数添加（检查调用次数）
        self.assertGreater(mock_parser.add_argument.call_count, 0)

    def test_cmd_method(self):
        subcommand = TokenizerSubcommand()
        args = argparse.Namespace()

        with patch("fastdeploy.entrypoints.cli.tokenizer.main") as mock_main:
            subcommand.cmd(args)
            mock_main.assert_called_once_with(args)


class TestCmdInit(unittest.TestCase):
    """测试cmd_init函数"""

    def test_cmd_init_returns_list(self):
        result = cmd_init()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TokenizerSubcommand)


class TestGetVocabSize(unittest.TestCase):
    """测试get_vocab_size函数"""

    def test_with_vocab_size_attribute(self):
        mock_tokenizer = MagicMock()
        # 使用PropertyMock来正确模拟属性
        type(mock_tokenizer).vocab_size = PropertyMock(return_value=1000)
        result = get_vocab_size(mock_tokenizer)
        self.assertEqual(result, 1000)

    def test_with_get_vocab_size_method(self):
        mock_tokenizer = MagicMock()
        # 确保vocab_size属性不存在，让代码使用get_vocab_size方法
        delattr(mock_tokenizer, "vocab_size")
        mock_tokenizer.get_vocab_size.return_value = 2000
        result = get_vocab_size(mock_tokenizer)
        self.assertEqual(result, 2000)

    def test_with_no_methods_available(self):
        mock_tokenizer = MagicMock()
        # 移除可能的方法
        delattr(mock_tokenizer, "vocab_size")
        delattr(mock_tokenizer, "get_vocab_size")
        result = get_vocab_size(mock_tokenizer)
        self.assertEqual(result, 100295)  # 默认值

    def test_exception_handling(self):
        mock_tokenizer = MagicMock()
        # 模拟两个方法都抛出异常
        type(mock_tokenizer).vocab_size = PropertyMock(side_effect=Exception("Error"))
        mock_tokenizer.get_vocab_size.side_effect = Exception("Error")
        result = get_vocab_size(mock_tokenizer)
        self.assertEqual(result, 0)  # 默认值


class TestGetTokenizerInfo(unittest.TestCase):
    """测试get_tokenizer_info函数"""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        type(self.mock_tokenizer).vocab_size = PropertyMock(return_value=1000)
        type(self.mock_tokenizer).name_or_path = PropertyMock(return_value="test/model")
        type(self.mock_tokenizer).model_max_length = PropertyMock(return_value=512)

        # 特殊token
        type(self.mock_tokenizer).bos_token = PropertyMock(return_value="<s>")
        type(self.mock_tokenizer).eos_token = PropertyMock(return_value="</s>")
        type(self.mock_tokenizer).unk_token = PropertyMock(return_value="<unk>")
        type(self.mock_tokenizer).sep_token = PropertyMock(return_value="<sep>")
        type(self.mock_tokenizer).pad_token = PropertyMock(return_value="<pad>")
        type(self.mock_tokenizer).cls_token = PropertyMock(return_value="<cls>")
        type(self.mock_tokenizer).mask_token = PropertyMock(return_value="<mask>")

        # 特殊token ID
        type(self.mock_tokenizer).bos_token_id = PropertyMock(return_value=1)
        type(self.mock_tokenizer).eos_token_id = PropertyMock(return_value=2)
        type(self.mock_tokenizer).unk_token_id = PropertyMock(return_value=3)
        type(self.mock_tokenizer).sep_token_id = PropertyMock(return_value=4)
        type(self.mock_tokenizer).pad_token_id = PropertyMock(return_value=0)
        type(self.mock_tokenizer).cls_token_id = PropertyMock(return_value=5)
        type(self.mock_tokenizer).mask_token_id = PropertyMock(return_value=6)

    def test_normal_case(self):
        info = get_tokenizer_info(self.mock_tokenizer)

        self.assertEqual(info["vocab_size"], 1000)
        self.assertEqual(info["model_name"], "test/model")
        self.assertEqual(info["tokenizer_type"], "MagicMock")
        self.assertEqual(info["model_max_length"], 512)

        # 检查特殊token
        self.assertEqual(info["special_tokens"]["bos_token"], "<s>")
        self.assertEqual(info["special_token_ids"]["bos_token_id"], 1)

    def test_exception_handling(self):
        # 模拟在获取属性时抛出异常
        with patch("fastdeploy.entrypoints.cli.tokenizer.get_vocab_size", side_effect=Exception("Test error")):
            info = get_tokenizer_info(self.mock_tokenizer)
            self.assertIn("error", info)
            self.assertIn("Test error", info["error"])


class TestGetVocabDict(unittest.TestCase):
    """测试get_vocab_dict函数"""

    def test_vocab_attribute(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {"hello": 1, "world": 2}
        result = get_vocab_dict(mock_tokenizer)
        self.assertEqual(result, {"hello": 1, "world": 2})

    def test_get_vocab_method(self):
        mock_tokenizer = MagicMock()
        # 确保vocab属性不存在，让代码使用get_vocab方法
        delattr(mock_tokenizer, "vocab")
        mock_tokenizer.get_vocab.return_value = {"a": 1, "b": 2}
        result = get_vocab_dict(mock_tokenizer)
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_tokenizer_vocab(self):
        mock_tokenizer = MagicMock()
        # 确保vocab和get_vocab都不存在
        delattr(mock_tokenizer, "vocab")
        delattr(mock_tokenizer, "get_vocab")

        mock_inner_tokenizer = MagicMock()
        mock_inner_tokenizer.vocab = {"x": 1}
        mock_tokenizer.tokenizer = mock_inner_tokenizer

        result = get_vocab_dict(mock_tokenizer)
        self.assertEqual(result, {"x": 1})

    def test_encoder_attribute(self):
        mock_tokenizer = MagicMock()
        # 确保其他属性都不存在
        delattr(mock_tokenizer, "vocab")
        delattr(mock_tokenizer, "get_vocab")
        delattr(mock_tokenizer, "tokenizer")

        mock_tokenizer.encoder = {"token": 0}
        result = get_vocab_dict(mock_tokenizer)
        self.assertEqual(result, {"token": 0})

    def test_no_vocab_available(self):
        mock_tokenizer = MagicMock()
        # 移除所有可能的属性
        delattr(mock_tokenizer, "vocab")
        delattr(mock_tokenizer, "get_vocab")
        delattr(mock_tokenizer, "tokenizer")
        delattr(mock_tokenizer, "encoder")

        result = get_vocab_dict(mock_tokenizer)
        self.assertEqual(result, {})

    def test_exception_handling(self):
        mock_tokenizer = MagicMock()
        # 模拟所有方法都抛出异常
        mock_tokenizer.vocab = {"a": 1}
        mock_tokenizer.get_vocab.side_effect = Exception("Error")
        result = get_vocab_dict(mock_tokenizer)
        self.assertEqual(result, {"a": 1})


class TestExportVocabulary(unittest.TestCase):
    """测试export_vocabulary函数"""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab = {"hello": 1, "world": 2, "test": 3}

    def test_export_json_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "vocab.json")

            with patch("builtins.print") as mock_print:
                export_vocabulary(self.mock_tokenizer, file_path)

                # 验证文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                self.assertEqual(content, {"hello": 1, "world": 2, "test": 3})

                # 验证打印输出
                mock_print.assert_any_call(f"Vocabulary exported to: {file_path}")
                mock_print.assert_any_call("Total tokens: 3")

    def test_export_text_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "vocab.txt")

            with patch("builtins.print"):
                export_vocabulary(self.mock_tokenizer, file_path)

                # 验证文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                self.assertEqual(len(lines), 3)
                # 检查排序和格式 - 注意repr会添加引号
                self.assertIn("1\t'hello'", lines[0])
                self.assertIn("2\t'world'", lines[1])
                self.assertIn("3\t'test'", lines[2])

    def test_empty_vocabulary(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "vocab.json")

            with patch("builtins.print") as mock_print:
                export_vocabulary(mock_tokenizer, file_path)
                mock_print.assert_any_call("Warning: Could not retrieve vocabulary from tokenizer")

    def test_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "newdir", "vocab.json")

            with patch("builtins.print"):
                export_vocabulary(self.mock_tokenizer, file_path)

                # 验证目录被创建
                self.assertTrue(os.path.exists(os.path.dirname(file_path)))

    def test_exception_handling(self):
        with patch("pathlib.Path.mkdir", side_effect=Exception("Permission denied")):
            with patch("builtins.print") as mock_print:
                export_vocabulary(self.mock_tokenizer, "/invalid/path/vocab.json")
                mock_print.assert_any_call("Error exporting vocabulary: Permission denied")


class TestMainFunction(unittest.TestCase):
    """测试main函数"""

    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_preprocessor = MagicMock()
        self.mock_preprocessor.create_processor.return_value.tokenizer = self.mock_tokenizer

    def test_no_arguments(self):
        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode=None,
            vocab_size=False,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("builtins.print") as mock_print:
            main(args)
            mock_print.assert_called_with(
                "请至少指定一个参数：--encode, --decode, --vocab-size, --info, --export-vocab"
            )

    def test_encode_operation(self):
        self.mock_tokenizer.encode.return_value = [101, 102, 103]

        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode="hello world",
            decode=None,
            vocab_size=False,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("builtins.print") as mock_print:
                main(args)

                self.mock_tokenizer.encode.assert_called_once_with("hello world")
                mock_print.assert_any_call("Input text: hello world")
                mock_print.assert_any_call("Encoded tokens: [101, 102, 103]")

    def test_decode_operation_list_string(self):
        self.mock_tokenizer.decode.return_value = "decoded text"

        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode="[1,2,3]",
            vocab_size=False,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("builtins.print") as mock_print:
                main(args)

                self.mock_tokenizer.decode.assert_called_once_with([1, 2, 3])
                mock_print.assert_any_call("Decoded text: decoded text")

    def test_decode_operation_comma_string(self):
        self.mock_tokenizer.decode.return_value = "decoded text"

        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode="1,2,3",
            vocab_size=False,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("builtins.print"):
                main(args)

                self.mock_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_decode_operation_already_list(self):
        self.mock_tokenizer.decode.return_value = "decoded text"

        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode=[1, 2, 3],
            vocab_size=False,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("builtins.print"):
                main(args)
                self.mock_tokenizer.decode.assert_called_once_with([1, 2, 3])

    def test_decode_exception_handling(self):
        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode="invalid[1,2",  # 无效的字符串
            vocab_size=False,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("builtins.print") as mock_print:
                main(args)
                # 检查是否有包含"Error decoding tokens:"的打印调用
                error_calls = [
                    call for call in mock_print.call_args_list if call[0] and "Error decoding tokens:" in call[0][0]
                ]
                self.assertTrue(len(error_calls) > 0)

    def test_vocab_size_operation(self):
        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode=None,
            vocab_size=True,
            info=False,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("fastdeploy.entrypoints.cli.tokenizer.get_vocab_size", return_value=1000) as mock_get_size:
                with patch("builtins.print") as mock_print:
                    main(args)

                    mock_get_size.assert_called_once_with(self.mock_tokenizer)
                    mock_print.assert_any_call("Vocabulary size: 1000")

    def test_info_operation(self):
        mock_info = {"vocab_size": 1000, "model_name": "test"}

        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode=None,
            vocab_size=False,
            info=True,
            vocab_export=None,
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch(
                "fastdeploy.entrypoints.cli.tokenizer.get_tokenizer_info", return_value=mock_info
            ) as mock_get_info:
                with patch("builtins.print") as mock_print:
                    main(args)

                    mock_get_info.assert_called_once_with(self.mock_tokenizer)
                    mock_print.assert_any_call(json.dumps(mock_info, indent=2))

    def test_vocab_export_operation(self):
        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode=None,
            decode=None,
            vocab_size=False,
            info=False,
            vocab_export="/path/to/vocab.json",
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("fastdeploy.entrypoints.cli.tokenizer.export_vocabulary") as mock_export:
                with patch("builtins.print"):
                    main(args)

                    mock_export.assert_called_once_with(self.mock_tokenizer, "/path/to/vocab.json")

    def test_multiple_operations(self):
        self.mock_tokenizer.encode.return_value = [1]
        self.mock_tokenizer.decode.return_value = "test"

        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode="hello",
            decode="1",
            vocab_size=True,
            info=True,
            vocab_export="/path/to/vocab",
            enable_mm=False,
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=self.mock_preprocessor):
            with patch("fastdeploy.entrypoints.cli.tokenizer.get_vocab_size", return_value=100):
                with patch("fastdeploy.entrypoints.cli.tokenizer.get_tokenizer_info", return_value={"size": 100}):
                    with patch("fastdeploy.entrypoints.cli.tokenizer.export_vocabulary") as mock_export:
                        with patch("builtins.print") as mock_print:
                            main(args)

                            # 验证所有操作都被调用
                            self.assertEqual(self.mock_tokenizer.encode.call_count, 1)
                            self.assertEqual(self.mock_tokenizer.decode.call_count, 1)
                            mock_export.assert_called_once()

                            # 验证操作计数
                            mock_print.assert_any_call("Completed 5 operation(s)")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_workflow(self):
        # 测试完整的CLI工作流程
        subcommand = TokenizerSubcommand()
        mock_subparsers = MagicMock()
        mock_parser = MagicMock()
        mock_subparsers.add_parser.return_value = mock_parser

        # 初始化解析器
        subcommand.subparser_init(mock_subparsers)

        # 测试cmd方法
        args = argparse.Namespace(
            model_name_or_path="test/model", encode="test", decode=None, vocab_size=True, info=False, vocab_export=None
        )

        with patch("fastdeploy.entrypoints.cli.tokenizer.main") as mock_main:
            subcommand.cmd(args)
            mock_main.assert_called_once_with(args)

    def test_main_functionality(self):
        """测试main函数的整体功能"""
        args = argparse.Namespace(
            model_name_or_path="test/model",
            encode="hello",
            decode="1",
            vocab_size=True,
            info=True,
            vocab_export=None,
            enable_mm=False,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.decode.return_value = "hello"

        mock_preprocessor = MagicMock()
        mock_preprocessor.create_processor.return_value.tokenizer = mock_tokenizer

        with patch("fastdeploy.entrypoints.cli.tokenizer.InputPreprocessor", return_value=mock_preprocessor):
            with patch("fastdeploy.entrypoints.cli.tokenizer.get_vocab_size", return_value=1000):
                with patch("fastdeploy.entrypoints.cli.tokenizer.get_tokenizer_info", return_value={"info": "test"}):
                    with patch("builtins.print") as mock_print:
                        main(args)

                        # 验证基本功能正常
                        self.assertTrue(mock_print.called)
                        # 验证encode和decode被调用
                        mock_tokenizer.encode.assert_called_once_with("hello")
                        mock_tokenizer.decode.assert_called_once_with([1])


# class TestTokenizerCli(unittest.TestCase):
#     def setUp(self):
#         self.test_args = argparse.Namespace()
#         self.test_args.model_name_or_path = "baidu/ERNIE-4.5-0.3B-PT"
#         self.test_args.encode = "Hello, world!"
#         self.test_args.decode = "[1, 2, 3]"
#         self.test_args.vocab_size = True
#         self.test_args.info = True
#         self.tmpdir = tempfile.TemporaryDirectory()
#         self.test_args.vocab_export = os.path.join(self.tmpdir.name, "vocab.txt")

#     def tearDown(self):
#         self.tmpdir.cleanup()

#     def test_main(self):
#         result = main(self.test_args)
#         self.assertIsNotNone(result)
#         self.assertTrue(os.path.exists(self.test_args.vocab_export))


if __name__ == "__main__":
    unittest.main(verbosity=2)
