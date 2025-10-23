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

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path

from fastdeploy.config import ModelConfig
from fastdeploy.entrypoints.cli.types import CLISubcommand
from fastdeploy.input.preprocess import InputPreprocessor

if typing.TYPE_CHECKING:
    from fastdeploy.utils import FlexibleArgumentParser


class TokenizerSubcommand(CLISubcommand):
    """The `tokenizer` subcommand for the FastDeploy CLI."""

    name = "tokenizer"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        tokenizer_parser = subparsers.add_parser(
            name=self.name,
            help="Start the FastDeploy Tokenizer Server.",
            description="Start the FastDeploy Tokenizer Server.",
            usage="fastdeploy tokenizer [--encode/-e TEXT] [--decode/-d TEXT]",
        )

        # 添加通用参数
        tokenizer_parser.add_argument(
            "--model_name_or_path",
            "--model",
            "-m",
            type=str,
            default="baidu/ERNIE-4.5-0.3B-PT",
            help="Path to model or model identifier",
        )
        tokenizer_parser.add_argument("--enable-mm", "-mm", action="store_true", help="Enable multi-modal support")
        tokenizer_parser.add_argument("--vocab-size", "-vs", action="store_true", help="Show vocabulary size")
        tokenizer_parser.add_argument("--info", "-i", action="store_true", help="Show tokenizer information")
        tokenizer_parser.add_argument(
            "--vocab-export", "-ve", type=str, metavar="FILE", help="Export vocabulary to file"
        )
        tokenizer_parser.add_argument("--encode", "-e", default=None, help="Encode text to tokens")
        tokenizer_parser.add_argument("--decode", "-d", default=None, help="Decode tokens to text")

        return tokenizer_parser


def cmd_init() -> list[CLISubcommand]:
    return [TokenizerSubcommand()]


def get_vocab_size(tokenizer) -> int:
    """获取词表大小"""
    try:
        if hasattr(tokenizer, "vocab_size"):
            return tokenizer.vocab_size
        elif hasattr(tokenizer, "get_vocab_size"):
            return tokenizer.get_vocab_size()
        else:
            return 100295  # Ernie4_5Tokenizer的固定词表大小
    except Exception:
        return 0


def get_tokenizer_info(tokenizer) -> dict:
    """获取tokenizer的元信息"""
    info = {}

    try:
        # 基本属性
        info["vocab_size"] = get_vocab_size(tokenizer)

        # 模型类型和路径
        if hasattr(tokenizer, "name_or_path"):
            info["model_name"] = tokenizer.name_or_path

        # tokenizer类型
        info["tokenizer_type"] = type(tokenizer).__name__

        # 特殊符号
        special_tokens = {}
        for attr in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]:
            if hasattr(tokenizer, attr):
                token = getattr(tokenizer, attr)
                if token:
                    special_tokens[attr] = token
        info["special_tokens"] = special_tokens

        # 特殊token IDs
        special_token_ids = {}
        for attr in [
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
            "sep_token_id",
            "pad_token_id",
            "cls_token_id",
            "mask_token_id",
        ]:
            if hasattr(tokenizer, attr):
                token_id = getattr(tokenizer, attr)
                if token_id is not None:
                    special_token_ids[attr] = token_id
        info["special_token_ids"] = special_token_ids

        # 模型最大长度
        if hasattr(tokenizer, "model_max_length"):
            info["model_max_length"] = tokenizer.model_max_length

    except Exception as e:
        info["error"] = f"Failed to get tokenizer info: {e}"

    return info


def get_vocab_dict(tokenizer) -> dict:
    """获取词表字典"""
    try:
        if hasattr(tokenizer, "vocab"):
            return tokenizer.vocab
        elif hasattr(tokenizer, "get_vocab"):
            return tokenizer.get_vocab()
        elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "vocab"):
            return tokenizer.tokenizer.vocab
        elif hasattr(tokenizer, "encoder"):
            return tokenizer.encoder
        else:
            return {}
    except Exception:
        return {}


def export_vocabulary(tokenizer, file_path: str) -> None:
    """导出词表到文件"""
    try:
        vocab = get_vocab_dict(tokenizer)
        if not vocab:
            print("Warning: Could not retrieve vocabulary from tokenizer")
            return

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 根据文件扩展名选择格式
        if path.suffix.lower() == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
        else:
            # 默认格式：每行一个token
            with open(path, "w", encoding="utf-8") as f:
                for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                    # 处理不可打印字符
                    try:
                        f.write(f"{token_id}\t{repr(token)}\n")
                    except:
                        f.write(f"{token_id}\t<unprintable>\n")

        print(f"Vocabulary exported to: {file_path}")
        print(f"Total tokens: {len(vocab)}")

    except Exception as e:
        print(f"Error exporting vocabulary: {e}")


def main(args: argparse.Namespace) -> None:

    def print_separator(title=""):
        if title:
            print(f"\n{'='*50}")
            print(f" {title}")
            print(f"{'='*50}")
        else:
            print(f"\n{'='*50}")

    # 检查参数
    if not any([args.encode, args.decode, args.vocab_size, args.info, args.vocab_export]):
        print("请至少指定一个参数：--encode, --decode, --vocab-size, --info, --export-vocab")
        return

    # 初始化tokenizer
    preprocessor = InputPreprocessor(model_config=ModelConfig({"model": args.model_name_or_path}))
    tokenizer = preprocessor.create_processor().tokenizer

    # 执行操作
    operations_count = 0

    if args.encode:
        print_separator("ENCODING")
        print(f"Input text: {args.encode}")
        encoded_text = tokenizer.encode(args.encode)
        print(f"Encoded tokens: {encoded_text}")
        operations_count += 1

    if args.decode:
        print_separator("DECODING")
        print(f"Input tokens: {args.decode}")
        try:
            if isinstance(args.decode, str):
                if args.decode.startswith("[") and args.decode.endswith("]"):
                    tokens = eval(args.decode)
                else:
                    tokens = [int(x.strip()) for x in args.decode.split(",")]
            else:
                tokens = args.decode

            decoded_text = tokenizer.decode(tokens)
            print(f"Decoded text: {decoded_text}")
        except Exception as e:
            print(f"Error decoding tokens: {e}")
        operations_count += 1

    if args.vocab_size:
        print_separator("VOCABULARY SIZE")
        print(f"Vocabulary size: {get_vocab_size(tokenizer)}")
        operations_count += 1

    if args.info:
        print_separator("TOKENIZER INFO")
        print(json.dumps(get_tokenizer_info(tokenizer), indent=2))
        operations_count += 1

    if args.vocab_export:
        print_separator("EXPORT VOCABULARY")
        export_vocabulary(tokenizer, args.vocab_export)
        operations_count += 1

    print_separator()
    print(f"Completed {operations_count} operation(s)")
