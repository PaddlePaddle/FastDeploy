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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/cli/run_batch.py

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata

from fastdeploy.entrypoints.cli.types import CLISubcommand
from fastdeploy.utils import (
    FASTDEPLOY_SUBCMD_PARSER_EPILOG,
    FlexibleArgumentParser,
    show_filtered_argument_or_group_from_help,
)


class RunBatchSubcommand(CLISubcommand):
    """The `run-batch` subcommand for FastDeploy CLI."""

    name = "run-batch"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from fastdeploy.entrypoints.openai.run_batch import main as run_batch_main

        print("FastDeploy batch processing API version", importlib.metadata.version("fastdeploy-gpu"))
        print(args)
        asyncio.run(run_batch_main(args))

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        from fastdeploy.entrypoints.openai.run_batch import make_arg_parser

        run_batch_parser = subparsers.add_parser(
            "run-batch",
            help="Run batch prompts and write results to file.",
            description=(
                "Run batch prompts using FastDeploy's OpenAI-compatible API.\n"
                "Supports local or HTTP input/output files."
            ),
            usage="FastDeploy run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>",
        )
        run_batch_parser = make_arg_parser(run_batch_parser)
        show_filtered_argument_or_group_from_help(run_batch_parser, ["run-batch"])
        run_batch_parser.epilog = FASTDEPLOY_SUBCMD_PARSER_EPILOG
        return run_batch_parser


def cmd_init() -> list[CLISubcommand]:
    return [RunBatchSubcommand()]
