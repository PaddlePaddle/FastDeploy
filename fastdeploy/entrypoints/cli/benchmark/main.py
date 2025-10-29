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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/cli/benchmark/main.py

from __future__ import annotations

import argparse
import subprocess
import sys
import typing

from fastdeploy.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from fastdeploy.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from fastdeploy.utils import FlexibleArgumentParser


FD_SUBCMD_PARSER_EPILOG = (
    "Tip: Use `fastdeploy [serve|run-batch|bench <bench_type>] "
    "--help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help=ModelConfig\n"
    "   - To view a single argument:    --help=max-num-seqs\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup\n"
    "   - To view help with pager:      --help=page"
)


def _output_with_pager(text: str):
    """Output text using scrolling view if available and appropriate."""

    pagers = ["less -R", "more"]
    for pager_cmd in pagers:
        try:
            proc = subprocess.Popen(pager_cmd.split(), stdin=subprocess.PIPE, text=True)
            proc.communicate(input=text)
            return
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            continue

    # No pager worked, fall back to normal print
    print(text)


def show_filtered_argument_or_group_from_help(parser: argparse.ArgumentParser, subcommand_name: list[str]):

    # Only handle --help=<keyword> for the current subcommand.
    # Since subparser_init() runs for all subcommands during CLI setup,
    # we skip processing if the subcommand name is not in sys.argv.
    # sys.argv[0] is the program name. The subcommand follows.
    # e.g., for `vllm bench latency`,
    # sys.argv is `['vllm', 'bench', 'latency', ...]`
    # and subcommand_name is "bench latency".
    if len(sys.argv) <= len(subcommand_name) or sys.argv[1 : 1 + len(subcommand_name)] != subcommand_name:
        return

    for arg in sys.argv:
        if arg.startswith("--help="):
            search_keyword = arg.split("=", 1)[1]

            # Enable paged view for full help
            if search_keyword == "page":
                help_text = parser.format_help()
                _output_with_pager(help_text)
                sys.exit(0)

            # List available groups
            if search_keyword == "listgroup":
                output_lines = ["\nAvailable argument groups:"]
                for group in parser._action_groups:
                    if group.title and not group.title.startswith("positional arguments"):
                        output_lines.append(f"  - {group.title}")
                        if group.description:
                            output_lines.append("    " + group.description.strip())
                        output_lines.append("")
                _output_with_pager("\n".join(output_lines))
                sys.exit(0)

            # For group search
            formatter = parser._get_formatter()
            for group in parser._action_groups:
                if group.title and group.title.lower() == search_keyword.lower():
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    _output_with_pager(formatter.format_help())
                    sys.exit(0)

            # For single arg
            matched_actions = []

            for group in parser._action_groups:
                for action in group._group_actions:
                    # search option name
                    if any(search_keyword.lower() in opt.lower() for opt in action.option_strings):
                        matched_actions.append(action)

            if matched_actions:
                header = f"\nParameters matching '{search_keyword}':\n"
                formatter = parser._get_formatter()
                formatter.add_arguments(matched_actions)
                _output_with_pager(header + formatter.format_help())
                sys.exit(0)

            print(f"\nNo group or parameter matching '{search_keyword}'")
            print("Tip: use `--help=listgroup` to view all groups.")
            sys.exit(1)


class BenchmarkSubcommand(CLISubcommand):
    """The `bench` subcommand for the vLLM CLI."""

    name = "bench"
    help = "fastdeploy bench subcommand."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        args.dispatch_function(args)

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        bench_parser = subparsers.add_parser(
            self.name, help=self.help, description=self.help, usage="fastdeploy bench <bench_type> [options]"
        )
        bench_subparsers = bench_parser.add_subparsers(required=True, dest="bench_type")

        for cmd_cls in BenchmarkSubcommandBase.__subclasses__():
            cmd_subparser = bench_subparsers.add_parser(
                cmd_cls.name,
                help=cmd_cls.help,
                description=cmd_cls.help,
                usage=f"fastdeploy bench {cmd_cls.name} [options]",
            )
            cmd_subparser.set_defaults(dispatch_function=cmd_cls.cmd)
            cmd_cls.add_cli_args(cmd_subparser)
            show_filtered_argument_or_group_from_help(cmd_subparser, ["bench", cmd_cls.name])
            cmd_subparser.epilog = FD_SUBCMD_PARSER_EPILOG
        return bench_parser


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkSubcommand()]
