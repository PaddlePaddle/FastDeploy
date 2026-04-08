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

# This file is modified from https://github.com/vllm-project/vllm/entrypoints/cli/collect_env.py

from __future__ import annotations

import argparse
import typing

from fastdeploy.collect_env import main as collect_env_main
from fastdeploy.entrypoints.cli.types import CLISubcommand

if typing.TYPE_CHECKING:
    from fastdeploy.utils import FlexibleArgumentParser


class CollectEnvSubcommand(CLISubcommand):
    """The `collect-env` subcommand for the FastDeploy CLI."""

    name = "collect-env"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """Collect information about the environment."""
        collect_env_main()

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        return subparsers.add_parser(
            "collect-env",
            help="Start collecting environment information.",
            description="Start collecting environment information.",
            usage="fastdeploy collect-env",
        )


def cmd_init() -> list[CLISubcommand]:
    return [CollectEnvSubcommand()]
