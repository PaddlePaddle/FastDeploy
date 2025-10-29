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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/latency.py

import argparse

from fastdeploy.benchmarks.latency import add_cli_args, main
from fastdeploy.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    """The `latency` subcommand for fastdeploy bench."""

    name = "latency"
    help = "Benchmark the latency of a single batch of requests."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
