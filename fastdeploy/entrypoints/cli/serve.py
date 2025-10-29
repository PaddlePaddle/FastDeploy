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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/cli/serve.py

import argparse
import atexit
import os
import signal
import subprocess
import sys

from fastdeploy.entrypoints.cli.types import CLISubcommand
from fastdeploy.entrypoints.openai.utils import make_arg_parser
from fastdeploy.utils import FlexibleArgumentParser


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the fastdeploy CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        env = os.environ.copy()
        cmd = [
            sys.executable,
            "-m",
            "fastdeploy.entrypoints.openai.api_server",
            *sys.argv[2:],
        ]

        # 启动子进程
        proc = subprocess.Popen(cmd, env=env)
        print(f"Starting server (PID: {proc.pid})")

        # 定义清理函数
        def cleanup():
            """终止子进程并确保资源释放"""
            if proc.poll() is None:  # 检查子进程是否仍在运行
                print(f"\nTerminating child process (PID: {proc.pid})...")
                proc.terminate()  # 发送终止信号

        # 注册退出时的清理函数
        atexit.register(cleanup)
        # 设置信号处理

        def signal_handler(signum, frame):
            cleanup()
            sys.exit(0)

        # 捕获 SIGINT (Ctrl+C) 和 SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        # 主进程阻塞等待子进程
        proc.wait()

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            name=self.name,
            help="Start the FastDeploy OpenAI Compatible API server.",
            description="Start the FastDeploy OpenAI Compatible API server.",
            usage="fastdeploy serve [options]",
        )
        serve_parser = make_arg_parser(serve_parser)
        serve_parser.add_argument("--config", help="Read CLI options from a config file. Must be a YAML file")
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]
