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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/cli/main.py
from __future__ import annotations

from fastdeploy import __version__


def main():
    import fastdeploy.entrypoints.cli.benchmark.main
    import fastdeploy.entrypoints.cli.collect_env
    import fastdeploy.entrypoints.cli.openai
    import fastdeploy.entrypoints.cli.run_batch
    import fastdeploy.entrypoints.cli.serve
    import fastdeploy.entrypoints.cli.tokenizer
    from fastdeploy.utils import FlexibleArgumentParser

    CMD_MODULES = [
        fastdeploy.entrypoints.cli.run_batch,
        fastdeploy.entrypoints.cli.tokenizer,
        fastdeploy.entrypoints.cli.openai,
        fastdeploy.entrypoints.cli.benchmark.main,
        fastdeploy.entrypoints.cli.serve,
        fastdeploy.entrypoints.cli.collect_env,
    ]

    parser = FlexibleArgumentParser(description="FastDeploy CLI")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
