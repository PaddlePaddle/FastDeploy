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

import argparse
import json
import logging
import subprocess
import sys
from functools import partial
from typing import Union

import pkg_resources

from fastdeploy.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def try_parse_json(value: str) -> Union[str, dict, None]:
    """尝试解析JSON格式的字符串"""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings.")
        return value


class BenchmarkEvalSubcommand(BenchmarkSubcommandBase):
    """The `eval` subcommand for fastdeploy bench."""

    name = "eval"
    help = "Run evaluation using lm-evaluation-harness."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`")
        parser.add_argument(
            "--tasks",
            "-t",
            default=None,
            type=str,
            metavar="task1,task2",
            help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
        )
        parser.add_argument(
            "--model_args",
            "-a",
            default="",
            type=try_parse_json,
            help="""Comma separated string or JSON formatted arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32` or '{"pretrained":"EleutherAI/pythia-160m","dtype":"float32"}'""",
        )
        parser.add_argument(
            "--num_fewshot",
            "-f",
            type=int,
            default=None,
            metavar="N",
            help="Number of examples in few-shot context",
        )
        parser.add_argument(
            "--batch_size",
            "-b",
            type=str,
            default=1,
            metavar="auto|auto:N|N",
            help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
        )
        parser.add_argument(
            "--max_batch_size",
            type=int,
            default=None,
            metavar="N",
            help="Maximal batch size to try with --batch_size auto.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Device to use (e.g. cuda, cuda:0, cpu).",
        )
        parser.add_argument(
            "--output_path",
            "-o",
            default=None,
            type=str,
            metavar="DIR|DIR/file.json",
            help="Path where result metrics will be saved. Can be either a directory or a .json file. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
        )
        parser.add_argument(
            "--limit",
            "-L",
            type=float,
            default=None,
            metavar="N|0<N<1",
            help="Limit the number of examples per task. "
            "If <1, limit is a percentage of the total number of examples.",
        )
        parser.add_argument(
            "--samples",
            "-E",
            default=None,
            type=str,
            metavar="/path/to/json",
            help='JSON string or path to JSON file containing doc indices of selected examples to test. Format: {"task_name":[indices],...}',
        )
        parser.add_argument(
            "--use_cache",
            "-c",
            type=str,
            default=None,
            metavar="DIR",
            help="A path to a sqlite db file for caching model responses. `None` if not caching.",
        )
        parser.add_argument(
            "--cache_requests",
            type=str,
            default=None,
            choices=["true", "refresh", "delete"],
            help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
        )
        parser.add_argument(
            "--check_integrity",
            action="store_true",
            help="Whether to run the relevant part of the test suite for the tasks.",
        )
        parser.add_argument(
            "--write_out",
            "-w",
            action="store_true",
            default=False,
            help="Prints the prompt for the first few documents.",
        )
        parser.add_argument(
            "--log_samples",
            "-s",
            action="store_true",
            default=False,
            help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
        )
        parser.add_argument(
            "--system_instruction",
            type=str,
            default=None,
            help="System instruction to be used in the prompt",
        )
        parser.add_argument(
            "--apply_chat_template",
            type=str,
            nargs="?",
            const=True,
            default=False,
            help=(
                "If True, apply chat template to the prompt. "
                "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt. "
                "To apply a specific template from the available list of templates, provide the template name as an argument. "
                "E.g. `--apply_chat_template template_name`"
            ),
        )
        parser.add_argument(
            "--fewshot_as_multiturn",
            action="store_true",
            default=False,
            help="If True, uses the fewshot as a multi-turn conversation",
        )
        parser.add_argument(
            "--show_config",
            action="store_true",
            default=False,
            help="If True, shows the the full config of all tasks at the end of the evaluation.",
        )
        parser.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="DIR",
            help="Additional path to include if there are external tasks to include.",
        )
        parser.add_argument(
            "--gen_kwargs",
            type=try_parse_json,
            default=None,
            help=(
                "Either comma delimited string or JSON formatted arguments for model generation on greedy_until tasks,"
                """ e.g. '{"temperature":0.7,"until":["hello"]}' or temperature=0,top_p=0.1."""
            ),
        )
        parser.add_argument(
            "--verbosity",
            "-v",
            type=str.upper,
            default=None,
            metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
            help="(Deprecated) Controls logging verbosity level. Use the `LOGLEVEL` environment variable instead. Set to DEBUG for detailed output when testing or adding new task configurations.",
        )
        parser.add_argument(
            "--wandb_args",
            type=str,
            default="",
            help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
        )
        parser.add_argument(
            "--wandb_config_args",
            type=str,
            default="",
            help="Comma separated string arguments passed to wandb.config.update. Use this to trace parameters that aren't already traced by default. eg. `lr=0.01,repeats=3",
        )
        parser.add_argument(
            "--hf_hub_log_args",
            type=str,
            default="",
            help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
        )
        parser.add_argument(
            "--predict_only",
            "-x",
            action="store_true",
            default=False,
            help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
        )
        default_seed_string = "0,1234,1234,1234"
        parser.add_argument(
            "--seed",
            type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
            default=default_seed_string,  # for backward compatibility
            help=(
                "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
                "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
                "respectively, or a single integer to set the same seed for all four.\n"
                f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
                "(for backward compatibility).\n"
                "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
                "Here numpy's seed is not set since the second value is `None`.\n"
                "E.g, `--seed 42` sets all four seeds to 42."
            ),
        )
        parser.add_argument(
            "--trust_remote_code",
            action="store_true",
            help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
        )
        parser.add_argument(
            "--confirm_run_unsafe_code",
            action="store_true",
            help="Confirm that you understand the risks of running unsafe code for tasks that require it",
        )
        parser.add_argument(
            "--metadata",
            type=json.loads,
            default=None,
            help="""JSON string metadata to pass to task configs, for example '{"max_seq_lengths":[4096,8192]}'. Will be merged with model_args. Can also be set in task config.""",
        )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """构建并执行lm-eval命令"""
        # 检查lm_eval版本是否为0.4.9.1
        try:
            version = pkg_resources.get_distribution("lm_eval").version
            if version != "0.4.9.1":
                print(
                    f"Warning: lm_eval version {version} is installed, but version 0.4.9.1 is required.\n"
                    "Please install the correct version with:\n"
                    "pip install lm_eval==0.4.9.1",
                    file=sys.stderr,
                )
                sys.exit(1)
        except pkg_resources.DistributionNotFound:
            print(
                "Error: lm_eval is not installed. Please install version 0.4.9.1 with:\n"
                "pip install lm_eval==0.4.9.1",
                file=sys.stderr,
            )
            sys.exit(1)

        cmd = ["lm-eval"]
        if args.model:
            cmd.extend(["--model", args.model])

        if args.model:
            cmd.extend(["--tasks", args.tasks])

        if args.model_args:
            if isinstance(args.model_args, dict):
                model_args = ",".join(f"{k}={v}" for k, v in args.model_args.items())
            else:
                model_args = args.model_args
            cmd.extend(["--model_args", model_args])

        if args.gen_kwargs:
            if isinstance(args.gen_kwargs, dict):
                gen_args = ",".join(f"{k}={v}" for k, v in args.gen_kwargs.items())
            else:
                gen_args = args.gen_kwargs
            cmd.extend(["--gen_kwargs", gen_args])

        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])

        if args.output_path:
            cmd.extend(["--output_path", args.output_path])

        if args.write_out:
            cmd.append("--write_out")
        if args.num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(args.num_fewshot)])
        if args.max_batch_size is not None:
            cmd.extend(["--max_batch_size", str(args.max_batch_size)])
        if args.device:
            cmd.extend(["--device", args.device])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.samples:
            cmd.extend(["--samples", args.samples])
        if args.use_cache:
            cmd.extend(["--use_cache", args.use_cache])
        if args.cache_requests:
            cmd.extend(["--cache_requests", args.cache_requests])
        if args.check_integrity:
            cmd.append("--check_integrity")
        if args.write_out:
            cmd.append("--write_out")
        if args.log_samples:
            cmd.append("--log_samples")
        if args.system_instruction:
            cmd.extend(["--system_instruction", args.system_instruction])
        if args.apply_chat_template:
            if args.apply_chat_template is True:
                cmd.append("--apply_chat_template")
            else:
                cmd.extend(["--apply_chat_template", args.apply_chat_template])
        if args.fewshot_as_multiturn:
            cmd.append("--fewshot_as_multiturn")
        if args.show_config:
            cmd.append("--show_config")
        if args.include_path:
            cmd.extend(["--include_path", args.include_path])
        if args.verbosity:
            cmd.extend(["--verbosity", args.verbosity])
        if args.wandb_args:
            cmd.extend(["--wandb_args", args.wandb_args])
        if args.wandb_config_args:
            cmd.extend(["--wandb_config_args", args.wandb_config_args])
        if args.hf_hub_log_args:
            cmd.extend(["--hf_hub_log_args", args.hf_hub_log_args])
        if args.predict_only:
            cmd.append("--predict_only")
        if args.seed:
            if isinstance(args.seed, list):
                seed_arg = ",".join(str(x) for x in args.seed)
            else:
                seed_arg = str(args.seed)
            cmd.extend(["--seed", seed_arg])
        if args.trust_remote_code:
            cmd.append("--trust_remote_code")
        if args.confirm_run_unsafe_code:
            cmd.append("--confirm_run_unsafe_code")
        if args.metadata:
            if isinstance(args.metadata, dict):
                metadata_arg = json.dumps(args.metadata)
            else:
                metadata_arg = str(args.metadata)
            cmd.extend(["--metadata", metadata_arg])
        # 打印执行的命令
        print("Executing command:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running lm-eval: {e}", file=sys.stderr)
            sys.exit(e.returncode)
        except FileNotFoundError:
            print("Error: lm-eval not found. Please install lm-evaluation-harness first.", file=sys.stderr)
            sys.exit(1)
