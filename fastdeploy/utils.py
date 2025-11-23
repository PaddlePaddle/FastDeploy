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
import asyncio
import codecs
import importlib
import json
import logging
import os
import pickle
import random
import re
import socket
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from importlib.metadata import PackageNotFoundError, distribution
from logging.handlers import BaseRotatingHandler
from pathlib import Path
from typing import Any, Literal, TypeVar, Union

import numpy as np
import paddle
import requests
import yaml
from aistudio_sdk.snapshot_download import snapshot_download as aistudio_download
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from tqdm import tqdm
from typing_extensions import TypeIs, assert_never

from fastdeploy import envs
from fastdeploy.entrypoints.openai.protocol import ErrorInfo, ErrorResponse
from fastdeploy.logger.logger import FastDeployLogger

T = TypeVar("T")
from typing import Callable, List, Optional

# [N,2] -> every line is [config_name, enable_xxx_name]
# Make sure enable_xxx equal to config.enable_xxx
ARGS_CORRECTION_LIST = [
    ["early_stop_config", "enable_early_stop"],
]

FASTDEPLOY_SUBCMD_PARSER_EPILOG = (
    "Tip: Use `fastdeploy [serve|run-batch|bench <bench_type>] "
    "--help=<keyword>` to explore arguments from help.\n"
    "   - To view a argument group:     --help=ModelConfig\n"
    "   - To view a single argument:    --help=max-num-seqs\n"
    "   - To search by keyword:         --help=max\n"
    "   - To list all groups:           --help=listgroup\n"
    "   - To view help with pager:      --help=page"
)


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


class EngineError(Exception):
    """Base exception class for engine errors"""

    def __init__(self, message, error_code=400):
        super().__init__(message)
        self.error_code = error_code


class ParameterError(Exception):
    def __init__(self, param: str, message: str):
        self.param = param
        self.message = message
        super().__init__(message)


class ExceptionHandler:

    # 全局异常兜底处理
    @staticmethod
    async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
        error = ErrorResponse(error=ErrorInfo(message=str(exc), type=ErrorType.INTERNAL_ERROR))
        return JSONResponse(content=error.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    # 处理请求参数验证异常
    @staticmethod
    async def handle_request_validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        if not errors:
            message = str(exc)
            param = None
        else:
            first_error = errors[0]
            loc = first_error.get("loc", [])
            param = loc[-1] if loc else None
            message = first_error.get("msg", str(exc))
        err = ErrorResponse(
            error=ErrorInfo(
                message=message,
                type=ErrorType.INVALID_REQUEST_ERROR,
                code=ErrorCode.MISSING_REQUIRED_PARAMETER if param == "messages" else ErrorCode.INVALID_VALUE,
                param=param,
            )
        )
        api_server_logger.error(f"invalid_request_error: {request.url} {param} {message}")
        return JSONResponse(content=err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


class ErrorType(str, Enum):
    INVALID_REQUEST_ERROR = "invalid_request_error"
    TIMEOUT_ERROR = "timeout_error"
    SERVER_ERROR = "server_error"
    INTERNAL_ERROR = "internal_error"
    API_CONNECTION_ERROR = "api_connection_error"


class ErrorCode(str, Enum):
    INVALID_VALUE = "invalid_value"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    MODEL_NOT_SUPPORT = "model_not_support"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    INTERNAL_ERROR = "internal_error"


class ColoredFormatter(logging.Formatter):
    """自定义日志格式器，用于控制台输出带颜色"""

    COLOR_CODES = {
        logging.WARNING: 33,  # 黄色
        logging.ERROR: 31,  # 红色
        logging.CRITICAL: 31,  # 红色
    }

    def format(self, record):
        color_code = self.COLOR_CODES.get(record.levelno, 0)
        prefix = f"\033[{color_code}m"
        suffix = "\033[0m"
        message = super().format(record)
        if color_code:
            message = f"{prefix}{message}{suffix}"
        return message


class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    like `logging.TimedRotatingFileHandler`, but this class support multi-process
    """

    def __init__(
        self,
        filename,
        backupCount=0,
        encoding="utf-8",
        delay=False,
        utc=False,
        **kwargs,
    ):
        """
            初始化 RotatingFileHandler 对象。

        Args:
            filename (str): 日志文件的路径，可以是相对路径或绝对路径。
            backupCount (int, optional, default=0): 保存的备份文件数量，默认为 0，表示不保存备份文件。
            encoding (str, optional, default='utf-8'): 编码格式，默认为 'utf-8'。
            delay (bool, optional, default=False): 是否延迟写入，默认为 False，表示立即写入。
            utc (bool, optional, default=False): 是否使用 UTC 时区，默认为 False，表示不使用 UTC 时区。
            kwargs (dict, optional): 其他参数将被传递给 BaseRotatingHandler 类的 init 方法。

        Raises:
            TypeError: 如果 filename 不是 str 类型。
            ValueError: 如果 backupCount 小于等于 0。
        """
        self.backup_count = backupCount
        self.utc = utc
        self.suffix = "%Y-%m-%d"
        self.base_log_path = Path(filename)
        self.base_filename = self.base_log_path.name
        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(self.current_filename)
        BaseRotatingHandler.__init__(self, filename, "a", encoding, delay)

    def shouldRollover(self, record):
        """
        check scroll through the log
        """
        if self.current_filename != self._compute_fn():
            return True
        return False

    def doRollover(self):
        """
        scroll log
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(self.current_filename)

        if not self.delay:
            self.stream = self._open()

        self.delete_expired_files()

    def _compute_fn(self):
        """
        Calculate the log file name corresponding current time
        """
        return self.base_filename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        """
        open new log file
        """
        if self.encoding is None:
            stream = open(str(self.current_log_path), self.mode)
        else:
            stream = codecs.open(str(self.current_log_path), self.mode, self.encoding)

        if self.base_log_path.exists():
            try:
                if not self.base_log_path.is_symlink() or os.readlink(self.base_log_path) != self.current_filename:
                    os.remove(self.base_log_path)
            except OSError:
                pass

        try:
            os.symlink(self.current_filename, str(self.base_log_path))
        except OSError:
            pass
        return stream

    def delete_expired_files(self):
        """
        delete expired log files
        """
        if self.backup_count <= 0:
            return

        file_names = os.listdir(str(self.base_log_path.parent))
        result = []
        prefix = self.base_filename + "."
        plen = len(prefix)
        for file_name in file_names:
            if file_name[:plen] == prefix:
                suffix = file_name[plen:]
                if re.match(r"^\d{4}-\d{2}-\d{2}(\.\w+)?$", suffix):
                    result.append(file_name)
        if len(result) < self.backup_count:
            result = []
        else:
            result.sort()
            result = result[: len(result) - self.backup_count]

        for file_name in result:
            os.remove(str(self.base_log_path.with_name(file_name)))


def chunk_list(lst: list[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def str_to_datetime(date_string):
    """
    string to datetime class object
    """
    if "." in date_string:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    else:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


def datetime_diff(datetime_start, datetime_end):
    """
    Calculate the difference between two dates and times(s)

    Args:
        datetime_start (Union[str, datetime.datetime]): start time
        datetime_end (Union[str, datetime.datetime]): end time

    Returns:
        float: date time difference(s)
    """
    if isinstance(datetime_start, str):
        datetime_start = str_to_datetime(datetime_start)
    if isinstance(datetime_end, str):
        datetime_end = str_to_datetime(datetime_end)
    if datetime_end > datetime_start:
        cost = datetime_end - datetime_start
    else:
        cost = datetime_start - datetime_end
    return cost.total_seconds()


def download_file(url, save_path):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(url)}",
        )

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
        return True
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(f"Download failed: {e!s}")


def extract_tar(tar_path, output_dir):
    """Extract tar file with progress tracking"""
    try:
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting files") as pbar:
                for member in members:
                    tar.extract(member, path=output_dir)
                    pbar.update(1)
        print(f"Successfully extracted to: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e!s}")


def set_random_seed(seed: int) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)


def get_limited_max_value(max_value):
    def validator(value):
        value = float(value)
        if value > max_value:
            raise argparse.ArgumentTypeError(f"The value cannot exceed {max_value}")
        return value

    return validator


def download_model(url, output_dir, temp_tar):
    """
    下载模型，并将其解压到指定目录。

    Args:
        url (str): 模型文件的URL地址。
        output_dir (str): 模型文件要保存的目录路径。
        temp_tar (str, optional): 临时保存模型文件的TAR包名称，默认为'temp.tar'.

    Raises:
        Exception: 如果下载或解压过程中出现任何错误，都会抛出Exception异常。

    Returns:
        None - 无返回值，只是在下载和解压过程中进行日志输出和清理临时文件。
    """
    try:
        temp_tar = os.path.join(output_dir, temp_tar)
        # Download the file
        llm_logger.info(f"\nStarting download from: {url} {temp_tar}")
        download_file(url, temp_tar)
        # Extract the archive
        print("\nExtracting files...")
        extract_tar(temp_tar, output_dir)

    except Exception:
        # Cleanup on failure
        if os.path.exists(temp_tar):
            os.remove(temp_tar)
        raise Exception(
            f"""Failed to get model from {url}, please recheck the model name from
            https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md"""
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_tar):
            os.remove(temp_tar)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """
    Extend argparse.ArgumentParser to support loading parameters from YAML files.
    """

    def __init__(self, *args, config_arg="--config", sep="_", **kwargs):
        super().__init__(*args, **kwargs)
        self.sep = sep

        # Create parser to prase yaml file
        self.tmp_parser = argparse.ArgumentParser(add_help=False)
        self.tmp_parser.add_argument(config_arg, type=str, help="Path to YAML config file")

    def parse_args(self, args=None, namespace=None):
        tmp_ns, remaining_args = self.tmp_parser.parse_known_args(args=args)
        config_path = tmp_ns.config

        config = {}
        if config_path:
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)
                config = loaded_config

        # Get declared parameters
        defined_actions = {action.dest: action for action in self._actions}
        filtered_config = {k: v for k, v in config.items() if k in defined_actions}

        # Set parameters
        if namespace is None:
            namespace = argparse.Namespace()
        for key, value in filtered_config.items():
            action = defined_actions[key]
            if action.type is not None and isinstance(value, (str, int, float)):
                try:
                    str_value = str(value).strip()
                    if str_value == "":
                        converted = None
                    else:
                        converted = action.type(str_value)
                    value = converted
                except Exception as e:
                    llm_logger.error(f"Error converting '{key}' with value '{value}': {e}")
            setattr(namespace, key, value)
        args = super().parse_args(args=remaining_args, namespace=namespace)

        # Args correction
        for config_name, flag_name in ARGS_CORRECTION_LIST:
            if hasattr(args, config_name) and hasattr(args, flag_name):
                # config is a dict
                config = getattr(args, config_name, None)
                if config is not None and flag_name in config.keys():
                    setattr(args, flag_name, config[flag_name])
        return args


def resolve_obj_from_strname(strname: str):
    module_name, obj_name = strname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def check_unified_ckpt(model_dir):
    """
    Check if the model is a PaddleNLP unified checkpoint
    """
    model_files = list()
    all_files = os.listdir(model_dir)
    for x in all_files:
        if x.startswith("model") and x.endswith(".safetensors"):
            model_files.append(x)

    is_unified_ckpt = len(model_files) > 0
    if not is_unified_ckpt:
        return False

    if len(model_files) == 1 and model_files[0] == "model.safetensors":
        return True

    try:
        # check all the file exists
        safetensors_num = int(model_files[0].strip(".safetensors").split("-")[-1])
        flags = [0] * safetensors_num
        for x in model_files:
            current_index = int(x.strip(".safetensors").split("-")[1])
            flags[current_index - 1] = 1
        assert sum(flags) == len(
            model_files
        ), f"Number of safetensor files should be {len(model_files)}, but now it's {sum(flags)}"
    except Exception as e:
        raise Exception(f"Failed to check unified checkpoint, details: {e}.")
    return is_unified_ckpt


def get_host_ip():
    """
    Get host IP address
    """
    ip = socket.gethostbyname(socket.gethostname())
    return ip


def get_random_port():
    while True:
        port = random.randint(49152, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue


def is_port_available(host, port):
    """
    Check the port is available
    """
    import errno
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            return True


def singleton(cls):
    """
    Singleton decorator for a class.
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def print_gpu_memory_use(gpu_id: int, title: str) -> None:
    """Print memory usage"""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()

    paddle_max_reserved = paddle.device.cuda.max_memory_reserved(gpu_id)
    paddle_max_allocated = paddle.device.cuda.max_memory_allocated(gpu_id)
    paddle_reserved = paddle.device.cuda.memory_reserved(gpu_id)
    paddle_allocated = paddle.device.cuda.memory_allocated(gpu_id)

    print(
        f"\n{title}:",
        f"\n\tDevice Total memory(GiB): {meminfo.total / 1024.0 / 1024.0 / 1024.0}",
        f"\n\tDevice Used memory(GiB): {meminfo.used / 1024.0 / 1024.0 / 1024.0}",
        f"\n\tDevice Free memory(GiB): {meminfo.free / 1024.0 / 1024.0 / 1024.0}",
        f"\n\tPaddle max memory Reserved(GiB): {paddle_max_reserved / 1024.0 / 1024.0 / 1024.0}",
        f"\n\tPaddle max memory Allocated(GiB): {paddle_max_allocated / 1024.0 / 1024.0 / 1024.0}",
        f"\n\tPaddle memory Reserved(GiB): {paddle_reserved / 1024.0 / 1024.0 / 1024.0}",
        f"\n\tPaddle memory Allocated(GiB): {paddle_allocated / 1024.0 / 1024.0 / 1024.0}",
    )


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def none_or_str(value):
    """
    Keep parameters None, not the string "None".
    """
    return None if value == "None" else value


def retrive_model_from_server(model_name_or_path, revision="master"):
    """
    Download pretrained model from AIStudio, MODELSCOPE or HUGGINGFACE automatically
    """
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    model_source = envs.FD_MODEL_SOURCE
    local_path = envs.FD_MODEL_CACHE
    repo_id = model_name_or_path
    if model_source == "AISTUDIO":
        try:
            if repo_id.lower().strip().startswith("baidu"):
                repo_id = "PaddlePaddle" + repo_id.strip()[5:]
            if local_path is None:
                local_path = f'{os.getenv("HOME")}'
            local_path = f"{local_path}/{repo_id}"
            aistudio_download(repo_id=repo_id, revision=revision, local_dir=local_path)
            model_name_or_path = local_path
        except requests.exceptions.ConnectTimeout:
            if os.path.exists(local_path):
                llm_logger.error(
                    f"Failed to connect to aistudio, but detected that the model directory {local_path} exists. Attempting to start."
                )
                return local_path
        except Exception:
            raise Exception(
                f"The {revision} of {model_name_or_path} is not exist. Please check the model name or revision."
            )
    elif model_source == "MODELSCOPE":
        try:
            from modelscope.hub.snapshot_download import (
                snapshot_download as modelscope_download,
            )

            if repo_id.lower().strip().startswith("baidu"):
                repo_id = "PaddlePaddle" + repo_id.strip()[5:]
            if local_path is None:
                local_path = f'{os.getenv("HOME")}'
            local_path = f"{local_path}/{repo_id}"
            modelscope_download(repo_id=repo_id, revision=revision, local_dir=local_path)
            model_name_or_path = local_path
        except requests.exceptions.ConnectTimeout:
            if os.path.exists(local_path):
                llm_logger.error(
                    f"Failed to connect to modelscope, but detected that the model directory {local_path} exists. Attempting to start."
                )
                return local_path
        except Exception:
            raise Exception(
                f"The {revision} of {model_name_or_path} is not exist. Please check the model name or revision."
            )
    elif model_source == "HUGGINGFACE":
        try:
            from huggingface_hub._snapshot_download import (
                snapshot_download as huggingface_download,
            )

            if revision == "master":
                revision = "main"
            repo_id = model_name_or_path
            if repo_id.lower().strip().startswith("PaddlePaddle"):
                repo_id = "baidu" + repo_id.strip()[12:]
            if local_path is None:
                local_path = f'{os.getenv("HOME")}'
            local_path = f"{local_path}/{repo_id}"
            huggingface_download(repo_id=repo_id, revision=revision, local_dir=local_path)
            model_name_or_path = local_path
        except Exception:
            raise Exception(
                f"The {revision} of {model_name_or_path} is not exist. Please check the model name or revision."
            )
    else:
        raise ValueError(
            f"Unsupported model source: {model_source}, please choose one of ['MODELSCOPE', 'AISTUDIO', 'HUGGINGFACE']"
        )
    return model_name_or_path


def is_list_of(
    value: object,
    typ: Union[type[T], tuple[type[T], ...]],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    """
    Check if the value is a list of specified type.

    Args:
        value: The value to check.
        typ: The type or tuple of types to check against.
        check: The check mode, either "first" or "all".

    Returns:
        Whether the value is a list of specified type.
    """
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


def import_from_path(module_name: str, file_path: Union[str, os.PathLike]):
    """
    Import a Python file according to its file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def is_package_installed(package_name):
    try:
        distribution(package_name)
        return True
    except PackageNotFoundError:
        return False


def version():
    """
    Prints the contents of the version.txt file located in the parent directory of this script.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    version_file_path = os.path.join(current_dir, "version.txt")

    content = "Unknown"
    try:
        with open(version_file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        llm_logger.error("[version.txt] Not Found!")
    return content


def current_package_version():
    """
    读取version.txt文件,解析出fastdeploy version对应的版本号

    Args:
    Returns:
        str: fastdeploy版本号,如果解析失败返回Unknown
    """
    fd_version = "Unknown"
    try:
        content = version()
        if content == "Unknown":
            return fd_version

        # 按行分割内容
        lines = content.strip().split("\n")
        # 查找包含"fastdeploy version:"的行
        for line in lines:
            if line.startswith("fastdeploy version:"):
                # 提取版本号部分
                fd_version = line.split("fastdeploy version:")[1].strip()
                return fd_version
        llm_logger.warning("fastdeploy version not found in version.txt")
        # 如果没有找到对应的行，返回None
        return fd_version
    except Exception as e:
        llm_logger.error(f"Failed to parse fastdeploy version from version.txt: {e}")
        return fd_version


class DeprecatedOptionWarning(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        console_logger.warning(f"Deprecated option is detected: {option_string}, which may be removed later")
        setattr(namespace, self.dest, True)


DEPRECATED_ARGS = ["enable_mm"]


def deprecated_kwargs_warning(**kwargs):
    for arg in DEPRECATED_ARGS:
        if arg in kwargs:
            console_logger.warning(f"Deprecated argument is detected: {arg}, which may be removed later")


class StatefulSemaphore:
    __slots__ = ("_semaphore", "_max_value", "_acquired_count", "_last_reset")

    """
    StatefulSemaphore is a class that wraps an asyncio.Semaphore and provides additional stateful information.
    """

    def __init__(self, value: int):
        """
        StatefulSemaphore constructor
        """
        if value < 0:
            raise ValueError("Value must be non-negative.")
        self._semaphore = asyncio.Semaphore(value)
        self._max_value = value
        self._acquired_count = 0
        self._last_reset = time.monotonic()

    async def acquire(self):
        await self._semaphore.acquire()
        self._acquired_count += 1

    def release(self):
        self._semaphore.release()

        self._acquired_count = max(0, self._acquired_count - 1)

    def locked(self) -> bool:
        return self._semaphore.locked()

    @property
    def available(self) -> int:
        return self._max_value - self._acquired_count

    @property
    def acquired(self) -> int:
        return self._acquired_count

    @property
    def max_value(self) -> int:
        return self._max_value

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._last_reset

    def status(self) -> dict:
        return {
            "available": self.available,
            "acquired": self.acquired,
            "max_value": self.max_value,
            "uptime": round(self.uptime, 2),
        }


def parse_quantization(value: str):
    """
    Parse a JSON string into a dictionary.
    """
    try:
        return json.loads(value)
    except ValueError:
        return {"quantization": value}


# 日志使用全局访问点（兼容原有使用方式）
def get_logger(name, file_name=None, without_formater=False, print_to_console=False):
    """全局函数包装器，保持向后兼容"""
    return FastDeployLogger().get_logger(name, file_name, without_formater, print_to_console)


def check_download_links(bos_client, links, timeout=1):
    """
    check bos download links
    """
    for link in links:
        try:
            if link.startswith("bos://"):
                link = link.replace("bos://", "")

            bucket_name = "/".join(link.split("/")[1:-1])
            object_key = link.split("/")[-1]
            response = bos_client.get_object_meta_data(bucket_name, object_key)
            assert (
                int(response.metadata.content_length) > 0
            ), f"bos download length error, {response.metadata.content_length}"
        except Exception as e:
            return f"link {link} download error: {str(e)}"
    return None


def init_bos_client():
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.services.bos.bos_client import BosClient

    cfg = BceClientConfiguration(
        credentials=BceCredentials(envs.ENCODE_FEATURE_BOS_AK, envs.ENCODE_FEATURE_BOS_SK),
        endpoint=envs.ENCODE_FEATURE_ENDPOINT,
    )
    return BosClient(cfg)


def download_from_bos(bos_client, bos_links, retry: int = 0):
    """
    Download pickled objects from Baidu Object Storage (BOS).
    Args:
        bos_client: BOS client instance
        bos_links: Single link or list of BOS links in format "bos://bucket-name/path/to/object"
        retry: Number of times to retry on failure (only retries on network-related errors)
    Yields:
        tuple: (success: bool, data: np.ndarray | error_msg: str)
            - On success: (True, deserialized_data)
            - On failure: (False, error_message) and stops processing remaining links
    Security Note:
        Uses pickle deserialization. Only use with trusted data sources.
    """

    def _bos_download(bos_client, link):
        if link.startswith("bos://"):
            link = link.replace("bos://", "")

        bucket_name = "/".join(link.split("/")[1:-1])
        object_key = link.split("/")[-1]
        return bos_client.get_object_as_string(bucket_name, object_key)

    if not isinstance(bos_links, list):
        bos_links = [bos_links]

    for link in bos_links:
        try:
            response = _bos_download(bos_client, link)
            yield True, pickle.loads(response)
        except Exception:
            # Only retry on network-related or timeout exceptions
            exceptions_msg = str(traceback.format_exc())

            if "request rate is too high" not in exceptions_msg or retry <= 0:
                yield False, f"Failed to download {link}: {exceptions_msg}"
                break

            for attempt in range(retry):
                try:
                    llm_logger.warning(f"Retry attempt {attempt + 1}/{retry} for {link}")
                    response = _bos_download(bos_client, link)
                    yield True, pickle.loads(response)
                    break
                except Exception:
                    if attempt == retry - 1:  # Last attempt failed
                        yield False, f"Failed after {retry} retries for {link}: {str(traceback.format_exc())}"
            break


llm_logger = get_logger("fastdeploy", "fastdeploy.log")
data_processor_logger = get_logger("data_processor", "data_processor.log")
scheduler_logger = get_logger("scheduler", "scheduler.log")
api_server_logger = get_logger("api_server", "api_server.log")
console_logger = get_logger("console", "console.log", print_to_console=True)
spec_logger = get_logger("speculate", "speculate.log")
zmq_client_logger = get_logger("zmq_client", "zmq_client.log")
trace_logger = FastDeployLogger().get_trace_logger("trace_logger", "trace_logger.log")
router_logger = get_logger("router", "router.log")


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:

    def _parse_type(val: str) -> T:
        try:
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Value {val} cannot be converted to {return_type}.") from e

    return _parse_type


def optional_type(return_type: Callable[[str], T]) -> Callable[[str], Optional[T]]:

    def _optional_type(val: str) -> Optional[T]:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type


def to_numpy(tasks: List[Any]):
    """
    Convert PaddlePaddle tensors in multimodal inputs to NumPy arrays.

    Args:
        tasks: List of tasks containing multimodal inputs.
    """
    try:
        for task in tasks:
            if not hasattr(task, "multimodal_inputs"):
                continue
            images = task.multimodal_inputs.get("images", None)
            if isinstance(images, paddle.Tensor):
                llm_logger.debug(f"Convert image to numpy, shape: {images.shape}")
                task.multimodal_inputs["images"] = images.numpy()

            list_keys = [
                "image_features",
                "video_features",
                "audio_features",
            ]
            for key in list_keys:
                value = task.multimodal_inputs.get(key, None)
                if value is None:
                    continue
                if isinstance(value, list):
                    task.multimodal_inputs[key] = [v.numpy() for v in value]
    except Exception as e:
        llm_logger.warning(f"Failed to convert to numpy: {e}")


def to_tensor(tasks: List[Any]):
    """
    Convert NumPy arrays in multimodal inputs to Paddle tensors.

    Args:
        tasks (tuple): ([request], bsz)
    """
    try:
        for task in tasks:
            multimodal_inputs = getattr(task, "multimodal_inputs", None)
            if not multimodal_inputs:
                continue
            # tensor keys
            tensor_keys = [
                "images",
                "patch_idx",
                "token_type_ids",
                "position_ids",
                "attention_mask_offset",
            ]

            list_keys = [
                "image_features",
                "video_features",
                "audio_features",
            ]

            llm_logger.debug(f"Converting multimodal inputs to tensor...{tensor_keys + list_keys}")

            for key in tensor_keys:
                value = multimodal_inputs.get(key)
                if value is None:
                    continue
                if not isinstance(value, paddle.Tensor):
                    multimodal_inputs[key] = paddle.to_tensor(value)

            for key in list_keys:
                value = multimodal_inputs.get(key)
                if value is None:
                    continue
                if isinstance(value, list):
                    multimodal_inputs[key] = [paddle.to_tensor(v) for v in value]
    except Exception as e:
        llm_logger.warning(f"Tensor conversion failed: {type(e).__name__}: {e}")
