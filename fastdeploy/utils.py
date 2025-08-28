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
import logging
import os
import random
import re
import socket
import sys
import tarfile
import time
from datetime import datetime
from logging.handlers import BaseRotatingHandler
from pathlib import Path
from typing import Literal, TypeVar, Union

import numpy as np
import paddle
import requests
import yaml
from aistudio_sdk.snapshot_download import snapshot_download as aistudio_download
from tqdm import tqdm
from typing_extensions import TypeIs, assert_never

from fastdeploy import envs
from fastdeploy.logger.logger import FastDeployLogger

T = TypeVar("T")

# [N,2] -> every line is [config_name, enable_xxx_name]
# Make sure enable_xxx equal to config.enable_xxx
ARGS_CORRECTION_LIST = [["early_stop_config", "enable_early_stop"], ["graph_optimization_config", "use_cudagraph"]]


class EngineError(Exception):
    """Base exception class for engine errors"""

    def __init__(self, message, error_code=400):
        super().__init__(message)
        self.error_code = error_code


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


# def get_logger(name, file_name, without_formater=False, print_to_console=False):
#     """
#     get logger
#     """
#     log_dir = envs.FD_LOG_DIR
#     if not os.path.exists(log_dir):
#         os.mkdir(log_dir)
#     is_debug = int(envs.FD_DEBUG)
#     logger = logging.getLogger(name)
#     if is_debug:
#         logger.setLevel(level=logging.DEBUG)
#     else:
#         logger.setLevel(level=logging.INFO)

#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)

#     LOG_FILE = f"{log_dir}/{file_name}"
#     backup_count = int(envs.FD_LOG_BACKUP_COUNT)
#     handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)
#     formatter = ColoredFormatter("%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s")

#     console_handler = logging.StreamHandler()
#     if not without_formater:
#         handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     if print_to_console:
#         logger.addHandler(console_handler)
#     handler.propagate = False
#     console_handler.propagate = False
#     return logger


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
        f"\n\tDevice Total memory: {meminfo.total}",
        f"\n\tDevice Used memory: {meminfo.used}",
        f"\n\tDevice Free memory: {meminfo.free}",
        f"\n\tPaddle max memory Reserved: {paddle_max_reserved}",
        f"\n\tPaddle max memory Allocated: {paddle_max_allocated}",
        f"\n\tPaddle memory Reserved: {paddle_reserved}",
        f"\n\tPaddle memory Allocated: {paddle_allocated}",
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


# 日志使用全局访问点（兼容原有使用方式）
def get_logger(name, file_name=None, without_formater=False, print_to_console=False):
    """全局函数包装器，保持向后兼容"""
    return FastDeployLogger().get_logger(name, file_name, without_formater, print_to_console)


llm_logger = get_logger("fastdeploy", "fastdeploy.log")
data_processor_logger = get_logger("data_processor", "data_processor.log")
scheduler_logger = get_logger("scheduler", "scheduler.log")
api_server_logger = get_logger("api_server", "api_server.log")
console_logger = get_logger("console", "console.log", print_to_console=True)
spec_logger = get_logger("speculate", "speculate.log")
