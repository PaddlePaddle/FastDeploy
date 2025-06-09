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

Utility functions and classes for FastDeploy server operations.
This module provides:
- Custom logging handlers and formatters
- File download and extraction utilities
- Configuration parsing helpers
- Various helper functions for server operations
"""

import argparse
import codecs
import importlib
import logging
import os
import re
import socket
import tarfile
import time
from datetime import datetime
from logging.handlers import BaseRotatingHandler
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


class EngineError(Exception):
    """Base exception class for engine-related errors.
    
    Attributes:
        message (str): Human-readable error description
        error_code (int): HTTP-style error code (default: 400)
    """

    def __init__(self, message, error_code=400):
        super().__init__(message)
        self.error_code = error_code


class ColoredFormatter(logging.Formatter):
    """Custom log formatter that adds color to console output.
    
    Colors different log levels for better visibility:
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red
    """
    COLOR_CODES = {
        logging.WARNING: 33,  # 黄色
        logging.ERROR: 31,  # 红色
        logging.CRITICAL: 31,  # 红色
    }

    def format(self, record):
        color_code = self.COLOR_CODES.get(record.levelno, 0)
        prefix = f'\033[{color_code}m'
        suffix = '\033[0m'
        message = super().format(record)
        if color_code:
            message = f"{prefix}{message}{suffix}"
        return message


class DailyRotatingFileHandler(BaseRotatingHandler):
    """Daily rotating file handler that supports multi-process logging.
    
    Similar to `logging.TimedRotatingFileHandler` but designed to work safely
    in multi-process environments.
    """

    def __init__(self,
                 filename,
                 backupCount=0,
                 encoding="utf-8",
                 delay=False,
                 utc=False,
                 **kwargs):
        """Initialize the rotating file handler.

        Args:
            filename (str): Path to the log file (can be relative or absolute)
            backupCount (int, optional): Number of backup files to keep. Defaults to 0.
            encoding (str, optional): File encoding. Defaults to "utf-8".
            delay (bool, optional): Delay file opening until first write. Defaults to False.
            utc (bool, optional): Use UTC timezone for rollover. Defaults to False.
            **kwargs: Additional arguments passed to BaseRotatingHandler.

        Raises:
            TypeError: If filename is not a string.
            ValueError: If backupCount is less than 0.
        """
        self.backup_count = backupCount
        self.utc = utc
        self.suffix = "%Y-%m-%d"
        self.base_log_path = Path(filename)
        self.base_filename = self.base_log_path.name
        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(
            self.current_filename)
        BaseRotatingHandler.__init__(self, filename, "a", encoding, delay)

    def shouldRollover(self, record):
        """Determine if a rollover should occur.
        
        Args:
            record (LogRecord): The log record being processed
            
        Returns:
            bool: True if rollover should occur, False otherwise
        """
        if self.current_filename != self._compute_fn():
            return True
        return False

    def doRollover(self):
        """Perform the actual rollover operation.
        
        Closes current file, creates new log file with current date suffix,
        and deletes any expired log files.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(
            self.current_filename)

        if not self.delay:
            self.stream = self._open()

        self.delete_expired_files()

    def _compute_fn(self):
        """Compute the current log filename with date suffix.
        
        Returns:
            str: Filename with current date suffix (format: filename.YYYY-MM-DD)
        """
        return self.base_filename + "." + time.strftime(
            self.suffix, time.localtime())

    def _open(self):
        """Open the current log file.
        
        Also creates a symlink from the base filename to the current log file.
        
        Returns:
            file object: The opened log file
        """
        if self.encoding is None:
            stream = open(str(self.current_log_path), self.mode)
        else:
            stream = codecs.open(str(self.current_log_path), self.mode,
                                 self.encoding)

        if self.base_log_path.exists():
            try:
                if (not self.base_log_path.is_symlink() or os.readlink(
                        self.base_log_path) != self.current_filename):
                    os.remove(self.base_log_path)
            except OSError:
                pass

        try:
            os.symlink(self.current_filename, str(self.base_log_path))
        except OSError:
            pass
        return stream

    def delete_expired_files(self):
        """Delete expired log files based on backup count.
        
        Only keeps the most recent backupCount files and deletes older ones.
        Does nothing if backupCount is <= 0.
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
            result = result[:len(result) - self.backup_count]

        for file_name in result:
            os.remove(str(self.base_log_path.with_name(file_name)))


def get_logger(name,
               file_name,
               without_formater=False,
               print_to_console=False):
    """Create and configure a logger instance.
    
    Args:
        name (str): Logger name
        file_name (str): Log file name (without path)
        without_formater (bool, optional): Skip adding formatter. Defaults to False.
        print_to_console (bool, optional): Also log to console. Defaults to False.
        
    Returns:
        Logger: Configured logger instance
    """
    log_dir = os.getenv("FD_LOG_DIR", default="log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    is_debug = int(os.getenv("FD_DEBUG", default="0"))
    logger = logging.getLogger(name)
    if is_debug:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    LOG_FILE = "{0}/{1}".format(log_dir, file_name)
    backup_count = int(os.getenv("FD_LOG_BACKUP_COUNT", "7"))
    handler = DailyRotatingFileHandler(LOG_FILE, backupCount=backup_count)
    formatter = ColoredFormatter(
        "%(levelname)-8s %(asctime)s %(process)-5s %(filename)s[line:%(lineno)d] %(message)s"
    )

    console_handler = logging.StreamHandler()
    if not without_formater:
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_to_console:
        logger.addHandler(console_handler)
    handler.propagate = False
    console_handler.propagate = False
    return logger


def str_to_datetime(date_string):
    """Convert string to datetime object.
    
    Supports both formats with and without microseconds.
    
    Args:
        date_string (str): Date string in format "YYYY-MM-DD HH:MM:SS" or 
                          "YYYY-MM-DD HH:MM:SS.microseconds"
                          
    Returns:
        datetime: Parsed datetime object
    """
    if "." in date_string:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
    else:
        return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


def datetime_diff(datetime_start, datetime_end):
    """Calculate time difference between two datetime points.
    
    Args:
        datetime_start (Union[str, datetime.datetime]): Start time
        datetime_end (Union[str, datetime.datetime]): End time
        
    Returns:
        float: Time difference in seconds (always positive)
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
    """Download a file from URL with progress bar.
    
    Args:
        url (str): File URL to download
        save_path (str): Local path to save the file
        
    Returns:
        bool: True if download succeeded
        
    Raises:
        RuntimeError: If download fails (file is deleted on failure)
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size,
                            unit='iB',
                            unit_scale=True,
                            desc=f"Downloading {os.path.basename(url)}")

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
        return True
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise RuntimeError(f"Download failed: {str(e)}")


def extract_tar(tar_path, output_dir):
    """Extract contents of a tar file with progress tracking.
    
    Args:
        tar_path (str): Path to tar file
        output_dir (str): Directory to extract files to
        
    Raises:
        RuntimeError: If extraction fails
    """
    try:
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting files") as pbar:
                for member in members:
                    tar.extract(member, path=output_dir)
                    pbar.update(1)
        print(f"Successfully extracted to: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {str(e)}")


def download_model(url, output_dir, temp_tar):
    """Download and extract a model from URL.
    
    Args:
        url (str): Model file URL
        output_dir (str): Directory to save extracted model
        temp_tar (str): Temporary tar filename for download
        
    Raises:
        Exception: If download or extraction fails
        RuntimeError: With link to model documentation if failure occurs
        
    Note:
        Cleans up temporary files even if operation fails
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
    """Extended ArgumentParser that supports loading parameters from YAML files.
    
    Supports nested configuration structures in YAML that get flattened
    into command-line style arguments.
    """

    def __init__(self, *args, config_arg='--config', sep='_', **kwargs):
        super().__init__(*args, **kwargs)
        self.sep = sep  # 用于展平嵌套字典的分隔符
        # 创建临时解析器，仅用于解析 --config 参数
        self.tmp_parser = argparse.ArgumentParser(add_help=False)
        self.tmp_parser.add_argument(config_arg,
                                     type=str,
                                     help='Path to YAML config file')

    def parse_args(self, args=None, namespace=None):
        """Parse arguments with support for YAML configuration files.
        
        Args:
            args: Argument strings to parse (default: sys.argv[1:])
            namespace: Namespace object to store attributes (default: new Namespace)
            
        Returns:
            Namespace: populated namespace object
            
        Note:
            Command line arguments override values from config file
        """
        # 使用临时解析器解析出 --config 参数
        tmp_ns, remaining_args = self.tmp_parser.parse_known_args(args=args)
        config_path = tmp_ns.config

        # 加载 YAML 文件并展平嵌套结构
        config = {}
        if config_path:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                config = self._flatten_dict(loaded_config)

        # 获取所有已定义参数的 dest 名称
        defined_dests = {action.dest for action in self._actions}

        # 过滤出已定义的参数
        filtered_config = {
            k: v
            for k, v in config.items() if k in defined_dests
        }

        # 创建或使用现有的命名空间对象
        if namespace is None:
            namespace = argparse.Namespace()

        # 将配置参数设置到命名空间
        for key, value in filtered_config.items():
            setattr(namespace, key, value)

        # 解析剩余参数并覆盖默认值
        return super().parse_args(args=remaining_args, namespace=namespace)

    def _flatten_dict(self, d):
        """Flatten nested dictionary into single level with joined keys.
        
        Args:
            d (dict): Nested dictionary to flatten
            
        Returns:
            dict: Flattened dictionary with keys joined by separator
        """

        def _flatten(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{self.sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return _flatten(d)


def resolve_obj_from_strname(strname: str):
    """Import and return an object from its full dotted path string.
    
    Args:
        strname (str): Full dotted path to object (e.g. "module.submodule.Class")
        
    Returns:
        object: The imported object
        
    Example:
        >>> resolve_obj_from_strname("os.path.join")
        <function join at 0x...>
    """
    module_name, obj_name = strname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def check_unified_ckpt(model_dir):
    """Check if directory contains a PaddleNLP unified checkpoint.
    
    Args:
        model_dir (str): Path to model directory
        
    Returns:
        bool: True if valid unified checkpoint, False otherwise
        
    Raises:
        Exception: If checkpoint appears corrupted
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
        safetensors_num = int(
            model_files[0].strip(".safetensors").split("-")[-1])
        flags = [0] * safetensors_num
        for x in model_files:
            current_index = int(x.strip(".safetensors").split("-")[1])
            flags[current_index - 1] = 1
        assert sum(flags) == len(
            model_files
        ), "Number of safetensor files should be {}, but now it's {}".format(
            len(model_files), sum(flags))
    except Exception as e:
        raise Exception(f"Failed to check unified checkpoint, details: {e}.")
    return is_unified_ckpt


def get_host_ip():
    """Get host machine's IP address.
    
    Returns:
        str: Host IP address
    """
    ip = socket.gethostbyname(socket.gethostname())
    return ip


def is_port_available(host, port):
    """Check if a network port is available for binding.
    
    Args:
        host (str): Hostname or IP address
        port (int): Port number
        
    Returns:
        bool: True if port is available, False if already in use
    """
    import errno
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                return False
            return True


llm_logger = get_logger("fastdeploy", "fastdeploy.log")
data_processor_logger = get_logger("data_processor", "data_processor.log")
api_server_logger = get_logger("api_server", "api_server.log")
console_logger = get_logger("console", "console.log", print_to_console=True)
