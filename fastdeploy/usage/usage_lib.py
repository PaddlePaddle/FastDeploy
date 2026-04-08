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

import datetime
import json
import multiprocessing
import os
import platform
import re
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import cpuinfo
import paddle
import psutil
import requests

from fastdeploy.config import FDConfig
from fastdeploy.platforms import current_platform
from fastdeploy.utils import api_server_logger, envs

_USAGE_STATS_ENABLED = None
_USAGE_STATS_SERVER = envs.FD_USAGE_STATS_SERVER
_GLOBAL_RUNTIME_DATA = dict[str, str | int | bool]()
_config_home = envs.FD_CONFIG_ROOT
_USAGE_STATS_JSON_PATH = os.path.join(_config_home, "usage_stats.json")
_USAGE_ENV_VARS_TO_COLLECT = [
    "ENABLE_V1_KVCACHE_SCHEDULER",
    "FD_DISABLE_CHUNKED_PREFILL",
    "FD_USE_HF_TOKENIZER",
    "FD_PLUGINS",
]


def set_runtime_usage_data(key: str, value: str | int | bool) -> None:
    """Set global usage data that will be sent with every usage heartbeat."""
    _GLOBAL_RUNTIME_DATA[key] = value


def is_usage_stats_enabled():
    """Determine whether or not we can send usage stats to the server.
    The logic is as follows:
    - By default, it should be enabled.
    - Three environment variables can disable it:
        - DO_NOT_TRACK=1
    """
    global _USAGE_STATS_ENABLED
    if _USAGE_STATS_ENABLED is None:
        do_not_track = envs.DO_NOT_TRACK

        _USAGE_STATS_ENABLED = not do_not_track
    return _USAGE_STATS_ENABLED


def get_current_timestamp_ns() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized."""
    if not paddle.is_compiled_with_cuda():
        return False
    return paddle.device.cuda.device_count() > 0


def cuda_get_device_properties(device, names: Sequence[str], init_cuda=False) -> tuple[Any, ...]:
    """Get specified CUDA device property values without initializing CUDA in
    the current process."""
    if init_cuda or cuda_is_initialized():
        try:
            props = paddle.device.cuda.get_device_properties(device)
            result = []
            for name in names:
                if name == "major":
                    value = props.major
                elif name == "minor":
                    value = props.minor
                elif name == "name":
                    value = props.name
                elif name == "total_memory":
                    value = props.total_memory
                elif name == "multi_processor_count":
                    value = props.multi_processor_count
                else:
                    value = getattr(props, name)
                result.append(value)
            return tuple(result)
        except Exception as e:
            api_server_logger.debug(f"Warning: Failed to get CUDA properties: {e}")
            return tuple([None] * len(names))

    # Run in subprocess to avoid initializing CUDA as a side effect.
    try:
        mp_ctx = multiprocessing.get_context("spawn")
    except ValueError:
        mp_ctx = multiprocessing.get_context()
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
        return executor.submit(cuda_get_device_properties, device, names, True).result()


def get_xpu_model():
    try:
        result = subprocess.run(["xpu-smi"], capture_output=True, text=True, timeout=5)

        if result.returncode != 0:
            return None

        pattern = r"^\|\s*(\d+)\s+(\w+)\s+\w+"
        lines = result.stdout.split("\n")

        for line in lines:
            match = re.search(pattern, line)
            if match:
                model = match.group(2)
                return model

        return "P800"
    except Exception:
        return "P800"


def get_cuda_version():
    try:
        result = os.popen("nvcc --version").read()
        if not result:
            return None

        regex = r"release (\S+),"
        match = re.search(regex, result)

        if match:
            return str(match.group(1))
        else:
            return None

    except Exception:
        return None


def cuda_device_count() -> int:
    if not paddle.device.is_compiled_with_cuda():
        return 0

    device_count = paddle.device.cuda.device_count()
    return device_count


def xpu_device_count() -> int:
    if not paddle.device.is_compiled_with_xpu():
        return 0

    device_count = paddle.device.xpu.device_count()
    return device_count


def detect_cloud_provider() -> str:
    if os.environ.get("SYS_JOB_NAME"):
        return "PDC"
    # Try detecting through vendor file
    vendor_files = [
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/chassis_asset_tag",
        "/sys/class/dmi/id/sys_vendor",
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        "amazon": "AWS",
        "microsoft corporation": "AZURE",
        "google": "GCP",
        "oraclecloud": "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider

    # Try detecting through environment variables
    env_to_cloud_provider = {
        "RUNPOD_DC_ID": "RUNPOD",
    }
    for env_var, provider in env_to_cloud_provider.items():
        if os.environ.get(env_var):
            return provider

    return "Unknown"


def simple_convert(obj):
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: simple_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [simple_convert(item) for item in obj]

    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except:
            return obj

    if hasattr(obj, "__dict__"):
        for method in ["to_dict", "to_json", "__getstate__", "as_dict"]:
            if hasattr(obj, method):
                result = getattr(obj, method)()
                if isinstance(result, dict):
                    return simple_convert(result)
                elif isinstance(result, str):
                    try:
                        return json.loads(result)
                    except:
                        return result

        try:
            return {k: simple_convert(v) for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            return str(obj)

    return str(obj)


class UsageMessage:
    """Collect platform information and send it to the usage stats server."""

    def __init__(self) -> None:

        self.uuid = str(uuid4())

        # Environment Information
        self.provider: str | None = None
        self.cpu_num: int | None = None
        self.cpu_type: str | None = None
        self.cpu_family_model_stepping: str | None = None
        self.total_memory: int | None = None
        self.architecture: str | None = None
        self.platform: str | None = None
        self.cuda_runtime: str | None = None
        self.gpu_num: int | None = None
        self.gpu_type: str | None = None
        self.gpu_memory_per_device: int | None = None
        self.env_var_json: str | None = None

        # FD Information
        self.model_architecture: str | None = None
        self.fd_version: str | None = None
        self.num_layers: int | None = None

        # Metadata
        self.log_time: int | None = None
        self.source: str | None = None
        self.config: str | None = None

    def report_usage(self, fd_config: FDConfig, extra_kvs: dict[str, Any] | None = None) -> None:
        t = Thread(
            target=self._report_usage_worker,
            args=(
                fd_config,
                extra_kvs,
            ),
            daemon=True,
        )
        t.start()

    def _report_usage_worker(self, fd_config: FDConfig, extra_kvs: dict[str, Any]) -> None:
        self._report_usage_once(fd_config, extra_kvs)
        self._report_continuous_usage()

    def _report_usage_once(self, fd_config: FDConfig, extra_kvs: dict[str, Any]):
        if current_platform.is_cuda_alike():
            self.gpu_num = cuda_device_count()
            self.gpu_type, self.gpu_memory_per_device = cuda_get_device_properties(0, ("name", "total_memory"))
        if current_platform.is_xpu():
            self.gpu_num = xpu_device_count()
            self.gpu_type = get_xpu_model()
            self.gpu_memory_per_device = paddle.device.xpu.memory_total()
        if current_platform.is_cuda():
            self.cuda_runtime = get_cuda_version()
        self.provider = detect_cloud_provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.total_memory = psutil.virtual_memory().total

        info = cpuinfo.get_cpu_info()
        self.cpu_num = info.get("count", None)
        self.cpu_type = info.get("brand_raw", "")
        self.cpu_family_model_stepping = ",".join(
            [
                str(info.get("family", "")),
                str(info.get("model", "")),
                str(info.get("stepping", "")),
            ]
        )
        self.env_var_json = json.dumps({env_var: getattr(envs, env_var) for env_var in _USAGE_ENV_VARS_TO_COLLECT})

        self.model_architecture = fd_config.model_config.architectures[0]
        from fastdeploy import __version__ as FD_VERSION

        self.fd_version = FD_VERSION
        self.log_time = get_current_timestamp_ns()
        self.source = envs.FD_USAGE_SOURCE

        self.config = json.dumps({k: simple_convert(v) for k, v in vars(fd_config).items()})
        data = vars(self)
        if extra_kvs:
            data.update(extra_kvs)
        self._write_to_file(data)
        self._send_to_server(data)

    def _send_to_server(self, data: dict[str, Any]) -> None:
        try:
            requests.post(url=_USAGE_STATS_SERVER, json=data)
        except requests.exceptions.RequestException as e:
            # silently ignore unless we are using debug log
            api_server_logger.debug(f"Failed to send usage data to server, errot: {str(e)}")

    def _report_continuous_usage(self):
        """Report usage every 10 minutes."""
        while True:
            time.sleep(600)
            data = {
                "uuid": self.uuid,
                "log_time": get_current_timestamp_ns(),
            }
            data.update(_GLOBAL_RUNTIME_DATA)

            self._write_to_file(data)
            self._send_to_server(data)

    def _write_to_file(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(_USAGE_STATS_JSON_PATH), exist_ok=True)
        Path(_USAGE_STATS_JSON_PATH).touch(exist_ok=True)
        with open(_USAGE_STATS_JSON_PATH, "a") as f:
            json.dump(data, f)
            f.write("\n")


def report_usage_stats(fd_config: FDConfig) -> None:
    """Report usage statistics if enabled."""
    if not is_usage_stats_enabled():
        return
    quant_val = fd_config.model_config.quantization
    if quant_val is None:
        quantization_str = None
    elif isinstance(quant_val, dict):
        quantization_str = quant_val.get("quantization")
    elif isinstance(quant_val, str):
        quantization_str = quant_val
    else:
        quantization_str = str(quant_val)
    usage_message = UsageMessage()
    usage_message.report_usage(
        fd_config,
        extra_kvs={
            "num_layers": fd_config.model_config.num_hidden_layers,
            "quantization": quantization_str,
            "block_size": fd_config.cache_config.block_size,
            "gpu_memory_utilization": fd_config.cache_config.gpu_memory_utilization,
            "enable_prefix_caching": fd_config.cache_config.enable_prefix_caching,
            "disable_custom_all_reduce": fd_config.parallel_config.disable_custom_all_reduce,
            "tensor_parallel_size": fd_config.parallel_config.tensor_parallel_size,
            "data_parallel_size": fd_config.parallel_config.data_parallel_size,
            "enable_expert_parallel": fd_config.parallel_config.enable_expert_parallel,
        },
    )
