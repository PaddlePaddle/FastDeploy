"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import asyncio
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Union

import aiohttp
import requests


class InstanceRole(Enum):
    MIXED = 0
    PREFILL = 1
    DECODE = 2


@dataclass
class InstanceInfo:
    role: Union[InstanceRole, str]
    host_ip: str
    port: Union[int, str]
    connector_port: Union[int, str] = 0
    engine_worker_queue_port: Union[int, str] = 0
    transfer_protocol: List[str] = field(default_factory=list)
    rdma_ports: Union[List[str], List[int]] = field(default_factory=list)
    device_ids: Union[List[str], List[int]] = field(default_factory=list)

    def __post_init__(self):
        """check and unify fields"""
        if isinstance(self.role, str):
            try:
                self.role = InstanceRole[self.role.upper()]
            except KeyError:
                raise ValueError(f"Invalid role string: {self.role}")
        elif not isinstance(self.role, InstanceRole):
            raise TypeError(f"role must be InstanceRole or str, got {type(self.role)}")

        for t in self.transfer_protocol:
            assert t in ["ipc", "rdma"], f"Invalid transfer_protocol: {self.transfer_protocol}"

        self.port = str(self.port)
        self.connector_port = str(self.connector_port)
        self.engine_worker_queue_port = str(self.engine_worker_queue_port)
        if self.rdma_ports:
            self.rdma_ports = [str(p) for p in self.rdma_ports]
        if self.device_ids:
            self.device_ids = [str(i) for i in self.device_ids]

    def to_dict(self):
        return {k: (v.name if isinstance(v, Enum) else v) for k, v in asdict(self).items()}

    def url(self) -> str:
        url = f"{self.host_ip}:{self.port}"
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        return url


def check_service_health(base_url: str, timeout: int = 3) -> bool:
    """
    Check the health status of a service.

    Args:
        base_url (str): The base URL of the service, e.g. "http://127.0.0.1:8080"
        timeout (int): Request timeout in seconds.

    Returns:
        bool: True if the service is healthy, False otherwise.
    """
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    url = f"{base_url.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False


async def check_service_health_async(base_url: str, timeout: int = 3) -> bool:
    """
    Asynchronously check the health status of a service.

    Args:
        base_url (str): The base URL of the service, e.g. "http://127.0.0.1:8080"
        timeout (int): Request timeout in seconds.

    Returns:
        bool: True if the service is healthy, False otherwise.
    """
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"

    url = f"{base_url.rstrip('/')}/health"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as resp:
                status = resp.status
                text = await resp.text()

                if status == 200:
                    print(f"[OK] Service is healthy ({status})")
                    return True
                else:
                    print(f"[WARN] Service not healthy ({status}): {text}")
                    return False
    except aiohttp.ClientError as e:
        print(f"[ERROR] Failed to connect to {url}: {e}")
        return False
    except asyncio.TimeoutError:
        print(f"[ERROR] Request to {url} timed out after {timeout}s")
        return False
