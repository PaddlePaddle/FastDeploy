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

import socket


def get_hostname_ip():
    """
    Get the system's hostname and primary IP address.

    Returns:
        tuple: A tuple containing:
            - hostname (str): The system's hostname
            - ip_address (str): The primary IP address associated with the hostname

    Raises:
        socket.gaierror: If the hostname cannot be resolved to an IP address
    """

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return hostname, ip_address
