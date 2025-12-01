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

"""
class for record metrics stats
"""

from dataclasses import dataclass


@dataclass
class ZMQMetricsStats:

    # ZMQ send
    msg_send_total: int = 0
    msg_send_failed_total: int = 0
    msg_bytes_send_total: int = 0

    # ZMQ receive
    msg_recv_total: int = 0
    msg_bytes_recv_total: int = 0

    # ZMQ latency
    zmq_latency: float = 0.0
