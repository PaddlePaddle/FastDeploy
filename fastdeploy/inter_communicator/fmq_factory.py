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

from fastdeploy.inter_communicator.fmq import FMQ


class FMQFactory:
    """
    Static factory for creating the four standard FMQ queues:
        1. q_a2e: api server --> engine
        2. q_e2w: engine --> worker
        3. q_w2e: worker --> engine
        4. q_e2a: engine --> api server

    NOTE:
    - FMQ instances are NOT reused
    - Each call creates a new FMQ object
    """

    # ------------------------------
    # API → Engine
    # ------------------------------
    @classmethod
    def q_a2e_producer(cls):
        return FMQ().queue("q_a2e", role="producer")

    @classmethod
    def q_a2e_consumer(cls):
        return FMQ().queue("q_a2e", role="consumer")

    # ------------------------------
    # Engine → Worker
    # ------------------------------
    @classmethod
    def q_e2w_producer(cls):
        return FMQ().queue("q_e2w", role="producer")

    @classmethod
    def q_e2w_consumer(cls):
        return FMQ().queue("q_e2w", role="consumer")

    # ------------------------------
    # Worker → Engine
    # ------------------------------
    @classmethod
    def q_w2e_producer(cls, name: str):
        return FMQ().queue(f"q_w2e_{name}", role="producer")

    @classmethod
    def q_w2e_consumer(cls, name: str):
        return FMQ().queue(f"q_w2e_{name}", role="consumer")

    # ------------------------------
    # Engine → API
    # ------------------------------
    @classmethod
    def q_e2a_producer(cls):
        return FMQ().queue("q_e2a", role="producer")

    @classmethod
    def q_e2a_consumer(cls):
        return FMQ().queue("q_e2a", role="consumer")
