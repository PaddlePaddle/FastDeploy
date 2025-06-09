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

import redis
from fastdeploy.utils import llm_logger
from .global_scheduler import GlobalScheduler
from .local_scheduler import LocalScheduler


class LocalSchedulerConfig:
    """
    LocalSchedulerConfig class
    """

    def __init__(self,
                 max_size: int = -1,
                 ttl: int = 900,
                 wait_response_timeout: float = 1,
                 **kwargs
                 ):
        self.max_size = max_size
        self.ttl = ttl
        self.wait_response_timeout = wait_response_timeout

    def check(self):
        """
        check config
        """
        assert self.wait_response_timeout > 0, \
            "LocalScheduler: `wait_response_timeout` must be greater than zero"
        assert self.ttl > self.wait_response_timeout, \
            "LocalScheduler: `ttl` must be greater than `wait_response_timeout`"

    def print(self):
        """
        print config
        """
        llm_logger.info("LocalScheduler Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class GlobalSchedulerConfig:
    """
    GlobalSchedulerConfig class
    """

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 6379,
                 db: int = 0,
                 password=None,
                 topic: str = "default",
                 ttl: int = 900,
                 wait_response_timeout: float = 1,
                 remote_write_time: int = 3,
                 **kwargs
                 ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.topic = topic
        self.ttl = ttl
        self.wait_response_timeout = wait_response_timeout
        self.remote_write_time = remote_write_time

    def check(self):
        """
        check config
        """
        assert self.wait_response_timeout > 0, \
            "GlobalScheduler: `wait_response_timeout` must be greater than zero"
        assert self.remote_write_time > 0, \
            "GlobalScheduler: `remote_write_time` must be greater than zero"
        assert self.ttl > self.remote_write_time, \
            "GlobalScheduler: `ttl` must be greater than `remote_write_time`"
        assert self.ttl > self.wait_response_timeout, \
            "GlobalScheduler: `ttl` must be greater than `wait_response_timeout`"

        r = redis.Redis(self.host, self.port, self.db, self.password)
        try:
            response = r.ping()
            if not response:
                raise Exception("connect to redis failed")
        finally:
            r.close()

    def print(self):
        """
        print config
        """
        llm_logger.info("GlobalScheduler Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class SchedulerConfig:
    """
    SchedulerConfig class
    """

    def __init__(self, name="local", **kwargs):
        self.name = name
        self.config = None

        if name == "local":
            self.config = LocalSchedulerConfig(**kwargs)

        if name == "global":
            self.config = GlobalSchedulerConfig(**kwargs)

    def check(self):
        """
        check config
        """
        if self.name not in ["local", "global"]:
            raise Exception(
                "SchedulerConfig: `name` must be `local` or `global`")

        self.config.check()

    def print(self):
        """
        print config
        """
        self.config.print()

    def scheduler(self):
        """
        create scheduler by config
        """

        if self.name == "global":
            return GlobalScheduler(host=self.config.host,
                                   port=self.config.port,
                                   db=self.config.db,
                                   password=self.config.password,
                                   topic=self.config.topic,
                                   ttl=self.config.ttl,
                                   remote_write_time=self.config.remote_write_time,
                                   wait_response_timeout=self.config.wait_response_timeout)

        return LocalScheduler(max_size=self.config.max_size,
                              ttl=self.config.ttl,
                              wait_response_timeout=self.config.wait_response_timeout
                              )
