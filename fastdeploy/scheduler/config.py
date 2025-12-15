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

from fastdeploy.utils import get_logger, llm_logger

config_logger = get_logger("config", "config.log")

from .dp_scheduler import DPScheduler
from .global_scheduler import GlobalScheduler
from .local_scheduler import LocalScheduler
from .splitwise_scheduler import SplitWiseScheduler, SplitWiseSchedulerConfig


class LocalSchedulerConfig:
    """
    Configuration class for LocalScheduler.

    Attributes:
        max_size: Maximum number of concurrent requests (-1 for unlimited)
        ttl: Time-to-live in seconds for request expiration
    """

    def __init__(
        self,
        max_size: int = -1,
        ttl: int = 900,
        max_model_len: int = 8192,
        enable_chunked_prefill: bool = False,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: int = 0,
        **kwargs,
    ):
        """
        Initialize LocalScheduler configuration.

        Args:
            max_size: Maximum concurrent requests (-1 for unlimited, 0 for disabled)
            ttl: Time-to-live in seconds for request expiration (default 900s)
            max_model_len: Maximum model context length in tokens
            enable_chunked_prefill: Whether to enable chunked prefill processing
            max_num_partial_prefills: Max partial prefill operations allowed
            max_long_partial_prefills: Max long-running partial prefill ops
            long_prefill_token_threshold: Token count threshold for long prefill
            **kwargs: Additional unused arguments (for forward compatibility)

        Note:
            - If long_prefill_token_threshold is 0, it's auto-calculated as 4% of max_model_len
            - See LocalScheduler class for implementation details
        """
        self.max_size = max_size
        self.ttl = ttl

        self.max_model_len = max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold
        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

    def check(self):
        """
        Validate the configuration values.

        Currently performs no validation as all values are acceptable.
        """
        pass

    def print(self):
        """
        Print the current configuration to logs.
        """
        config_logger.info("LocalScheduler Configuration Information :")
        for k, v in self.__dict__.items():
            config_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        config_logger.info("=============================================================")


class DPLocalSchedulerConfig(LocalSchedulerConfig):
    """
    Configuration class for DPLocalScheduler.
    Attributes:
        max_size: Maximum number of concurrent requests (-1 for unlimited)
        ttl: Time-to-live in seconds for request expiration
    """

    def __init__(
        self,
        max_size: int = -1,
        ttl: int = 900,
        max_model_len: int = 8192,
        enable_chunked_prefill: bool = False,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: int = 0,
        splitwise_role: str = "prefill",
        **kwargs,
    ):
        """
        Initialize LocalScheduler configuration.
        Args:
            max_size: Maximum concurrent requests (-1 for unlimited, 0 for disabled)
            ttl: Time-to-live in seconds for request expiration (default 900s)
            max_model_len: Maximum model context length in tokens
            enable_chunked_prefill: Whether to enable chunked prefill processing
            max_num_partial_prefills: Max partial prefill operations allowed
            max_long_partial_prefills: Max long-running partial prefill ops
            long_prefill_token_threshold: Token count threshold for long prefill
            **kwargs: Additional unused arguments (for forward compatibility)
        Note:
            - If long_prefill_token_threshold is 0, it's auto-calculated as 4% of max_model_len
            - See LocalScheduler class for implementation details
        """
        self.max_size = max_size
        self.ttl = ttl

        self.max_model_len = max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold
        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)
        self.splitwise_role = splitwise_role


class GlobalSchedulerConfig:
    """
    Configuration class for GlobalScheduler (Redis-based).

    Attributes:
        host: Redis server hostname
        port: Redis server port
        db: Redis database number
        password: Optional Redis password
        topic: Namespace prefix for queues
        ttl: Time-to-live in seconds for Redis keys
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password=None,
        topic: str = "default",
        ttl: int = 900,
        min_load_score: float = 3,
        max_model_len: int = 8192,
        load_shards_num: int = 1,
        enable_chunked_prefill: bool = False,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: int = 0,
        **kwargs,
    ):
        """
        Initialize GlobalScheduler (Redis-based) configuration.

        Args:
            host: Redis server hostname (default "127.0.0.1")
            port: Redis server port (default 6379)
            db: Redis database number (default 0)
            password: Optional Redis password
            topic: Namespace prefix for queues (default "default")
            ttl: Time-to-live in seconds for Redis keys (default 900s)
            min_load_score: Minimum load score for task assignment (default 3)
            max_model_len: Maximum model context length in tokens
            load_shards_num: Number of load balancing shards
            enable_chunked_prefill: Whether to enable chunked prefill processing
            max_num_partial_prefills: Max partial prefill operations allowed
            max_long_partial_prefills: Max long-running partial prefill ops
            long_prefill_token_threshold: Token count threshold for long prefill
            **kwargs: Additional unused arguments (for forward compatibility)

        Note:
            - If long_prefill_token_threshold is 0, it's auto-calculated as 4% of max_model_len
            - See GlobalScheduler class for implementation details
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.topic = topic
        self.ttl = ttl
        self.min_load_score = min_load_score
        self.load_shards_num = load_shards_num

        self.max_model_len = max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold
        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

    def check(self):
        """
        Validate the configuration by testing Redis connection.

        Raises:
            Exception: If connection to Redis fails
        """

        if self.ttl <= 0:
            raise ValueError("ttl should be greater than 60")
        if self.min_load_score < 1:
            raise ValueError("min_load_score should be greater than 0")
        if self.load_shards_num < 1:
            raise ValueError("load_shards_num should be greater than 0")

        r = redis.Redis(self.host, self.port, self.db, self.password)
        try:
            response = r.ping()
            if not response:
                raise Exception("connect to redis failed")
        finally:
            r.close()

    def print(self):
        """
        Print the current configuration to logs.
        """
        llm_logger.info("GlobalScheduler Configuration Information :")
        password = self.password
        self.password = "******"
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        self.password = password
        llm_logger.info("=============================================================")


class SchedulerConfig:
    """
    Factory class for scheduler configurations.

    Creates appropriate config based on scheduler type (local/global).
    """

    def __init__(self, args):
        """
        Initialize scheduler configuration factory.

        Args:
            args: Configuration parameters for the specific scheduler type

        Initializes:
            - Appropriate config object based on scheduler type
            - Validates configuration parameters

        Raises:
            Exception: If invalid scheduler type is specified
        """
        self.name = "local"  # "local" for LocalScheduler or "global" for GlobalScheduler
        self.max_num_batched_tokens = 2048  # base token_num for text inputs
        self.max_extra_num_batched_tokens = 16384  # extra token_num for multimodal inputs
        self.max_num_seqs = 34
        self.splitwise_role = "mixed"
        self.config = None

        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.name == "local":
            self.config = LocalSchedulerConfig(**args)

        if self.name == "global":
            self.config = GlobalSchedulerConfig(**args)

        if self.name == "splitwise":
            self.config = SplitWiseSchedulerConfig(**args)

        if self.name == "dp":
            self.config = DPLocalSchedulerConfig(**args)

    def check(self):
        """
        Validate the configuration.

        Raises:
            Exception: If invalid scheduler type is specified
        """
        if self.name not in ["local", "global", "splitwise", "dp"]:
            raise Exception(f"Unknown scheduler type {self.name}")

        self.config.check()

    def print(self):
        """
        Print the current configuration to logs.
        """
        self.config.print()

    def scheduler(self):
        """
        Create a scheduler instance based on the configuration.

        Returns:
            Initialized scheduler instance (LocalScheduler or GlobalScheduler)
        """
        llm_logger.info("Scheduler Type: %s" % self.name)

        if self.name == "global":
            return GlobalScheduler(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                topic=self.config.topic,
                ttl=self.config.ttl,
                min_load_score=self.config.min_load_score,
                load_shards_num=self.config.load_shards_num,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                max_num_partial_prefills=self.config.max_num_partial_prefills,
                max_long_partial_prefills=self.config.max_long_partial_prefills,
                long_prefill_token_threshold=self.config.long_prefill_token_threshold,
            )

        if self.name == "splitwise":
            return SplitWiseScheduler(self.config)

        if self.name == "dp":
            return DPScheduler(
                max_size=self.config.max_size,
                ttl=self.config.ttl,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
                max_num_partial_prefills=self.config.max_num_partial_prefills,
                max_long_partial_prefills=self.config.max_long_partial_prefills,
                long_prefill_token_threshold=self.config.long_prefill_token_threshold,
                splitwise_role=self.config.splitwise_role,
            )

        return LocalScheduler(
            max_size=self.config.max_size,
            ttl=self.config.ttl,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            max_num_partial_prefills=self.config.max_num_partial_prefills,
            max_long_partial_prefills=self.config.max_long_partial_prefills,
            long_prefill_token_threshold=self.config.long_prefill_token_threshold,
        )
