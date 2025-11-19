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

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import paddle.distributed as dist

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.utils import spec_logger


class Proposer(ABC):
    """
    Proposer Base Class.

    Used to provide an extensible interface for draft tokens within
    the speculative decoding framework
    """

    def __init__(self, fd_config: FDConfig):
        """
        Init Speculative proposer
        """
        fd_config.parallel_config.tp_group = None
        fd_config.parallel_config.ep_group = None
        self.fd_config = deepcopy(fd_config)
        fd_config.parallel_config.tp_group = dist.get_group(
            fd_config.parallel_config.data_parallel_rank + envs.FD_TP_GROUP_GID_OFFSET
        )
        fd_config.parallel_config.ep_group = dist.get_group(
            fd_config.parallel_config.data_parallel_size + envs.FD_TP_GROUP_GID_OFFSET
        )
        self.fd_config.parallel_config.tp_group = dist.get_group(
            fd_config.parallel_config.data_parallel_rank + envs.FD_TP_GROUP_GID_OFFSET
        )
        self.fd_config.parallel_config.ep_group = dist.get_group(
            fd_config.parallel_config.data_parallel_size + envs.FD_TP_GROUP_GID_OFFSET
        )
        self.parallel_config = self.fd_config.parallel_config
        self.model_config = self.fd_config.model_config
        self.speculative_config = self.fd_config.speculative_config
        self.cache_config = self.fd_config.cache_config
        self.quant_config = self.fd_config.quant_config
        self.graph_opt_config = self.fd_config.graph_opt_config
        self.scheduler_config = self.fd_config.scheduler_config

        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_model_len = self.model_config.max_model_len
        self.speculative_method = self.speculative_config.method
        self.max_draft_token_num = self.speculative_config.num_speculative_tokens
        self.num_model_steps = self.speculative_config.num_model_steps

        self.max_ngram_size = self.speculative_config.max_ngram_size
        self.min_ngram_size = self.speculative_config.min_ngram_size

        self.enable_mm = self.model_config.enable_mm

        spec_logger.info(f"Speculate config: {self.speculative_config}")

    def run(self, *args, **kwargs) -> Any:
        """
        Unified entry point for all proposer types.
        Dispatches to subclass-specific logic via `_run_impl`.
        """
        return self._run_impl(*args, **kwargs)

    @abstractmethod
    def _run_impl(self, *args, **kwargs) -> Any:
        """
        Implementation for different method
        """
        raise NotImplementedError

    def is_chunk_prefill_enabled(self) -> bool:
        """
        Check whether chunk-based prefill is enabled.
        Default is False.

        Returns:
            bool: True if chunk prefill is enabled; False otherwise.
        """
        return False
