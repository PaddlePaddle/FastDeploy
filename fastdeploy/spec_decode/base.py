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

from fastdeploy.config import FDConfig
from fastdeploy.utils import spec_logger


class Proposer(ABC):
    """
    Proposer Base Class.

    Used to provide an extensible interface for draft tokens within
    the speculative decoding framework
    """

    def __init__(self, cfg: FDConfig):
        """
        Init Speculative proposer
        """
        self.cfg = deepcopy(cfg)
        self.parallel_config = self.cfg.parallel_config
        self.model_config = self.cfg.model_config
        self.speculative_config = self.cfg.speculative_config
        self.kv_cache_config = self.cfg.kv_cache_config
        self.quant_config = self.cfg.quant_config

        self.max_num_seqs = self.parallel_config.max_num_seqs
        self.max_model_len = self.parallel_config.max_model_len
        self.speculative_method = self.speculative_config.method
        self.max_draft_token_num = self.speculative_config.num_speculative_tokens

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
        Implemention for different method
        """
        raise NotImplementedError
