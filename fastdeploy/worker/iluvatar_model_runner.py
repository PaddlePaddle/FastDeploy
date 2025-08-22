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

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention import IluvatarAttnBackend
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class IluvatarModelRunner(GPUModelRunner):
    def __init__(
        self,
        fd_config: FDConfig,
        device: str,  # logic device
        device_id: int,  # physical device id
        rank: int,
        local_rank: int,
    ):
        super(IluvatarModelRunner, self).__init__(
            fd_config=fd_config, device=device, device_id=device_id, rank=rank, local_rank=local_rank
        )
        assert not self.speculative_decoding, "Iluvatar does not support speculative decoding"
        assert self.guided_backend is None, "Iluvatar does not support guided decoding"
        assert not envs.ENABLE_V1_KVCACHE_SCHEDULER, "Iluvatar does not support v1 kvcache scheduler"
        assert not self.cache_config.enable_prefix_caching, "Iluvatar does not support prefix caching"

    def initialize_attn_backend(self) -> None:
        """
        Initialize attention backends
        """
        assert len(self.attn_backends) == 0

        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )
        attn_backend = IluvatarAttnBackend(
            self.fd_config,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=self.model_config.head_dim,
        )
        self.attn_backends.append(attn_backend)
