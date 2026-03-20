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

from functools import partial

import paddle

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


def _patch_before_model_runner():
    paddle.Tensor.pin_memory = paddle.Tensor.cpu
    paddle.device.cuda.create_event = partial(paddle.device.custom_device.create_event, device_type="iluvatar_gpu")

    def disable_record(self):
        pass

    paddle.device.custom_device.Event.record = disable_record

    def disable_synchronize(self):
        pass

    paddle.device.custom_device.Event.synchronize = disable_synchronize


_patch_before_model_runner()


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
        assert not self.cache_config.enable_prefix_caching, "Iluvatar does not support prefix caching"
        self.mla_cache = envs.FD_ATTENTION_BACKEND == "MLA_ATTN"
        assert not self.mla_cache, "Iluvatar does not support MLA"
        self.dsa_cache = envs.FD_ATTENTION_BACKEND == "DSA_ATTN"
        assert not self.dsa_cache, "Iluvatar does not support DSA_ATTN"
        if self.enable_mm:
            assert (
                not self.cache_config.enable_chunked_prefill
            ), "Iluvatar does not support chunked prefill for VL model"

        print(f"self.use_cudagraph={self.use_cudagraph}")
        # VL neox style = True
        emb_shape = self.share_inputs["rope_emb"].shape
        if emb_shape[-1] == self.model_config.head_dim // 2:
            emb_shape[-1] = self.model_config.head_dim
            self.share_inputs["rope_emb"] = paddle.full(
                shape=emb_shape,
                fill_value=0,
                dtype="float32",
            )

    def _initialize_attn_backend(self) -> None:
        """
        Initialize attention backends
        """
        assert (
            len(self.attn_backends) == 0
        ), f"attn_backends should be empty before initialization, got {len(self.attn_backends)} backends"

        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )
        attn_cls = get_attention_backend()
        attn_backend = attn_cls(
            self.fd_config,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=self.model_config.head_dim,
        )
        self.attn_backends.append(attn_backend)

    def initialize_kv_cache(self, profile: bool = False) -> None:
        super(IluvatarModelRunner, self).initialize_kv_cache(profile)
        paddle.device.empty_cache()

    def initialize_forward_meta(self, is_dummy_or_profile_run=False):
        super(IluvatarModelRunner, self).initialize_forward_meta(is_dummy_or_profile_run)
        only_decode = self.forward_meta.attn_backend.prefill_len == 0
        self.fd_config.model_config.moe_phase.phase = "decode" if only_decode else "prefill"

    def clear_cache(self):
        super(IluvatarModelRunner, self).clear_cache()
        paddle.device.empty_cache()

    def clear_parameters(self, pid):
        super(IluvatarModelRunner, self).clear_parameters(pid)
        paddle.device.empty_cache()
