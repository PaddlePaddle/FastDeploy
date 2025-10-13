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

from dataclasses import dataclass
from typing import Optional

import paddle

from fastdeploy.engine.pooling_params import PoolingParams


@dataclass
class PoolingCursor:
    index: list[int]
    first_token_indices_gpu: paddle.Tensor
    last_token_indices_gpu: paddle.Tensor
    prompt_lens_cpu: paddle.Tensor
    num_scheduled_tokens_cpu: paddle.Tensor

    def __getitem__(self, indices: slice):
        return PoolingCursor(
            index=self.index[indices],
            first_token_indices_gpu=self.first_token_indices_gpu[indices],
            last_token_indices_gpu=self.last_token_indices_gpu[indices],
            prompt_lens_cpu=self.prompt_lens_cpu[indices],
            num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
        )

    def is_partial_prefill(self):
        return not paddle.all(self.prompt_lens_cpu == self.num_scheduled_tokens_cpu).item()


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: paddle.Tensor  # CPU Tensor
    prompt_token_ids: Optional[paddle.Tensor]
    pooling_params: list[PoolingParams]
    pooling_cursor: Optional[PoolingCursor] = None

    def __getitem__(self, indices: slice):
        return PoolingMetadata(
            prompt_lens=self.prompt_lens[indices],
            prompt_token_ids=None if self.prompt_token_ids is None else self.prompt_token_ids[indices],
            pooling_params=self.pooling_params[indices],
            pooling_cursor=None if self.pooling_cursor is None else self.pooling_cursor[indices],
        )

    def build_pooling_cursor(self, num_scheduled_tokens: list[int], device: str):
        self.pooling_cursor = build_pooling_cursor(num_scheduled_tokens, self.prompt_lens, device)


def build_pooling_cursor(num_scheduled_tokens: list[int], prompt_lens: paddle.Tensor, device: str):
    assert len(prompt_lens) == len(num_scheduled_tokens)

    n_seq = len(num_scheduled_tokens)
    index = list(range(n_seq))
    num_scheduled_tokens = paddle.to_tensor(num_scheduled_tokens)
    cumsum = paddle.zeros([n_seq + 1], dtype="int64")

    paddle.cumsum(num_scheduled_tokens, axis=0, out=cumsum[1:])
    if device == "gpu":
        cumsum_device = cumsum.cuda()
    else:
        cumsum_device = cumsum
    return PoolingCursor(
        index=index,
        first_token_indices_gpu=cumsum_device[:n_seq],
        last_token_indices_gpu=cumsum_device[1:] - 1,
        prompt_lens_cpu=prompt_lens,
        num_scheduled_tokens_cpu=num_scheduled_tokens,
    )
