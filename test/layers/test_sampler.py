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

import paddle

from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler


def _create_fake_logits(batch_size: int, vocab_size: int) -> paddle.Tensor:
    fake_logits = paddle.full(shape=[batch_size, vocab_size],
                              fill_value=1e-2,
                              dtype="float32")
    return fake_logits


def _create_penalty_tensor(batch_size: int,
                           penalty_value: float) -> paddle.Tensor:
    return paddle.full(shape=[batch_size, 1],
                       fill_value=penalty_value,
                       dtype="float32")


def _create_tokens_tensor(
    batch_size: int,
    max_seq_len: int,
) -> paddle.Tensor:
    pre_token_ids = paddle.full(shape=[batch_size, max_seq_len],
                                fill_value=-1,
                                dtype="int64")
    return pre_token_ids


def _create_default_sampling_metadata(
    batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
) -> SamplingMetadata:

    fake_sampling_metadata = SamplingMetadata(
        temperature=paddle.full(shape=[batch_size, 1],
                                fill_value=0.9,
                                dtype="float32"),
        top_p=paddle.full(shape=[batch_size, 1],
                          fill_value=0.7,
                          dtype="float32"),
        step_idx=paddle.full(shape=[batch_size, 1],
                             fill_value=0,
                             dtype="int64"),
        pre_token_ids=_create_tokens_tensor(batch_size, max_seq_len),
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0),
        min_dec_lens=paddle.full(shape=[batch_size, 1],
                                 fill_value=min_seq_len,
                                 dtype="int64"),
        bad_words_token_ids=paddle.full(shape=[batch_size],
                                        fill_value=-1,
                                        dtype="int64"),
        eos_token_ids=paddle.full(shape=[batch_size],
                                  fill_value=-2,
                                  dtype="int64"),
    )
    return fake_sampling_metadata


def test_sampler():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024

    sampler = Sampler()
    logits = _create_fake_logits(batch_size, vocab_size)
    sampling_metadata = _create_default_sampling_metadata(
        batch_size, min_seq_len, max_seq_len)
    next_tokens = sampler(logits, sampling_metadata)
    print(next_tokens)


if __name__ == "__main__":
    test_sampler()
