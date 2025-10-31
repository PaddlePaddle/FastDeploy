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

from unittest.mock import Mock

import paddle

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import (
    MTPSampler,
    SpeculativeSampler,
)


def _create_fake_logits(batch_size: int, vocab_size: int) -> paddle.Tensor:
    fake_logits = paddle.rand(shape=[batch_size, vocab_size], dtype="float32")
    return fake_logits


def _create_penalty_tensor(batch_size: int, penalty_value: float) -> paddle.Tensor:
    return paddle.full(shape=[batch_size, 1], fill_value=penalty_value, dtype="float32")


def _create_tokens_tensor(
    batch_size: int,
    max_seq_len: int,
) -> paddle.Tensor:
    pre_token_ids = paddle.full(shape=[batch_size, max_seq_len], fill_value=-1, dtype="int64")
    return pre_token_ids


def _create_default_sampling_metadata(
    batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
    max_num_logprobs: int = None,
) -> SamplingMetadata:

    fake_sampling_metadata = SamplingMetadata(
        temperature=paddle.full(shape=[batch_size, 1], fill_value=0.9, dtype="float32"),
        top_p=paddle.full(shape=[batch_size, 1], fill_value=0.7, dtype="float32"),
        top_k=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int32"),
        prompt_ids=paddle.full(shape=[batch_size, max_seq_len], fill_value=0, dtype="int64"),
        prompt_lens=paddle.full(shape=[batch_size, 1], fill_value=5, dtype="int64"),
        step_idx=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        pre_token_ids=_create_tokens_tensor(batch_size, max_seq_len),
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0),
        min_dec_lens=paddle.full(shape=[batch_size, 1], fill_value=min_seq_len, dtype="int64"),
        bad_words_token_ids=paddle.full(shape=[batch_size], fill_value=-1, dtype="int64"),
        eos_token_ids=paddle.full(shape=[batch_size], fill_value=-2, dtype="int64"),
        min_p=paddle.randn([batch_size]),
        seed=paddle.to_tensor([[2025]]),
    )
    if max_num_logprobs is not None:
        fake_sampling_metadata.max_num_logprobs = max_num_logprobs
    return fake_sampling_metadata


def _create_fd_config(max_model_len):
    model_config: Mock = Mock()
    model_config.max_model_len = max_model_len
    speculative_config = SpeculativeConfig({})
    graph_opt_config = GraphOptimizationConfig({})
    scheduler_config = SchedulerConfig({})
    parallel_config = ParallelConfig({})
    cache_config = CacheConfig({})
    cache_config.cache_transfer_protocol = "rdma,ipc"
    cache_config.pd_comm_port = "2334"
    fd_config = FDConfig(
        model_config=model_config,
        speculative_config=speculative_config,
        graph_opt_config=graph_opt_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
    )

    return fd_config


def _create_share_inputs(max_num_seqs, max_draft_token_num, max_model_len, vocab_size):
    share_inputs = {}
    share_inputs["seq_lens_this_time"] = paddle.full([max_num_seqs, 1], 2, dtype="int32")
    share_inputs["output_cum_offsets"] = paddle.concat(
        [(max_model_len - share_inputs["seq_lens_this_time"][i]) * i for i in range(max_num_seqs)]
    )
    share_inputs["output_padding_offset"] = paddle.repeat_interleave(share_inputs["output_cum_offsets"], 2)
    share_inputs["accept_tokens"] = paddle.full(
        shape=[max_num_seqs, max_draft_token_num + 1], fill_value=0, dtype="int64"
    )
    share_inputs["accept_num"] = paddle.full(shape=[max_num_seqs], fill_value=1, dtype="int32")
    share_inputs["step_idx"] = paddle.full([max_num_seqs, 1], 1, dtype="int64")
    share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1], False, dtype="bool")
    share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
    share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 2, dtype="int32")
    share_inputs["draft_tokens"] = paddle.full(
        shape=[max_num_seqs, max_draft_token_num + 1], fill_value=0, dtype="int64"
    )
    share_inputs["max_dec_len"] = paddle.full([max_num_seqs, 1], max_model_len, dtype="int64")
    share_inputs["is_block_step"] = paddle.full([max_num_seqs], False, dtype="bool")
    share_inputs["actual_draft_token_num"] = paddle.full(
        shape=[max_num_seqs], fill_value=max_draft_token_num, dtype="int32"
    )

    share_inputs["batch_token_num"] = paddle.where(
        share_inputs["seq_lens_encoder"] != 0,
        paddle.ones_like(share_inputs["seq_lens_encoder"]),
        share_inputs["seq_lens_this_time"],
    ).squeeze(1)
    share_inputs["next_token_num"] = paddle.full(shape=[max_num_seqs], fill_value=0, dtype="int32")
    share_inputs["cu_batch_token_offset"] = paddle.concat(
        [paddle.to_tensor([0]), paddle.cumsum(share_inputs["accept_num"])]
    ).astype("int32")
    share_inputs["cu_next_token_offset"] = paddle.full(shape=[max_num_seqs + 1], fill_value=0, dtype="int32")
    share_inputs["substep"] = 0
    share_inputs["draft_logits"] = paddle.full(
        [max_num_seqs * (max_draft_token_num + 1), vocab_size], -1, dtype="float32"
    )

    return share_inputs


def test_speculative_sampler():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024
    max_model_len = 1024
    max_draft_token_num = 1

    fd_config = _create_fd_config(max_model_len)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len)
    logits = _create_fake_logits(batch_size * (max_draft_token_num + 1), vocab_size)
    share_inputs = _create_share_inputs(batch_size, max_draft_token_num, max_model_len, vocab_size)

    sampler = SpeculativeSampler(fd_config)
    sampler(logits, sampling_metadata, max_model_len, share_inputs)


def test_speculative_sampler_logprobs():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024
    max_model_len = 1024
    max_draft_token_num = 1

    fd_config = _create_fd_config(max_model_len)
    share_inputs = _create_share_inputs(batch_size, max_draft_token_num, max_model_len, vocab_size)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len, max_num_logprobs=0)
    sampling_metadata.share_inputs = share_inputs
    logits = _create_fake_logits(batch_size * (max_draft_token_num + 1), vocab_size)

    logprobs_mode_list = ["raw_logprobs", "raw_logits"]
    for logprobs_mode in logprobs_mode_list:
        fd_config.model_config.logprobs_mode = logprobs_mode
        sampler = SpeculativeSampler(fd_config)
        sampler(logits, sampling_metadata, max_model_len, share_inputs)


def test_mtp_sampler():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024
    max_model_len = 1024
    max_draft_token_num = 1

    fd_config = _create_fd_config(max_model_len)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len)
    logits = _create_fake_logits(batch_size * (max_draft_token_num + 1), vocab_size)

    share_inputs = _create_share_inputs(batch_size, max_draft_token_num, max_model_len, vocab_size)

    sampler = MTPSampler(fd_config)
    sampler(logits, sampling_metadata, max_model_len, share_inputs)


def test_mtp_sampler_logprobs():
    batch_size = 32
    vocab_size = 1024
    min_seq_len = 1
    max_seq_len = 1024
    max_model_len = 1024
    max_draft_token_num = 1

    fd_config = _create_fd_config(max_model_len)
    share_inputs = _create_share_inputs(batch_size, max_draft_token_num, max_model_len, vocab_size)
    sampling_metadata = _create_default_sampling_metadata(batch_size, min_seq_len, max_seq_len, max_num_logprobs=0)
    sampling_metadata.share_inputs = share_inputs
    logits = _create_fake_logits(batch_size * (max_draft_token_num + 1), vocab_size)

    logprobs_mode_list = ["raw_logprobs", "raw_logits"]
    for logprobs_mode in logprobs_mode_list:
        fd_config.model_config.logprobs_mode = logprobs_mode
        sampler = MTPSampler(fd_config)
        sampler(logits, sampling_metadata, max_model_len, share_inputs)


if __name__ == "__main__":
    test_speculative_sampler()
    test_speculative_sampler_logprobs()
    test_mtp_sampler()
    test_mtp_sampler_logprobs()
