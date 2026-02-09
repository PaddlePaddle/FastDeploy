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
    padding_sampling_params,
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
        bad_words_token_len=paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64"),
        eos_token_ids=paddle.full(shape=[batch_size], fill_value=-2, dtype="int64"),
        min_p=paddle.randn([batch_size]),
        seed=paddle.full(shape=[batch_size], fill_value=0, dtype="int64"),
    )
    if max_num_logprobs is not None:
        fake_sampling_metadata.max_num_logprobs = max_num_logprobs
    return fake_sampling_metadata


def _create_fd_config(max_model_len):
    model_config: Mock = Mock()
    model_config.max_model_len = max_model_len
    model_config.architectures = ["test_model"]
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

    cu_seqlens_q_output = [0] + paddle.cumsum(share_inputs["seq_lens_this_time"]).numpy().tolist()
    share_inputs["cu_seqlens_q_output"] = paddle.to_tensor(cu_seqlens_q_output).cast("int32")
    share_inputs["batch_id_per_token_output"] = paddle.arange(max_num_seqs, dtype="int32") * 2

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
    share_inputs["reasoning_status"] = paddle.zeros([max_num_seqs], dtype="int32")

    return share_inputs


def _create_padding_inputs():
    # batch_size = 3
    top_p = paddle.to_tensor([[0.9], [0.8], [0.7], [1.0]], dtype="float32")
    top_k = paddle.to_tensor([[10], [20], [30], [40]], dtype="int32")
    infer_seed = paddle.to_tensor([[100], [200], [300], [400]], dtype="int64")

    # decoder, encoder, decoder
    seq_lens_encoder = paddle.to_tensor([[0], [5], [0], [0]], dtype="int32")
    seq_lens_this_time = paddle.to_tensor([[3], [2], [0], [2]], dtype="int32")

    return top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder


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


def test_padding_sampling_params_basic():
    top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder = _create_padding_inputs()

    top_p_pad, top_k_pad, seed_pad = padding_sampling_params(
        top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder
    )

    # decoder(3) + encoder(1) + decoder(2) = 6
    assert top_p_pad.shape == [6, 1]
    assert top_k_pad.shape == [6, 1]
    assert seed_pad.shape == [6, 1]

    # top_p padding check
    expected_top_p = [0.9, 0.9, 0.9, 0.8, 1.0, 1.0]
    assert paddle.allclose(top_p_pad.squeeze(), paddle.to_tensor(expected_top_p, dtype="float32"))

    # top_k padding check
    expected_top_k = [10, 10, 10, 20, 40, 40]
    assert paddle.equal_all(top_k_pad.squeeze(), paddle.to_tensor(expected_top_k, dtype="int32"))


def test_padding_sampling_params_seed_offset():
    top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder = _create_padding_inputs()

    _, _, seed_pad = padding_sampling_params(top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder)

    # decoder(0): 100 + 4*k
    # encoder(1): 200 (no offset)
    # null
    # decoder(3): 400 + 4*k
    expected_seed = [
        100,
        104,
        108,  # first decoder seq (len=3)
        200,  # encoder
        400,
        404,  # second decoder seq (len=2)
    ]

    assert paddle.equal_all(seed_pad.squeeze(), paddle.to_tensor(expected_seed, dtype="int64"))


if __name__ == "__main__":
    test_speculative_sampler()
    test_speculative_sampler_logprobs()
    test_mtp_sampler()
    test_mtp_sampler_logprobs()
    test_padding_sampling_params_basic()
    test_padding_sampling_params_seed_offset()
