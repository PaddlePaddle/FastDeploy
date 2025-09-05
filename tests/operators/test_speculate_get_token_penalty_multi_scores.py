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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_get_token_penalty_multi_scores


def min_length_logits_process(
    logits,
    cur_len,
    min_len,
    eos_token_id,
    output_padding_offset,
    output_cum_offsets,
    token_num,
    bs,
    length,
    end_length,
    max_seq_len,
):
    for token_idx in range(token_num):
        bi = (token_idx + output_padding_offset[token_idx]) / max_seq_len
        bi = bi.astype(paddle.int32)
        if bi >= bs:
            continue
        query_start_token_idx = bi * max_seq_len - output_cum_offsets[bi]

        if cur_len[bi] < 0:
            continue
        if cur_len[bi] + (token_idx - query_start_token_idx) < min_len[bi]:
            for i in range(end_length):
                logits[token_idx][eos_token_id[i]] = -1e10


def update_repeat_times(
    pre_ids, cur_len, repeat_times, output_padding_offset, token_num, bs, length, length_id, max_seq_len
):
    for token_idx in range(token_num):
        bi = (token_idx + output_padding_offset[token_idx]) / max_seq_len
        bi = bi.astype(paddle.int32)
        if bi >= bs:
            continue
        if cur_len[bi] < 0:
            continue

        pre_ids_now = pre_ids[bi]
        repeat_times_now = repeat_times[token_idx]

        for i in range(length_id):
            id = pre_ids_now[i]
            if id < 0:
                break
            repeat_times_now[id] = repeat_times_now[id] + 1


def update_value_by_repeat_times(
    repeat_times,
    penalty_scores,
    frequency_score,
    presence_score,
    temperatures,
    logits,
    output_padding_offset,
    token_num,
    bs,
    length,
    max_seq_len,
):
    for token_idx in range(token_num):
        bi = (token_idx + output_padding_offset[token_idx]) / max_seq_len
        bi = bi.astype(paddle.int32)
        if bi >= bs:
            continue
        logits_now = logits[token_idx]
        repeat_times_now = repeat_times[token_idx]
        alpha = penalty_scores[bi]
        beta = frequency_score[bi]
        gamma = presence_score[bi]
        for i in range(length):
            times = repeat_times_now[i]
            logit_now = logits_now[i]
            if times != 0:
                logit_now = logit_now * alpha if logit_now < 0 else logit_now / alpha
                logit_now = logit_now - times * beta - gamma

            logits_now[i] = logit_now / temperatures[bi]


def ban_bad_words(logits, bad_words_list, output_padding_offset, token_num, bs, length, bad_words_length, max_seq_len):
    for token_idx in range(token_num):
        bi = (token_idx + output_padding_offset[token_idx]) / max_seq_len
        bi = bi.astype(paddle.int32)
        if bi >= bs:
            continue
        logits_now = logits[token_idx]
        for i in range(bad_words_length):
            bad_words_token_id = bad_words_list[i]
            if bad_words_token_id >= length or bad_words_token_id < 0:
                continue
            logits_now[bad_words_token_id] = -1e10


def speculate_get_token_penalty_multi_scores_ref(
    pre_ids,
    logits,
    penalty_scores,
    frequency_score,
    presence_score,
    temperatures,
    bad_tokens,
    cur_len,
    min_len,
    eos_token_id,
    seq_lens_this_time,
    output_padding_offset,
    output_cum_offsets,
    max_seq_len,
):
    shape = logits.shape
    repeat_times = paddle.full(shape, 0, dtype=paddle.int32)
    bs = seq_lens_this_time.shape[0]
    token_num = shape[0]
    length = shape[1]
    length_id = pre_ids.shape[1]
    length_bad_words = bad_tokens.shape[1]

    end_length = eos_token_id.shape[0]

    min_length_logits_process(
        logits,
        cur_len,
        min_len,
        eos_token_id,
        output_padding_offset,
        output_cum_offsets,
        token_num,
        bs,
        length,
        end_length,
        max_seq_len,
    )

    update_repeat_times(
        pre_ids, cur_len, repeat_times, output_padding_offset, token_num, bs, length, length_id, max_seq_len
    )

    update_value_by_repeat_times(
        repeat_times,
        penalty_scores,
        frequency_score,
        presence_score,
        temperatures,
        logits,
        output_padding_offset,
        token_num,
        bs,
        length,
        max_seq_len,
    )

    ban_bad_words(logits, bad_tokens, output_padding_offset, token_num, bs, length, length_bad_words, max_seq_len)


class TestSpeculateGetTokenPenaltyMultiScores(unittest.TestCase):
    def test_speculate_get_token_penalty_multi_scores(self):
        paddle.seed(2023)
        np.random.seed(2023)

        bs = 64
        max_seq_len = 1024  # 1024 #2048 #8192
        data_type = "float32"

        # prepare output_padding_offset and output_cum_offsets
        tokens = [1] * bs
        token_num = np.sum(tokens)
        output_padding_offset = []
        output_cum_offsets = [0]
        opo_offset = 0
        for bid in range(bs):
            ts = tokens[bid]
            for i in range(ts):
                output_padding_offset.append(opo_offset)
            opo_offset += max_seq_len - ts
            output_cum_offsets.append(opo_offset)
        output_cum_offsets = output_cum_offsets[:-1]
        output_padding_offset = paddle.to_tensor(output_padding_offset, "int32")
        output_cum_offsets = paddle.to_tensor(output_cum_offsets, "int32")

        # prepare pre_ids and logits
        pre_ids_len = 122
        logits_len = 110
        pre_ids = np.random.randint(1, logits_len, size=(bs, pre_ids_len))
        negative_start = np.random.randint(1, pre_ids_len + 1, size=(bs))
        for i in range(bs):
            pre_ids[:, negative_start[i] :] = -1
        pre_ids = paddle.to_tensor(pre_ids).astype("int64")
        logits = paddle.zeros([token_num, logits_len]).astype(data_type)
        # prepare other params
        penalty_scores = paddle.to_tensor(np.random.random([bs])).astype(data_type)
        frequency_scores = paddle.to_tensor(np.random.random([bs])).astype(data_type)
        presence_scores = paddle.to_tensor(np.random.random([bs])).astype(data_type)
        temperatures = paddle.to_tensor(np.random.random([bs])).astype("float32")
        bad_tokens = paddle.to_tensor(np.random.randint(1, 2, size=([bs, 1]))).astype("int64")
        cur_len = paddle.to_tensor(np.random.randint(1, 50, size=(bs))).astype("int64")
        min_len = paddle.to_tensor(np.random.randint(1, 50, size=(bs))).astype("int64")
        eos_token_id = paddle.to_tensor(np.random.randint(1, 64, size=(bs))).astype("int64")
        seq_len_this_time = paddle.to_tensor(
            np.random.randint(0, 1, size=(bs)), "int32"
        )  # value of seq_len_this_time is useless

        inputs = (
            pre_ids,
            logits,
            penalty_scores,
            frequency_scores,
            presence_scores,
            temperatures,
            bad_tokens,
            cur_len,
            min_len,
            eos_token_id,
            seq_len_this_time,
            output_padding_offset,
            output_cum_offsets,
            max_seq_len,
        )
        # inplace modify, not return data
        inputs_clone = [x.clone() if isinstance(x, paddle.Tensor) else x for x in inputs]
        speculate_get_token_penalty_multi_scores(*inputs)
        speculate_get_token_penalty_multi_scores_ref(*inputs_clone)

        np.testing.assert_allclose(inputs[1].numpy(), inputs_clone[1].numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
