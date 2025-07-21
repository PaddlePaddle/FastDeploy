# Copyright (c) 2025PaddlePaddle Authors. All Rights Reserved.
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
"""UT for air_topp_sampling kernel"""

import copy
import unittest

import numpy as np
import paddle


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        self.num_seqs = 4
        self.max_model_len = 32768
        self.vocab_size = 103424

        # prompt token
        prompt_ids = paddle.full(
            shape=[self.num_seqs, self.max_model_len],
            fill_value=0,
            dtype="int64",
        )
        prompt_lens = paddle.randint(low=0, high=100, shape=[self.num_seqs, 1], dtype="int64")
        fake_tokens = paddle.randint(
            low=3,
            high=self.vocab_size,
            shape=[self.num_seqs, self.max_model_len],
            dtype="int64",
        )
        for i in range(self.num_seqs):
            prompt_ids[i, : prompt_lens[i]] = fake_tokens[i, : prompt_lens[i]]

        # generated token
        pre_ids = paddle.full(
            shape=[self.num_seqs, self.max_model_len],
            fill_value=-1,
            dtype="int64",
        )
        step_idx = paddle.randint(low=0, high=100, shape=[self.num_seqs, 1], dtype="int64")
        fake_tokens = paddle.randint(
            low=3,
            high=self.vocab_size,
            shape=[self.num_seqs, self.max_model_len],
            dtype="int64",
        )
        for i in range(self.num_seqs):
            pre_ids[i, : step_idx[i]] = fake_tokens[i, : step_idx[i]]

        logits = paddle.randn([self.num_seqs, self.vocab_size]).cast("float32")

        penalty_score = paddle.ones([self.num_seqs, 1]) * 1.05
        frequency_score = paddle.ones([self.num_seqs, 1]) * 0.5
        presence_score = paddle.ones([self.num_seqs, 1]) * 0.3
        temperature = paddle.ones([self.num_seqs, 1]) * 0.8

        bad_tokens = paddle.to_tensor([[-1]]).cast("int64")
        min_dec_len = paddle.ones([self.num_seqs, 1]).cast("int64")
        eos_token_id = paddle.to_tensor([[2]]).cast("int64")

        self.input_data = {
            "prompt_ids": prompt_ids,
            "prompt_lens": prompt_lens,
            "pre_ids": pre_ids,
            "step_idx": step_idx,
            "logits": logits,
            "bad_tokens": bad_tokens,
            "min_dec_len": min_dec_len,
            "eos_token_id": eos_token_id,
            "penalty_score": penalty_score,
            "frequency_score": frequency_score,
            "presence_score": presence_score,
            "temperature": temperature,
        }

    def get_token_penalty_multi_scores_baseline(self):
        input_data = copy.deepcopy(self.input_data)
        logits = input_data["logits"]
        penalty_score = input_data["penalty_score"]
        frequency_score = input_data["frequency_score"]
        presence_score = input_data["presence_score"]
        temperature = input_data["temperature"]

        # min token penalties
        mask = input_data["step_idx"] < input_data["min_dec_len"]
        for bi, flag in enumerate(mask):
            if flag:
                logits[bi, input_data["eos_token_id"]] = -1e10

        # bad words exclusion
        for token in input_data["bad_tokens"]:
            if token < 0 or token > self.vocab_size:
                continue
            logits[:, token] = -1e10
        # all penalties
        prompt_ids = input_data["prompt_ids"]
        for i in range(self.num_seqs):
            prompt_ids[i, input_data["prompt_lens"][i] :] = -1
        prompt_repeat_times = paddle.zeros([self.num_seqs, self.vocab_size + 1]).cast("int64")
        prompt_repeat_times = paddle.put_along_axis(
            prompt_repeat_times,
            prompt_ids,
            paddle.ones_like(input_data["pre_ids"]),
            axis=1,
            reduce="add",
        )
        prompt_repeat_times = prompt_repeat_times[:, : self.vocab_size]
        prompt_mask = prompt_repeat_times > 0

        pre_ids = input_data["pre_ids"]
        pre_ids[pre_ids == -1] = self.vocab_size
        out_repeat_times = paddle.zeros([self.num_seqs, self.vocab_size + 1]).cast("int64")
        out_repeat_times = paddle.put_along_axis(
            out_repeat_times,
            pre_ids,
            paddle.ones_like(input_data["pre_ids"]),
            axis=1,
            reduce="add",
        )
        out_repeat_times = out_repeat_times[:, : self.vocab_size]
        output_mask = out_repeat_times > 0

        penalty_score = penalty_score.tile(self.vocab_size)
        logits[logits > 0] /= paddle.where(output_mask | prompt_mask, penalty_score, 1.0)[logits > 0]
        logits[logits <= 0] *= paddle.where(output_mask | prompt_mask, penalty_score, 1.0)[logits <= 0]
        logits -= frequency_score * out_repeat_times.cast("float32")
        logits -= presence_score * output_mask.cast("float32")

        # temperature
        logits /= temperature
        return logits

    def test_penalty_op(self):
        """ """
        baseline_out = self.get_token_penalty_multi_scores_baseline()
        from fastdeploy.model_executor.ops.gpu import get_token_penalty_multi_scores

        logits = get_token_penalty_multi_scores(
            self.input_data["pre_ids"],
            self.input_data["prompt_ids"],
            self.input_data["prompt_lens"],
            self.input_data["logits"],
            self.input_data["penalty_score"],
            self.input_data["frequency_score"],
            self.input_data["presence_score"],
            self.input_data["temperature"],
            self.input_data["bad_tokens"],
            self.input_data["step_idx"],
            self.input_data["min_dec_len"],
            self.input_data["eos_token_id"],
        )
        np.testing.assert_allclose(baseline_out.numpy(), logits.numpy(), rtol=1e-04, atol=1e-04)


if __name__ == "__main__":
    unittest.main()
