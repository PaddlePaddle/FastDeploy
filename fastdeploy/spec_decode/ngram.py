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

from typing import TYPE_CHECKING

import paddle

from fastdeploy.model_executor.ops.gpu import ngram_match

from .base import Proposer

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


class NgramProposer(Proposer):
    """
    Proposer for Ngram match method.

    Matching corresponding tokens in input and output as draft tokens.
    """

    def __init__(self, fd_config: "FDConfig"):
        super().__init__(fd_config)
        self.max_ngram_size = self.speculative_config.max_ngram_size
        self.input_ids_len = paddle.zeros(shape=[self.max_num_seqs, 1], dtype="int64").cpu()
        self.input_ids_len_gpu = paddle.zeros(shape=[self.max_num_seqs, 1], dtype="int64").cuda()

    def update(self, bid: int, seq_len: int):
        """
        update
        """
        self.input_ids_len[bid] = seq_len
        self.input_ids_len_gpu[bid] = seq_len

    def _run_impl(self, share_inputs):
        """
        run
        """
        ngram_match(
            share_inputs["input_ids_cpu"].cuda(),
            self.input_ids_len_gpu,
            share_inputs["token_ids_all"],
            share_inputs["prompt_lens"],
            share_inputs["step_idx"],
            share_inputs["actual_draft_token_num"],
            share_inputs["draft_tokens"],
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            share_inputs["max_dec_len"],
            self.max_ngram_size,
            self.max_draft_token_num,
        )
