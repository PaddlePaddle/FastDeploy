"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np
from fastdeploy.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

class SamplingState:
    def __init__(
        self, 
        max_num_seqs: int, 
        eos_tokens_len: int,
        max_stop_seqs_num: int,
        stop_seqs_max_len: int,
        bad_words_max_len: int,
        max_bad_words_num: int):

        self.max_num_seqs = max_num_seqs

        self.temperature = StagedWriteTensor(max_num_seqs, dtype=paddle.float32)
        self.top_k = StagedWriteTensor(max_num_seqs, dtype=paddle.int32)
        self.top_p = StagedWriteTensor(max_num_seqs, dtype=paddle.float32)
        self.min_p = StagedWriteTensor(max_num_seqs, dtype=paddle.float32)
        self.seeds = StagedWriteTensor(max_num_seqs, dtype=paddle.int64)

        self.repetition_penalty = StagedWriteTensor(max_num_seqs, dtype=paddle.float32)
        self.frequency_penalty = StagedWriteTensor(max_num_seqs, dtype=paddle.float32)
        self.presence_penalty = StagedWriteTensor(max_num_seqs, dtype=paddle.float32)
        self.use_penalty = np.zeros(max_num_seqs, dtype=bool)

        self.num_bad_words = StagedWriteTensor(max_num_seqs, dtype=paddle.int32)
        self.bad_word_token_ids = StagedWriteTensor(
            (max_num_seqs, bad_words_max_len * max_bad_words_num),
            dtype=paddle.int32
        )
        self.bad_word_offsets = StagedWriteTensor(
            (max_num_seqs, max_bad_words_num + 1),
            dtype=paddle.int32
        )

        self.max_lens = StagedWriteTensor(max_num_seqs, dtype=paddle.int32)
        self.min_lens = StagedWriteTensor(max_num_seqs, dtype=paddle.int32)
        
        self.num_stop_token_ids = StagedWriteTensor(max_num_seqs, dtype=paddle.int32)
        self.stop_token_ids = StagedWriteTensor(
            (max_num_seqs, max_stop_seqs_num * stop_seqs_max_len),
            dtype=paddle.int32
        )
        self.stop_token_offsets = StagedWriteTensor(
            (max_num_seqs, max_stop_seqs_num + 1),
            dtype=paddle.int32
        )

        self.num_logprobs = np.empty(self.max_num_seqs, dtype=np.int32)
    

    def add_request(self, req_idx: int, request) -> None:
        temperature = request.get("temperature", 1.0)
        top_k = request.get("top_k", 0)
        top_p = request.get("top_p", 1.0)
        min_p = request.get("min_p", 0.0)
        seed = request.get("seed", 0)
        self.temperature.stage_write_elem(req_idx, temperature)
        self.top_k.stage_write_elem(req_idx, top_k)
        self.top_p.stage_write_elem(req_idx, top_p)
        self.min_p.stage_write_elem(req_idx, min_p)
        self.seeds.stage_write_elem(req_idx, seed)

        repetition_penalty = request.get("repetition_penalty", 1.0)
        frequency_penalty = request.get("frequency_penalty", 0.0)
        presence_penalty = request.get("presence_penalty", 0.0)
        self.repetition_penalty.stage_write_elem(req_idx, repetition_penalty)
        self.frequency_penalty.stage_write_elem(req_idx, frequency_penalty)
        self.presence_penalty.stage_write_elem(req_idx, presence_penalty)
        self.use_penalty[req_idx] = (
            repetition_penalty != 1.0
            or frequency_penalty != 0.0
            or presence_penalty != 0.0
        )

        # Each element in bad_words_token_ids is treated as a single-token bad word.
        bad_words = request.get("bad_words_token_ids") or []
        if bad_words:
            self.bad_word_token_ids.stage_write(req_idx, 0, bad_words)
            offsets = list(range(len(bad_words) + 1))
            self.bad_word_offsets.stage_write(req_idx, 0, offsets)
        self.num_bad_words.stage_write_elem(req_idx, len(bad_words))

        self.max_lens.stage_write_elem(req_idx, request.get("max_tokens", 0))
        self.min_lens.stage_write_elem(req_idx, request.get("min_tokens", 1))
        stop_token_ids = request.get("stop_token_ids") or []
        if stop_token_ids:
            self.stop_token_ids.stage_write(req_idx, 0, stop_token_ids)
            offsets = list(range(len(stop_token_ids) + 1))
            self.stop_token_offsets.stage_write(req_idx, 0, offsets)
        self.num_stop_token_ids.stage_write_elem(req_idx, len(stop_token_ids))

    def apply_staged_writes(self) -> None:
        self.temperature.apply_write()
        self.top_k.apply_write()
        self.top_p.apply_write()
        self.min_p.apply_write()
        self.seeds.apply_write()
        self.repetition_penalty.apply_write()
        self.frequency_penalty.apply_write()
        self.presence_penalty.apply_write()
        self.bad_word_token_ids.apply_write()
        self.bad_word_offsets.apply_write()
        self.num_bad_words.apply_write()
        self.max_lens.apply_write()
        self.min_lens.apply_write()
        self.num_stop_token_ids.apply_write()
        self.stop_token_ids.apply_write()
        self.stop_token_offsets.apply_write()