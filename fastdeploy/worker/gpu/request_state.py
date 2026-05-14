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
from fastdeploy.worker.gpu.input_batch import InputBuffers
from fastdeploy.worker.gpu.gather_tokens_kernel import gather_tokens

class RequestState:
    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size

        self.req_id_to_index: dict[str, int] = {}
        self.idx_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_seqs))

        # prompt + output
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_seqs, self.max_model_len),
            dtype=paddle.int32,
            init_value=-1,
            uva_instead_of_gpu=True,
        )

        # 
        self.batched_input_ids = StagedWriteTensor(
            (self.max_num_seqs, self.max_num_batched_tokens),
            dtype=paddle.int32,
            init_value=-1
        )

        self.num_tokens_per_seq = np.zeros(self.max_num_seqs, dtype=np.int32)

        # prompt_len: Number of tokens in user-provided prompt.
        # prefill_len: Number of tokens passed into model runner.
        #   This can include prompt and additional partial output tokens,
        #   so prefill_len >= prompt_len.
        # Usually, prefill_len equals prompt_len, but in cases such as resumption after
        # preemption, prefill_len may be greater. Differentiating between these values
        # is crucial, as certain features such as prompt logprobs or frequency penalties
        # must treat prompt and output tokens separately.
        # Using UvaBackedTensor because these need frequent CPU access (np attribute)
        self.prompt_len = UvaBackedTensor(self.max_num_seqs, dtype=paddle.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_seqs, dtype=paddle.int32)
        self.total_len = StagedWriteTensor(self.max_num_seqs, dtype=paddle.int32)

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_seqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(self.max_num_seqs, dtype=paddle.int32)

        # Number of computing tokens.
        self.num_tokens_per_req = StagedWriteTensor(self.max_num_seqs, dtype=paddle.int32)

        # Last sampled tokens.
        self.last_sampled_tokens = paddle.full(self.max_num_seqs, -1, dtype=paddle.int32)

        # Draft tokens.
        self.draft_tokens = paddle.full(
            (self.max_num_seqs, self.num_speculative_steps),
            -1,
            dtype=paddle.int32,
        )
        self.draft_tokens_len = paddle.full(self.max_num_seqs, 0, dtype=paddle.int32)

        self.next_prefill_tokens = paddle.full(self.max_num_seqs, 0, dtype=paddle.int32)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_idx: int,
        num_tokens: int,
        prompt_len: int,
        prefill_len: int,
        all_token_ids: list[int],
        batched_input_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        self.num_tokens_per_seq[req_idx] = num_tokens
        self.prompt_len.np[req_idx] = prompt_len
        self.prefill_len.np[req_idx] = prefill_len
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.batched_input_ids.stage_write(req_idx, 0, batched_input_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.batched_input_ids.apply_write()
        self.num_computed_tokens.apply_write()

    # def remove_request(self, req_idx: str) -> bool:
    #     req_id = self.idx_to_req_id.pop(req_idx, None)
    #     if req_id is None:
    #         # Request not found.
    #         return False
    #     return True

    def exist_prefill(self) -> bool:
        running_idx = self.num_tokens_per_seq > 0
        return len(running_idx) > 0 and np.any(
            self.num_computed_prefill_tokens[running_idx]
            < self.prefill_len.np[running_idx]
        )

    def exist_decode(self) -> bool:
        running_idx = self.num_tokens_per_seq > 0
        return len(running_idx) > 0 and np.any(
            self.num_computed_prefill_tokens[running_idx]
            >= self.prefill_len.np[running_idx]
        )

    def get_num_prefills(self) -> int:
        running_idx = self.num_tokens_per_seq > 0
        return np.sum(self.num_computed_prefill_tokens[running_idx] < self.prefill_len[running_idx])
    
    def get_num_num_decodes(self) -> int:
        running_idx = self.num_tokens_per_seq > 0
        return np.sum(self.num_computed_prefill_tokens[running_idx] >= self.prefill_len[running_idx])