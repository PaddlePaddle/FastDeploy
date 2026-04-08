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

import numpy as np

from fastdeploy.config import FDConfig
from fastdeploy.utils import spec_logger

from .base import Proposer

try:
    from arctic_inference.suffix_decoding import SuffixDecodingCache
except ImportError:
    SuffixDecodingCache = None


class SuffixProposer(Proposer):
    """
    Proposer for Suffix Decoding method.

    Uses SuffixDecodingCache to generate draft tokens based on suffix tree matching.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        if SuffixDecodingCache is None:
            raise ImportError(
                "arctic_inference.suffix_decoding is not available. Please install via `pip install arctic-inference==0.1.2`"
            )

        # Initialize SuffixDecodingCache
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=self.speculative_config.suffix_decoding_max_tree_depth,
            max_cached_requests=self.speculative_config.suffix_decoding_max_cached_requests,
        )

        self.max_tree_depth = self.speculative_config.suffix_decoding_max_tree_depth
        self.max_spec_factor = self.speculative_config.suffix_decoding_max_spec_factor
        self.min_token_prob = self.speculative_config.suffix_decoding_min_token_prob

        # Track active requests: req_id -> idx mapping
        self.req_id_to_idx = {}
        self.idx_to_req_id = {}
        self.context_tokens = np.full(
            (self.max_num_seqs, self.max_model_len),
            -1,
            dtype=np.int32,
        )
        self.ban_tokens = set([101031, 101032, 101033])

    def _update_request_mapping(self, idx: int, req_id: str):
        """
        Update the mapping between request ID and batch index.

        Args:
            req_id: Request identifier
            idx: Batch index
        """
        # Clean up old mapping if exists
        if idx in self.idx_to_req_id:
            old_req_id = self.idx_to_req_id[idx]
            if old_req_id in self.req_id_to_idx:
                del self.req_id_to_idx[old_req_id]

        # Set new mapping
        self.req_id_to_idx[req_id] = idx
        self.idx_to_req_id[idx] = req_id

    def start_request(self, idx: int, req_id: str, prompt_token_ids: list[int]):
        """
        Start a new request in the suffix cache.

        Args:
            req_id: Request identifier
            prompt_token_ids: List of prompt token IDs
        """
        if req_id in self.suffix_cache.active_requests:
            # Request already active, skip
            return

        prompt_array = np.array(prompt_token_ids, dtype=np.int32)
        if not prompt_array.flags["CONTIGUOUS"]:
            prompt_array = np.ascontiguousarray(prompt_array)

        self.context_tokens[idx, :] = -1
        self.context_tokens[idx, : len(prompt_token_ids)] = prompt_array
        self._update_request_mapping(idx, req_id)
        if req_id not in self.suffix_cache.active_requests:
            if req_id in self.suffix_cache.cached_requests:
                # Reset the suffix cache for current req_id
                self.suffix_cache.evict_cached_response(req_id)
                spec_logger.debug(f"[SuffixDecoding] Reset suffix cache for request {req_id}.")
            self.suffix_cache.start_request(req_id, prompt_array)
            spec_logger.debug(f"[SuffixDecoding] Start request {req_id}.")

    def stop_request(self, req_id: str):
        """
        Stop a request in the suffix cache.

        Args:
            req_id: Request identifier
        """
        if req_id in self.suffix_cache.active_requests:
            self.suffix_cache.stop_request(req_id)

        # Clean up mappings
        if req_id in self.req_id_to_idx:
            idx = self.req_id_to_idx[req_id]

            del self.req_id_to_idx[req_id]
            if idx in self.idx_to_req_id:
                del self.idx_to_req_id[idx]

        spec_logger.debug(f"[SuffixDecoding] Stop request {req_id}.")

    def add_active_response(self, req_id: str, token_ids: list[int]):
        """
        Add newly sampled tokens to the suffix cache for a request.

        Args:
            req_id: Request identifier
            token_ids: List of newly sampled token IDs
        """
        if req_id not in self.suffix_cache.active_requests:
            return

        token_array = np.array(token_ids, dtype=np.int32)
        if not token_array.flags["CONTIGUOUS"]:
            token_array = np.ascontiguousarray(token_array)

        self.suffix_cache.add_active_response(req_id, token_array)

    def _run_impl(self, share_inputs):

        stop_flags_cpu = share_inputs["stop_flags"].cpu().numpy().flatten()
        is_block_step_cpu = share_inputs["is_block_step"].cpu().numpy().flatten()
        accept_tokens_cpu = share_inputs["accept_tokens"].cpu()
        accept_num_cpu = share_inputs["accept_num"].cpu().numpy().flatten()
        seq_lens_encoder = share_inputs["seq_lens_encoder"].cpu().numpy().flatten()
        seq_lens_decoder = share_inputs["seq_lens_decoder"].cpu().numpy().flatten()

        draft_tokens_cpu = share_inputs["draft_tokens"].cpu()
        seq_lens_this_time_cpu = share_inputs["seq_lens_this_time"].cpu()

        total_lens = seq_lens_encoder + seq_lens_decoder
        batch_size = seq_lens_this_time_cpu.shape[0]

        for bid in range(batch_size):
            req_id = self.idx_to_req_id.get(bid)
            # 1. Stop condition has the highest priority
            if stop_flags_cpu[bid]:
                seq_lens_this_time_cpu[bid] = 0
                draft_tokens_cpu[bid, :] = -1
                if not is_block_step_cpu[bid]:
                    if req_id is not None and req_id in self.suffix_cache.active_requests:
                        self.stop_request(req_id)
                continue
            else:
                seq_lens_this_time_cpu[bid] = 1
                draft_tokens_cpu[bid, 1:] = -1
            # 2. Skip some cases
            num_tokens = total_lens[bid]
            max_spec_tokens = min(
                self.max_draft_token_num,
                self.max_model_len - num_tokens - 1,
            )
            if max_spec_tokens <= 1:
                continue
            if req_id is None:
                continue

            # 3. Add accept tokens to context
            acc_num = int(accept_num_cpu[bid])
            assert (
                acc_num > 0
            ), f"Request {req_id} (bid {bid}) must have at least one accepted token, but got {acc_num}."
            if acc_num > 0:
                token_ids = accept_tokens_cpu[bid, :acc_num]
                ctx_start = seq_lens_decoder[bid] - acc_num
                self.context_tokens[bid, ctx_start : ctx_start + acc_num] = token_ids
                self.add_active_response(req_id, token_ids)

            # 4. Get context
            start = max(0, num_tokens - self.max_tree_depth)
            ctx = self.context_tokens[bid, start:num_tokens]
            ctx = ctx[ctx >= 0]
            if ctx.size == 0:
                continue
            if not ctx.flags["CONTIGUOUS"]:
                ctx = np.ascontiguousarray(ctx, dtype=np.int32)
            else:
                ctx = ctx.astype(np.int32, copy=False)

            # 5. Speculate
            draft = self.suffix_cache.speculate(
                req_id,
                ctx,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            token_ids = draft.token_ids

            counter = 0
            for token in token_ids:
                if token in self.ban_tokens:
                    break
                else:
                    counter += 1
            if counter > 0:
                draft_tokens_cpu[bid, 1 : 1 + counter] = np.array(token_ids[:counter])
                draft_tokens_cpu[bid, 1 + counter :] = -1
                seq_lens_this_time_cpu[bid] = 1 + counter

        share_inputs["draft_tokens"][:] = draft_tokens_cpu.cuda()
        share_inputs["seq_lens_this_time"][:] = seq_lens_this_time_cpu.cuda()
