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

import unittest

import paddle

from fastdeploy.model_executor.forward_meta import ForwardMeta


class TOYGPUModelRunner:
    def __init__(self):
        self.forward_meta: ForwardMeta = None

        self.max_num_seqs = 64
        self.max_model_len = 1024
        self.pre_max_block_num = 16
        # Not the tensor in real sense, just for make ForwardMeta
        self.share_inputs = {}

        self.share_inputs["input_ids"] = paddle.full(
            [self.max_num_seqs, self.max_model_len],
            0,
            dtype="int64",
        )
        self.share_inputs["ids_remove_padding"] = paddle.full(
            [self.max_num_seqs * self.max_model_len],
            0,
            dtype="int64",
        )
        self.share_inputs["decoder_batch_ids"] = None
        self.share_inputs["decoder_tile_ids_per_batch"] = None
        self.share_inputs["decoder_num_blocks_cpu"] = None
        self.share_inputs["max_len_tensor_cpu"] = None
        self.share_inputs["seq_lens_encoder"] = paddle.full([self.max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_decoder"] = paddle.full([self.max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_this_time"] = paddle.full([self.max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["batch_id_per_token"] = paddle.full(
            [self.max_num_seqs * self.max_model_len, 1], 0, dtype="int32"
        )
        self.share_inputs["cu_seqlens_q"] = paddle.full([self.max_num_seqs + 1, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_k"] = paddle.full([self.max_num_seqs + 1, 1], 0, dtype="int32")
        self.share_inputs["block_tables"] = paddle.full([self.max_num_seqs, self.pre_max_block_num], -1, dtype="int32")
        self.share_inputs["caches"] = [
            paddle.full([self.max_num_seqs, 4, self.max_model_len, self.pre_max_block_num], 0, dtype="int32")
        ] * 16

    def initialize_forward_meta(self):
        """
        Initialize forward meta
        """
        # Ignore the attentionbackbend for simplify
        self.forward_meta = ForwardMeta(
            ids_remove_padding=self.share_inputs["ids_remove_padding"],
            # rotary_embs=self.share_inputs["rope_emb"],# Ignore the rope_emb for simplify
            # attn_backend=self.attn_backends[0],# Ignore the attn_backbend for simplify
            decoder_batch_ids=self.share_inputs["decoder_batch_ids"],
            decoder_tile_ids_per_batch=self.share_inputs["decoder_tile_ids_per_batch"],
            decoder_num_blocks_cpu=self.share_inputs["decoder_num_blocks_cpu"],
            max_len_tensor_cpu=self.share_inputs["max_len_tensor_cpu"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            batch_id_per_token=self.share_inputs["batch_id_per_token"],
            cu_seqlens_q=self.share_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.share_inputs["cu_seqlens_k"],
            block_tables=self.share_inputs["block_tables"],
            caches=self.share_inputs["caches"],
        )


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize the test environment
        """
        self.runner = TOYGPUModelRunner()

    def test_case(self):
        """
        Check if the CustomAllreduce function works properly.
        """
        print(
            "in test/model_executor/test_forward_meta_str.py, forward_meta :", self.runner.forward_meta
        )  # Get None, Not Error
        self.runner.initialize_forward_meta()
        print(
            "in test/model_executor/test_forward_meta_str.py, forward_meta :", self.runner.forward_meta
        )  # Get information


if __name__ == "__main__":
    unittest.main()
