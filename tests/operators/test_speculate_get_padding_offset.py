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

from fastdeploy.model_executor.ops.gpu import speculate_get_padding_offset


def ref_speculate_get_padding_offset(cum_offsets, seq_lens, max_seq_len, token_num_data):
    bsz = seq_lens.shape[0]

    padding_offset = np.zeros([token_num_data], dtype=np.int32)
    batch_id_per_token = np.zeros([token_num_data], dtype=np.int32)
    cum_offsets_out = np.zeros([bsz], dtype=np.int32)
    cu_seqlens_q = np.zeros([bsz + 1], dtype=np.int32)
    cu_seqlens_k = np.zeros([bsz + 1], dtype=np.int32)

    modified_indices = {
        "padding_offset": [],
        "cum_offsets_out": [],
        "cu_seqlens_q": [],
        "cu_seqlens_k": [],
    }

    cu_seqlens_q[0] = 0
    cu_seqlens_k[0] = 0
    modified_indices["cu_seqlens_q"].append(0)
    modified_indices["cu_seqlens_k"].append(0)

    for bi in range(bsz):
        cum_offset = 0 if bi == 0 else cum_offsets[bi - 1]
        cum_offsets_out[bi] = cum_offset
        modified_indices["cum_offsets_out"].append(bi)

        for i in range(seq_lens[bi]):
            idx = bi * max_seq_len - cum_offset + i
            if idx >= 0 and idx < token_num_data:
                if idx == 0:
                    print(idx, bi, cum_offset)
                padding_offset[idx] = cum_offset
                batch_id_per_token[idx] = bi
                modified_indices["padding_offset"].append(idx)

        cum_seq_len = (bi + 1) * max_seq_len - cum_offsets[bi]
        cu_seqlens_q[bi + 1] = cum_seq_len
        cu_seqlens_k[bi + 1] = cum_seq_len
        modified_indices["cu_seqlens_q"].append(bi + 1)
        modified_indices["cu_seqlens_k"].append(bi + 1)

    return (
        padding_offset,
        cum_offsets_out,
        cu_seqlens_q,
        cu_seqlens_k,
        modified_indices,
        batch_id_per_token,
    )


class TestSpeculateGetPaddingOffset(unittest.TestCase):
    def test_speculate_get_padding_offset(self):
        test_case = {
            "bsz": 4,
            "max_seq_len": 10,
            "token_num_data": 32,
            "cum_offsets": np.array([2, 5, 8, 12], dtype=np.int32),
            "seq_lens": np.array([8, 5, 7, 6], dtype=np.int32),
            "seq_lens_encoder": np.array([1, 0, 1, 0], dtype=np.int32),
        }

        max_draft_tokens = 4

        input_ids = np.random.randint(0, 1000, (test_case["bsz"], test_case["max_seq_len"]), dtype=np.int64)
        draft_tokens = np.random.randint(0, 1000, (test_case["bsz"], max_draft_tokens), dtype=np.int64)
        token_num = np.array([test_case["token_num_data"]], dtype=np.int64)

        input_ids_tensor = paddle.to_tensor(input_ids)
        draft_tokens_tensor = paddle.to_tensor(draft_tokens)
        cum_offsets_tensor = paddle.to_tensor(test_case["cum_offsets"])
        seq_lens_tensor = paddle.to_tensor(test_case["seq_lens"])
        seq_lens_encoder_tensor = paddle.to_tensor(test_case["seq_lens_encoder"])
        token_num_tensor = paddle.to_tensor(token_num)

        (
            x_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids_tensor,
            draft_tokens_tensor,
            cum_offsets_tensor,
            token_num_tensor,
            seq_lens_tensor,
            seq_lens_encoder_tensor,
        )

        (
            ref_padding_offset,
            ref_cum_offsets_out,
            ref_cu_seqlens_q,
            ref_cu_seqlens_k,
            modified_indices,
            ref_batch_id_per_token,
        ) = ref_speculate_get_padding_offset(
            test_case["cum_offsets"],
            test_case["seq_lens"],
            test_case["max_seq_len"],
            test_case["token_num_data"],
        )

        output_arrays = {
            "batch_id_per_token": batch_id_per_token.numpy(),
            "cu_seqlens_q": cu_seqlens_q.numpy(),
            "cu_seqlens_k": cu_seqlens_k.numpy(),
        }

        ref_arrays = {
            "batch_id_per_token": ref_batch_id_per_token,
            "cu_seqlens_q": ref_cu_seqlens_q,
            "cu_seqlens_k": ref_cu_seqlens_k,
        }

        for key in output_arrays:
            np.testing.assert_allclose(output_arrays[key], ref_arrays[key])


if __name__ == "__main__":
    unittest.main()
