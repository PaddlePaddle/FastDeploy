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

from fastdeploy.model_executor.ops.gpu import get_padding_offset


class TestGetPaddingOffset(unittest.TestCase):
    def test_get_padding_offset(self):
        seq_lens = np.array([4, 3, 6], "int32").reshape(-1, 1)
        token_num = np.sum(seq_lens)
        input_ids = np.array(
            [[8, 7, 8, 2, 0, 0, 0, 0, 0, 0], [4, 5, 5, 0, 0, 0, 0, 0, 0, 0], [7, 6, 1, 7, 2, 6, 0, 0, 0, 0]], "int64"
        )
        (
            x_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(token_num),
            paddle.to_tensor(seq_lens),
        )

        ref_x_remove_padding = np.array([8, 7, 8, 2, 4, 5, 5, 7, 6, 1, 7, 2, 6], "int64")
        ref_batch_id_per_token = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], "int32")
        ref_cu_seqlens_q = np.array([0, 4, 7, 13], "int32")
        ref_cu_seqlens_k = np.array([0, 4, 7, 13], "int32")

        np.testing.assert_allclose(x_remove_padding.numpy(), ref_x_remove_padding)
        np.testing.assert_allclose(batch_id_per_token.numpy(), ref_batch_id_per_token)
        np.testing.assert_allclose(cu_seqlens_q.numpy(), ref_cu_seqlens_q)
        np.testing.assert_allclose(cu_seqlens_k.numpy(), ref_cu_seqlens_k)


if __name__ == "__main__":
    unittest.main()
