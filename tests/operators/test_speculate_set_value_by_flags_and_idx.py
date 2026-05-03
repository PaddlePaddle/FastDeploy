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

from fastdeploy.model_executor.ops.gpu import speculate_set_value_by_flags_and_idx

paddle.seed(42)
np.random.seed(42)


def _reference(
    token_ids_all,
    prompt_lens,
    accept_tokens,
    accept_num,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
):
    """NumPy reference: write accepted draft tokens into token_ids_all buffer."""
    out_ids = token_ids_all.copy()
    out_accept = accept_num.copy()
    out_dec = seq_lens_decoder.copy()
    bs = seq_lens_this_time.shape[0]
    for i in range(bs):
        if stop_flags[i]:
            out_accept[i] = 0
            out_dec[i] = 0
        else:
            if seq_lens_decoder[i] == 0 and seq_lens_encoder[i] == 0:
                continue
            if step_idx[i] > 0:
                pl = int(prompt_lens[i, 0] if prompt_lens.ndim == 2 else prompt_lens[i])
                for j in range(int(accept_num[i])):
                    out_ids[i, pl + int(step_idx[i]) - j] = accept_tokens[i, int(accept_num[i]) - 1 - j]
    return out_ids, out_accept, out_dec


def _build_inputs(batch_size, max_len, max_draft, stop_ratio=0.3):
    """Create random inputs for the op."""
    ids = np.full((batch_size, max_len), -1, dtype="int64")
    plens = np.random.randint(0, max_len // 4, (batch_size, 1)).astype("int64")
    atoks = np.random.randint(100, 50000, (batch_size, max_draft)).astype("int64")
    anum = np.random.randint(0, max_draft + 1, (batch_size,)).astype("int32")
    stop = np.random.choice([True, False], (batch_size,), p=[stop_ratio, 1 - stop_ratio])
    slt = np.ones(batch_size, dtype="int32")
    sle = np.random.randint(0, 3, (batch_size,)).astype("int32")
    sld = np.random.randint(0, 3, (batch_size,)).astype("int32")
    step = np.zeros(batch_size, dtype="int64")
    for i in range(batch_size):
        hi = max_len - int(plens[i, 0]) - max_draft - 1
        step[i] = np.random.randint(max_draft, max(hi, max_draft + 1))
    return ids, plens, atoks, anum, stop, slt, sle, sld, step


def _check_output(tc, inputs_np):
    """Run GPU op, compare all 3 inplace-modified tensors against reference."""
    ref_ids, ref_an, ref_sd = _reference(*inputs_np)
    tensors = [paddle.to_tensor(x) for x in inputs_np]
    speculate_set_value_by_flags_and_idx(*tensors)
    np.testing.assert_array_equal(tensors[0].numpy(), ref_ids)
    np.testing.assert_array_equal(tensors[3].numpy(), ref_an)
    np.testing.assert_array_equal(tensors[7].numpy(), ref_sd)


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "GPU required")
class TestSpeculateSetValue(unittest.TestCase):
    """Correctness tests for speculate_set_value_by_flags_and_idx."""

    def setUp(self):
        paddle.set_device("gpu")

    def test_correctness(self):
        """Random batch with mixed stopped/active sequences."""
        _check_output(self, _build_inputs(32, 256, 4))

    def test_with_prompt_offset(self):
        """Non-zero prompt_lens shift write positions correctly."""
        bs, max_len = 2, 64
        ids = np.full((bs, max_len), -1, dtype="int64")
        plens = np.array([[10], [20]], dtype="int64")
        atoks = np.array([[500, 600, 700], [800, 900, 1000]], dtype="int64")
        anum = np.array([2, 3], dtype="int32")
        stop = np.array([False, False])
        slt = np.ones(bs, dtype="int32")
        sle = np.zeros(bs, dtype="int32")
        sld = np.array([2, 3], dtype="int32")
        step = np.array([5, 10], dtype="int64")
        inp = (ids, plens, atoks, anum, stop, slt, sle, sld, step)
        _check_output(self, inp)

    def test_mixed_stopped_active(self):
        """Stopped sequences zero out accept_num/seq_lens_decoder."""
        bs, max_len = 4, 64
        ids = np.full((bs, max_len), -1, dtype="int64")
        plens = np.zeros((bs, 1), dtype="int64")
        atoks = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype="int64")
        anum = np.array([1, 2, 1, 2], dtype="int32")
        stop = np.array([False, True, False, True])
        slt = np.ones(bs, dtype="int32")
        sle = np.array([1, 1, 1, 1], dtype="int32")
        sld = np.array([2, 3, 1, 4], dtype="int32")
        step = np.array([5, 6, 7, 8], dtype="int64")
        inp = (ids, plens, atoks, anum, stop, slt, sle, sld, step)
        _check_output(self, inp)


if __name__ == "__main__":
    unittest.main()
