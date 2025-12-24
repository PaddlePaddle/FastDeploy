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

# 假设这是你的自定义算子
from fastdeploy.model_executor.ops.xpu import speculate_update_v3


def gen_inputs(
    max_bsz=512,  # 与 CUDA BlockSize 对齐
    max_draft_tokens=16,
    real_bsz=123,  # 可自调；须 ≤ max_bsz
    seed=2022,
):
    """生成随机测试输入数据"""
    rng = np.random.default_rng(seed)

    # 基本张量
    seq_lens_encoder = rng.integers(0, 3, size=max_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(0, 20, size=max_bsz, dtype=np.int32)
    not_need_stop = rng.integers(0, 1, size=1, dtype=np.bool_)
    draft_tokens = rng.integers(0, 1000, size=(max_bsz, max_draft_tokens), dtype=np.int64)
    actual_draft_nums = rng.integers(1, max_draft_tokens, size=max_bsz, dtype=np.int32)
    accept_tokens = rng.integers(0, 1000, size=(max_bsz, max_draft_tokens), dtype=np.int64)
    accept_num = rng.integers(1, max_draft_tokens, size=max_bsz, dtype=np.int32)
    stop_flags = rng.integers(0, 2, size=max_bsz, dtype=np.bool_)
    is_block_step = rng.integers(0, 2, size=max_bsz, dtype=np.bool_)
    stop_nums = np.array([5], dtype=np.int64)  # 阈值随意

    # seq_lens_this_time 仅取 real_bsz 长度
    seq_lens_this_time = rng.integers(1, max_draft_tokens + 1, size=real_bsz, dtype=np.int32)

    paddle.set_device("xpu:0")
    data_xpu = {
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder),
        "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder),
        "not_need_stop": paddle.to_tensor(not_need_stop).cpu(),
        "draft_tokens": paddle.to_tensor(draft_tokens),
        "actual_draft_token_nums": paddle.to_tensor(actual_draft_nums),
        "accept_tokens": paddle.to_tensor(accept_tokens),
        "accept_num": paddle.to_tensor(accept_num),
        "stop_flags": paddle.to_tensor(stop_flags),
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time),
        "is_block_step": paddle.to_tensor(is_block_step),
        "stop_nums": paddle.to_tensor(stop_nums),
    }

    paddle.set_device("cpu")
    data_cpu = {
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder),
        "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder),
        "not_need_stop": paddle.to_tensor(not_need_stop),
        "draft_tokens": paddle.to_tensor(draft_tokens),
        "actual_draft_token_nums": paddle.to_tensor(actual_draft_nums),
        "accept_tokens": paddle.to_tensor(accept_tokens),
        "accept_num": paddle.to_tensor(accept_num),
        "stop_flags": paddle.to_tensor(stop_flags),
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time),
        "is_block_step": paddle.to_tensor(is_block_step),
        "stop_nums": paddle.to_tensor(stop_nums),
    }
    return data_xpu, data_cpu


class TestSpeculateUpdateV3(unittest.TestCase):
    """测试 speculate_update_v3 算子"""

    def test_op_vs_golden(self, max_bsz=512, max_draft_tokens=16, real_bsz=123):
        """
        核心测试：比较自定义算子的输出与纯 NumPy 参考实现的输出。
        """
        # 1. gen inputs for cpu/xpu
        data_xpu, data_cpu = gen_inputs(max_bsz=max_bsz, max_draft_tokens=max_draft_tokens, real_bsz=real_bsz)

        # 3. run xpu kernel
        speculate_update_v3(**data_xpu)

        # 4. run cpu kernel
        speculate_update_v3(**data_cpu)

        # 5. format outputs
        outputs_xpu = [
            data_xpu["seq_lens_encoder"].cpu().numpy(),
            data_xpu["seq_lens_decoder"].cpu().numpy(),
            data_xpu["not_need_stop"].cpu().numpy(),
            data_xpu["draft_tokens"].cpu().numpy(),
            data_xpu["actual_draft_token_nums"].cpu().numpy(),
        ]

        outputs_cpu = [
            data_cpu["seq_lens_encoder"].numpy(),
            data_cpu["seq_lens_decoder"].numpy(),
            data_cpu["not_need_stop"].numpy(),
            data_cpu["draft_tokens"].numpy(),
            data_cpu["actual_draft_token_nums"].numpy(),
        ]
        output_names = [
            "seq_lens_encoder",
            "seq_lens_decoder",
            "not_need_stop",
            "draft_tokens",
            "actual_draft_token_nums",
        ]

        # 6. check outputs
        for name, pd_out, np_out in zip(output_names, outputs_xpu, outputs_cpu):
            with self.subTest(output_name=name):
                np.testing.assert_allclose(
                    pd_out,
                    np_out,
                    atol=0,
                    rtol=1e-6,
                    err_msg=f"Output mismatch for tensor '{name}'.\nPaddle Output:\n{pd_out}\nGolden Output:\n{np_out}",
                )


if __name__ == "__main__":
    unittest.main()
