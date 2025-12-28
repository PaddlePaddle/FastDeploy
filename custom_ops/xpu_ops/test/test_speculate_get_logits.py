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

from fastdeploy.model_executor.ops.xpu import speculate_get_logits

# 固定随机种子，保证测试可复现
np.random.seed(2023)
paddle.seed(2023)


def generate_test_data():
    """
    生成测试数据的辅助函数。
    这部分逻辑从 pytest 的 fixture 转换而来，作为一个普通函数供测试方法调用。
    """
    real_bsz = 64
    vocab_size = 2 * 1024
    max_seq_len = 8 * 1024

    # 生成原始测试数据（完全复用原有逻辑）
    seq_lens_encoder = np.random.randint(0, 2, [real_bsz], dtype=np.int32)
    seq_lens_this_time = np.random.randint(1, max_seq_len, [real_bsz], dtype=np.int32)
    draft_logits_seqlen = 0
    logits_seqlen = 0
    for i in range(real_bsz):
        if seq_lens_encoder[i] > 0:
            draft_logits_seqlen += 2
            logits_seqlen += 1
        else:
            draft_logits_seqlen += seq_lens_this_time[i]
            logits_seqlen += seq_lens_this_time[i]

    draft_logits = np.zeros([draft_logits_seqlen, vocab_size], dtype=np.float32)
    next_token_num = np.zeros([real_bsz], dtype=np.int32)
    batch_token_num = np.zeros([real_bsz], dtype=np.int32)
    cu_next_token_offset = np.zeros([real_bsz], dtype=np.int32)
    cu_batch_token_offset = np.zeros([real_bsz], dtype=np.int32)
    logits = np.random.rand(logits_seqlen, vocab_size).astype(np.float32)
    first_token_logits = np.random.rand(real_bsz, vocab_size).astype(np.float32)

    paddle.set_device("cpu")
    # 转换为 paddle tensor（保持原有逻辑）
    data_cpu = {
        "draft_logits": paddle.to_tensor(draft_logits),
        "next_token_num": paddle.to_tensor(next_token_num),
        "batch_token_num": paddle.to_tensor(batch_token_num),
        "cu_next_token_offset": paddle.to_tensor(cu_next_token_offset),
        "cu_batch_token_offset": paddle.to_tensor(cu_batch_token_offset),
        "logits": paddle.to_tensor(logits),
        "first_token_logits": paddle.to_tensor(first_token_logits),
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time),
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder),
    }

    paddle.set_device("xpu:0")
    data_xpu = {
        "draft_logits": paddle.to_tensor(draft_logits),
        "next_token_num": paddle.to_tensor(next_token_num),
        "batch_token_num": paddle.to_tensor(batch_token_num),
        "cu_next_token_offset": paddle.to_tensor(cu_next_token_offset),
        "cu_batch_token_offset": paddle.to_tensor(cu_batch_token_offset),
        "logits": paddle.to_tensor(logits),
        "first_token_logits": paddle.to_tensor(first_token_logits),
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time),
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder),
    }

    # 恢复默认设备，避免影响其他测试
    paddle.set_device("cpu")

    return data_cpu, data_xpu


def speculate_get_logits_execution(test_data):
    """测试函数的执行性和输出合理性"""

    # 执行目标函数（核心测试步骤）
    speculate_get_logits(**test_data)

    return test_data


class TestSpeculateGetLogits(unittest.TestCase):
    """
    测试类，继承自 unittest.TestCase。
    所有以 'test_' 开头的方法都会被视为测试用例。
    """

    def assert_test_data_equal(self, test_data1, test_data2, rtol=1e-05, atol=1e-08, target_keys=None):
        """
        自定义的断言方法，用于比较两个 test_data 结构和数据。
        在 unittest 中，自定义断言通常以 'assert' 开头。
        """
        # 1. 先校验两个 test_data 的字段名完全一致
        keys1 = set(test_data1.keys())
        keys2 = set(test_data2.keys())
        self.assertEqual(
            keys1,
            keys2,
            msg=f"两个 test_data 字段不一致！\n仅在第一个中存在：{keys1 - keys2}\n仅在第二个中存在：{keys2 - keys1}",
        )

        # 2. 逐字段校验数据
        if target_keys is not None and isinstance(target_keys, list):
            local_target_key = target_keys
        else:
            local_target_key = keys1
        for key in local_target_key:
            data1 = test_data1[key]
            data2 = test_data2[key]

            # 区分：paddle Tensor（需转 numpy）和 普通标量/数组（直接使用）
            if isinstance(data1, paddle.Tensor):
                np1 = data1.detach().cpu().numpy()
            else:
                np1 = np.asarray(data1)

            if isinstance(data2, paddle.Tensor):
                np2 = data2.detach().cpu().numpy()
            else:
                np2 = np.asarray(data2)

            # 3. 校验数据
            if np1.dtype in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8):
                # 布尔/整数型：必须完全相等
                np.testing.assert_array_equal(np1, np2, err_msg=f"字段 {key} 数据不一致！")
            else:
                # 浮点型：允许 rtol/atol 范围内的误差
                np.testing.assert_allclose(np1, np2, rtol=rtol, atol=atol, err_msg=f"字段 {key} 浮点数据不一致！")

        print("✅ 两个 test_data 结构和数据完全一致！")

    def test_speculate_get_logits(self):
        """
        核心测试用例方法。
        该方法会调用 generate_test_data 获取数据，
        分别在 CPU 和 XPU 上执行测试函数，
        并使用自定义的断言方法比较结果。
        """
        print("\nRunning test: test_speculate_get_logits")

        # 1. 获取测试数据
        data_cpu, data_xpu = generate_test_data()

        # 2. 执行测试函数
        result_xpu = speculate_get_logits_execution(data_xpu)
        result_cpu = speculate_get_logits_execution(data_cpu)

        # 3. 断言结果一致
        target_keys = ["draft_logits", "batch_token_num", "cu_batch_token_offset"]
        self.assert_test_data_equal(result_cpu, result_xpu, target_keys=target_keys)


if __name__ == "__main__":
    # 使用 unittest 的主程序来运行所有测试用例
    unittest.main()
