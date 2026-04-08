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

import os
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import speculate_step_paddle

# 固定随机种子，保证测试可复现
np.random.seed(2023)
paddle.seed(2023)


def generate_test_data():
    """
    生成测试数据的辅助函数。
    这部分逻辑从 pytest 的 fixture 转换而来，作为一个普通函数供测试方法调用。
    """
    # max_bs = 128
    max_bs = 8
    bs = max_bs
    max_seq_len = 8192
    block_size = 64
    block_bs = 8
    block_ratio = 0.75
    max_draft_tokens = 1
    encoder_decoder_block_num = 1

    # 生成原始测试数据（完全复用原有逻辑）
    stop_flags = np.random.randint(0, 2, [max_bs]).astype("bool")
    seq_lens_this_time = np.zeros([bs], "int32")
    seq_lens_encoder = np.zeros([max_bs], "int32")
    seq_lens_decoder = np.zeros([max_bs], "int32")
    accept_num = np.random.randint(1, 3, [max_bs]).astype("int32")
    for i in range(bs):
        seq_lens_decoder[i] = 2 + i * 2
        seq_lens_this_time[i] = 1

    ori_seq_lens_encoder = np.zeros([max_bs], "int32")
    ori_seq_lens_encoder[:] = seq_lens_decoder[:] // 2
    step_idx = (seq_lens_decoder - ori_seq_lens_encoder).astype("int64")

    max_block_num = block_bs * max_seq_len // block_size
    free_list_len = int(max_block_num * (1 - block_ratio))
    free_list_len = np.full([1], free_list_len, "int32")
    free_list = np.arange(
        max_block_num - 1, max_block_num - free_list_len.item() - 1, -1, dtype="int32"  # 加 .item() 转为标量
    )
    encoder_block_lens = np.zeros([max_bs], "int32")
    used_list_len = np.zeros([max_bs], "int32")
    block_tables = np.full([max_bs, 128], -1, "int32")
    encoder_block_id = 0

    for i in range(bs):
        enc_block_num = (ori_seq_lens_encoder[i] + block_size - 1) // block_size
        encoder_block_lens[i] = enc_block_num
        dec_block_num = (seq_lens_decoder[i] + block_size - 1) // block_size - enc_block_num
        used_list_len[i] = dec_block_num
        block_tables[i, :enc_block_num] = np.arange(encoder_block_id, encoder_block_id + enc_block_num, 1, "int32")
        encoder_block_id += enc_block_num
        if dec_block_num > 0:
            block_tables[i, enc_block_num : enc_block_num + dec_block_num] = free_list[
                free_list_len[0] - 1 - dec_block_num : free_list_len[0] - 1
            ]
            free_list[free_list_len[0] - 1 - dec_block_num : free_list_len[0] - 1] = -1
            free_list_len[0] -= dec_block_num

    assert free_list_len[0] >= 0, "free_list_len should not be negative"

    is_block_step = np.zeros([max_bs], "bool")
    is_block_step[:bs] = np.random.randint(0, 2, [bs]).astype("bool")
    step_block_list = np.full([max_bs], -1, "int32")
    step_lens = np.full([1], 0, "int32")

    for i in range(bs):
        if is_block_step[i]:
            step_block_list[step_lens[0]] = i
            step_lens[0] += 1

    recover_lens = np.full([1], 0, "int32")
    recover_block_list = np.full([max_bs], -1, "int32")
    need_block_len = np.full([1], 0, "int32")
    need_block_list = np.full([max_bs], -1, "int32")

    input_ids = np.random.randint(0, 1000, [max_bs, max_seq_len], "int64")
    pre_ids = np.random.randint(0, 1000, [max_bs, max_seq_len], "int64")
    next_tokens = np.random.randint(0, 1000, [max_bs], "int64")
    first_token_ids = np.random.randint(0, 1000, [max_bs], "int64")

    paddle.set_device("cpu")
    # 转换为 paddle tensor（保持原有逻辑）
    data_cpu = {
        "stop_flags": paddle.to_tensor(stop_flags),
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time),
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder),
        "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder),
        "ori_seq_lens_encoder": paddle.to_tensor(ori_seq_lens_encoder),
        "block_tables": paddle.to_tensor(block_tables),
        "encoder_block_lens": paddle.to_tensor(encoder_block_lens),
        "is_block_step": paddle.to_tensor(is_block_step),
        "step_block_list": paddle.to_tensor(step_block_list),
        "step_lens": paddle.to_tensor(step_lens),
        "recover_block_list": paddle.to_tensor(recover_block_list),
        "recover_lens": paddle.to_tensor(recover_lens),
        "need_block_list": paddle.to_tensor(need_block_list),
        "need_block_len": paddle.to_tensor(need_block_len),
        "used_list_len": paddle.to_tensor(used_list_len),
        "free_list_len": paddle.to_tensor(free_list_len),
        "free_list": paddle.to_tensor(free_list),
        "input_ids": paddle.to_tensor(input_ids),
        "pre_ids": paddle.to_tensor(pre_ids),
        "step_idx": paddle.to_tensor(step_idx),
        "next_tokens": paddle.to_tensor(next_tokens),
        "first_token_ids": paddle.to_tensor(first_token_ids),
        "accept_num": paddle.to_tensor(accept_num),
        "block_size": block_size,
        "encoder_decoder_block_num": encoder_decoder_block_num,
        "max_draft_tokens": max_draft_tokens,
    }

    paddle.set_device("xpu:0")
    data_xpu = {
        "stop_flags": paddle.to_tensor(stop_flags),
        "seq_lens_this_time": paddle.to_tensor(seq_lens_this_time),
        "seq_lens_encoder": paddle.to_tensor(seq_lens_encoder),
        "seq_lens_decoder": paddle.to_tensor(seq_lens_decoder),
        "ori_seq_lens_encoder": paddle.to_tensor(ori_seq_lens_encoder),
        "block_tables": paddle.to_tensor(block_tables),
        "encoder_block_lens": paddle.to_tensor(encoder_block_lens),
        "is_block_step": paddle.to_tensor(is_block_step),
        "step_block_list": paddle.to_tensor(step_block_list),
        "step_lens": paddle.to_tensor(step_lens),
        "recover_block_list": paddle.to_tensor(recover_block_list),
        "recover_lens": paddle.to_tensor(recover_lens),
        "need_block_list": paddle.to_tensor(need_block_list),
        "need_block_len": paddle.to_tensor(need_block_len),
        "used_list_len": paddle.to_tensor(used_list_len),
        "free_list_len": paddle.to_tensor(free_list_len),
        "free_list": paddle.to_tensor(free_list),
        "input_ids": paddle.to_tensor(input_ids),
        "pre_ids": paddle.to_tensor(pre_ids),
        "step_idx": paddle.to_tensor(step_idx),
        "next_tokens": paddle.to_tensor(next_tokens),
        "first_token_ids": paddle.to_tensor(first_token_ids),
        "accept_num": paddle.to_tensor(accept_num),
        "block_size": block_size,
        "encoder_decoder_block_num": encoder_decoder_block_num,
        "max_draft_tokens": max_draft_tokens,
    }

    # 恢复默认设备，避免影响其他测试
    paddle.set_device("cpu")

    return data_cpu, data_xpu


def speculate_step_paddle_execution(test_data):
    """测试 speculate_step_paddle 函数的执行性和输出合理性"""
    # 提取输入数据
    stop_flags = test_data["stop_flags"]  # 克隆避免影响夹具数据
    seq_lens_this_time = test_data["seq_lens_this_time"]
    ori_seq_lens_encoder = test_data["ori_seq_lens_encoder"]
    seq_lens_encoder = test_data["seq_lens_encoder"]
    seq_lens_decoder = test_data["seq_lens_decoder"]
    block_tables = test_data["block_tables"]
    encoder_block_lens = test_data["encoder_block_lens"]
    is_block_step = test_data["is_block_step"]
    step_block_list = test_data["step_block_list"]
    step_lens = test_data["step_lens"]
    recover_block_list = test_data["recover_block_list"]
    recover_lens = test_data["recover_lens"]
    need_block_list = test_data["need_block_list"]
    need_block_len = test_data["need_block_len"]
    used_list_len = test_data["used_list_len"]
    free_list = test_data["free_list"]
    free_list_len = test_data["free_list_len"]
    input_ids = test_data["input_ids"]
    pre_ids = test_data["pre_ids"]
    step_idx = test_data["step_idx"]
    next_tokens = test_data["next_tokens"]
    first_token_ids = test_data["first_token_ids"]
    accept_num = test_data["accept_num"]
    block_size = test_data["block_size"]
    encoder_decoder_block_num = test_data["encoder_decoder_block_num"]
    max_draft_tokens = test_data["max_draft_tokens"]

    # 可选：打印执行前关键信息（如需调试可开启）
    if os.environ.get("STEP_TEST_DEBUG", "0") == "1":
        print("-" * 50 + "before step op" + "-" * 50)
        # ... (省略打印内容以保持简洁)

    # 执行目标函数（核心测试步骤）
    speculate_step_paddle(
        stop_flags,
        seq_lens_this_time,
        ori_seq_lens_encoder,
        seq_lens_encoder,
        seq_lens_decoder,
        block_tables,
        encoder_block_lens,
        is_block_step,
        step_block_list,
        step_lens,
        recover_block_list,
        recover_lens,
        need_block_list,
        need_block_len,
        used_list_len,
        free_list,
        free_list_len,
        input_ids,
        pre_ids,
        step_idx,
        next_tokens,
        first_token_ids,
        accept_num,
        block_size,
        encoder_decoder_block_num,
        max_draft_tokens,
    )

    # 可选：打印执行后关键信息（如需调试可开启）
    if os.environ.get("STEP_TEST_DEBUG", "0") == "1":
        print("-" * 50 + "after step op" + "-" * 50)
        # ... (省略打印内容以保持简洁)

    return test_data


class TestSpeculateStepPaddle(unittest.TestCase):
    """
    测试类，继承自 unittest.TestCase。
    所有以 'test_' 开头的方法都会被视为测试用例。
    """

    def assert_test_data_equal(self, test_data1, test_data2, rtol=1e-05, atol=1e-08):
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
        for key in keys1:
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

    def test_speculate_step_paddle_execution(self):
        """
        核心测试用例方法。
        该方法会调用 generate_test_data 获取数据，
        分别在 CPU 和 XPU 上执行测试函数，
        并使用自定义的断言方法比较结果。
        """
        print("\nRunning test: test_speculate_step_paddle_execution")

        # 1. 获取测试数据
        data_cpu, data_xpu = generate_test_data()

        # 2. 执行测试函数
        result_xpu = speculate_step_paddle_execution(data_xpu)
        result_cpu = speculate_step_paddle_execution(data_cpu)

        # 3. 断言结果一致
        self.assert_test_data_equal(result_xpu, result_cpu)


if __name__ == "__main__":
    # 使用 unittest 的主程序来运行所有测试用例
    unittest.main()
