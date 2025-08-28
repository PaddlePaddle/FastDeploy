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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import speculate_get_seq_lens_output  # 假设已编译并导入


def run_seq_lens_test(device="cpu"):
    """运行序列长度测试函数"""
    paddle.seed(42)
    np.random.seed(42)

    if device == "cpu":
        paddle.set_device(device)
    elif device == "xpu":
        paddle.set_device(device)
    else:
        raise ValueError(f"Invalid device: {device}")

    # 创建不同尺寸的随机测试数据
    batch_sizes = [1, 4, 16, 64, 128, 192, 256]
    results = []
    test_times = 100
    for _ in range(test_times):
        for bsz in batch_sizes:
            # 生成随机输入张量
            seq_lens_this_time = paddle.randint(0, 10, shape=(bsz,), dtype="int32")
            seq_lens_encoder = paddle.randint(0, 10, shape=(bsz,), dtype="int32")
            seq_lens_decoder = paddle.randint(0, 10, shape=(bsz,), dtype="int32")

            # 记录输入值用于调试
            input_values = [
                seq_lens_this_time.numpy().copy(),
                seq_lens_encoder.numpy().copy(),
                seq_lens_decoder.numpy().copy(),
            ]
            # 运行算子
            seq_lens_output = speculate_get_seq_lens_output(seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)[0]

            # 收集结果
            results.append((input_values, seq_lens_output.numpy()))

    return results


if __name__ == "__main__":
    print("\n运行XPU测试...")
    xpu_results = run_seq_lens_test("xpu")

    print("运行CPU测试...")
    cpu_results = run_seq_lens_test("cpu")

    print("\n比较结果...")
    all_pass = True

    # 逐个批次比较结果
    for i, (cpu_data, xpu_data) in enumerate(zip(cpu_results, xpu_results)):
        # 解包数据
        cpu_inputs, cpu_output = cpu_data
        xpu_inputs, xpu_output = xpu_data

        # 比较输入数据是否相同
        for j in range(3):
            if not np.array_equal(cpu_inputs[j], xpu_inputs[j]):
                print(f"错误: 批次 #{i+1} 输入 {j} 不同 (CPU vs XPU)")
                print(f"CPU输入: {cpu_inputs[j]}")
                print(f"XPU输入: {xpu_inputs[j]}")
                all_pass = False

        # 比较输出结果是否相同
        if not np.array_equal(cpu_output, xpu_output):
            print(f"\n错误: 批次 #{i+1} 输出不同 (CPU vs XPU)")
            print(f"CPU输出: {cpu_output}")
            print(f"XPU输出: {xpu_output}")

            # 打印差异详情
            diff_indices = np.where(cpu_output != xpu_output)[0]
            for idx in diff_indices:
                print(f"索引 {idx}: CPU输出={cpu_output[idx]}, XPU输出={xpu_output[idx]}")
                print(
                    f"对应输入: this_time={cpu_inputs[0][idx]}, "
                    f"encoder={cpu_inputs[1][idx]}, decoder={cpu_inputs[2][idx]}"
                )
            all_pass = False
        else:
            print(f"批次 #{i+1} 结果匹配")

    if all_pass:
        print("\n所有测试通过! CPU和XPU结果完全一致")
    else:
        print("\n测试失败: 发现不一致的结果")
        exit(1)
