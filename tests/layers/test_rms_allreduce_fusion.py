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

import time
import unittest
from unittest.mock import Mock

import numpy as np
import paddle
import paddle.distributed as dist


class TestFlashInferAllReduceResidualRMSNorm(unittest.TestCase):
    """测试 FlashInfer AllReduce + Residual + RMSNorm 融合算子"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        if paddle.is_compiled_with_cuda():
            paddle.set_device("gpu")
        else:
            paddle.set_device("cpu")
        dist.init_parallel_env()

    def setUp(self):
        """每个测试用例的初始化"""
        # 固定随机种子，确保可复现性
        paddle.seed(42)
        np.random.seed(42)

        self.dtype = paddle.float32
        self.token_num = 128
        self.hidden_dim = 768
        self.eps = 1e-6
        self.epsilon = 1e-6
        self.max_token_num = 2048

        # 创建 mock FDConfig
        self.fd_config = Mock()
        self.fd_config.parallel_config = Mock()
        self.fd_config.parallel_config.tensor_parallel_size = dist.get_world_size()
        self.begin_norm_axis = 1

        # 性能测试参数 - 增加迭代次数提高稳定性
        self.warmup_iterations = 20  # 增加warmup
        self.test_iterations = 200  # 增加测试迭代

    def tearDown(self):
        """清理资源"""
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
            paddle.device.cuda.synchronize()

    def create_test_tensors(self):
        """创建测试用的张量"""
        input_tensor = paddle.randn([self.token_num, self.hidden_dim], dtype=self.dtype)
        residual = paddle.randn([self.token_num, self.hidden_dim], dtype=self.dtype)
        weight = paddle.randn([self.hidden_dim], dtype=self.dtype)
        return input_tensor, residual, weight

    def compute_reference_output(self, input_tensor, residual, weight, eps):
        """参考实现：手动计算 AllReduce + Residual + RMSNorm"""
        # # Step 1: AllReduce (在单卡情况下就是原值)
        # allreduce_out = input_tensor.clone()
        # 添加 all reduce 算子
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        # Step 2: Add residual
        residual_out = input_tensor + residual

        # Step 3: RMSNorm
        variance = residual_out.pow(2).mean(axis=-1, keepdim=True)
        norm_out = residual_out * paddle.rsqrt(variance + eps)
        norm_out = norm_out * weight

        # dist.all_reduce(residual_out, op=dist.ReduceOp.SUM)
        return norm_out, residual_out

    def paddle_rms_fuse(self, input_tensor, residual, weight, eps):
        from paddle.incubate.nn.functional import fused_rms_norm

        # 添加 all reduce 算子
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
        out_fused = fused_rms_norm(
            input_tensor,
            norm_weight=weight,
            norm_bias=None,
            epsilon=eps,
            begin_norm_axis=self.begin_norm_axis,
            bias=None,
            residual=residual,
        )

        return out_fused[0], out_fused[1]

    def flashinfer_rms_fuse(self, input_tensor, residual, weight, eps):
        """FlashInfer融合算子"""
        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        norm_out, residual_out = flashinfer_allreduce_residual_rmsnorm(
            fd_config=self.fd_config,
            input_tensor=input_tensor,
            residual=residual,
            weight=weight,
            eps=eps,
            max_token_num=self.max_token_num,
            use_oneshot=False,
        )
        return norm_out, residual_out

    def benchmark_function(self, func, *args, name="", **kwargs):
        """
        改进的性能基准测试
        - 增加GPU频率稳定等待
        - 使用中位数而非平均值（更稳定）
        - 过滤异常值
        """
        # 强制GPU频率稳定
        if paddle.is_compiled_with_cuda():
            for _ in range(5):
                paddle.device.cuda.synchronize()
                time.sleep(0.01)

        # Warmup - 充分预热
        for _ in range(self.warmup_iterations):
            result = func(*args, **kwargs)
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

        # 额外等待，确保GPU稳定
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
            time.sleep(0.1)

        # 正式测试
        times = []
        for i in range(self.test_iterations):
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

            start = time.perf_counter()
            result = func(*args, **kwargs)

            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

            end = time.perf_counter()
            elapsed = (end - start) * 1000  # 转换为毫秒
            times.append(elapsed)

        times = np.array(times)

        # 使用IQR方法过滤异常值
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_times = times[(times >= lower_bound) & (times <= upper_bound)]

        # 如果过滤后数据太少，使用原始数据
        if len(filtered_times) < self.test_iterations * 0.5:
            filtered_times = times

        # 统计信息
        avg_time = np.mean(filtered_times)
        median_time = np.median(filtered_times)
        std_time = np.std(filtered_times)
        min_time = np.min(filtered_times)
        max_time = np.max(filtered_times)
        cv = (std_time / avg_time) * 100  # 变异系数 (%)

        print(f"\n{'='*70}")
        print(f"Performance Benchmark: {name}")
        print(f"{'='*70}")
        print(f"Iterations: {len(filtered_times)}/{self.test_iterations} (after {self.warmup_iterations} warmup)")
        print(f"Median:     {median_time:.4f} ms  (most stable metric)")
        print(f"Average:    {avg_time:.4f} ms")
        print(f"Std Dev:    {std_time:.4f} ms  (CV: {cv:.2f}%)")
        print(f"Min:        {min_time:.4f} ms")
        print(f"Max:        {max_time:.4f} ms")
        print(f"{'='*70}\n")

        # 返回中位数（更稳定）和结果
        return median_time, result

    def test_accuracy_fused_vs_reference(self):
        """测试融合算子与参考实现的准确性"""
        input_tensor, residual, weight = self.create_test_tensors()
        reference_output, ref_res = self.compute_reference_output(
            input_tensor.clone(), residual.clone(), weight.clone(), self.eps
        )
        fused_output, paddle_res = self.paddle_rms_fuse(
            input_tensor.clone(), residual.clone(), weight.clone(), self.eps
        )
        flashinfer_output, flashinfer_res = self.flashinfer_rms_fuse(
            input_tensor.clone(), residual.clone(), weight.clone(), self.eps
        )
        # 验证结果
        np.testing.assert_allclose(fused_output.numpy(), reference_output.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_res.numpy(), paddle_res.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(flashinfer_output.numpy(), reference_output.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ref_res.numpy(), flashinfer_res.numpy(), rtol=1e-5, atol=1e-5)


class TestFlashInferWorkspaceManager(unittest.TestCase):
    """测试 FlashInferWorkspaceManager"""

    def setUp(self):
        """初始化"""
        from fastdeploy.model_executor.layers.flashinfer_comm_fusion import (
            FlashInferWorkspaceManager,
        )

        self.manager = FlashInferWorkspaceManager()

    def test_initialization(self):
        """测试初始化状态"""
        self.assertIsNone(self.manager.workspace_tensor)
        self.assertIsNone(self.manager.ipc_handles)
        self.assertIsNone(self.manager.world_size)
        self.assertIsNone(self.manager.rank)
        self.assertFalse(self.manager.initialized)

    def test_cleanup(self):
        """测试清理功能"""
        self.manager.cleanup()
        self.assertFalse(self.manager.initialized)
        self.assertIsNone(self.manager.workspace_tensor)


if __name__ == "__main__":
    unittest.main(verbosity=2)

    # 多卡运行示例:
    # python -m paddle.distributed.launch --gpus=0,1 test_flashinfer.py
