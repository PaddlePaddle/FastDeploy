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

import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.cache_manager.cache_metrics import CacheMetrics
from fastdeploy.output.token_processor import TokenProcessor


class TestCoverageFix(unittest.TestCase):
    @patch("fastdeploy.cache_manager.cache_metrics.main_process_metrics")
    def test_cache_metrics_update_history(self, mock_main_process_metrics):
        """
        测试 CacheMetrics._update_history_hit_metrics 方法。

        目标：确保 main_process_metrics 的 .set() 方法被正确调用，覆盖第 58-61 行。
        """
        print("\nRunning test for CacheMetrics._update_history_hit_metrics...")
        metrics = CacheMetrics()

        # 准备数据以避免除零错误
        metrics.req_count = 20
        metrics.hit_req_count = 10
        metrics.total_token_num = 1000
        metrics.total_cpu_matched_token_num = 250
        metrics.total_gpu_matched_token_num = 350
        metrics.matched_token_num = metrics.total_cpu_matched_token_num + metrics.total_gpu_matched_token_num

        # 调用目标方法
        metrics._update_history_hit_metrics()

        # 断言 Prometheus 指标的 set 方法是否被正确的值调用
        mock_main_process_metrics.hit_req_rate.set.assert_called_once_with(0.5)  # 10 / 20
        mock_main_process_metrics.hit_token_rate.set.assert_called_once_with(0.6)  # 600 / 1000
        mock_main_process_metrics.cpu_hit_token_rate.set.assert_called_once_with(0.25)  # 250 / 1000
        mock_main_process_metrics.gpu_hit_token_rate.set.assert_called_once_with(0.35)  # 350 / 1000

        print("Test for CacheMetrics passed.")

    def setUp(self):
        """为 TokenProcessor 测试设置通用的 mock 对象。"""
        self.mock_cfg = MagicMock()
        self.mock_cached_generated_tokens = MagicMock()
        self.mock_engine_worker_queue = MagicMock()
        self.mock_split_connector = MagicMock()
        self.mock_resource_manager = MagicMock()

        self.processor = TokenProcessor(
            cfg=self.mock_cfg,
            cached_generated_tokens=self.mock_cached_generated_tokens,
            engine_worker_queue=self.mock_engine_worker_queue,
            split_connector=self.mock_split_connector,
        )
        self.processor.resource_manager = self.mock_resource_manager

    # 使用 patch 来模拟 token_processor 模块中引用的 main_process_metrics
    @patch("fastdeploy.output.token_processor.main_process_metrics")
    def test_recycle_resources_updates_metrics(self, mock_main_process_metrics):
        """
        测试 TokenProcessor._recycle_resources 方法。

        目标：确保 available_batch_size 等指标被更新，覆盖第 285 行左右的代码。
        """
        print("\nRunning test for TokenProcessor._recycle_resources (metric update)...")

        # 1. 准备测试数据和 mock 行为
        task_id = "request-456"
        index = 0
        mock_task = MagicMock()

        # 配置 resource_manager 的 mock 返回值
        self.mock_resource_manager.available_batch.return_value = 8
        self.mock_resource_manager.total_block_number.return_value = 1024
        self.mock_resource_manager.max_num_seqs = 16

        # _recycle_resources 方法内部会操作这些列表/字典
        self.mock_resource_manager.tasks_list = [mock_task]
        self.mock_resource_manager.stop_flags = [False]

        # 为了避免 del self.tokens_counter[task_id] 抛出 KeyError
        self.processor.tokens_counter[task_id] = 5

        # 调用目标方法
        self.processor._recycle_resources(task_id=task_id, index=index, task=mock_task, result=None, is_prefill=False)

        # 核心断言：验证 available_batch_size 指标是否被正确设置
        mock_main_process_metrics.available_batch_size.set.assert_called_once_with(8)

        print("Test for TokenProcessor passed.")


if __name__ == "__main__":
    unittest.main()
