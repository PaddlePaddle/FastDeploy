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

import os
import sys
import unittest
from unittest.mock import Mock, patch

# 添加路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from fastdeploy.engine.expert_service import ExpertService, start_data_parallel_service


class TestExpertService(unittest.TestCase):
    """测试 ExpertService 类"""

    def setUp(self):
        """设置测试环境"""
        # 创建模拟配置对象
        self.mock_cfg = Mock()
        self.mock_cfg.parallel_config = Mock()
        self.mock_cfg.parallel_config.data_parallel_size = 1
        self.mock_cfg.parallel_config.local_engine_worker_queue_port = 8080
        self.mock_cfg.parallel_config.engine_worker_queue_port = [8080, 8081]
        self.mock_cfg.cache_config = Mock()
        self.mock_cfg.cache_config.num_gpu_blocks_override = None
        self.mock_cfg.scheduler_config = Mock()
        self.mock_cfg.scheduler_config.name = "default"
        self.mock_cfg.scheduler_config.splitwise_role = "mixed"
        self.mock_cfg.host_ip = "127.0.0.1"
        self.mock_cfg.register_info = {}
        self.mock_cfg.worker_num_per_node = 1
        self.mock_cfg.nnode = 1
        self.mock_cfg.local_device_ids = [0]

    @patch("fastdeploy.engine.expert_service.EngineService")
    @patch("fastdeploy.engine.expert_service.get_logger")
    @patch("fastdeploy.engine.expert_service.llm_logger")
    def test_expert_service_init_single_dp(self, mock_llm_logger, mock_get_logger, mock_engine_service):
        """测试单数据并行模式下的初始化"""
        local_data_parallel_id = 0

        # 创建 ExpertService 实例
        expert_service = ExpertService(self.mock_cfg, local_data_parallel_id)

        # 验证配置设置
        self.assertEqual(expert_service.cfg, self.mock_cfg)

        # 验证日志设置
        self.assertEqual(expert_service.llm_logger, mock_llm_logger)

        # 验证 EngineService 初始化
        mock_engine_service.assert_called_once_with(self.mock_cfg, True)

    @patch("fastdeploy.engine.expert_service.EngineService")
    @patch("fastdeploy.engine.expert_service.get_logger")
    @patch("fastdeploy.engine.expert_service.envs")
    def test_expert_service_init_multi_dp(self, mock_envs, mock_get_logger, mock_engine_service):
        """测试多数据并行模式下的初始化"""
        # 设置多数据并行配置
        self.mock_cfg.parallel_config.data_parallel_size = 2
        mock_envs.FD_ENABLE_MULTI_API_SERVER = False
        mock_envs.FD_ENABLE_INTERNAL_ADAPTER = False

        local_data_parallel_id = 1
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # 创建 ExpertService 实例
        expert_service = ExpertService(self.mock_cfg, local_data_parallel_id)

        # 验证配置更新
        self.assertEqual(expert_service.cfg.parallel_config.local_data_parallel_id, local_data_parallel_id)

        # 验证多DP模式下的日志设置
        mock_get_logger.assert_called_once_with("fastdeploy", f"fastdeploy_dprank{local_data_parallel_id}.log")

    @patch("fastdeploy.engine.expert_service.EngineService")
    @patch("fastdeploy.engine.expert_service.time")
    @patch("fastdeploy.engine.expert_service.threading")
    @patch("fastdeploy.engine.expert_service.envs")
    def test_start_method(self, mock_envs, mock_threading, mock_time, mock_engine_service):
        mock_envs.FD_ENABLE_RETURN_TEXT = False
        mock_envs.FD_ENABLE_MULTI_API_SERVER = False

        local_data_parallel_id = 0

        mock_process = Mock()
        mock_process.pid = 1234

        # 关键：设在实例 mock 上
        mock_engine_instance = mock_engine_service.return_value
        mock_engine_instance.start_cache_service.return_value = [mock_process]

        expert_service = ExpertService(self.mock_cfg, local_data_parallel_id)

        with patch("fastdeploy.engine.expert_service.IPCSignal") as mock_ipc_signal:
            mock_ipc_instance = Mock()
            mock_ipc_instance.value = [100]
            mock_ipc_signal.return_value = mock_ipc_instance

            result = expert_service.start(None, local_data_parallel_id)

        # 验证用的是 EngineService 的实例 mock
        mock_engine_instance.start.assert_called_once()
        mock_engine_instance.start_zmq_service.assert_called_once_with(
            self.mock_cfg.parallel_config.engine_worker_queue_port[local_data_parallel_id]
        )
        mock_engine_instance.start_cache_service.assert_called_once()

        self.assertTrue(result)

    @patch("fastdeploy.engine.expert_service.EngineService")
    @patch("fastdeploy.engine.expert_service.IPCSignal")
    @patch("fastdeploy.engine.expert_service.time")
    def test_reset_kvcache_blocks(self, mock_time, mock_ipc_signal, mock_engine_service):
        """测试重置KV缓存块功能"""
        local_data_parallel_id = 0

        # 创建 ExpertService 实例
        expert_service = ExpertService(self.mock_cfg, local_data_parallel_id)
        expert_service.llm_logger = Mock()
        expert_service.engine = Mock()
        expert_service.engine.resource_manager = Mock()

        # 设置模拟信号
        mock_signal_instance = Mock()
        mock_signal_instance.value = [100]  # 模拟已获取的块数
        expert_service.get_profile_block_num_signal = mock_signal_instance

        # 调用 reset_kvcache_blocks
        expert_service.reset_kvcache_blocks()

        # 验证缓存配置重置
        self.mock_cfg.cache_config.reset.assert_called_once_with(100)
        expert_service.engine.resource_manager.reset_cache_config.assert_called_once_with(self.mock_cfg.cache_config)

    @patch("fastdeploy.engine.expert_service.EngineService")
    @patch("fastdeploy.engine.expert_service.os")
    @patch("fastdeploy.engine.expert_service.signal")
    def test_exit_sub_services(self, mock_signal, mock_os, mock_engine_service):
        """测试退出子服务功能"""
        local_data_parallel_id = 0
        pgid = 5678

        # 创建 ExpertService 实例
        expert_service = ExpertService(self.mock_cfg, local_data_parallel_id)
        expert_service.llm_logger = Mock()

        # 设置模拟缓存管理进程
        mock_process = Mock()
        mock_process.pid = 1234
        expert_service.cache_manager_processes = [mock_process]
        mock_os.getpgid.return_value = pgid

        # 设置模拟引擎资源管理器
        expert_service.engine = Mock()
        expert_service.engine.resource_manager = Mock()
        expert_service.engine.resource_manager.cache_manager = Mock()
        expert_service.engine.resource_manager.cache_manager.shm_cache_task_flag_broadcast = Mock()

        # 设置模拟ZMQ服务器
        expert_service.zmq_server = Mock()

        # 调用退出方法
        expert_service._exit_sub_services()

        # 验证缓存管理器清理
        expert_service.engine.resource_manager.cache_manager.shm_cache_task_flag_broadcast.clear.assert_called_once()
        mock_os.getpgid.assert_called_once_with(1234)
        mock_os.killpg.assert_called_once_with(pgid, mock_signal.SIGTERM)

        # 验证ZMQ服务器关闭
        expert_service.zmq_server.close.assert_called_once()

    @patch("fastdeploy.engine.expert_service.ExpertService")
    @patch("fastdeploy.engine.expert_service.threading")
    @patch("fastdeploy.engine.expert_service.time")
    @patch("fastdeploy.engine.expert_service.traceback")
    def test_start_data_parallel_service_success(self, mock_traceback, mock_time, mock_threading, mock_expert_service):
        """测试启动数据并行服务的成功情况"""
        mock_cfg = Mock()
        local_data_parallel_id = 0

        # 模拟 ExpertService 实例
        mock_expert_instance = Mock()
        mock_expert_service.return_value = mock_expert_instance

        # 模拟线程
        mock_thread_instance = Mock()
        mock_threading.Thread.return_value = mock_thread_instance

        # 调用函数
        start_data_parallel_service(mock_cfg, local_data_parallel_id)

        # 验证 ExpertService 创建和启动
        mock_expert_service.assert_called_once_with(mock_cfg, local_data_parallel_id, start_queue=False)
        mock_expert_instance.start.assert_called_once_with(None, local_data_parallel_id, None, None)

    @patch("fastdeploy.engine.expert_service.ExpertService")
    @patch("fastdeploy.engine.expert_service.llm_logger")
    @patch("fastdeploy.engine.expert_service.traceback")
    def test_start_data_parallel_service_exception(self, mock_traceback, mock_llm_logger, mock_expert_service):
        """测试启动数据并行服务的异常情况"""
        mock_cfg = Mock()
        local_data_parallel_id = 0

        # 模拟 ExpertService 启动失败
        mock_expert_instance = Mock()
        mock_expert_instance.start.side_effect = Exception("Test exception")
        mock_expert_service.return_value = mock_expert_instance

        # 模拟 traceback
        mock_traceback.format_exc.return_value = "Traceback details"

        # 调用函数并验证没有抛出异常
        try:
            start_data_parallel_service(mock_cfg, local_data_parallel_id)
        except Exception:
            self.fail("start_data_parallel_service should handle exceptions gracefully")

        # 验证异常被记录
        mock_llm_logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
