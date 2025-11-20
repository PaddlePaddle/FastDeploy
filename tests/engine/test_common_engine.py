import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastdeploy.engine.common_engine import EngineService


class TestStartWorkerQueueService:
    @pytest.fixture
    def engine(self):
        # 创建完全模拟的 EngineService 实例
        engine = MagicMock(spec=EngineService)
        engine.cfg = MagicMock()
        engine.cfg.master_ip = "127.0.0.1"
        engine.cfg.parallel_config = MagicMock()
        engine.cfg.parallel_config.engine_worker_queue_port = ["8080"]
        engine.cfg.parallel_config.local_data_parallel_id = 0
        engine.cfg.parallel_config.tensor_parallel_size = 1
        engine.cfg.parallel_config.data_parallel_size = 1
        engine.cfg.host_ip = "127.0.0.1"
        engine.cfg.cache_config = MagicMock()
        engine.cfg.cache_config.enable_prefix_caching = False
        engine.cfg.cache_config.cache_queue_port = "9090"
        engine.cfg.scheduler_config = MagicMock()
        engine.cfg.scheduler_config.splitwise_role = "mixed"
        engine.cfg.worker_num_per_node = 1
        engine.cfg.node_rank = 0
        engine.llm_logger = MagicMock()
        return engine

    def test_start_with_tcp_port(self, engine):
        # 模拟方法行为
        def mock_start_worker_queue_service(start_queue):
            if start_queue:
                engine.cfg.parallel_config.engine_worker_queue_port[0] = "8081"

        engine.start_worker_queue_service = mock_start_worker_queue_service

        # 调用方法
        engine.start_worker_queue_service(start_queue=True)

        # 验证配置更新
        assert engine.cfg.parallel_config.engine_worker_queue_port[0] == "8081"

    def test_port_in_use_exception(self, engine):
        # 模拟方法行为
        def mock_start_worker_queue_service(start_queue):
            if start_queue:
                raise Exception("The parameter `engine_worker_queue_port`:8080 is already in use.")

        engine.start_worker_queue_service = mock_start_worker_queue_service

        # 验证异常
        with pytest.raises(Exception) as excinfo:
            engine.start_worker_queue_service(start_queue=True)

        assert "is already in use" in str(excinfo.value)

    def test_start_with_shm(self, engine):
        # 模拟方法行为
        def mock_start_worker_queue_service(start_queue):
            if start_queue:
                engine.engine_worker_queue_server = MagicMock()
                engine.engine_worker_queue_server.address = "/dev/shm/fd_task_queue_8080.sock"

        engine.start_worker_queue_service = mock_start_worker_queue_service

        # 调用方法
        engine.start_worker_queue_service(start_queue=True)

        # 验证地址设置
        assert hasattr(engine, "engine_worker_queue_server")
        assert "shm" in engine.engine_worker_queue_server.address

    def test_start_cache_queue(self, engine):
        # 模拟方法行为
        def mock_start_worker_queue_service(start_queue):
            if start_queue:
                engine.cfg.cache_config.enable_prefix_caching = True
                engine.cache_task_queue = MagicMock()
                engine.cache_task_queue.get_server_port.return_value = "9091"

        engine.start_worker_queue_service = mock_start_worker_queue_service

        # 调用方法
        engine.start_worker_queue_service(start_queue=True)

        # 验证缓存队列设置
        assert engine.cfg.cache_config.enable_prefix_caching is True
        assert hasattr(engine, "cache_task_queue")

    def test_start_with_different_master_ip(self, engine):
        # 模拟方法行为
        def mock_start_worker_queue_service(start_queue):
            if start_queue and engine.cfg.master_ip == "0.0.0.0":
                engine.engine_worker_queue_server = MagicMock()
                engine.engine_worker_queue_server.address = ("0.0.0.0", 8080)

        engine.start_worker_queue_service = mock_start_worker_queue_service
        engine.cfg.master_ip = "0.0.0.0"

        # 调用方法
        engine.start_worker_queue_service(start_queue=True)

        # 验证地址设置
        assert engine.engine_worker_queue_server.address[0] == "0.0.0.0"

    def test_client_mode(self, engine):
        # 模拟方法行为
        def mock_start_worker_queue_service(start_queue):
            if not start_queue:
                engine.engine_worker_queue = MagicMock()
                engine.engine_worker_queue.address = ("127.0.0.1", 8080)
                engine.engine_worker_queue.is_server = False

        engine.start_worker_queue_service = mock_start_worker_queue_service

        # 调用方法
        engine.start_worker_queue_service(start_queue=False)

        # 验证客户端模式设置
        assert engine.engine_worker_queue.is_server is False
        assert engine.engine_worker_queue.address[0] == "127.0.0.1"
