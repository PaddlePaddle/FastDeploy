from unittest.mock import Mock, patch

import pytest

from fastdeploy.engine.common_engine import EngineService


class TestExitSubServicesFMQ:
    """测试 _exit_sub_services 方法中 fmq_a2e_consumer 清理逻辑"""

    @pytest.fixture
    def mock_engine_service(self):
        """创建模拟的 EngineService 实例"""
        engine = Mock(spec=EngineService)
        engine.llm_logger = Mock()
        engine.running = True
        engine.use_async_llm = True
        # 添加 _exit_sub_services 方法中需要的 signal 属性
        engine.exist_task_signal = Mock()
        engine.exist_swapped_task_signal = Mock()
        engine.worker_healthy_live_signal = Mock()
        engine.cache_ready_signal = Mock()
        engine.swap_space_ready_signal = Mock()
        engine.exist_prefill_task_signal = Mock()
        engine.model_weights_status_signal = Mock()
        engine.prefix_tree_status_signal = Mock()
        engine.kv_cache_status_signal = Mock()
        engine.worker_ready_signal = Mock()
        engine.loaded_model_signal = Mock()
        return engine

    @patch("fastdeploy.engine.common_engine.llm_logger")
    def test_fmq_a2e_consumer_with_socket_close_success(self, mock_llm_logger, mock_engine_service):
        """测试 fmq_a2e_consumer 有 socket 且关闭成功的情况"""
        # 模拟 fmq_a2e_consumer 对象，有 socket 属性
        mock_socket = Mock()
        mock_fmq_consumer = Mock()
        mock_fmq_consumer.socket = mock_socket
        mock_engine_service.fmq_a2e_consumer = mock_fmq_consumer

        # 创建真实的函数并调用
        real_function = EngineService._exit_sub_services
        bound_method = real_function.__get__(mock_engine_service, EngineService)

        # 执行函数
        bound_method()

        # 验证 socket.close() 被调用
        mock_socket.close.assert_called_once()
        mock_llm_logger.info.assert_any_call("FMQ consumer socket closed successfully.")
        # 验证 finally 块中 fmq_a2e_consumer 被设置为 None
        assert mock_engine_service.fmq_a2e_consumer is None

    @patch("fastdeploy.engine.common_engine.llm_logger")
    def test_fmq_a2e_consumer_with_socket_close_exception(self, mock_llm_logger, mock_engine_service):
        """测试 fmq_a2e_consumer 有 socket 但关闭时抛出异常的情况"""
        # 模拟 fmq_a2e_consumer 对象，有 socket 属性
        mock_socket = Mock()
        mock_socket.close.side_effect = Exception("Socket close failed")
        mock_fmq_consumer = Mock()
        mock_fmq_consumer.socket = mock_socket
        mock_engine_service.fmq_a2e_consumer = mock_fmq_consumer

        # 创建真实的函数并调用
        real_function = EngineService._exit_sub_services
        bound_method = real_function.__get__(mock_engine_service, EngineService)

        # 执行函数
        bound_method()

        # 验证 socket.close() 被调用
        mock_socket.close.assert_called_once()
        # 验证异常被捕获并记录 error 日志
        mock_llm_logger.error.assert_called_once()
        error_call_args = mock_llm_logger.error.call_args[0][0]
        assert "Error closing fmq_consumer: Socket close failed" in error_call_args
        # 验证 finally 块中 fmq_a2e_consumer 仍然被设置为 None
        assert mock_engine_service.fmq_a2e_consumer is None

    @patch("fastdeploy.engine.common_engine.llm_logger")
    def test_fmq_a2e_consumer_no_socket(self, mock_llm_logger, mock_engine_service):
        """测试 fmq_a2e_consumer 没有 socket 属性的情况"""
        # 模拟 fmq_a2e_consumer 对象，没有 socket 属性
        mock_fmq_consumer = Mock()
        # 移除 socket 属性
        del mock_fmq_consumer.socket
        mock_engine_service.fmq_a2e_consumer = mock_fmq_consumer

        # 创建真实的函数并调用
        real_function = EngineService._exit_sub_services
        bound_method = real_function.__get__(mock_engine_service, EngineService)

        # 执行函数
        bound_method()

        # 验证没有调用 socket.close()
        # 验证没有记录 "FMQ consumer socket closed successfully." 日志
        info_calls = [call[0][0] for call in mock_llm_logger.info.call_args_list]
        assert "FMQ consumer socket closed successfully." not in info_calls
        # 验证 finally 块中 fmq_a2e_consumer 被设置为 None
        assert mock_engine_service.fmq_a2e_consumer is None

    @patch("fastdeploy.engine.common_engine.llm_logger")
    def test_fmq_a2e_consumer_socket_none(self, mock_llm_logger, mock_engine_service):
        """测试 fmq_a2e_consumer 有 socket 属性但 socket 为 None 的情况"""
        # 模拟 fmq_a2e_consumer 对象，socket 为 None
        mock_fmq_consumer = Mock()
        mock_fmq_consumer.socket = None
        mock_engine_service.fmq_a2e_consumer = mock_fmq_consumer

        # 创建真实的函数并调用
        real_function = EngineService._exit_sub_services
        bound_method = real_function.__get__(mock_engine_service, EngineService)

        # 执行函数
        bound_method()

        # 验证没有调用 socket.close()
        # 验证没有记录 "FMQ consumer socket closed successfully." 日志
        info_calls = [call[0][0] for call in mock_llm_logger.info.call_args_list]
        assert "FMQ consumer socket closed successfully." not in info_calls
        # 验证 finally 块中 fmq_a2e_consumer 被设置为 None
        assert mock_engine_service.fmq_a2e_consumer is None

    @patch("fastdeploy.engine.common_engine.llm_logger")
    def test_fmq_a2e_consumer_none(self, mock_llm_logger, mock_engine_service):
        """测试 fmq_a2e_consumer 为 None 的情况"""
        # 设置 fmq_a2e_consumer 为 None
        mock_engine_service.fmq_a2e_consumer = None

        # 创建真实的函数并调用
        real_function = EngineService._exit_sub_services
        bound_method = real_function.__get__(mock_engine_service, EngineService)

        # 执行函数
        bound_method()

        # 验证 fmq_a2e_consumer 保持为 None
        assert mock_engine_service.fmq_a2e_consumer is None
        # 验证没有调用任何 socket 相关方法
        info_calls = [call[0][0] for call in mock_llm_logger.info.call_args_list]
        assert "FMQ consumer socket closed successfully." not in info_calls
        error_calls = [call[0][0] for call in mock_llm_logger.error.call_args_list]
        assert not any("Error closing fmq_consumer" in call for call in error_calls)

    @patch("fastdeploy.engine.common_engine.llm_logger")
    def test_fmq_a2e_consumer_hasattr_false(self, mock_llm_logger, mock_engine_service):
        """测试对象没有 fmq_a2e_consumer 属性的情况"""
        # 确保对象没有 fmq_a2e_consumer 属性
        if hasattr(mock_engine_service, "fmq_a2e_consumer"):
            delattr(mock_engine_service, "fmq_a2e_consumer")

        # 创建真实的函数并调用
        real_function = EngineService._exit_sub_services
        bound_method = real_function.__get__(mock_engine_service, EngineService)

        # 执行函数
        bound_method()

        # 验证没有调用任何 socket 相关方法
        info_calls = [call[0][0] for call in mock_llm_logger.info.call_args_list]
        assert "FMQ consumer socket closed successfully." not in info_calls
        error_calls = [call[0][0] for call in mock_llm_logger.error.call_args_list]
        assert not any("Error closing fmq_consumer" in call for call in error_calls)
