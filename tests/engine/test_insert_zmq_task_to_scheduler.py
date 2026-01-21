from unittest.mock import AsyncMock, Mock, patch

import pytest

from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.request import Request, RequestMetrics


class TestInsertZmqTaskToScheduler:
    """测试 _insert_zmq_task_to_scheduler 函数的单元测试"""

    @pytest.fixture
    def mock_engine_service(self):
        """创建模拟的 EngineService 实例"""
        engine = Mock(spec=EngineService)
        engine.running = True
        engine.cfg = Mock()
        engine.cfg.scheduler_config = Mock()
        engine.cfg.scheduler_config.splitwise_role = "prefill"
        engine.cfg.model_config = Mock()
        engine.cfg.model_config.enable_mm = False
        engine.llm_logger = Mock()
        engine.scheduler = Mock()
        engine.guided_decoding_checker = None
        engine.fmq_a2e_consumer = None

        # 模拟 added_requests 字典
        engine.added_requests = {}

        return engine

    @pytest.fixture
    def sample_request_data(self):
        """示例请求数据"""
        return {
            "request_id": "test_req_123",
            "prompt": "Hello, world!",
            "prompt_token_ids": [1, 2, 3, 4],
            "prompt_token_ids_len": 4,
            "messages": None,
            "history": None,
            "tools": None,
            "system": None,
            "eos_token_ids": [2],
            "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 100},
            "user": "test_user",
        }

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_with_internal_adapter_json(
        self, mock_engine_service, sample_request_data
    ):
        """测试 FD_ENABLE_INTERNAL_ADAPTER=True 且 enable_mm=False 的情况（JSON模式）"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                    with patch("fastdeploy.engine.common_engine.trace_print") as mock_trace:
                        # 模拟 recv_request_server
                        mock_recv_server = Mock()
                        mock_recv_server.receive_json_once.return_value = (None, sample_request_data)
                        mock_engine_service.recv_request_server = mock_recv_server

                        # 模拟 Request.from_dict 返回
                        mock_request = Mock(spec=Request)
                        mock_request.request_id = "test_req_123"
                        mock_request.metrics = RequestMetrics()
                        mock_from_dict.return_value = mock_request
                        mock_trace = mock_trace if mock_trace is not None else mock_trace

                        # 模拟 scheduler.put_requests 返回
                        mock_engine_service.scheduler.put_requests.return_value = [("test_req_123", None)]

                        # 模拟 metrics
                        mock_metrics.requests_number = Mock()
                        mock_metrics.num_requests_waiting = Mock()

                        # 创建真实的函数并调用
                        real_function = EngineService._insert_zmq_task_to_scheduler
                        bound_method = real_function.__get__(mock_engine_service, EngineService)

                        # 由于这是一个无限循环的函数，我们需要模拟只运行一次
                        # 使用 side_effect 来控制循环退出
                        call_count = 0

                        def mock_receive_json_once(block):
                            nonlocal call_count
                            call_count += 1
                            if call_count == 1:
                                return (None, sample_request_data)
                            else:
                                # 模拟 Context was terminated 错误来退出循环
                                return (Exception("Context was terminated"), None)

                        mock_recv_server.receive_json_once.side_effect = mock_receive_json_once

                        # 执行函数
                        await bound_method()

                        # 验证调用
                        mock_recv_server.receive_json_once.assert_called()
                        mock_from_dict.assert_called_once_with(sample_request_data)
                        mock_engine_service.scheduler.put_requests.assert_called_once()
                        mock_metrics.requests_number.inc.assert_called_once()
                        mock_metrics.num_requests_waiting.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_with_internal_adapter_pyobj(
        self, mock_engine_service, sample_request_data
    ):
        """测试 FD_ENABLE_INTERNAL_ADAPTER=True 且 enable_mm=True 的情况（PyObj模式）"""
        mock_engine_service.cfg.model_config.enable_mm = True

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                    with patch("fastdeploy.engine.common_engine.trace_print") as mock_trace:
                        # 模拟 recv_request_server
                        mock_recv_server = Mock()
                        mock_recv_server.receive_pyobj_once.return_value = (None, sample_request_data)
                        mock_engine_service.recv_request_server = mock_recv_server

                        # 模拟 Request.from_dict 返回
                        mock_request = Mock(spec=Request)
                        mock_request.request_id = "test_req_123"
                        mock_request.metrics = RequestMetrics()
                        mock_from_dict.return_value = mock_request
                        mock_trace = mock_trace if mock_trace is not None else mock_trace
                        # 模拟 scheduler.put_requests 返回
                        mock_engine_service.scheduler.put_requests.return_value = [("test_req_123", None)]

                        # 模拟 metrics
                        mock_metrics.requests_number = Mock()
                        mock_metrics.num_requests_waiting = Mock()

                        # 创建真实的函数并调用
                        real_function = EngineService._insert_zmq_task_to_scheduler
                        bound_method = real_function.__get__(mock_engine_service, EngineService)

                        call_count = 0

                        def mock_receive_pyobj_once(block):
                            nonlocal call_count
                            call_count += 1
                            if call_count == 1:
                                return (None, sample_request_data)
                            else:
                                return (Exception("Context was terminated"), None)

                        mock_recv_server.receive_pyobj_once.side_effect = mock_receive_pyobj_once

                        # 执行函数
                        await bound_method()

                        # 验证调用
                        mock_recv_server.receive_pyobj_once.assert_called()
                        mock_from_dict.assert_called_once_with(sample_request_data)
                        mock_engine_service.scheduler.put_requests.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_without_internal_adapter(
        self, mock_engine_service, sample_request_data
    ):
        """测试 FD_ENABLE_INTERNAL_ADAPTER=False 的情况（FMQ模式）"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
            with patch("fastdeploy.engine.common_engine.FMQFactory") as mock_fmq_factory:
                with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                    with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                        # 模拟 FMQ consumer
                        mock_consumer = AsyncMock()
                        mock_msg = Mock()
                        mock_msg.payload = sample_request_data
                        mock_fmq_factory.q_a2e_consumer.return_value = mock_consumer
                        mock_engine_service.fmq_a2e_consumer = mock_consumer

                        # 模拟 Request.from_dict 返回
                        mock_request = Mock(spec=Request)
                        mock_request.request_id = "test_req_123"
                        mock_request.metrics = RequestMetrics()
                        mock_from_dict.return_value = mock_request

                        # 模拟 scheduler.put_requests 返回
                        mock_engine_service.scheduler.put_requests.return_value = [("test_req_123", None)]

                        # 模拟 metrics
                        mock_metrics.requests_number = Mock()
                        mock_metrics.num_requests_waiting = Mock()

                        # 创建真实的函数并调用
                        real_function = EngineService._insert_zmq_task_to_scheduler
                        bound_method = real_function.__get__(mock_engine_service, EngineService)

                        call_count = 0

                        async def mock_get():
                            nonlocal call_count
                            call_count += 1
                            if call_count == 1:
                                return mock_msg
                            else:
                                # 抛出异常来退出循环，而不是返回 None
                                raise Exception("FMQ connection terminated")

                        mock_consumer.get.side_effect = mock_get

                        # 执行函数
                        await bound_method()

                        # 验证调用
                        mock_consumer.get.assert_called()
                        mock_from_dict.assert_called_once_with(sample_request_data)
                        mock_engine_service.scheduler.put_requests.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_decode_role_early_return(self, mock_engine_service):
        """测试 splitwise_role='decode' 时的早期返回"""
        mock_engine_service.cfg.scheduler_config.splitwise_role = "decode"

        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            # 创建真实的函数并调用
            real_function = EngineService._insert_zmq_task_to_scheduler
            bound_method = real_function.__get__(mock_engine_service, EngineService)

            # 执行函数，应该立即返回
            await bound_method()

            # 验证没有进行任何网络调用
            assert (
                not hasattr(mock_engine_service, "recv_request_server")
                or mock_engine_service.recv_request_server is None
            )

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_request_error(self, mock_engine_service, sample_request_data):
        """测试请求解析错误的情况"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                    # 模拟 recv_request_server
                    mock_recv_server = Mock()
                    mock_recv_server.receive_json_once.return_value = (None, sample_request_data)
                    mock_engine_service.recv_request_server = mock_recv_server

                    # 模拟 Request.from_dict 抛出异常
                    mock_from_dict.side_effect = Exception("Invalid request data")

                    # 模拟 _send_error_response
                    mock_engine_service._send_error_response = Mock()

                    # 模拟 scheduler.put_requests 返回
                    mock_engine_service.scheduler.put_requests.return_value = []

                    # 模拟 metrics
                    mock_metrics.requests_number = Mock()

                    # 创建真实的函数并调用
                    real_function = EngineService._insert_zmq_task_to_scheduler
                    bound_method = real_function.__get__(mock_engine_service, EngineService)

                    call_count = 0

                    def mock_receive_json_once(block):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            return (None, sample_request_data)
                        else:
                            return (Exception("Context was terminated"), None)

                    mock_recv_server.receive_json_once.side_effect = mock_receive_json_once

                    # 执行函数
                    await bound_method()

                    # 验证错误处理
                    mock_from_dict.assert_called_once_with(sample_request_data)
                    mock_engine_service._send_error_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_guided_decoding_error(self, mock_engine_service, sample_request_data):
        """测试 guided_decoding_checker 错误的情况"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                    # 模拟 recv_request_server
                    mock_recv_server = Mock()
                    mock_recv_server.receive_json_once.return_value = (None, sample_request_data)
                    mock_engine_service.recv_request_server = mock_recv_server

                    # 模拟 Request.from_dict 返回
                    mock_request = Mock(spec=Request)
                    mock_request.request_id = "test_req_123"
                    mock_request.metrics = RequestMetrics()
                    mock_from_dict.return_value = mock_request

                    # 模拟 guided_decoding_checker
                    mock_checker = Mock()
                    mock_checker.schema_format.return_value = (mock_request, "Schema validation error")
                    mock_engine_service.guided_decoding_checker = mock_checker

                    # 模拟 _send_error_response
                    mock_engine_service._send_error_response = Mock()

                    # 模拟 scheduler.put_requests 返回
                    mock_engine_service.scheduler.put_requests.return_value = []

                    # 模拟 metrics
                    mock_metrics.requests_number = Mock()

                    # 创建真实的函数并调用
                    real_function = EngineService._insert_zmq_task_to_scheduler
                    bound_method = real_function.__get__(mock_engine_service, EngineService)

                    call_count = 0

                    def mock_receive_json_once(block):
                        nonlocal call_count
                        call_count += 1
                        if call_count == 1:
                            return (None, sample_request_data)
                        else:
                            return (Exception("Context was terminated"), None)

                    mock_recv_server.receive_json_once.side_effect = mock_receive_json_once

                    # 执行函数
                    await bound_method()

                    # 验证 guided_decoding_checker 调用
                    mock_checker.schema_format.assert_called_once_with(mock_request)
                    mock_engine_service._send_error_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_fmq_get_exception(self, mock_engine_service):
        """测试 FMQ consumer.get() 异常的情况"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", False):
            with patch("fastdeploy.engine.common_engine.FMQFactory") as mock_fmq_factory:
                # 模拟 FMQ consumer
                mock_consumer = AsyncMock()
                mock_consumer.get.side_effect = Exception("FMQ connection error")
                mock_fmq_factory.q_a2e_consumer.return_value = mock_consumer
                mock_engine_service.fmq_a2e_consumer = mock_consumer

                # 创建真实的函数并调用
                real_function = EngineService._insert_zmq_task_to_scheduler
                bound_method = real_function.__get__(mock_engine_service, EngineService)

                # 执行函数，应该因为异常而退出循环
                await bound_method()

                # 验证 consumer.get 被调用
                mock_consumer.get.assert_called()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_zmq_context_terminated(self, mock_engine_service):
        """测试 ZMQ context 终止的正常关闭情况"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            # 模拟 recv_request_server
            mock_recv_server = Mock()
            mock_recv_server.receive_json_once.return_value = (Exception("Context was terminated"), None)
            mock_engine_service.recv_request_server = mock_recv_server

            # 创建真实的函数并调用
            real_function = EngineService._insert_zmq_task_to_scheduler
            bound_method = real_function.__get__(mock_engine_service, EngineService)

            # 执行函数，应该因为 Context was terminated 而正常退出
            await bound_method()

            # 验证 receive_json_once 被调用
            mock_recv_server.receive_json_once.assert_called()
            # 验证记录了 info 日志
            mock_engine_service.llm_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_multiple_requests(self, mock_engine_service, sample_request_data):
        """测试处理多个请求的情况"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                    # 模拟 recv_request_server
                    mock_recv_server = Mock()
                    mock_engine_service.recv_request_server = mock_recv_server

                    # 模拟 Request.from_dict 返回
                    mock_request = Mock(spec=Request)
                    mock_request.request_id = "test_req_123"
                    mock_request.metrics = RequestMetrics()
                    mock_from_dict.return_value = mock_request

                    # 模拟 scheduler.put_requests 返回
                    mock_engine_service.scheduler.put_requests.return_value = [("test_req_123", None)]

                    # 模拟 metrics
                    mock_metrics.requests_number = Mock()
                    mock_metrics.num_requests_waiting = Mock()

                    # 创建真实的函数并调用
                    real_function = EngineService._insert_zmq_task_to_scheduler
                    bound_method = real_function.__get__(mock_engine_service, EngineService)

                    call_count = 0

                    def mock_receive_json_once(block):
                        nonlocal call_count
                        call_count += 1
                        if call_count <= 2:  # 处理两个请求
                            return (None, sample_request_data)
                        else:
                            return (Exception("Context was terminated"), None)

                    mock_recv_server.receive_json_once.side_effect = mock_receive_json_once

                    # 执行函数
                    await bound_method()

                    # 验证多次调用
                    assert mock_recv_server.receive_json_once.call_count == 3
                    assert mock_from_dict.call_count == 2
                    assert mock_engine_service.scheduler.put_requests.call_count == 2
                    assert mock_metrics.requests_number.inc.call_count == 2
                    assert mock_metrics.num_requests_waiting.inc.call_count == 2

    @pytest.mark.asyncio
    async def test_insert_zmq_task_to_scheduler_block_parameter(self, mock_engine_service, sample_request_data):
        """测试 block 参数的逻辑"""
        with patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True):
            with patch("fastdeploy.engine.common_engine.Request.from_dict") as mock_from_dict:
                with patch("fastdeploy.engine.common_engine.main_process_metrics") as mock_metrics:
                    # 模拟 recv_request_server
                    mock_recv_server = Mock()
                    mock_engine_service.recv_request_server = mock_recv_server

                    # 模拟 Request.from_dict 返回
                    mock_request = Mock(spec=Request)
                    mock_request.request_id = "test_req_123"
                    mock_request.metrics = RequestMetrics()
                    mock_from_dict.return_value = mock_request

                    # 模拟 scheduler.put_requests 返回成功，请求会从 added_requests 中移除
                    mock_engine_service.scheduler.put_requests.return_value = [("test_req_123", None)]
                    mock_engine_service._send_error_response = Mock()

                    # 模拟 metrics
                    mock_metrics.requests_number = Mock()
                    mock_metrics.num_requests_waiting = Mock()

                    # 创建真实的函数并调用
                    real_function = EngineService._insert_zmq_task_to_scheduler
                    bound_method = real_function.__get__(mock_engine_service, EngineService)

                    calls = []

                    def mock_receive_json_once(block):
                        calls.append(block)
                        if len(calls) == 1:
                            # 第一次调用，added_requests 为空，block 应该是 True
                            return (None, sample_request_data)
                        elif len(calls) == 2:
                            # 第二次调用，added_requests 为空（因为第一个请求成功处理并移除），block 应该是 True
                            return (None, sample_request_data)
                        else:
                            return (Exception("Context was terminated"), None)

                    mock_recv_server.receive_json_once.side_effect = mock_receive_json_once

                    # 执行函数
                    await bound_method()

                    # 验证 block 参数的逻辑
                    assert calls[0] is True  # 第一次调用，added_requests 为空
                    assert calls[1] is True  # 第二次调用，added_requests 也为空（第一个请求已成功处理）
