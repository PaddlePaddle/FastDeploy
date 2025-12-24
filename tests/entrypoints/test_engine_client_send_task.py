from unittest.mock import AsyncMock, Mock, patch

import pytest

from fastdeploy.entrypoints.engine_client import EngineClient


class TestEngineClientSendTask:
    """测试 EngineClient._send_task 方法"""

    @pytest.fixture
    def sample_task(self):
        """示例任务数据"""
        return {"request_id": "test_req_123", "prompt": "Hello, world!", "max_tokens": 100}

    @pytest.mark.asyncio
    @patch("fastdeploy.entrypoints.engine_client.api_server_logger")
    @patch("fastdeploy.entrypoints.engine_client.FMQFactory")
    async def test_send_task_success_with_new_producer(self, mock_fmq_factory, mock_logger, sample_task):
        """测试成功发送任务，需要创建新的 producer"""
        # 创建真实的 EngineClient 实例
        client = EngineClient.__new__(EngineClient)
        client.fmq_a2e_producer = None  # 初始为 None

        # 模拟 FMQ producer
        mock_producer = AsyncMock()
        mock_fmq_factory.q_a2e_producer.return_value = mock_producer

        # 执行函数
        await client._send_task(sample_task)

        # 验证 producer 被创建
        mock_fmq_factory.q_a2e_producer.assert_called_once()
        # 验证 producer.put 被调用
        mock_producer.put.assert_called_once_with(sample_task)
        # 验证 fmq_a2e_producer 被设置
        assert client.fmq_a2e_producer == mock_producer
        # 验证没有记录错误日志
        mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    @patch("fastdeploy.entrypoints.engine_client.api_server_logger")
    async def test_send_task_success_with_existing_producer(self, mock_logger, sample_task):
        """测试成功发送任务，使用现有的 producer"""
        # 创建真实的 EngineClient 实例
        client = EngineClient.__new__(EngineClient)

        # 模拟现有的 FMQ producer
        mock_producer = AsyncMock()
        client.fmq_a2e_producer = mock_producer

        # 执行函数
        await client._send_task(sample_task)

        # 验证 producer.put 被调用
        mock_producer.put.assert_called_once_with(sample_task)
        # 验证没有记录错误日志
        mock_logger.error.assert_not_called()

    @pytest.mark.asyncio
    @patch("fastdeploy.entrypoints.engine_client.api_server_logger")
    @patch("fastdeploy.entrypoints.engine_client.FMQFactory")
    async def test_send_task_producer_put_exception(self, mock_fmq_factory, mock_logger, sample_task):
        """测试 producer.put 抛出异常的情况"""
        # 创建真实的 EngineClient 实例
        client = EngineClient.__new__(EngineClient)
        client.fmq_a2e_producer = None

        # 模拟 FMQ producer
        mock_producer = AsyncMock()
        mock_producer.put.side_effect = Exception("Connection failed")
        mock_fmq_factory.q_a2e_producer.return_value = mock_producer

        # 执行函数
        await client._send_task(sample_task)

        # 验证 producer 被创建
        mock_fmq_factory.q_a2e_producer.assert_called_once()
        # 验证 producer.put 被调用
        mock_producer.put.assert_called_once_with(sample_task)
        # 验证异常被捕获并记录错误日志
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "Failed to send task via FMQ: Connection failed" in error_call_args

    @pytest.mark.asyncio
    @patch("fastdeploy.entrypoints.engine_client.api_server_logger")
    @patch("fastdeploy.entrypoints.engine_client.FMQFactory")
    async def test_send_task_get_producer_exception(self, mock_fmq_factory, mock_logger, sample_task):
        """测试 _get_producer 抛出异常的情况"""
        # 创建真实的 EngineClient 实例
        client = EngineClient.__new__(EngineClient)
        client.fmq_a2e_producer = None

        # 模拟 FMQFactory 抛出异常
        mock_fmq_factory.q_a2e_producer.side_effect = Exception("Factory initialization failed")

        # 执行函数
        await client._send_task(sample_task)

        # 验证 FMQFactory 被调用
        mock_fmq_factory.q_a2e_producer.assert_called_once()
        # 验证异常被捕获并记录错误日志
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "Failed to send task via FMQ: Factory initialization failed" in error_call_args

    def test_get_producer_returns_existing(self):
        """测试 _get_producer 方法返回现有 producer"""
        # 创建真实的 EngineClient 实例
        client = EngineClient.__new__(EngineClient)

        # 模拟现有的 FMQ producer
        mock_producer = Mock()
        client.fmq_a2e_producer = mock_producer

        # 执行函数
        result = client._get_producer()

        # 验证返回现有的 producer
        assert result == mock_producer

    @patch("fastdeploy.entrypoints.engine_client.FMQFactory")
    def test_get_producer_creates_new(self, mock_fmq_factory):
        """测试 _get_producer 方法创建新的 producer"""
        # 创建真实的 EngineClient 实例
        client = EngineClient.__new__(EngineClient)
        client.fmq_a2e_producer = None

        # 模拟 FMQ producer
        mock_producer = Mock()
        mock_fmq_factory.q_a2e_producer.return_value = mock_producer

        # 执行函数
        result = client._get_producer()

        # 验证 FMQFactory 被调用
        mock_fmq_factory.q_a2e_producer.assert_called_once()
        # 验证返回新的 producer
        assert result == mock_producer
        # 验证 fmq_a2e_producer 被设置
        assert client.fmq_a2e_producer == mock_producer
