import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from fastdeploy.metrics.metrics_middleware import EXCLUDE_PATHS, PrometheusMiddleware


@pytest.fixture
def mock_request():
    return MagicMock(spec=Request)


@pytest.fixture
def mock_call_next():
    return AsyncMock()


def run_async(coro):
    """同步运行异步函数的辅助函数"""
    return asyncio.run(coro)


def test_dispatch_with_excluded_path(mock_request, mock_call_next):
    """测试排除路径的情况"""
    mock_request.url.path = "/health"  # 使用实际的排除路径
    mock_request.method = "GET"
    mock_call_next.return_value = "response"

    middleware = PrometheusMiddleware(MagicMock())
    result = run_async(middleware.dispatch(mock_request, mock_call_next))

    mock_call_next.assert_called_once_with(mock_request)
    assert result == "response"


def test_dispatch_successful_request(mock_request, mock_call_next):
    """测试成功请求的指标记录"""
    mock_request.url.path = "/test"
    mock_request.method = "POST"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_call_next.return_value = mock_response

    mock_metrics = MagicMock()
    with (
        patch("fastdeploy.metrics.metrics_middleware.main_process_metrics", mock_metrics),
        patch("time.time", side_effect=[1000, 1001.5]),
    ):  # 模拟1.5秒处理时间

        middleware = PrometheusMiddleware(MagicMock())
        result = run_async(middleware.dispatch(mock_request, mock_call_next))

    mock_call_next.assert_called_once_with(mock_request)
    assert result == mock_response

    # 验证指标记录
    mock_metrics.http_requests_total.labels.assert_called_once_with(method="POST", path="/test", status_code=200)
    mock_metrics.http_requests_total.labels().inc.assert_called_once()

    mock_metrics.http_request_duration_seconds.labels.assert_called_once_with(method="POST", path="/test")
    mock_metrics.http_request_duration_seconds.labels().observe.assert_called_once_with(1.5)


def test_dispatch_with_exception(mock_request, mock_call_next):
    """测试请求抛出异常的情况"""
    mock_request.url.path = "/error"
    mock_request.method = "GET"
    mock_call_next.side_effect = Exception("Test error")

    mock_metrics = MagicMock()
    with (
        patch("fastdeploy.metrics.metrics_middleware.main_process_metrics", mock_metrics),
        patch("time.time", side_effect=[1000, 1002]),
    ):  # 模拟2秒处理时间

        middleware = PrometheusMiddleware(MagicMock())
        with pytest.raises(Exception, match="Test error"):
            run_async(middleware.dispatch(mock_request, mock_call_next))

    # 验证即使抛出异常也记录了指标
    mock_metrics.http_requests_total.labels.assert_called_once_with(method="GET", path="/error", status_code=500)
    mock_metrics.http_requests_total.labels().inc.assert_called_once()

    mock_metrics.http_request_duration_seconds.labels.assert_called_once_with(method="GET", path="/error")
    mock_metrics.http_request_duration_seconds.labels().observe.assert_called_once_with(2.0)


def test_all_excluded_paths(mock_request, mock_call_next):
    """测试所有排除路径"""
    mock_call_next.return_value = "response"

    for path in EXCLUDE_PATHS:
        mock_request.url.path = path
        mock_request.method = "GET"

        middleware = PrometheusMiddleware(MagicMock())
        result = run_async(middleware.dispatch(mock_request, mock_call_next))

        assert result == "response"
        mock_call_next.assert_called_with(mock_request)
        mock_call_next.reset_mock()
