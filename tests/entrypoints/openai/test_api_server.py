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

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import JSONResponse, Response, StreamingResponse

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelList,
)


class TestAPIServerFunctions(unittest.TestCase):
    """Unit tests for API server utility functions"""

    @patch('fastdeploy.entrypoints.openai.api_server.args')
    @patch('fastdeploy.entrypoints.openai.api_server.is_port_available')
    @patch('fastdeploy.entrypoints.openai.api_server.uvicorn')
    @patch('fastdeploy.entrypoints.openai.api_server.api_server_logger')
    @patch('fastdeploy.entrypoints.openai.api_server.fd_start_span')
    def test_launch_api_server_success(self, mock_fd_start, mock_logger, mock_uvicorn, mock_port_check, mock_args):
        """Test successful API server launch"""
        # Setup mocks
        mock_args.host = "0.0.0.0"
        mock_args.port = 8000
        mock_args.workers = 1
        mock_args.__dict__ = {"host": "0.0.0.0", "port": 8000, "workers": 1}
        mock_port_check.return_value = True
        
        from fastdeploy.entrypoints.openai.api_server import launch_api_server
        
        # Test successful launch
        launch_api_server()
        
        mock_port_check.assert_called_once_with("0.0.0.0", 8000)
        mock_fd_start.assert_called_once_with("FD_START")
        mock_uvicorn.run.assert_called_once()

    @patch('fastdeploy.entrypoints.openai.api_server.args')
    @patch('fastdeploy.entrypoints.openai.api_server.is_port_available')
    def test_launch_api_server_port_in_use(self, mock_port_check, mock_args):
        """Test API server launch with port already in use"""
        mock_args.host = "0.0.0.0"
        mock_args.port = 8000
        mock_port_check.return_value = False
        
        from fastdeploy.entrypoints.openai.api_server import launch_api_server
        
        with self.assertRaises(Exception) as context:
            launch_api_server()
        
        self.assertIn("port:8000 is already in use", str(context.exception))

    @patch('fastdeploy.entrypoints.openai.api_server.args')
    @patch('fastdeploy.entrypoints.openai.api_server.is_port_available')
    @patch('fastdeploy.entrypoints.openai.api_server.cleanup_prometheus_files')
    @patch('fastdeploy.entrypoints.openai.api_server.threading')
    def test_launch_metrics_server_success(self, mock_threading, mock_cleanup, mock_port_check, mock_args):
        """Test successful metrics server launch"""
        mock_args.host = "0.0.0.0"
        mock_args.metrics_port = 9090
        mock_port_check.return_value = True
        mock_cleanup.return_value = "/tmp/prometheus"
        mock_thread = MagicMock()
        mock_threading.Thread.return_value = mock_thread
        
        from fastdeploy.entrypoints.openai.api_server import launch_metrics_server
        
        with patch('time.sleep'):
            launch_metrics_server()
        
        mock_port_check.assert_called_once_with("0.0.0.0", 9090)
        mock_cleanup.assert_called_once_with(True)
        mock_thread.start.assert_called_once()

    def test_wrap_streaming_generator(self):
        """Test streaming generator wrapper"""
        from fastdeploy.entrypoints.openai.api_server import wrap_streaming_generator
        
        # Mock semaphore
        mock_semaphore = MagicMock()
        
        async def mock_generator():
            yield "chunk1"
            yield "chunk2"
        
        with patch('fastdeploy.entrypoints.openai.api_server.connection_semaphore', mock_semaphore):
            wrapped = wrap_streaming_generator(mock_generator())
            
            # Test that generator works and semaphore is released
            async def test_wrapper():
                chunks = []
                async for chunk in wrapped():
                    chunks.append(chunk)
                return chunks
            
            # Run the async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chunks = loop.run_until_complete(test_wrapper())
                self.assertEqual(chunks, ["chunk1", "chunk2"])
                mock_semaphore.release.assert_called_once()
            finally:
                loop.close()


class TestAPIEndpoints(unittest.TestCase):
    """Unit tests for API endpoints"""

    def setUp(self):
        """Set up test environment"""
        self.mock_app = MagicMock()
        self.mock_engine_client = MagicMock()
        self.mock_chat_handler = AsyncMock()
        self.mock_completion_handler = AsyncMock()
        self.mock_model_handler = AsyncMock()
        
        self.mock_app.state.engine_client = self.mock_engine_client
        self.mock_app.state.chat_handler = self.mock_chat_handler
        self.mock_app.state.completion_handler = self.mock_completion_handler
        self.mock_app.state.model_handler = self.mock_model_handler
        self.mock_app.state.dynamic_load_weight = False

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_health_endpoint_success(self, mock_app):
        """Test health endpoint with healthy service"""
        mock_app.state.engine_client.check_health.return_value = (True, "OK")
        mock_app.state.engine_client.is_workers_alive.return_value = (True, "OK")
        
        from fastdeploy.entrypoints.openai.api_server import health
        
        mock_request = MagicMock()
        response = health(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_health_endpoint_engine_unhealthy(self, mock_app):
        """Test health endpoint with unhealthy engine"""
        mock_app.state.engine_client.check_health.return_value = (False, "Engine down")
        
        from fastdeploy.entrypoints.openai.api_server import health
        
        mock_request = MagicMock()
        response = health(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 404)

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_health_endpoint_workers_down(self, mock_app):
        """Test health endpoint with workers down"""
        mock_app.state.engine_client.check_health.return_value = (True, "OK")
        mock_app.state.engine_client.is_workers_alive.return_value = (False, "Workers down")
        
        from fastdeploy.entrypoints.openai.api_server import health
        
        mock_request = MagicMock()
        response = health(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 304)

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_list_routes(self, mock_app):
        """Test list routes endpoint"""
        # Mock routes
        mock_route1 = MagicMock()
        mock_route1.path = "/v1/chat/completions"
        mock_route1.methods = {"POST", "GET"}
        mock_route1.tags = ["chat"]
        
        mock_route2 = MagicMock()
        mock_route2.path = "/health"
        mock_route2.methods = {"GET"}
        
        mock_route3 = MagicMock()
        mock_route3.path = "/v1/models"
        mock_route3.methods = {"GET"}
        mock_route3.tags = []
        
        mock_app.routes = [mock_route1, mock_route2, mock_route3]
        
        from fastdeploy.entrypoints.openai.api_server import list_all_routes
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(list_all_routes())
            
            self.assertIn("routes", result)
            routes = result["routes"]
            
            # Should only include /v1 routes
            self.assertEqual(len(routes), 2)
            
            # Check first route
            route1 = next(r for r in routes if r["path"] == "/v1/chat/completions")
            self.assertEqual(sorted(route1["methods"]), ["GET", "POST"])
            self.assertEqual(route1["tags"], ["chat"])
            
            # Check second route
            route2 = next(r for r in routes if r["path"] == "/v1/models")
            self.assertEqual(route2["methods"], ["GET"])
            self.assertEqual(route2["tags"], [])
        finally:
            loop.close()

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_ping_endpoint(self, mock_app):
        """Test ping endpoint delegates to health"""
        mock_app.state.engine_client.check_health.return_value = (True, "OK")
        mock_app.state.engine_client.is_workers_alive.return_value = (True, "OK")
        
        from fastdeploy.entrypoints.openai.api_server import ping
        
        mock_request = MagicMock()
        response = ping(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_update_model_weight_enabled(self, mock_app):
        """Test update model weight with dynamic loading enabled"""
        mock_app.state.dynamic_load_weight = True
        mock_app.state.engine_client.update_model_weight.return_value = (True, "Success")
        
        from fastdeploy.entrypoints.openai.api_server import update_model_weight
        
        mock_request = MagicMock()
        response = update_model_weight(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_update_model_weight_disabled(self, mock_app):
        """Test update model weight with dynamic loading disabled"""
        mock_app.state.dynamic_load_weight = False
        
        from fastdeploy.entrypoints.openai.api_server import update_model_weight
        
        mock_request = MagicMock()
        response = update_model_weight(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 404)
        self.assertIn("Dynamic Load Weight Disabled", response.body.decode())

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_clear_load_weight_success(self, mock_app):
        """Test clear model weight successfully"""
        mock_app.state.dynamic_load_weight = True
        mock_app.state.engine_client.clear_load_weight.return_value = (True, "Success")
        
        from fastdeploy.entrypoints.openai.api_server import clear_load_weight
        
        mock_request = MagicMock()
        response = clear_load_weight(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)

    @patch('fastdeploy.entrypoints.openai.api_server.app')
    def test_clear_load_weight_failed(self, mock_app):
        """Test clear model weight failure"""
        mock_app.state.dynamic_load_weight = True
        mock_app.state.engine_client.clear_load_weight.return_value = (False, "Failed")
        
        from fastdeploy.entrypoints.openai.api_server import clear_load_weight
        
        mock_request = MagicMock()
        response = clear_load_weight(mock_request)
        
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()