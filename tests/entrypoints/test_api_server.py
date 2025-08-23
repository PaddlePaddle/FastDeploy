"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""

import unittest
from unittest.mock import MagicMock, patch

import fastdeploy.entrypoints.api_server as api_server_module


class TestApiServer(unittest.TestCase):
    """Test case for API server functionality"""

    def setUp(self):
        """Set up test environment"""
        # Reset the global llm_engine
        api_server_module.llm_engine = None

    @patch('fastdeploy.entrypoints.api_server.EngineArgs')
    @patch('fastdeploy.entrypoints.api_server.LLMEngine')
    def test_init_app_success(self, mock_llm_engine, mock_engine_args):
        """Test successful app initialization"""
        # Mock the arguments
        mock_args = MagicMock()
        
        # Mock EngineArgs.from_cli_args
        mock_engine_args_instance = MagicMock()
        mock_engine_args.from_cli_args.return_value = mock_engine_args_instance
        
        # Mock LLMEngine.from_engine_args and start()
        mock_engine_instance = MagicMock()
        mock_engine_instance.start.return_value = True
        mock_llm_engine.from_engine_args.return_value = mock_engine_instance
        
        # Call init_app
        result = api_server_module.init_app(mock_args)
        
        # Verify results
        self.assertTrue(result)
        self.assertEqual(api_server_module.llm_engine, mock_engine_instance)
        mock_engine_args.from_cli_args.assert_called_once_with(mock_args)
        mock_llm_engine.from_engine_args.assert_called_once_with(mock_engine_args_instance)
        mock_engine_instance.start.assert_called_once()

    @patch('fastdeploy.entrypoints.api_server.EngineArgs')
    @patch('fastdeploy.entrypoints.api_server.LLMEngine')
    def test_init_app_failure(self, mock_llm_engine, mock_engine_args):
        """Test failed app initialization"""
        # Mock the arguments
        mock_args = MagicMock()
        
        # Mock EngineArgs.from_cli_args
        mock_engine_args_instance = MagicMock()
        mock_engine_args.from_cli_args.return_value = mock_engine_args_instance
        
        # Mock LLMEngine.from_engine_args and start() to fail
        mock_engine_instance = MagicMock()
        mock_engine_instance.start.return_value = False
        mock_llm_engine.from_engine_args.return_value = mock_engine_instance
        
        # Call init_app
        result = api_server_module.init_app(mock_args)
        
        # Verify results
        self.assertFalse(result)
        self.assertEqual(api_server_module.llm_engine, mock_engine_instance)
        mock_engine_instance.start.assert_called_once()

    def test_app_instance_exists(self):
        """Test that FastAPI app instance exists"""
        self.assertIsNotNone(api_server_module.app)
        # Verify it's a FastAPI instance
        self.assertEqual(type(api_server_module.app).__name__, 'FastAPI')

    def test_global_llm_engine_initial_state(self):
        """Test that global llm_engine starts as None"""
        # After setUp, llm_engine should be None
        self.assertIsNone(api_server_module.llm_engine)

    @patch('fastdeploy.entrypoints.api_server.EngineArgs')
    @patch('fastdeploy.entrypoints.api_server.LLMEngine')
    def test_init_app_modifies_global_state(self, mock_llm_engine, mock_engine_args):
        """Test that init_app modifies global llm_engine state"""
        # Setup mocks
        mock_args = MagicMock()
        mock_engine_args_instance = MagicMock()
        mock_engine_args.from_cli_args.return_value = mock_engine_args_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.start.return_value = True
        mock_llm_engine.from_engine_args.return_value = mock_engine_instance
        
        # Verify initial state
        self.assertIsNone(api_server_module.llm_engine)
        
        # Call init_app
        result = api_server_module.init_app(mock_args)
        
        # Verify global state changed
        self.assertTrue(result)
        self.assertIsNotNone(api_server_module.llm_engine)
        self.assertEqual(api_server_module.llm_engine, mock_engine_instance)

    def test_module_imports(self):
        """Test that required modules can be imported"""
        # Verify that the module has the expected attributes
        required_attributes = ['app', 'llm_engine', 'init_app']
        
        for attr in required_attributes:
            self.assertTrue(hasattr(api_server_module, attr), 
                          f"Module missing required attribute: {attr}")

    def test_module_constants(self):
        """Test that module constants are properly defined"""
        # Test that the app instance is properly configured
        self.assertIsNotNone(api_server_module.app)
        
        # Test that llm_engine starts as None (before initialization)
        api_server_module.llm_engine = None  # Reset to ensure test independence
        self.assertIsNone(api_server_module.llm_engine)

    @patch('fastdeploy.entrypoints.api_server.api_server_logger')
    @patch('fastdeploy.entrypoints.api_server.EngineArgs')
    @patch('fastdeploy.entrypoints.api_server.LLMEngine')
    def test_init_app_logging(self, mock_llm_engine, mock_engine_args, mock_logger):
        """Test that init_app logs appropriately"""
        # Setup for successful initialization
        mock_args = MagicMock()
        mock_engine_args_instance = MagicMock()
        mock_engine_args.from_cli_args.return_value = mock_engine_args_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.start.return_value = True
        mock_llm_engine.from_engine_args.return_value = mock_engine_instance
        
        # Call init_app
        result = api_server_module.init_app(mock_args)
        
        # Verify success logging
        self.assertTrue(result)
        mock_logger.info.assert_called_with("FastDeploy LLM engine initialized!")

    @patch('fastdeploy.entrypoints.api_server.api_server_logger')
    @patch('fastdeploy.entrypoints.api_server.EngineArgs')
    @patch('fastdeploy.entrypoints.api_server.LLMEngine')
    def test_init_app_failure_logging(self, mock_llm_engine, mock_engine_args, mock_logger):
        """Test that init_app logs errors on failure"""
        # Setup for failed initialization
        mock_args = MagicMock()
        mock_engine_args_instance = MagicMock()
        mock_engine_args.from_cli_args.return_value = mock_engine_args_instance
        
        mock_engine_instance = MagicMock()
        mock_engine_instance.start.return_value = False
        mock_llm_engine.from_engine_args.return_value = mock_engine_instance
        
        # Call init_app
        result = api_server_module.init_app(mock_args)
        
        # Verify failure logging
        self.assertFalse(result)
        mock_logger.error.assert_called_with("Failed to initialize FastDeploy LLM engine, service exit now!")


class TestApiServerEndpoints(unittest.TestCase):
    """Test case for API server endpoints"""

    def test_health_endpoint_function_exists(self):
        """Test that health endpoint function exists"""
        self.assertTrue(hasattr(api_server_module, 'health'))
        
        # Verify it's callable
        self.assertTrue(callable(api_server_module.health))

    def test_generate_endpoint_function_exists(self):
        """Test that generate endpoint function exists"""
        self.assertTrue(hasattr(api_server_module, 'generate'))
        
        # Verify it's callable
        self.assertTrue(callable(api_server_module.generate))

    def test_fastapi_routes_registered(self):
        """Test that routes are registered with FastAPI app"""
        app = api_server_module.app
        
        # Get all routes
        routes = [route for route in app.routes]
        route_paths = [route.path for route in routes if hasattr(route, 'path')]
        
        # Verify health endpoint exists
        self.assertIn('/health', route_paths)
        
        # Verify generate endpoint exists
        self.assertIn('/generate', route_paths)


if __name__ == "__main__":
    unittest.main()